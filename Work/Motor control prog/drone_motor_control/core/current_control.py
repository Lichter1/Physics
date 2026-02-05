"""
Current control module for I(PWM) calibration and current-based motor control.
"""
import json
import os
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict

try:
    import numpy as np
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class CalibrationPoint:
    """Single I(PWM) measurement point."""
    pwm: int
    current_amps: float
    timestamp: str  # ISO format


@dataclass
class CalibrationCurve:
    """Stores a calibration relationship."""
    motor_id: int
    created_at: str  # ISO format
    points: List[CalibrationPoint]
    pwm_min: int
    pwm_max: int


class CurrentControlModule:
    """
    Provides current-based motor control through I(PWM) calibration.

    Workflow:
    1. Run linear PWM sweep (calibration run)
    2. Parse ESC telemetry log to extract I(PWM) data
    3. Generate and store calibration curve
    4. Use calibration for current-to-PWM lookup
    """

    # Regex for parsing ESC telemetry lines
    # Format: 2026-02-04 09:17:43,309 esc4:{'throttle_in': 1000.0, ..., 'current': 0.0, ...}
    ESC_LINE_PATTERN = re.compile(
        r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) esc(\d+):\{(.+)\}$'
    )

    def __init__(self, calibration_dir: str = "data/calibration/",
                 log_manager=None):
        """
        Initialize CurrentControlModule.

        Args:
            calibration_dir: Directory to store calibration curves
            log_manager: LogManager instance for log discovery
        """
        self.calibration_dir = calibration_dir
        self.log_manager = log_manager
        self._calibrations: Dict[int, CalibrationCurve] = {}
        self._interpolators: Dict[int, Any] = {}  # scipy interpolators

        # Load existing calibrations
        self._load_all_calibrations()

    def _load_all_calibrations(self):
        """Load all saved calibration curves."""
        if not os.path.exists(self.calibration_dir):
            return

        for filename in os.listdir(self.calibration_dir):
            if filename.startswith('motor_') and filename.endswith('_calibration.json'):
                try:
                    motor_id = int(filename.split('_')[1])
                    curve = self.load_calibration(motor_id)
                    if curve:
                        self._calibrations[motor_id] = curve
                        self._build_interpolator(motor_id)
                except (ValueError, IndexError):
                    continue

    def generate_calibration_profile(self,
                                      pwm_min: int = 1000,
                                      pwm_max: int = 1800,
                                      rate: float = 20) -> Dict[str, Any]:
        """
        Generate a linear ramp profile for calibration.

        Args:
            pwm_min: Starting PWM (typically 1000)
            pwm_max: Ending PWM (safe max for calibration)
            rate: PWM change rate per second (slow for accurate readings)

        Returns:
            Profile configuration dict for execution
        """
        return {
            "profile": "Linear Ramp",
            "params": {
                "pwm_start": pwm_min,
                "pwm_end": pwm_max,
                "rate": rate
            }
        }

    def parse_esc_telemetry_line(self, line: str) -> Optional[Tuple[datetime, int, Dict]]:
        """
        Parse a single ESC telemetry line.

        Args:
            line: Line from ESC telemetry log

        Returns:
            Tuple of (timestamp, motor_id, data_dict) or None if not a valid ESC line
        """
        match = self.ESC_LINE_PATTERN.match(line.strip())
        if not match:
            return None

        timestamp_str, motor_id_str, data_str = match.groups()

        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
            motor_id = int(motor_id_str)

            # Parse the dict string (it's Python literal format)
            # Convert to valid JSON by replacing single quotes with double quotes
            data_str = data_str.replace("'", '"')
            data = json.loads('{' + data_str + '}')

            return timestamp, motor_id, data
        except (ValueError, json.JSONDecodeError):
            return None

    def parse_calibration_run(self,
                              esc_telemetry_path: str,
                              start_time: datetime,
                              end_time: datetime,
                              motor_id: int) -> Optional[CalibrationCurve]:
        """
        Parse ESC telemetry log to extract I(PWM) relationship.

        Args:
            esc_telemetry_path: Path to ESC telemetry log
            start_time: Calibration run start time
            end_time: Calibration run end time
            motor_id: Which motor (1-4) to extract

        Returns:
            CalibrationCurve with I(PWM) data, or None on error
        """
        points = []
        seen_pwm = set()  # Track unique PWM values

        try:
            with open(esc_telemetry_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    result = self.parse_esc_telemetry_line(line)
                    if result is None:
                        continue

                    timestamp, esc_id, data = result

                    # Filter by time range
                    if timestamp < start_time or timestamp > end_time:
                        continue

                    # Filter by motor ID
                    if esc_id != motor_id:
                        continue

                    # Extract PWM and current
                    throttle_in = data.get('throttle_in')
                    current = data.get('current')

                    if throttle_in is None or current is None:
                        continue

                    pwm = int(throttle_in)
                    current_amps = float(current)

                    # Only add unique PWM values (average if duplicates)
                    if pwm not in seen_pwm:
                        seen_pwm.add(pwm)
                        points.append(CalibrationPoint(
                            pwm=pwm,
                            current_amps=current_amps,
                            timestamp=timestamp.isoformat()
                        ))

        except IOError as e:
            print(f"Error reading ESC telemetry: {e}")
            return None

        if not points:
            print("No calibration points found in time range")
            return None

        # Sort by PWM
        points.sort(key=lambda p: p.pwm)

        # Create calibration curve
        curve = CalibrationCurve(
            motor_id=motor_id,
            created_at=datetime.now().isoformat(),
            points=points,
            pwm_min=points[0].pwm,
            pwm_max=points[-1].pwm
        )

        return curve

    def _build_interpolator(self, motor_id: int):
        """Build scipy interpolator for a calibration curve."""
        if not HAS_SCIPY:
            return

        curve = self._calibrations.get(motor_id)
        if not curve or not curve.points:
            return

        pwm_values = [p.pwm for p in curve.points]
        current_values = [p.current_amps for p in curve.points]

        if len(pwm_values) < 2:
            return

        try:
            # Create interpolator for PWM -> Current
            self._interpolators[motor_id] = {
                'pwm_to_current': interp1d(
                    pwm_values, current_values,
                    kind='linear',
                    fill_value='extrapolate'
                ),
                'pwm_values': np.array(pwm_values),
                'current_values': np.array(current_values)
            }
        except Exception as e:
            print(f"Error building interpolator for motor {motor_id}: {e}")

    def fit_curve(self, curve: CalibrationCurve) -> CalibrationCurve:
        """
        Store calibration curve and build interpolator.

        Args:
            curve: Calibration curve to fit

        Returns:
            The same curve (for chaining)
        """
        self._calibrations[curve.motor_id] = curve
        self._build_interpolator(curve.motor_id)
        return curve

    def current_to_pwm(self, motor_id: int, target_current: float) -> Optional[int]:
        """
        Look up PWM value for a target current.

        Uses inverse interpolation on the I(PWM) curve.

        Args:
            motor_id: Motor to look up (1-4)
            target_current: Desired current in amps

        Returns:
            PWM value to achieve target current, or None if out of range
        """
        if not HAS_SCIPY:
            return self._current_to_pwm_simple(motor_id, target_current)

        interp_data = self._interpolators.get(motor_id)
        if not interp_data:
            return None

        pwm_values = interp_data['pwm_values']
        current_values = interp_data['current_values']

        # Check if target is in range
        if target_current < current_values.min() or target_current > current_values.max():
            print(f"Target current {target_current}A is outside calibration range "
                  f"[{current_values.min():.2f}, {current_values.max():.2f}]")
            return None

        # Find PWM for target current using inverse lookup
        # Since current generally increases with PWM, we can search for the crossing point
        try:
            # Find indices where current crosses target
            for i in range(len(current_values) - 1):
                if (current_values[i] <= target_current <= current_values[i + 1] or
                    current_values[i] >= target_current >= current_values[i + 1]):
                    # Linear interpolation between points
                    if current_values[i + 1] != current_values[i]:
                        ratio = (target_current - current_values[i]) / (current_values[i + 1] - current_values[i])
                        pwm = pwm_values[i] + ratio * (pwm_values[i + 1] - pwm_values[i])
                        return int(round(pwm))

            return None
        except Exception as e:
            print(f"Error in current_to_pwm lookup: {e}")
            return None

    def _current_to_pwm_simple(self, motor_id: int, target_current: float) -> Optional[int]:
        """Simple linear search fallback when scipy is not available."""
        curve = self._calibrations.get(motor_id)
        if not curve or not curve.points:
            return None

        points = sorted(curve.points, key=lambda p: p.pwm)

        # Find closest points
        for i in range(len(points) - 1):
            if (points[i].current_amps <= target_current <= points[i + 1].current_amps or
                points[i].current_amps >= target_current >= points[i + 1].current_amps):
                # Linear interpolation
                diff_current = points[i + 1].current_amps - points[i].current_amps
                if diff_current != 0:
                    ratio = (target_current - points[i].current_amps) / diff_current
                    pwm = points[i].pwm + ratio * (points[i + 1].pwm - points[i].pwm)
                    return int(round(pwm))

        return None

    def pwm_to_current(self, motor_id: int, pwm: int) -> Optional[float]:
        """
        Look up expected current for a PWM value.

        Args:
            motor_id: Motor to look up (1-4)
            pwm: PWM value

        Returns:
            Expected current in amps, or None if not calibrated
        """
        if not HAS_SCIPY:
            return self._pwm_to_current_simple(motor_id, pwm)

        interp_data = self._interpolators.get(motor_id)
        if not interp_data:
            return None

        try:
            current = float(interp_data['pwm_to_current'](pwm))
            return max(0, current)  # Current can't be negative
        except Exception:
            return None

    def _pwm_to_current_simple(self, motor_id: int, pwm: int) -> Optional[float]:
        """Simple linear search fallback."""
        curve = self._calibrations.get(motor_id)
        if not curve or not curve.points:
            return None

        points = sorted(curve.points, key=lambda p: p.pwm)

        for i in range(len(points) - 1):
            if points[i].pwm <= pwm <= points[i + 1].pwm:
                diff_pwm = points[i + 1].pwm - points[i].pwm
                if diff_pwm != 0:
                    ratio = (pwm - points[i].pwm) / diff_pwm
                    current = points[i].current_amps + ratio * (points[i + 1].current_amps - points[i].current_amps)
                    return max(0, current)

        return None

    def save_calibration(self, curve: CalibrationCurve) -> str:
        """
        Save calibration curve to disk.

        Args:
            curve: Calibration curve to save

        Returns:
            Path to saved file
        """
        os.makedirs(self.calibration_dir, exist_ok=True)

        filename = f"motor_{curve.motor_id}_calibration.json"
        filepath = os.path.join(self.calibration_dir, filename)

        data = {
            'motor_id': curve.motor_id,
            'created_at': curve.created_at,
            'pwm_min': curve.pwm_min,
            'pwm_max': curve.pwm_max,
            'points': [asdict(p) for p in curve.points]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        return filepath

    def load_calibration(self, motor_id: int) -> Optional[CalibrationCurve]:
        """
        Load calibration for a motor.

        Args:
            motor_id: Motor ID to load

        Returns:
            CalibrationCurve or None if not found
        """
        filename = f"motor_{motor_id}_calibration.json"
        filepath = os.path.join(self.calibration_dir, filename)

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            points = [
                CalibrationPoint(
                    pwm=p['pwm'],
                    current_amps=p['current_amps'],
                    timestamp=p['timestamp']
                )
                for p in data['points']
            ]

            return CalibrationCurve(
                motor_id=data['motor_id'],
                created_at=data['created_at'],
                points=points,
                pwm_min=data['pwm_min'],
                pwm_max=data['pwm_max']
            )
        except (IOError, json.JSONDecodeError, KeyError) as e:
            print(f"Error loading calibration for motor {motor_id}: {e}")
            return None

    def get_calibration_plot_data(self, motor_id: int) -> Optional[Dict[str, Any]]:
        """
        Return data for plotting I(PWM) curve in frontend.

        Args:
            motor_id: Motor to get data for

        Returns:
            Dict with 'pwm' and 'current' arrays, or None if not calibrated
        """
        curve = self._calibrations.get(motor_id)
        if not curve or not curve.points:
            return None

        points = sorted(curve.points, key=lambda p: p.pwm)

        return {
            'motor_id': motor_id,
            'created_at': curve.created_at,
            'pwm': [p.pwm for p in points],
            'current': [p.current_amps for p in points],
            'pwm_min': curve.pwm_min,
            'pwm_max': curve.pwm_max
        }

    def list_calibrations(self) -> List[Dict[str, Any]]:
        """
        List all available calibrations.

        Returns:
            List of calibration info dicts
        """
        return [
            {
                'motor_id': curve.motor_id,
                'created_at': curve.created_at,
                'pwm_min': curve.pwm_min,
                'pwm_max': curve.pwm_max,
                'num_points': len(curve.points)
            }
            for curve in self._calibrations.values()
        ]

    def has_calibration(self, motor_id: int) -> bool:
        """Check if a motor has calibration data."""
        return motor_id in self._calibrations
