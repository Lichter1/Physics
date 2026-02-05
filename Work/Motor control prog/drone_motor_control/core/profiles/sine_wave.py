"""
Sine wave PWM profile.
"""
import math
from typing import Dict, Any, List, Tuple
from .base_profile import BaseProfile, ProfilePoint


class SineWaveProfile(BaseProfile):
    """
    Generates a sinusoidal PWM pattern.
    PWM(t) = pwm_center + amplitude * sin(2 * pi * frequency * t)
    """

    @property
    def name(self) -> str:
        return "Sine Wave"

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "pwm_center": {
                "type": "int",
                "min": 1000,
                "max": 2000,
                "default": 1500,
                "label": "Center PWM"
            },
            "amplitude": {
                "type": "int",
                "min": 50,
                "max": 500,
                "default": 200,
                "label": "Amplitude"
            },
            "frequency_hz": {
                "type": "float",
                "min": 0.01,
                "max": 5,
                "default": 0.5,
                "label": "Frequency (Hz)"
            },
            "cycles": {
                "type": "int",
                "min": 1,
                "max": 100,
                "default": 5,
                "label": "Number of Cycles"
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        pwm_center = params.get("pwm_center")
        amplitude = params.get("amplitude")
        frequency = params.get("frequency_hz")
        cycles = params.get("cycles")

        if pwm_center is None:
            return False, "Center PWM is required"
        if amplitude is None:
            return False, "Amplitude is required"
        if frequency is None:
            return False, "Frequency is required"
        if cycles is None:
            return False, "Number of cycles is required"

        try:
            pwm_center = int(pwm_center)
            amplitude = int(amplitude)
            frequency = float(frequency)
            cycles = int(cycles)
        except (ValueError, TypeError) as e:
            return False, f"Invalid parameter type: {e}"

        if not 1000 <= pwm_center <= 2000:
            return False, f"Center PWM must be between 1000 and 2000, got {pwm_center}"
        if amplitude < 50 or amplitude > 500:
            return False, f"Amplitude must be between 50 and 500, got {amplitude}"
        if frequency <= 0 or frequency > 5:
            return False, f"Frequency must be between 0.01 and 5 Hz, got {frequency}"
        if cycles < 1 or cycles > 100:
            return False, f"Cycles must be between 1 and 100, got {cycles}"

        # Check that PWM stays in valid range
        pwm_min = pwm_center - amplitude
        pwm_max = pwm_center + amplitude
        if pwm_min < 1000:
            return False, f"PWM would go below 1000 (min: {pwm_min}). Reduce amplitude or increase center."
        if pwm_max > 2000:
            return False, f"PWM would exceed 2000 (max: {pwm_max}). Reduce amplitude or decrease center."

        return True, ""

    def generate(self, params: Dict[str, Any], resolution_ms: int = 50) -> List[ProfilePoint]:
        valid, error = self.validate_params(params)
        if not valid:
            raise ValueError(error)

        pwm_center = int(params["pwm_center"])
        amplitude = int(params["amplitude"])
        frequency = float(params["frequency_hz"])
        cycles = int(params["cycles"])
        resolution_sec = resolution_ms / 1000.0

        # Calculate duration
        duration = cycles / frequency

        points = []
        t = 0.0
        while t <= duration:
            # Calculate PWM at this time
            pwm = pwm_center + int(amplitude * math.sin(2 * math.pi * frequency * t))
            # Clamp to valid range (should be unnecessary if validation passed)
            pwm = max(1000, min(2000, pwm))
            points.append(ProfilePoint(time_sec=round(t, 3), pwm=pwm))
            t += resolution_sec

        return points

    def get_duration(self, params: Dict[str, Any]) -> float:
        frequency = float(params.get("frequency_hz", 0.5))
        cycles = int(params.get("cycles", 5))
        if frequency <= 0:
            return 0.0
        return cycles / frequency
