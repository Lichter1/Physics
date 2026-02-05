"""
Linear ramp PWM profile.
"""
from typing import Dict, Any, List, Tuple
from .base_profile import BaseProfile, ProfilePoint


class LinearRampProfile(BaseProfile):
    """
    Linearly ramps PWM from start to end value at a specified rate.
    Based on mode 2 from the existing motor control script.
    """

    @property
    def name(self) -> str:
        return "Linear Ramp"

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "pwm_start": {
                "type": "int",
                "min": 1000,
                "max": 2000,
                "default": 1200,
                "label": "Start PWM"
            },
            "pwm_end": {
                "type": "int",
                "min": 1000,
                "max": 2000,
                "default": 1800,
                "label": "End PWM"
            },
            "rate": {
                "type": "float",
                "min": 1,
                "max": 500,
                "default": 50,
                "label": "Rate (PWM/sec)"
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        pwm_start = params.get("pwm_start")
        pwm_end = params.get("pwm_end")
        rate = params.get("rate")

        if pwm_start is None:
            return False, "Start PWM is required"
        if pwm_end is None:
            return False, "End PWM is required"
        if rate is None:
            return False, "Rate is required"

        try:
            pwm_start = int(pwm_start)
            pwm_end = int(pwm_end)
            rate = float(rate)
        except (ValueError, TypeError) as e:
            return False, f"Invalid parameter type: {e}"

        if not 1000 <= pwm_start <= 2000:
            return False, f"Start PWM must be between 1000 and 2000, got {pwm_start}"
        if not 1000 <= pwm_end <= 2000:
            return False, f"End PWM must be between 1000 and 2000, got {pwm_end}"
        if pwm_start == pwm_end:
            return False, "Start and end PWM must be different"
        if rate <= 0:
            return False, f"Rate must be positive, got {rate}"

        return True, ""

    def generate(self, params: Dict[str, Any], resolution_ms: int = 50) -> List[ProfilePoint]:
        valid, error = self.validate_params(params)
        if not valid:
            raise ValueError(error)

        pwm_start = int(params["pwm_start"])
        pwm_end = int(params["pwm_end"])
        rate = float(params["rate"])
        resolution_sec = resolution_ms / 1000.0

        # Calculate duration
        duration = abs(pwm_end - pwm_start) / rate
        direction = 1 if pwm_end > pwm_start else -1

        points = []
        t = 0.0
        while t <= duration:
            # Calculate PWM at this time
            pwm = pwm_start + direction * int(rate * t)
            # Clamp to range
            pwm = max(min(pwm, max(pwm_start, pwm_end)), min(pwm_start, pwm_end))
            points.append(ProfilePoint(time_sec=round(t, 3), pwm=pwm))
            t += resolution_sec

        # Ensure final point is exactly at end PWM
        if points and points[-1].pwm != pwm_end:
            points.append(ProfilePoint(time_sec=round(duration, 3), pwm=pwm_end))

        return points

    def get_duration(self, params: Dict[str, Any]) -> float:
        pwm_start = int(params.get("pwm_start", 1200))
        pwm_end = int(params.get("pwm_end", 1800))
        rate = float(params.get("rate", 50))
        if rate <= 0:
            return 0.0
        return abs(pwm_end - pwm_start) / rate
