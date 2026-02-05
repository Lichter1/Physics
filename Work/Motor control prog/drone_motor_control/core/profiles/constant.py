"""
Constant PWM profile.
"""
from typing import Dict, Any, List, Tuple
from .base_profile import BaseProfile, ProfilePoint


class ConstantProfile(BaseProfile):
    """
    Maintains a constant PWM value for a specified duration.
    """

    @property
    def name(self) -> str:
        return "Constant PWM"

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "pwm": {
                "type": "int",
                "min": 1000,
                "max": 2000,
                "default": 1500,
                "label": "PWM Value"
            },
            "duration_sec": {
                "type": "float",
                "min": 0.1,
                "max": 300,
                "default": 10,
                "label": "Duration (sec)"
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        pwm = params.get("pwm")
        duration = params.get("duration_sec")

        if pwm is None:
            return False, "PWM value is required"
        if duration is None:
            return False, "Duration is required"

        try:
            pwm = int(pwm)
            duration = float(duration)
        except (ValueError, TypeError) as e:
            return False, f"Invalid parameter type: {e}"

        if not 1000 <= pwm <= 2000:
            return False, f"PWM must be between 1000 and 2000, got {pwm}"
        if not 0.1 <= duration <= 300:
            return False, f"Duration must be between 0.1 and 300 seconds, got {duration}"

        return True, ""

    def generate(self, params: Dict[str, Any], resolution_ms: int = 50) -> List[ProfilePoint]:
        valid, error = self.validate_params(params)
        if not valid:
            raise ValueError(error)

        pwm = int(params["pwm"])
        duration = float(params["duration_sec"])
        resolution_sec = resolution_ms / 1000.0

        points = []
        t = 0.0
        while t <= duration:
            points.append(ProfilePoint(time_sec=round(t, 3), pwm=pwm))
            t += resolution_sec

        # Ensure final point is exactly at duration
        if points and points[-1].time_sec < duration:
            points.append(ProfilePoint(time_sec=duration, pwm=pwm))

        return points

    def get_duration(self, params: Dict[str, Any]) -> float:
        return float(params.get("duration_sec", 0))
