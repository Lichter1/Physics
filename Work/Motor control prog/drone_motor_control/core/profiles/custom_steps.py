"""
Custom step sequence PWM profile.
"""
from typing import Dict, Any, List, Tuple
from .base_profile import BaseProfile, ProfilePoint


class CustomStepsProfile(BaseProfile):
    """
    Executes a sequence of user-defined PWM steps.
    Based on mode 3 from the existing motor control script.
    """

    @property
    def name(self) -> str:
        return "Custom Steps"

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "steps": {
                "type": "array",
                "label": "Step Sequence",
                "items": {
                    "pwm": {
                        "type": "int",
                        "min": 1000,
                        "max": 2000,
                        "label": "PWM"
                    },
                    "duration_sec": {
                        "type": "float",
                        "min": 0.1,
                        "max": 60,
                        "label": "Duration (sec)"
                    }
                },
                "min_items": 1,
                "max_items": 20
            },
            "loop": {
                "type": "bool",
                "default": False,
                "label": "Loop Forever"
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        steps = params.get("steps")

        if steps is None or not isinstance(steps, list):
            return False, "Steps array is required"

        if len(steps) == 0:
            return False, "At least one step is required"

        if len(steps) > 20:
            return False, f"Maximum 20 steps allowed, got {len(steps)}"

        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                return False, f"Step {i+1} must be an object"

            pwm = step.get("pwm")
            duration = step.get("duration_sec")

            if pwm is None:
                return False, f"Step {i+1}: PWM is required"
            if duration is None:
                return False, f"Step {i+1}: Duration is required"

            try:
                pwm = int(pwm)
                duration = float(duration)
            except (ValueError, TypeError) as e:
                return False, f"Step {i+1}: Invalid parameter type: {e}"

            if not 1000 <= pwm <= 2000:
                return False, f"Step {i+1}: PWM must be between 1000 and 2000, got {pwm}"
            if not 0.1 <= duration <= 60:
                return False, f"Step {i+1}: Duration must be between 0.1 and 60 seconds, got {duration}"

        return True, ""

    def generate(self, params: Dict[str, Any], resolution_ms: int = 50) -> List[ProfilePoint]:
        """
        Generate profile for one iteration of the step sequence.
        Loop handling is done at execution time, not generation time.
        """
        valid, error = self.validate_params(params)
        if not valid:
            raise ValueError(error)

        steps = params["steps"]
        resolution_sec = resolution_ms / 1000.0

        points = []
        current_time = 0.0

        for step in steps:
            pwm = int(step["pwm"])
            duration = float(step["duration_sec"])
            step_end = current_time + duration

            # Generate points for this step
            t = current_time
            while t < step_end:
                points.append(ProfilePoint(time_sec=round(t, 3), pwm=pwm))
                t += resolution_sec

            current_time = step_end

        # Add final point
        if steps:
            last_pwm = int(steps[-1]["pwm"])
            points.append(ProfilePoint(time_sec=round(current_time, 3), pwm=last_pwm))

        return points

    def get_duration(self, params: Dict[str, Any]) -> float:
        steps = params.get("steps", [])
        total = 0.0
        for step in steps:
            total += float(step.get("duration_sec", 0))
        return total

    def is_looping(self, params: Dict[str, Any]) -> bool:
        """Check if this profile should loop."""
        return bool(params.get("loop", False))
