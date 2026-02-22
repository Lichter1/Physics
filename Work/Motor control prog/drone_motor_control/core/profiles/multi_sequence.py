"""
Multi-sequence profile that composes multiple profiles together.
"""
from typing import Dict, Any, List, Tuple
from .base_profile import BaseProfile, ProfilePoint


class MultiSequenceProfile(BaseProfile):
    """
    Executes multiple profile sequences in order with optional looping.

    This is a meta-profile that composes other profiles together, allowing
    users to create complex multi-step profiles like "ramp up → hold → ramp down".
    """

    def __init__(self, profile_engine):
        """
        Initialize multi-sequence profile with reference to profile engine.

        Args:
            profile_engine: ProfileEngine instance to look up other profiles
        """
        self._profile_engine = profile_engine

    @property
    def name(self) -> str:
        return "Multi-Sequence"

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Multi-sequence parameters.

        Note: The 'sequences' parameter has a special type that will be handled
        by custom UI in the frontend.
        """
        return {
            "sequences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "profile": {"type": "string"},
                        "params": {"type": "object"}
                    }
                },
                "label": "Sequence Chain",
                "default": []
            },
            "loop_count": {
                "type": "int",
                "min": 0,
                "default": 0,
                "label": "Loop Count (0=no loop, -1=infinite)"
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate multi-sequence parameters.

        Checks:
        - sequences array exists and is not empty
        - each sequence has valid profile name
        - each sequence's params are valid for its profile
        - loop_count is valid
        """
        sequences = params.get("sequences")
        if not sequences:
            return False, "At least one sequence is required"

        if not isinstance(sequences, list):
            return False, "Sequences must be an array"

        if len(sequences) == 0:
            return False, "Sequences array cannot be empty"

        # Validate each sequence
        for i, seq in enumerate(sequences):
            # Check structure
            if not isinstance(seq, dict):
                return False, f"Sequence {i + 1} must be an object"

            profile_name = seq.get("profile")
            if not profile_name:
                return False, f"Sequence {i + 1} missing profile name"

            seq_params = seq.get("params", {})
            if not isinstance(seq_params, dict):
                return False, f"Sequence {i + 1} params must be an object"

            # Get profile instance
            profile = self._profile_engine.get_profile(profile_name)
            if not profile:
                return False, f"Sequence {i + 1}: Unknown profile '{profile_name}'"

            # Validate sequence parameters
            valid, error = profile.validate_params(seq_params)
            if not valid:
                return False, f"Sequence {i + 1} ({profile_name}): {error}"

        # Validate loop_count
        loop_count = params.get("loop_count", 0)
        try:
            loop_count = int(loop_count)
        except (ValueError, TypeError):
            return False, "Loop count must be an integer"

        if loop_count < -1:
            return False, "Loop count must be >= -1 (0=no loop, -1=infinite)"

        return True, ""

    def generate(self, params: Dict[str, Any], resolution_ms: int = 50) -> List[ProfilePoint]:
        """
        Generate combined profile points for ONE iteration.

        The loop handling is done by the execution engine, not here.
        This method generates a single pass through all sequences.

        Args:
            params: Must contain 'sequences' array
            resolution_ms: Time resolution in milliseconds

        Returns:
            List of ProfilePoint objects for one iteration
        """
        # Validate first
        valid, error = self.validate_params(params)
        if not valid:
            raise ValueError(error)

        sequences = params["sequences"]
        all_points = []
        time_offset = 0.0

        for i, seq in enumerate(sequences):
            profile_name = seq["profile"]
            seq_params = seq["params"]

            # Get the profile instance
            profile = self._profile_engine.get_profile(profile_name)
            if not profile:
                raise ValueError(f"Unknown profile: {profile_name}")

            # Generate points for this sequence
            try:
                points = profile.generate(seq_params, resolution_ms)
            except Exception as e:
                raise ValueError(f"Sequence {i + 1} ({profile_name}) generation failed: {e}")

            if not points:
                raise ValueError(f"Sequence {i + 1} ({profile_name}) generated no points")

            # Offset time by accumulated duration
            for point in points:
                all_points.append(ProfilePoint(
                    time_sec=round(point.time_sec + time_offset, 3),
                    pwm=point.pwm
                ))

            # Update time offset for next sequence
            time_offset += points[-1].time_sec

        return all_points

    def get_duration(self, params: Dict[str, Any]) -> float:
        """
        Calculate total duration for ONE iteration.

        Returns sum of all sequence durations.
        """
        sequences = params.get("sequences", [])
        total_duration = 0.0

        for seq in sequences:
            profile_name = seq.get("profile")
            seq_params = seq.get("params", {})

            profile = self._profile_engine.get_profile(profile_name)
            if profile:
                try:
                    total_duration += profile.get_duration(seq_params)
                except Exception:
                    # If duration calculation fails, skip this sequence
                    pass

        return total_duration

    def get_sequence_boundaries(self, params: Dict[str, Any]) -> Tuple[List[float], List[str]]:
        """
        Get time boundaries and labels for each sequence.

        Returns:
            Tuple of (boundaries, labels)
            - boundaries: List of time points where sequences start
            - labels: List of human-readable labels for each sequence
        """
        sequences = params.get("sequences", [])
        boundaries = []
        labels = []
        time_offset = 0.0

        for i, seq in enumerate(sequences):
            profile_name = seq.get("profile")
            seq_params = seq.get("params", {})

            boundaries.append(time_offset)
            labels.append(f"Seq {i + 1}: {profile_name}")

            profile = self._profile_engine.get_profile(profile_name)
            if profile:
                try:
                    time_offset += profile.get_duration(seq_params)
                except Exception:
                    pass

        return boundaries, labels
