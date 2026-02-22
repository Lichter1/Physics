"""
Profile engine for managing and executing motion profiles.
"""
import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict

from .profiles import (
    BaseProfile,
    ProfilePoint,
    ConstantProfile,
    LinearRampProfile,
    SineWaveProfile,
    CustomStepsProfile,
    MultiSequenceProfile
)


@dataclass
class ExecutionState:
    """Current execution state."""
    running: bool
    elapsed_sec: float
    total_sec: float
    current_pwm: int
    progress_pct: float
    session_id: str
    error: Optional[str] = None
    failed_motors: Optional[List[int]] = None
    current_iteration: Optional[int] = None
    total_iterations: Optional[int] = None


class ProfileEngine:
    """
    Central registry and execution engine for motion profiles.
    """

    def __init__(self):
        self._profiles: Dict[str, BaseProfile] = {}
        self._running = False
        self._abort_requested = False
        self._current_session_id: Optional[str] = None
        self._register_built_in_profiles()

    def _register_built_in_profiles(self):
        """Register all built-in profile types."""
        self.register(ConstantProfile())
        self.register(LinearRampProfile())
        self.register(SineWaveProfile())
        self.register(CustomStepsProfile())
        # Multi-sequence needs reference to engine for composing other profiles
        self.register(MultiSequenceProfile(self))

    def register(self, profile: BaseProfile):
        """Register a new profile type."""
        self._profiles[profile.name] = profile

    def list_profiles(self) -> List[Dict[str, Any]]:
        """Return list of available profiles with their parameter schemas."""
        return [p.to_dict() for p in self._profiles.values()]

    def get_profile(self, profile_name: str) -> Optional[BaseProfile]:
        """Get a profile by name."""
        return self._profiles.get(profile_name)

    def generate_preview(self, profile_name: str, params: Dict[str, Any],
                         resolution_ms: int = 50) -> Dict[str, Any]:
        """
        Generate preview data for charting.

        Args:
            profile_name: Name of the profile
            params: Profile parameters
            resolution_ms: Time resolution

        Returns:
            Dict with 'data' (list of points) and 'duration' (total seconds)
        """
        profile = self._profiles.get(profile_name)
        if not profile:
            raise ValueError(f"Unknown profile: {profile_name}")

        # Validate parameters
        valid, error = profile.validate_params(params)
        if not valid:
            raise ValueError(error)

        points = profile.generate(params, resolution_ms)
        duration = profile.get_duration(params)

        result = {
            "data": [{"t": p.time_sec, "pwm": p.pwm} for p in points],
            "duration": duration
        }

        # Add sequence boundaries for multi-sequence profiles
        if isinstance(profile, MultiSequenceProfile):
            boundaries, labels = profile.get_sequence_boundaries(params)
            result["sequence_boundaries"] = boundaries
            result["sequence_labels"] = labels

        return result

    @property
    def is_running(self) -> bool:
        """Check if a profile is currently executing."""
        return self._running

    def abort(self):
        """Request abort of current execution."""
        if self._running:
            self._abort_requested = True

    async def execute(
        self,
        profile_name: str,
        params: Dict[str, Any],
        motor_ids: List[int],
        controller,  # MAVLinkController
        command_logger,  # CommandLogger
        progress_callback: Optional[Callable[[ExecutionState], None]] = None,
        resolution_ms: int = 50
    ) -> str:
        """
        Execute a profile with real-time logging.

        Args:
            profile_name: Name of the profile to execute
            params: Profile parameters
            motor_ids: List of motor IDs to control
            controller: MAVLinkController instance
            command_logger: CommandLogger instance
            progress_callback: Optional callback for progress updates
            resolution_ms: Execution resolution in milliseconds

        Returns:
            Session ID for the execution
        """
        if self._running:
            raise RuntimeError("A profile is already executing")

        profile = self._profiles.get(profile_name)
        if not profile:
            raise ValueError(f"Unknown profile: {profile_name}")

        # Validate parameters
        valid, error = profile.validate_params(params)
        if not valid:
            raise ValueError(error)

        # Generate profile points
        points = profile.generate(params, resolution_ms)
        if not points:
            raise ValueError("Profile generated no points")

        duration = profile.get_duration(params)

        # Check for looping
        # Support: loop_count parameter (for multi-sequence) or is_looping method (for custom steps)
        loop_count_param = params.get('loop_count', 0)  # 0 = no loop, -1 = infinite, N > 0 = loop N times
        is_looping = False

        # Check if profile has custom looping logic (CustomStepsProfile)
        if isinstance(profile, CustomStepsProfile) and profile.is_looping(params):
            is_looping = True
            loop_count_param = -1  # Infinite loop for backwards compatibility

        # Multi-sequence or explicit loop_count parameter
        if loop_count_param != 0:
            is_looping = True

        # Start session
        self._running = True
        self._abort_requested = False
        session_id = command_logger.start_session(profile_name, motor_ids)
        self._current_session_id = session_id

        try:
            # Disable motor functions for direct control
            controller.disable_motor_functions()

            start_time = time.time()
            point_index = 0
            current_iteration = 0
            last_pwm = None

            while self._running and not self._abort_requested:
                elapsed = time.time() - start_time

                # For looping, reset when we complete a cycle
                if is_looping and point_index >= len(points):
                    current_iteration += 1

                    # Check loop limit (-1 = infinite, 0 = no loop, N > 0 = loop N times)
                    if loop_count_param > 0 and current_iteration >= loop_count_param:
                        break  # Finished all iterations

                    # Reset for next iteration
                    point_index = 0
                    start_time = time.time()
                    elapsed = 0

                # Check if we're done (non-looping or exceeded iterations)
                if not is_looping and point_index >= len(points):
                    break

                # Find current point based on elapsed time
                while point_index < len(points) - 1 and points[point_index + 1].time_sec <= elapsed:
                    point_index += 1

                current_point = points[min(point_index, len(points) - 1)]
                pwm = current_point.pwm

                # Send PWM command if changed
                if pwm != last_pwm:
                    controller.set_selected_motors_pwm(motor_ids, pwm)
                    command_logger.log_command(motor_ids, pwm, profile_name)
                    last_pwm = pwm

                # Calculate progress
                if is_looping:
                    # For looping, show current position in cycle
                    progress_pct = (elapsed / duration * 100) if duration > 0 else 0
                else:
                    progress_pct = min(100, (elapsed / duration * 100)) if duration > 0 else 100

                # Report progress
                if progress_callback:
                    state = ExecutionState(
                        running=True,
                        elapsed_sec=elapsed,
                        total_sec=duration,
                        current_pwm=pwm,
                        progress_pct=progress_pct,
                        session_id=session_id,
                        current_iteration=current_iteration if is_looping else None,
                        total_iterations=loop_count_param if is_looping else None
                    )
                    progress_callback(state)

                # Sleep for resolution interval
                await asyncio.sleep(resolution_ms / 1000.0)

        except Exception as e:
            print(f"Execution error: {e}")
            raise
        finally:
            # Always stop motors when done (with verification)
            self._running = False

            # Verified stop with retry
            success, failed = controller.stop_all_motors_verified(timeout=5.0)

            # Final progress update
            if progress_callback:
                # Include error information if stop failed
                error_msg = None
                if not success:
                    error_msg = f"Failed to stop motors {failed}. Check MAVLink connection."
                    print(f"[ERROR] {error_msg}")

                state = ExecutionState(
                    running=False,
                    elapsed_sec=duration,
                    total_sec=duration,
                    current_pwm=1000,
                    progress_pct=100,
                    session_id=session_id,
                    error=error_msg,
                    failed_motors=failed if not success else None
                )
                progress_callback(state)

            # End logging session
            command_logger.end_session()
            self._current_session_id = None

        return session_id

    def execute_sync(
        self,
        profile_name: str,
        params: Dict[str, Any],
        motor_ids: List[int],
        controller,
        command_logger,
        progress_callback: Optional[Callable[[ExecutionState], None]] = None,
        resolution_ms: int = 50
    ) -> str:
        """
        Synchronous wrapper for execute().
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.execute(
                    profile_name, params, motor_ids,
                    controller, command_logger,
                    progress_callback, resolution_ms
                )
            )
        finally:
            loop.close()
