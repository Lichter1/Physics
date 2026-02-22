"""
MAVLink controller for drone motor communication.

Based on the existing implementation in Board PWM controller with input (3).py
"""
import threading
import time
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass

# For running blocking pymavlink operations in native threads
try:
    import eventlet.tpool
    HAS_EVENTLET = True
except ImportError:
    HAS_EVENTLET = False

try:
    from pymavlink import mavutil
except ImportError:
    mavutil = None


@dataclass
class ConnectionStatus:
    """MAVLink connection status."""
    connected: bool
    system_id: Optional[int] = None
    component_id: Optional[int] = None
    last_heartbeat: Optional[float] = None


class MAVLinkController:
    """
    Handles MAVLink communication with the flight controller.

    Connection: TCP to MAVProxy/flight controller
    Protocol: MAVLink via pymavlink
    """

    def __init__(self, connection_string: str = "tcp:127.0.0.1:8888", baud: int = 115220):
        """
        Initialize the MAVLink controller.

        Args:
            connection_string: MAVLink connection string (e.g., "tcp:127.0.0.1:8888")
            baud: Baud rate (used for serial connections)
        """
        self.connection_string = connection_string
        self.baud = baud
        self._connection = None
        self._connected = False
        self._target_system = None
        self._target_component = None
        self._lock = threading.Lock()
        self._last_heartbeat = None

    @property
    def is_connected(self) -> bool:
        """Check if connected to MAVLink."""
        return self._connected and self._connection is not None

    def get_status(self) -> ConnectionStatus:
        """Get current connection status."""
        return ConnectionStatus(
            connected=self.is_connected,
            system_id=self._target_system,
            component_id=self._target_component,
            last_heartbeat=self._last_heartbeat
        )

    def _blocking_connect(self, timeout: float) -> dict:
        """
        Perform blocking MAVLink connection (runs in native thread via tpool).

        Returns dict with connection result to avoid sharing objects across threads.
        """
        try:
            print(f"Connecting to: {self.connection_string}")
            connection = mavutil.mavlink_connection(
                self.connection_string,
                self.baud,
                autoreconnect=True
            )

            # Wait for heartbeat
            print("Waiting for heartbeat...")
            msg = connection.wait_heartbeat(timeout=timeout)
            if msg is None:
                print("Timeout waiting for heartbeat")
                return {'success': False, 'error': 'Timeout waiting for heartbeat'}

            print(f"Heartbeat from system {connection.target_system} component {connection.target_component}")
            return {
                'success': True,
                'connection': connection,
                'target_system': connection.target_system,
                'target_component': connection.target_component
            }

        except Exception as e:
            print(f"Connection failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def connect(self, timeout: float = 10.0) -> bool:
        """
        Establish MAVLink connection and wait for heartbeat.

        Args:
            timeout: Timeout in seconds to wait for heartbeat

        Returns:
            True if connection successful, False otherwise
        """
        if mavutil is None:
            raise RuntimeError("pymavlink is not installed")

        with self._lock:
            try:
                # Run blocking connection in native thread to avoid eventlet issues
                if HAS_EVENTLET:
                    result = eventlet.tpool.execute(self._blocking_connect, timeout)
                else:
                    result = self._blocking_connect(timeout)

                if not result['success']:
                    print(f"Connection failed: {result.get('error', 'Unknown error')}")
                    self._connection = None
                    self._connected = False
                    return False

                self._connection = result['connection']
                self._target_system = result['target_system']
                self._target_component = result['target_component']
                self._connected = True
                self._last_heartbeat = time.time()

                return True

            except Exception as e:
                print(f"Connection failed: {e}")
                import traceback
                traceback.print_exc()
                self._connection = None
                self._connected = False
                return False

    def disconnect(self):
        """Disconnect from MAVLink."""
        with self._lock:
            if self._connection:
                try:
                    self._connection.close()
                except Exception:
                    pass
            self._connection = None
            self._connected = False
            self._target_system = None
            self._target_component = None
            print("Disconnected from MAVLink")

    def disable_motor_functions(self):
        """
        Disable servo functions for motors 1-4.
        This allows direct PWM control.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to MAVLink")

        with self._lock:
            motor_dis = 0
            for i in range(1, 5):
                self._connection.mav.param_set_send(
                    self._target_system,
                    self._target_component,
                    bytes(f"SERVO{i}_FUNCTION", "utf-8"),
                    motor_dis,
                    mavutil.mavlink.MAV_PARAM_TYPE_INT32
                )
            print("Motor functions disabled for direct PWM control")

    def set_motor_pwm(self, motor_id: int, pwm: int):
        """
        Set PWM for a single motor.

        Args:
            motor_id: Motor ID (1-4)
            pwm: PWM value (1000-2000)

        Raises:
            ValueError: If motor_id or pwm is out of range
            RuntimeError: If not connected
        """
        if not 1 <= motor_id <= 4:
            raise ValueError(f"Motor ID must be 1-4, got {motor_id}")
        if not 1000 <= pwm <= 2000:
            raise ValueError(f"PWM must be 1000-2000, got {pwm}")
        if not self.is_connected:
            raise RuntimeError("Not connected to MAVLink")

        with self._lock:
            self._connection.mav.command_long_send(
                self._target_system,
                self._target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,  # confirmation
                motor_id,  # servo/channel number
                pwm,  # PWM value
                0, 0, 0, 0, 0  # unused parameters
            )

    def set_all_motors_pwm(self, pwm: int):
        """
        Set PWM for all 4 motors.

        Args:
            pwm: PWM value (1000-2000)
        """
        for i in range(1, 5):
            self.set_motor_pwm(i, pwm)

    def set_selected_motors_pwm(self, motor_ids: List[int], pwm: int):
        """
        Set PWM for selected motors.

        Args:
            motor_ids: List of motor IDs (1-4)
            pwm: PWM value (1000-2000)
        """
        for motor_id in motor_ids:
            self.set_motor_pwm(motor_id, pwm)

    def _wait_for_command_ack(self, command: int, timeout: float = 2.0) -> Optional[bool]:
        """
        Wait for COMMAND_ACK message for a specific command.

        Args:
            command: MAVLink command ID to wait for (e.g., MAV_CMD_DO_SET_SERVO)
            timeout: Maximum time to wait for ACK in seconds

        Returns:
            True if ACK received and accepted, False if NACK, None if timeout
        """
        if not self.is_connected:
            return None

        start_time = time.time()

        while time.time() - start_time < timeout:
            # Receive message with short timeout
            msg = self._connection.recv_match(type='COMMAND_ACK', blocking=True, timeout=0.1)

            if msg and msg.command == command:
                # Check result
                # MAV_RESULT_ACCEPTED = 0, anything else is failure
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    return True
                else:
                    print(f"[WARN] Command {command} rejected: result={msg.result}")
                    return False

        # Timeout
        return None

    def set_motor_pwm_verified(self, motor_id: int, pwm: int, timeout: float = 2.0, max_retries: int = 3) -> bool:
        """
        Set PWM for a single motor with acknowledgment verification.

        Sends PWM command and waits for COMMAND_ACK. Retries with exponential backoff
        if no ACK received or command rejected.

        Args:
            motor_id: Motor ID (1-4)
            pwm: PWM value (1000-2000)
            timeout: Timeout per attempt in seconds
            max_retries: Maximum number of retry attempts

        Returns:
            True if command acknowledged successfully, False otherwise

        Raises:
            ValueError: If motor_id or pwm is out of range
            RuntimeError: If not connected
        """
        if not 1 <= motor_id <= 4:
            raise ValueError(f"Motor ID must be 1-4, got {motor_id}")
        if not 1000 <= pwm <= 2000:
            raise ValueError(f"PWM must be 1000-2000, got {pwm}")
        if not self.is_connected:
            raise RuntimeError("Not connected to MAVLink")

        backoff_delays = [0.5, 1.0, 2.0]  # Exponential backoff

        for attempt in range(max_retries):
            try:
                with self._lock:
                    # Send command
                    self._connection.mav.command_long_send(
                        self._target_system,
                        self._target_component,
                        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                        0,  # confirmation
                        motor_id,  # servo/channel number
                        pwm,  # PWM value
                        0, 0, 0, 0, 0  # unused parameters
                    )

                # Wait for ACK (outside lock to allow message reception)
                ack_result = self._wait_for_command_ack(mavutil.mavlink.MAV_CMD_DO_SET_SERVO, timeout)

                if ack_result is True:
                    # Success
                    return True
                elif ack_result is False:
                    # Explicit NACK - command rejected
                    print(f"[WARN] Motor {motor_id} PWM command rejected, attempt {attempt + 1}/{max_retries}")
                else:
                    # Timeout - no ACK received
                    print(f"[WARN] Motor {motor_id} PWM command timeout, attempt {attempt + 1}/{max_retries}")

                # Retry with backoff (if not last attempt)
                if attempt < max_retries - 1:
                    delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    time.sleep(delay)

            except Exception as e:
                print(f"[ERROR] Exception during verified PWM send: {e}")
                if attempt < max_retries - 1:
                    time.sleep(backoff_delays[min(attempt, len(backoff_delays) - 1)])

        # All retries exhausted
        print(f"[ERROR] Failed to set motor {motor_id} PWM to {pwm} after {max_retries} attempts")
        return False

    def stop_all_motors_verified(self, timeout: float = 5.0) -> Tuple[bool, List[int]]:
        """
        Stop all motors with verification.

        Sets all 4 motors to PWM 1000 (stop) with ACK verification.

        Args:
            timeout: Total timeout for all motor stop commands

        Returns:
            Tuple of (all_stopped: bool, failed_motor_ids: List[int])
            - all_stopped: True if all motors stopped successfully
            - failed_motor_ids: List of motor IDs that failed to stop
        """
        if not self.is_connected:
            print("[ERROR] Cannot stop motors: Not connected to MAVLink")
            return False, [1, 2, 3, 4]

        failed = []
        start_time = time.time()
        timeout_per_motor = timeout / 4  # Divide total timeout among motors

        for motor_id in range(1, 5):
            # Check if we've exceeded total timeout
            if time.time() - start_time > timeout:
                print(f"[ERROR] Total timeout exceeded, motors {motor_id} to 4 not stopped")
                failed.extend(range(motor_id, 5))
                break

            # Try to stop this motor
            remaining_time = timeout - (time.time() - start_time)
            motor_timeout = min(timeout_per_motor, remaining_time)

            success = self.set_motor_pwm_verified(motor_id, 1000, timeout=motor_timeout, max_retries=3)
            if not success:
                failed.append(motor_id)

        all_stopped = len(failed) == 0
        if all_stopped:
            print("[OK] All motors stopped successfully (verified)")
        else:
            print(f"[ERROR] Failed to stop motors: {failed}")

        return all_stopped, failed

    def emergency_stop(self):
        """
        Emergency stop - set all motors to minimum PWM (1000).
        Now uses verified stop for better reliability.
        """
        print("[!] Emergency stop triggered!")
        try:
            if self.is_connected:
                # Use verified stop with shorter timeout for emergency
                success, failed = self.stop_all_motors_verified(timeout=3.0)
                if success:
                    print("All motors stopped (emergency)")
                else:
                    print(f"[WARN] Emergency stop incomplete, motors {failed} may still be running")
                    # Fallback: send raw commands without waiting
                    with self._lock:
                        for motor_id in failed:
                            self._connection.mav.command_long_send(
                                self._target_system,
                                self._target_component,
                                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                                0, motor_id, 1000, 0, 0, 0, 0, 0
                            )
                    print("Emergency stop fallback commands sent")
        except Exception as e:
            print(f"Error during emergency stop: {e}")
