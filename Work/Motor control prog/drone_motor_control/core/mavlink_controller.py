"""
MAVLink controller for drone motor communication.

Based on the existing implementation in Board PWM controller with input (3).py
"""
import threading
import time
from typing import List, Optional, Callable
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

    def emergency_stop(self):
        """
        Emergency stop - set all motors to minimum PWM (1000).
        """
        print("[!] Emergency stop triggered!")
        try:
            if self.is_connected:
                with self._lock:
                    for i in range(1, 5):
                        self._connection.mav.command_long_send(
                            self._target_system,
                            self._target_component,
                            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                            0, i, 1000, 0, 0, 0, 0, 0
                        )
                print("All motors set to 1000 (stopped)")
        except Exception as e:
            print(f"Error during emergency stop: {e}")
