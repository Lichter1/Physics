"""
Command logger for recording PWM commands during motor operations.
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class CommandLogEntry:
    """Single PWM command record."""
    timestamp: str  # ISO format string
    elapsed_sec: float
    motor_ids: List[int]
    pwm: int
    profile_name: str


class CommandLogger:
    """
    Logs PWM commands during motor operations.

    Output format: JSON Lines for easy parsing
    """

    def __init__(self, output_dir: str = "data/command_logs/"):
        """
        Initialize CommandLogger.

        Args:
            output_dir: Directory to save command logs
        """
        self.output_dir = output_dir
        self._entries: List[CommandLogEntry] = []
        self._start_time: Optional[datetime] = None
        self._current_file: Optional[str] = None
        self._profile_name: Optional[str] = None
        self._motor_ids: Optional[List[int]] = None
        self._session_id: Optional[str] = None

    @property
    def current_session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self._session_id

    @property
    def start_time(self) -> Optional[datetime]:
        """Get session start time."""
        return self._start_time

    @property
    def end_time(self) -> Optional[datetime]:
        """Get session end time (time of last entry)."""
        if self._entries:
            return datetime.fromisoformat(self._entries[-1].timestamp)
        return self._start_time

    def start_session(self, profile_name: str, motor_ids: List[int]) -> str:
        """
        Initialize a new logging session.

        Args:
            profile_name: Name of the profile being executed
            motor_ids: List of motor IDs being controlled

        Returns:
            Session ID (used for identifying this run)
        """
        self._start_time = datetime.now()
        self._profile_name = profile_name
        self._motor_ids = motor_ids
        self._entries = []

        # Generate session ID from timestamp
        self._session_id = self._start_time.strftime("%Y_%m_%d__%H_%M_%S")

        # Create output file path
        filename = f"{self._session_id}_commands.jsonl"
        self._current_file = os.path.join(self.output_dir, filename)

        # Ensure directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Write session header
        header = {
            "type": "session_start",
            "session_id": self._session_id,
            "timestamp": self._start_time.isoformat(),
            "profile_name": profile_name,
            "motor_ids": motor_ids
        }
        self._write_json_line(header)

        return self._session_id

    def log_command(self, motor_ids: List[int], pwm: int, profile_name: str):
        """
        Record a PWM command.

        Args:
            motor_ids: Motor IDs that received the command
            pwm: PWM value sent
            profile_name: Name of the active profile
        """
        if self._start_time is None:
            return  # No active session

        now = datetime.now()
        entry = CommandLogEntry(
            timestamp=now.isoformat(),
            elapsed_sec=round((now - self._start_time).total_seconds(), 3),
            motor_ids=motor_ids,
            pwm=pwm,
            profile_name=profile_name
        )
        self._entries.append(entry)

        # Write immediately for crash safety
        record = {
            "type": "command",
            **asdict(entry)
        }
        self._write_json_line(record)

    def end_session(self) -> Optional[str]:
        """
        Finalize logging session.

        Returns:
            Path to the log file, or None if no session was active
        """
        if self._start_time is None:
            return None

        end_time = datetime.now()

        # Write session footer
        footer = {
            "type": "session_end",
            "session_id": self._session_id,
            "timestamp": end_time.isoformat(),
            "total_commands": len(self._entries),
            "duration_sec": round((end_time - self._start_time).total_seconds(), 3)
        }
        self._write_json_line(footer)

        log_path = self._current_file

        # Reset state
        self._start_time = None
        self._current_file = None
        self._profile_name = None
        self._motor_ids = None
        self._session_id = None
        self._entries = []

        return log_path

    def _write_json_line(self, data: Dict[str, Any]):
        """Write a JSON line to the current log file."""
        if self._current_file:
            try:
                with open(self._current_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data) + '\n')
            except IOError as e:
                print(f"Error writing to command log: {e}")

    def get_session_data(self) -> List[Dict[str, Any]]:
        """Return all entries for current session as dicts."""
        return [asdict(e) for e in self._entries]

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available command log sessions.

        Returns:
            List of session info dicts
        """
        sessions = []

        if not os.path.exists(self.output_dir):
            return sessions

        for filename in os.listdir(self.output_dir):
            if filename.endswith('_commands.jsonl'):
                filepath = os.path.join(self.output_dir, filename)
                session_info = self._read_session_info(filepath)
                if session_info:
                    sessions.append(session_info)

        # Sort by timestamp (newest first)
        sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return sessions

    def _read_session_info(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Read session info from a command log file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if first_line:
                    data = json.loads(first_line)
                    if data.get('type') == 'session_start':
                        data['file_path'] = filepath
                        return data
        except (IOError, json.JSONDecodeError):
            pass
        return None

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a complete session's data.

        Args:
            session_id: Session ID to load

        Returns:
            Dict with session info and commands, or None if not found
        """
        filename = f"{session_id}_commands.jsonl"
        filepath = os.path.join(self.output_dir, filename)

        if not os.path.exists(filepath):
            return None

        result = {
            'session_id': session_id,
            'commands': [],
            'start': None,
            'end': None
        }

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get('type') == 'session_start':
                        result['start'] = data
                    elif data.get('type') == 'session_end':
                        result['end'] = data
                    elif data.get('type') == 'command':
                        result['commands'].append(data)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading session {session_id}: {e}")
            return None

        return result
