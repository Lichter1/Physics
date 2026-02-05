"""
Log manager for discovering and extracting drone log files.
"""
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LogFile:
    """Represents a discovered log file."""
    path: str
    log_type: str  # 'esc_telemetry', 'temps_server', 'pdb_server', 'general'
    timestamp: datetime
    size_bytes: int


class LogManager:
    """
    Manages discovery and extraction of drone log files.

    Log location: /media/ssd/

    File naming patterns (observed):
    - YYYY_MM_DD__HH_MM_SS__esc_telemetry.log
    - YYYY_MM_DD__HH_MM_SS__temps_server.log
    - YYYY_MM_DD__HH_MM_SS_pdb_server.log
    - YYYY_MM_DD__HH_MM_SS.log (general)
    """

    # Regex patterns for log file types
    PATTERNS = {
        'esc_telemetry': re.compile(r'(\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})__esc_telemetry\.log$'),
        'temps_server': re.compile(r'(\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})__temps_server\.log$'),
        'pdb_server': re.compile(r'(\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})_pdb_server\.log$'),
        'general': re.compile(r'(\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})\.log$')
    }

    # Timestamp format in filenames
    FILENAME_TIME_FORMAT = "%Y_%m_%d__%H_%M_%S"

    # Regex for parsing timestamps in log content
    LOG_TIMESTAMP_PATTERN = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')

    def __init__(self, ssd_path: str = "/media/ssd/"):
        """
        Initialize LogManager.

        Args:
            ssd_path: Path to SSD where logs are stored
        """
        self.ssd_path = ssd_path

    def _parse_filename_timestamp(self, filename: str) -> Tuple[Optional[str], Optional[datetime]]:
        """
        Parse log type and timestamp from filename.

        Returns:
            Tuple of (log_type, timestamp) or (None, None) if not a log file
        """
        for log_type, pattern in self.PATTERNS.items():
            match = pattern.search(filename)
            if match:
                timestamp_str = match.group(1)
                try:
                    timestamp = datetime.strptime(timestamp_str, self.FILENAME_TIME_FORMAT)
                    return log_type, timestamp
                except ValueError:
                    continue
        return None, None

    def discover_all_logs(self) -> Dict[str, List[LogFile]]:
        """
        Discover all log files on SSD, grouped by type.

        Returns:
            Dict mapping log_type to list of LogFile objects, sorted by timestamp (newest first)
        """
        result = {
            'esc_telemetry': [],
            'temps_server': [],
            'pdb_server': [],
            'general': []
        }

        if not os.path.exists(self.ssd_path):
            return result

        try:
            for filename in os.listdir(self.ssd_path):
                filepath = os.path.join(self.ssd_path, filename)
                if not os.path.isfile(filepath):
                    continue

                log_type, timestamp = self._parse_filename_timestamp(filename)
                if log_type and timestamp:
                    try:
                        size = os.path.getsize(filepath)
                        log_file = LogFile(
                            path=filepath,
                            log_type=log_type,
                            timestamp=timestamp,
                            size_bytes=size
                        )
                        result[log_type].append(log_file)
                    except OSError:
                        continue

            # Sort each list by timestamp (newest first)
            for log_type in result:
                result[log_type].sort(key=lambda x: x.timestamp, reverse=True)

        except OSError as e:
            print(f"Error scanning log directory: {e}")

        return result

    def discover_latest_logs(self) -> Dict[str, Optional[LogFile]]:
        """
        Find the most recent log file of each type.

        Returns:
            Dict mapping log_type to LogFile (or None if not found)
        """
        all_logs = self.discover_all_logs()
        return {
            log_type: logs[0] if logs else None
            for log_type, logs in all_logs.items()
        }

    def discover_logs_near_time(self, target_time: datetime,
                                 tolerance_minutes: int = 10) -> Dict[str, Optional[LogFile]]:
        """
        Find log files closest to a target timestamp.

        Args:
            target_time: Target timestamp to match
            tolerance_minutes: Maximum time difference allowed

        Returns:
            Dict mapping log_type to closest LogFile within tolerance
        """
        all_logs = self.discover_all_logs()
        tolerance = timedelta(minutes=tolerance_minutes)
        result = {}

        for log_type, logs in all_logs.items():
            closest = None
            min_diff = None

            for log_file in logs:
                diff = abs(log_file.timestamp - target_time)
                if diff <= tolerance:
                    if min_diff is None or diff < min_diff:
                        min_diff = diff
                        closest = log_file

            result[log_type] = closest

        return result

    def _parse_line_timestamp(self, line: str) -> Optional[datetime]:
        """
        Extract timestamp from a log line.

        All log types use format: YYYY-MM-DD HH:MM:SS,mmm
        """
        match = self.LOG_TIMESTAMP_PATTERN.match(line)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
            except ValueError:
                pass
        return None

    def extract_time_slice(self, log_file: LogFile,
                           start_time: datetime,
                           end_time: datetime,
                           margin_seconds: float = 5.0) -> str:
        """
        Extract lines from a log file within a time range.

        Args:
            log_file: The log file to extract from
            start_time: Operation start time
            end_time: Operation end time
            margin_seconds: Seconds to add before/after

        Returns:
            Extracted log content as string
        """
        adjusted_start = start_time - timedelta(seconds=margin_seconds)
        adjusted_end = end_time + timedelta(seconds=margin_seconds)

        extracted_lines = []

        try:
            with open(log_file.path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    timestamp = self._parse_line_timestamp(line)
                    if timestamp:
                        if adjusted_start <= timestamp <= adjusted_end:
                            extracted_lines.append(line)
                    # For lines without timestamps (continuation), include if we're in range
                    elif extracted_lines:
                        extracted_lines.append(line)
        except IOError as e:
            print(f"Error reading log file: {e}")

        return ''.join(extracted_lines)

    def extract_and_save(self, log_file: LogFile,
                         start_time: datetime,
                         end_time: datetime,
                         output_dir: str,
                         margin_seconds: float = 5.0) -> Optional[str]:
        """
        Extract time slice and save to output directory.

        Args:
            log_file: The log file to extract from
            start_time: Operation start time
            end_time: Operation end time
            output_dir: Directory to save extracted file
            margin_seconds: Seconds to add before/after

        Returns:
            Path to saved file, or None on error
        """
        content = self.extract_time_slice(log_file, start_time, end_time, margin_seconds)

        if not content:
            return None

        # Create output filename
        timestamp_str = start_time.strftime("%Y_%m_%d__%H_%M_%S")
        output_filename = f"{timestamp_str}_extracted_{log_file.log_type}.log"
        output_path = os.path.join(output_dir, output_filename)

        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return output_path
        except IOError as e:
            print(f"Error saving extracted log: {e}")
            return None

    def extract_all_logs(self, start_time: datetime,
                         end_time: datetime,
                         output_dir: str,
                         margin_seconds: float = 5.0,
                         log_types: Optional[List[str]] = None) -> Dict[str, Optional[str]]:
        """
        Extract time slices from all (or specified) log types.

        Args:
            start_time: Operation start time
            end_time: Operation end time
            output_dir: Directory to save extracted files
            margin_seconds: Seconds to add before/after
            log_types: List of log types to extract (None = all)

        Returns:
            Dict mapping log_type to output path (or None if not found/extracted)
        """
        latest_logs = self.discover_logs_near_time(start_time, tolerance_minutes=60)

        if log_types is None:
            log_types = list(self.PATTERNS.keys())

        result = {}
        for log_type in log_types:
            log_file = latest_logs.get(log_type)
            if log_file:
                output_path = self.extract_and_save(
                    log_file, start_time, end_time, output_dir, margin_seconds
                )
                result[log_type] = output_path
            else:
                result[log_type] = None

        return result
