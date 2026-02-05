"""
Base class for motion profiles.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class ProfilePoint:
    """Single point in a motion profile."""
    time_sec: float
    pwm: int


class BaseProfile(ABC):
    """
    Abstract base class for all motion profiles.

    Subclass this to add new profile types. Each profile must implement:
    - name: Human-readable profile name
    - parameters: Parameter schema for UI generation
    - generate(): Generate time-series of PWM values
    - validate_params(): Validate parameters
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable profile name."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Return parameter schema for UI generation.

        Each parameter should have:
        - type: "int", "float", "bool", or "array"
        - min/max: For numeric types
        - default: Default value
        - label: Human-readable label (optional)
        """
        pass

    @abstractmethod
    def generate(self, params: Dict[str, Any], resolution_ms: int = 50) -> List[ProfilePoint]:
        """
        Generate time-series of PWM values.

        Args:
            params: Profile-specific parameters
            resolution_ms: Time resolution in milliseconds

        Returns:
            List of ProfilePoint objects
        """
        pass

    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate parameters.

        Args:
            params: Parameters to validate

        Returns:
            Tuple of (valid, error_message)
        """
        pass

    def get_duration(self, params: Dict[str, Any]) -> float:
        """
        Calculate total duration in seconds.

        Default implementation generates the profile and returns the last time point.
        Override for efficiency if duration can be calculated directly.
        """
        points = self.generate(params)
        if points:
            return points[-1].time_sec
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile metadata to dictionary."""
        return {
            "name": self.name,
            "parameters": self.parameters
        }
