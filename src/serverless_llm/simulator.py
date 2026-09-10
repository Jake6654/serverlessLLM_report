"""Core data models for the server lifecycle sumulator"""

from dataclasses import dataclass
from enum import Enum

class ServerState(str, Enum):
  """Possible lifecycle states of the simulated model server"""

  OFF = "off"
  STARTING = "starting"
  READY = "ready"


class InvalidStateTransitionError(RuntimeError):
    """Raised when the server cannot perform an action in its state."""


@dataclass(frozen=True)
class ServerTiming:
  """Fixed timing assumptuions used by the initial simulator"""

  startup_duration_seconds: float
  request_duration_seconds: float
  
  def __post_init__(self) -> None:
      """Reject invalid timing values immediately."""

      if (
          type(self.startup_duration_seconds) not in (int, float) 
          or self.startup_duration_seconds <=0
      ):
          raise ValueError(
            "startup_duration_seconds must be a positive number"
        )
      
      if (
            type(self.request_duration_seconds) not in (int, float)
            or self.request_duration_seconds <= 0
        ):
            raise ValueError(
                "request_duration_seconds must be a positive number"
            )

@dataclass
class SimulatedServer:
    """Mutable runtime state of one simulated model server."""

    timing: ServerTiming
    state: ServerState = ServerState.OFF
    current_time_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Reject an invalid initial server state."""

        if not isinstance(self.timing, ServerTiming):
            raise ValueError("timing must be a ServerTiming object")

        if not isinstance(self.state, ServerState):
            raise ValueError("state must be a ServerState value")

        if (
            type(self.current_time_seconds) not in (int, float)
            or self.current_time_seconds < 0
        ):
            raise ValueError(
                "current_time_seconds must be a non-negative number"
            )

    @property
    def is_off(self) -> bool:
        """Return whether the server is completely stopped."""

        return self.state is ServerState.OFF
    
    @property
    def is_ready(self) -> bool:
        return self.state is ServerState.READY
    
    @property
    def is_running(self) -> bool:
        """Return whether the serve process has started"""

        return self.state in {
            ServerState.STARTING,
            ServerState.READY,
        }
    
    def start(self) -> None:
        """Start loading model from the OFF state"""

        self._require_state(
            expected=ServerState.OFF,
            action="start",
        )

        # Starting beings now, but loading time has not passed yet
        self.state = ServerState.STARTING

    def mark_ready(self) -> None:
        """Finishing model loading and move into the READY state."""

        self._require_state(
            expected=ServerState.STARTING,
            action="mark ready",
        )

        # Move the simulation clock forward by the model loading time.
        self.current_time_seconds += float(
            self.timing.startup_duration_seconds
        )
        self.state = ServerState.READY

    def stop(self) -> None:
        """Stop a READY server and release its simulated resources."""

        self._require_state(
            expected=ServerState.READY,
            action="stop",
        )

        # Shutdown is instantanueous in the initial simulator
        self.state = ServerState.OFF

    def _require_state(
        self,
        *,
        expected: ServerState,
        action: str,
    ) -> None:
        """Check that an action is valid in the current state."""

        if self.state is not expected:
            raise InvalidStateTransitionError(
                f"cannot {action} server while state is "
                f"{self.state.value}; expected {expected.value}"
            )