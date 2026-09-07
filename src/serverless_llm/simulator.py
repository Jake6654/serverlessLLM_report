"""Core data models for the server lifecycle sumulator"""

from dataclasses import dataclass
from enum import Enum

class ServerState(str, Enum):
  """Possible lifecycle states of the simulated model server"""

  OFF = "off"
  STARTING = "starting"
  READY = "ready"


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
