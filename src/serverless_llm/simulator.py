"""Core data models for the server lifecycle sumulator"""

from dataclasses import dataclass
from enum import Enum

from serverless_llm.workload import RequestEvent

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

@dataclass(frozen=True)
class RequestResult:
    """Recorded timing result for one completed request."""

    request_id: int
    arrival_time_seconds: float
    started_at_seconds: float
    completed_at_seconds: float
    cold_start: bool

    def __post_init__(self) -> None:
        """Reject invalid request timing results."""

        if type(self.request_id) is not int or self.request_id < 1:
            raise ValueError("request_id must be a positive integer")

        if (
            type(self.arrival_time_seconds) not in (int, float)
            or self.arrival_time_seconds < 0
        ):
            raise ValueError(
                "arrival_time_seconds must be a non-negative number"
            )

        if (
            type(self.started_at_seconds) not in (int, float)
            or self.started_at_seconds < 0
        ):
            raise ValueError(
                "started_at_seconds must be a non-negative number"
            )

        if (
            type(self.completed_at_seconds) not in (int, float)
            or self.completed_at_seconds < 0
        ):
            raise ValueError(
                "completed_at_seconds must be a non-negative number"
            )

        if self.started_at_seconds < self.arrival_time_seconds:
            raise ValueError(
                "started_at_seconds cannot be earlier than "
                "arrival_time_seconds"
            )

        if self.completed_at_seconds < self.started_at_seconds:
            raise ValueError(
                "completed_at_seconds cannot be earlier than "
                "started_at_seconds"
            )

        if type(self.cold_start) is not bool:
            raise ValueError("cold_start must be a boolean")
        
    @property
    def waiting_time_seconds(self) -> float:
        """Return the time spent waiting before processing"""

        return float(
            self.started_at_seconds - self.arrival_time_seconds
        )
    
    @property
    def processing_time_seconds(self) -> float:
        """Return the time spent processing the request"""

        return float(
            self.completed_at_seconds - self.started_at_seconds
        )
    
    @property
    def total_latency_seconds(self) -> float:
        """Return the complete request latency."""

        return float(
            self.completed_at_seconds - self.arrival_time_seconds
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
    
    def advance_to(self, target_time_seconds: float) -> None:
        """Move the simluation clock forward to a target time."""

        if(
            type(target_time_seconds) not in (int, float)
            or target_time_seconds < 0
        ):
            raise ValueError(
                "target_time_seconds must be a non-negative number"
            )
        
        if target_time_seconds < self.current_time_seconds:
            raise ValueError(
                "target_time_seconds cannot move simulation time backwards"
            )
        
        # Equal timestamps are allowed, but time can never move backwards
        self.current_time_seconds = float(target_time_seconds)
    
    def process_request(self, event: RequestEvent) -> RequestResult:
        """Process one request and return its simulated timing result."""

        if not isinstance(event, RequestEvent):
            raise ValueError("event must be a RequestEvent object")
        
        arrival_time_seconds = float(event.scheduled_at_seconds)

        # Skip idle time when the request arrives after the current clock
        if arrival_time_seconds > self.current_time_seconds:
            self.advance_to(arrival_time_seconds)

        # Remember the original state before a possible server startup
        cold_start = self.is_off
        if self.is_off:
            self.start()
            self.mark_ready()
        elif self.state is ServerState.STARTING:
            raise InvalidStateTransitionError(
                "cannot process request while state is starting"
            )

        started_at_seconds = self.current_time_seconds

        # Simulate the time required to run inference
        self.current_time_seconds += float(
            self.timing.request_duration_seconds
        )

        completed_at_seconds = self.current_time_seconds

        return RequestResult(
            request_id=event.request_id,
            arrival_time_seconds=arrival_time_seconds,
            started_at_seconds=started_at_seconds,
            completed_at_seconds=completed_at_seconds,
            cold_start=cold_start,
        )

    
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

    # 메서드 앞의 _는 이 메서드가 클래스 내부에서 사용하기 위한 private helper라는 관례이다
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