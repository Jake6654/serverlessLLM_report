"""Core data models for the server lifecycle sumulator"""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite

from serverless_llm.workload import RequestEvent

class ServerState(str, Enum):
    """Possible lifecycle states of the simulated model server"""

    OFF = "off"
    STARTING = "starting"
    READY = "ready"


class InvalidStateTransitionError(RuntimeError):
    """Raised when the server cannot perform an action in its state."""

@dataclass(frozen=True)
class LifecycleEvent:
    """Server state immdeidately after a transition at a virtual time"""

    at_seconds: float
    state: ServerState

    def __post_init__(self)-> None:
        """Reject an invalid lifecycle event. """

        if (
            type(self.at_seconds) not in (int, float)
            or not isfinite(self.at_seconds)
            or self.at_seconds < 0
        ):
            raise ValueError(
                "at_seconds must be a finite non-negative number"
            )
        
        if not isinstance(self.state, ServerState):
            raise ValueError("state must be a ServerState value")
        
        

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
    _lifecycle_event: list[LifecycleEvent] = field(
        default_factory=list,
        init=False, # 사용자가 constructor 을 총해 lifecycle 기록을 직접 넣지 못하게함 
        repr=False, # 서버 객체를 출력할 때 긴 lifecycle list를 기본 출력에서 제외
    )

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
        # Record the server's initial state at the beginning of the simulation.
        self._lifecycle_event.append(
            LifecycleEvent(
                at_seconds=0.0,
                state=self.state,
            )
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
    @property
    def lifecycle_events(self) -> tuple[LifecycleEvent, ...]:
        """Return a read-only snapshot of the lifecycle timeline."""

        # 내부에서는 수정가능한 리스트를 사용하지만 외부에서는 수정할수없는 tuple 을 사용
        return tuple(self._lifecycle_event)
    
    def state_at(self, at_seconds:float) -> ServerState:
        """Return the server state at an observed simulation time."""

        if (
            type(at_seconds) not in (int, float)
            or not isfinite(at_seconds)
            or at_seconds< 0
            or at_seconds > self.current_time_seconds
        ):
            raise ValueError(
                "at_seconds must be within observed simulation time"
            )
    
        state = self._lifecycle_event[0].state

        for event in self._lifecycle_event:
            # 찾는 시각보다 미래의 event 을 만나면 반복을 중단
            if event.at_seconds > at_seconds:
                break

            state = event.state
        return state
    
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
        # ServerState 가 off 이면 True on 이면 False
        cold_start = (
            # 괄호를 사용한 이유는 코드가 길어졌을 때 여러 줄로 나눠 읽기 쉽게 하기 위해서이다.
            self.state_at(arrival_time_seconds)
            is not ServerState.READY
        )
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
        self._record_state()

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
        self._record_state()

    def stop(self) -> None:
        """Stop a READY server and release its simulated resources."""

        self._require_state(
            expected=ServerState.READY,
            action="stop",
        )

        # Shutdown is instantanueous in the initial simulator
        self.state = ServerState.OFF
        self._record_state()
    
    def _record_state(self) -> None:
        """Record the current state at the current simulation time."""

        self._lifecycle_event.append(
            LifecycleEvent(
                at_seconds=float(self.current_time_seconds),
                state=self.state,
            )
        )

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
