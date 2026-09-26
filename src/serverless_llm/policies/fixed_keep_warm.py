
from dataclasses import dataclass, field
from math import isfinite

from serverless_llm.policies.base import WarmPolicy
from serverless_llm.simulator import (
    InvalidStateTransitionError,
    RequestResult,
    SimulatedServer,
)
from serverless_llm.workload import (
    RequestEvent,
    WorkloadTrace,
)

@dataclass
class FixedKeepWarmPolicy(WarmPolicy):
    """Keep the server ready for a fixed time after queue drain"""

    timeout_seconds: float

    _shutdown_deadline_seconds: float | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate and normalize the configured timeout."""

        if (
            type(self.timeout_seconds) not in (int, float)
            or not isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError(
                "timeout_seconds must be positive and finite"
            )

        self.timeout_seconds = float(self.timeout_seconds)


    @property
    def name(self) -> str:
        return "fixed_keep_warm"

    @property
    def shutdown_deadline_seconds(self) -> float | None:
        return self._shutdown_deadline_seconds

    def before_workload(
            self,
            server: SimulatedServer,
            trace: WorkloadTrace
    ) -> None:
        """Start the workload with an OFF server and no deadline."""

        if not server.is_off:
          raise InvalidStateTransitionError(
              "fixed keep-warm policy requires the server to be "
              "OFF before the workload begins"
          )

        # A policy object may be reused, so clear any previous dealine
        self._shutdown_deadline_seconds = None

    def before_request(
        self,
        server: SimulatedServer,
        event: RequestEvent,
    ) -> None:
        """Apply a scheduled shutdown before processing a new request."""

        if server.is_off:
            # An OFF server cannot have an active shutdown timer.
            self._shutdown_deadline_seconds = None
            return

        if not server.is_ready:
            raise InvalidStateTransitionError(
                "fixed keep-warm policy requires the server to be "
                "OFF or READY before a request"
            )

        deadline = self.shutdown_deadline_seconds

        #A READY server may have no dealine while draining requests
        # that had already arrived
        if deadline is None:
            return

        # Requests at exactly the dealine are processed while warm
        if event.scheduled_at_seconds > deadline:
            # The timeout happened before request arrived
            server.advance_to(deadline)
            server.stop()

        # Whether the request arrived before or after the deadline,
        # the previous timer has now been consumed or cancelled.
        self._shutdown_deadline_seconds = None

    def after_request(
        self,
        server: SimulatedServer,
        event: RequestEvent,
        result: RequestResult,
        *,
        has_pending_requests: bool,
    ) -> None:
        """Schedule shutdown after the already-arrived queue is drained."""

        if not server.is_ready:
            raise InvalidStateTransitionError(
                "fixed keep-warm policy requires the server to be "
                "READY after request processing"
            )

        if type(has_pending_requests) is not bool:
            raise ValueError(
                "has_pending_requests must be a boolean"
            )

        if has_pending_requests:
            # Process waiting requests immediately without a timer.
            self._shutdown_deadline_seconds = None
            return

        # The queue is empty, so begin the fixed keep-warm window.
        self._shutdown_deadline_seconds = (
            result.completed_at_seconds
            + self.timeout_seconds
        )

    def after_workload(
            self,
            server: SimulatedServer,
            trace: WorkloadTrace,
            results: tuple[RequestResult, ...],
    ) -> None:
        """Stop the server and cancel its timer after the workload"""

        # All policies use the same observation window. Therefore, the
        # server is stopped at workload completion instead of waiting
        # for the final timeout to expire.
        if server.is_ready:
            server.stop()

        if not server.is_off:
            raise InvalidStateTransitionError(
                "fixed keep-warm policy requires the server to be "
                "OFF after the workload finishes"
            )

        self._shutdown_deadline_seconds = None



