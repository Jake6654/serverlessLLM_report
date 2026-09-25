"""Always-on warm-state policy"""

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

class AlwaysOnPolicy(WarmPolicy):
    """Keep the model server ready for the entire workload"""

    @property
    def name(self) -> str:
        """Return the stable experiemnt name for this policy"""

        return "always_on"
    
    def before_workload(
            self,
            server:SimulatedServer,
            trace: WorkloadTrace,
    ) -> None:
        """Start the server and finishing loading before requests begin"""

        if server.is_off:
            server.start()

        if not server.is_ready:
            server.mark_ready()

    def before_request(
            self,
            server: SimulatedServer,
            event: RequestEvent,
    ) -> None:
        """Verify that every request finds an already-ready server"""

        # Always-On must never allow a request to trigger startup
        if not server.is_ready:
            raise InvalidStateTransitionError(
                "always-on policy requires the server to be READY "
                "before every request"
            )
        
    
    def after_request(
            self,
            server: SimulatedServer,
            event: RequestEvent,
            result: RequestResult,
            *,
            has_pending_requests: bool,
    ) -> None:
        """Keep the server ready after a reuqest completes"""

        # process_request() must leave an Always-on server is READY
        if not server.is_ready:
            raise InvalidStateTransitionError(
                "always-on policy requires the server to remain READY "
                "after every request"
            )
        
    def after_workload(
            self,
            server: SimulatedServer,
            trace: WorkloadTrace,
            results: tuple[RequestResult, ...]
    ) -> None:
        """Stop the server after the complete workload finishes"""

        # Reaching workload completion in another state indicates that
        # the Always-On lifecycle invariant was broken.
        if not server.is_ready:
            raise InvalidStateTransitionError(
                "always-on policy requires the server to be READY "
                "when the workload finishes"
            )

        server.stop()