"""Naive serverless warm-state policy."""

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

class NaiveServerlessPolicy(WarmPolicy):
  """Stop the server immediately whenever the request queue is empty"""

  @property
  def name(self) -> str:
    return "naive_serverless"

  def before_workload(
      self,
      server: SimulatedServer,
      trace: WorkloadTrace,
  ) -> None:
    # Naive serverless dose not pre-ward the model
    # That means the first request must trigger server startup
    if not server.is_off:
      raise InvalidStateTransitionError(
        "naive serverless policy requires the server to be "
        "OFF before the workload begins"
      )


  def before_request(
      self,
      server: SimulatedServer,
      event: RequestEvent,
  ) -> None:

      # OFF means this request will trigger a cold start.
      # READY means an earlier request kept the server alive because
      # another request was already waiting in the queue.
      if not server.is_off and not server.is_ready:
          raise InvalidStateTransitionError(
              "naive serverless policy requires the server to be "
              "OFF or READY before a request"
          )

  def after_request(
        self,
        server: SimulatedServer,
        event: RequestEvent,
        result: RequestResult,
        *,
        has_pending_requests: bool,
  ) -> None:

      # process_request() must finish with the model in READY state
      if not server.is_ready:
            raise InvalidStateTransitionError(
                "naive serverless policy requires the server to be "
                "READY after request processing"
            )

      # Keep the server ready only while draining requests that have
      # already arrived. Future requests do not justify staying warm.
      if not has_pending_requests:
          server.stop()

  def after_workload(
        self,
        server: SimulatedServer,
        trace: WorkloadTrace,
        results: tuple[RequestResult, ...],
    ) -> None:
        """Ensure that no server remains active after the workload."""

        # This is a final cleanup guard. Normally, the last call to
        # after_request() has already stopped the server.
        if server.is_ready:
            server.stop()

        if not server.is_off:
            raise InvalidStateTransitionError(
                "naive serverless policy requires the server to be "
                "OFF after the workload finishes"
            )