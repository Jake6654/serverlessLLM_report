"""Run a workload under a warm-state management policy"""

"""Run a workload under a warm-state management policy."""

from dataclasses import replace

from serverless_llm.policies import WarmPolicy
from serverless_llm.simulator import (
    RequestResult,
    SimulatedServer,
)
from serverless_llm.workload import (
    RequestEvent,
    WorkloadTrace,
)

def run_policy_workload(
    server: SimulatedServer,
    trace: WorkloadTrace,
    policy: WarmPolicy,
) -> tuple[RequestResult, ...]:
    """Run one workload using a warm-state management policy."""

    # Lifecycle hooks require a real simulated server.
    if not isinstance(server, SimulatedServer):
        raise ValueError(
            "server must be a SimulatedServer object"
        )

    # The runner needs an immutable, time-ordered workload.
    if not isinstance(trace, WorkloadTrace):
        raise ValueError(
            "trace must be a WorkloadTrace object"
        )

    # Reject objects that do not implement the policy contract.
    if not isinstance(policy, WarmPolicy):
        raise ValueError(
            "policy must be a WarmPolicy object"
        )

    # Give the policy a chance to prepare the server.
    #
    # Always-On starts and loads the model here, so the simulation
    # clock advances by the configured startup duration.
    policy.before_workload(
        server,
        trace,
    )

    # Workload timestamps are relative to the moment request replay begins
    # For always-On, this is after model startup has completed
    workload_started_at_seconds = server.current_time_seconds

    # Convert every relative trace event into an absolute simulation event
    absolute_events = tuple(
        replace(
            relative_event,
            scheduled_at_seconds =(
                workload_started_at_seconds
                + relative_event.scheduled_at_seconds
            ),
        )

        for relative_event in trace.events
    )

    # Results are collected incrementally while requests are processed/
    results: list[RequestResult] = []

    for index, absolute_event in enumerate(absolute_events):
        # Apply policy actions that must happen before this request
        policy.before_request(
            server,
            absolute_event,
        )

        # Process the request using its absolute simulation timestamp
        result = server.process_request(
            absolute_event
        )

        # Because events are ordered, checking the next event is enough
        # last request cannot over the length of list
        if index + 1 < len(absolute_events):
            next_event = absolute_events[index + 1]

            has_pending_requests = (
                next_event.scheduled_at_seconds <= result.completed_at_seconds
            )

        else:
            has_pending_requests = False


        # Let the policy react to request completion
        policy.after_request(
            server,
            absolute_event,
            result,
            has_pending_requests=has_pending_requests,
        )


        results.append(result)

        # Freeze the completed results before passing them outside the runner.
    completed_results = tuple(results)

    # Let the policy perform final cleanup, such as stopping the server.
    policy.after_workload(
        server,
        trace,
        completed_results,
    )

    return completed_results
