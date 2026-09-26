"""Unit tests for the Fixed Keep-Warm policy."""

import pytest

from serverless_llm.policies import FixedKeepWarmPolicy
from serverless_llm.simulator import (
    InvalidStateTransitionError,
    LifecycleEvent,
    RequestResult,
    ServerState,
    ServerTiming,
    SimulatedServer,
)
from serverless_llm.workload import (
    RequestEvent,
    generate_steady_workload,
)


def make_server() -> SimulatedServer:
    """Create a deterministic OFF server."""

    return SimulatedServer(
        timing=ServerTiming(
            startup_duration_seconds=12.0,
            request_duration_seconds=2.0,
        )
    )


def make_ready_server(
    *,
    current_time_seconds: float = 14.0,
) -> SimulatedServer:
    """Create a READY server at a chosen observed time."""

    server = make_server()
    server.start()
    server.mark_ready()
    server.advance_to(current_time_seconds)
    return server


def make_event(
    *,
    scheduled_at_seconds: float,
) -> RequestEvent:
    """Create one request event at an absolute simulation time."""

    return RequestEvent(
        request_id=1,
        scheduled_at_seconds=scheduled_at_seconds,
        prompt_id="default",
    )


def make_result(
    *,
    completed_at_seconds: float = 14.0,
) -> RequestResult:
    """Create a request result that completed on a READY server."""

    return RequestResult(
        request_id=1,
        arrival_time_seconds=0.0,
        started_at_seconds=completed_at_seconds - 2.0,
        completed_at_seconds=completed_at_seconds,
        cold_start=True,
    )


def make_trace():
    """Create a minimal valid workload trace."""

    return generate_steady_workload(
        total_requests=1,
        interval_seconds=5.0,
        random_seed=42,
    )


def schedule_deadline(
    policy: FixedKeepWarmPolicy,
    server: SimulatedServer,
) -> None:
    """Schedule shutdown after a request completed at 14 seconds."""

    policy.after_request(
        server,
        make_event(scheduled_at_seconds=0.0),
        make_result(),
        has_pending_requests=False,
    )


def test_policy_normalizes_timeout_and_starts_without_deadline() -> None:
    """Store timeout as a float and expose no initial timer."""

    policy = FixedKeepWarmPolicy(timeout_seconds=30)

    assert policy.name == "fixed_keep_warm"
    assert policy.timeout_seconds == 30.0
    assert policy.shutdown_deadline_seconds is None


@pytest.mark.parametrize(
    "timeout_seconds",
    [0, -1, float("inf"), float("nan"), True, "30", None],
)
def test_policy_rejects_invalid_timeout(
    timeout_seconds: object,
) -> None:
    """Require a positive finite numeric timeout."""

    with pytest.raises(ValueError, match="positive and finite"):
        FixedKeepWarmPolicy(
            timeout_seconds=timeout_seconds,
        )


def test_before_workload_requires_off_server() -> None:
    """Reject a warm server at the beginning of an experiment."""

    server = make_ready_server()
    policy = FixedKeepWarmPolicy(timeout_seconds=30.0)

    with pytest.raises(
        InvalidStateTransitionError,
        match="OFF before the workload",
    ):
        policy.before_workload(server, make_trace())


def test_after_request_schedules_deadline_when_queue_is_empty() -> None:
    """Add the timeout to request completion time."""

    server = make_ready_server()
    policy = FixedKeepWarmPolicy(timeout_seconds=30.0)

    schedule_deadline(policy, server)

    assert policy.shutdown_deadline_seconds == 44.0
    assert server.state is ServerState.READY


def test_after_request_does_not_schedule_while_requests_are_pending() -> None:
    """Drain already-arrived requests without starting a timer."""

    server = make_ready_server()
    policy = FixedKeepWarmPolicy(timeout_seconds=30.0)
    schedule_deadline(policy, server)

    policy.after_request(
        server,
        make_event(scheduled_at_seconds=5.0),
        make_result(),
        has_pending_requests=True,
    )

    assert policy.shutdown_deadline_seconds is None
    assert server.state is ServerState.READY


def test_after_request_rejects_non_boolean_pending_flag() -> None:
    """Prevent ambiguous queue state from changing policy behavior."""

    server = make_ready_server()
    policy = FixedKeepWarmPolicy(timeout_seconds=30.0)

    with pytest.raises(ValueError, match="must be a boolean"):
        policy.after_request(
            server,
            make_event(scheduled_at_seconds=0.0),
            make_result(),
            has_pending_requests=1,
        )


def test_request_before_deadline_cancels_shutdown() -> None:
    """Keep the server warm when a request arrives inside the timeout."""

    server = make_ready_server()
    policy = FixedKeepWarmPolicy(timeout_seconds=30.0)
    schedule_deadline(policy, server)

    policy.before_request(
        server,
        make_event(scheduled_at_seconds=40.0),
    )

    assert server.state is ServerState.READY
    assert server.current_time_seconds == 14.0
    assert policy.shutdown_deadline_seconds is None


def test_request_at_deadline_is_still_warm() -> None:
    """Give request arrival priority over shutdown at equal time."""

    server = make_ready_server()
    policy = FixedKeepWarmPolicy(timeout_seconds=30.0)
    schedule_deadline(policy, server)

    policy.before_request(
        server,
        make_event(scheduled_at_seconds=44.0),
    )

    assert server.state is ServerState.READY
    assert policy.shutdown_deadline_seconds is None


def test_request_after_deadline_stops_server_at_deadline() -> None:
    """Apply an expired timer before a later request is processed."""

    server = make_ready_server()
    policy = FixedKeepWarmPolicy(timeout_seconds=10.0)
    schedule_deadline(policy, server)

    policy.before_request(
        server,
        make_event(scheduled_at_seconds=30.0),
    )

    assert server.state is ServerState.OFF
    assert server.current_time_seconds == 24.0
    assert policy.shutdown_deadline_seconds is None
    assert server.lifecycle_events[-1] == LifecycleEvent(
        at_seconds=24.0,
        state=ServerState.OFF,
    )


def test_before_request_rejects_starting_server() -> None:
    """Reject request handling before startup has completed."""

    server = make_server()
    server.start()
    policy = FixedKeepWarmPolicy(timeout_seconds=30.0)

    with pytest.raises(
        InvalidStateTransitionError,
        match="OFF or READY",
    ):
        policy.before_request(
            server,
            make_event(scheduled_at_seconds=0.0),
        )


def test_after_workload_stops_server_and_clears_deadline() -> None:
    """Leave no warm server or timer after the observation window."""

    server = make_ready_server()
    policy = FixedKeepWarmPolicy(timeout_seconds=30.0)
    schedule_deadline(policy, server)

    policy.after_workload(
        server,
        make_trace(),
        (make_result(),),
    )

    assert server.state is ServerState.OFF
    assert server.current_time_seconds == 14.0
    assert policy.shutdown_deadline_seconds is None

