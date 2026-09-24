"""Tests for the Always-On warm-state policy."""

import pytest

from serverless_llm.policies import (
    AlwaysOnPolicy,
    WarmPolicy,
)
from serverless_llm.simulator import (
    InvalidStateTransitionError,
    LifecycleEvent,
    ServerState,
    ServerTiming,
    SimulatedServer,
)
from serverless_llm.workload import generate_steady_workload


def make_server() -> SimulatedServer:
    """Create a deterministic server for policy tests."""

    return SimulatedServer(
        timing=ServerTiming(
            startup_duration_seconds=12.0,
            request_duration_seconds=2.0,
        )
    )


def make_trace():
    """Create a minimal valid workload trace."""

    return generate_steady_workload(
        total_requests=1,
        interval_seconds=5.0,
        random_seed=42,
    )


def test_warm_policy_cannot_be_instantiated() -> None:
    """Verify that the base policy remains an abstract contract."""

    with pytest.raises(TypeError, match="abstract"):
        WarmPolicy()


def test_always_on_policy_has_stable_name() -> None:
    """Expose a stable identifier for result directories."""

    policy = AlwaysOnPolicy()

    assert policy.name == "always_on"


def test_before_workload_prepares_server() -> None:
    """Load the model before workload request timing begins."""

    server = make_server()
    trace = make_trace()
    policy = AlwaysOnPolicy()

    policy.before_workload(
        server,
        trace,
    )

    assert server.state is ServerState.READY
    assert server.current_time_seconds == 12.0

    assert server.lifecycle_events == (
        LifecycleEvent(
            at_seconds=0.0,
            state=ServerState.OFF,
        ),
        LifecycleEvent(
            at_seconds=0.0,
            state=ServerState.STARTING,
        ),
        LifecycleEvent(
            at_seconds=12.0,
            state=ServerState.READY,
        ),
    )


def test_before_workload_accepts_already_ready_server() -> None:
    """Avoid restarting a server that is already ready."""

    server = SimulatedServer(
        timing=ServerTiming(
            startup_duration_seconds=12.0,
            request_duration_seconds=2.0,
        ),
        state=ServerState.READY,
    )
    trace = make_trace()
    policy = AlwaysOnPolicy()

    policy.before_workload(
        server,
        trace,
    )

    assert server.state is ServerState.READY
    assert server.current_time_seconds == 0.0


def test_before_request_rejects_non_ready_server() -> None:
    """Prevent a request from triggering startup under Always-On."""

    server = make_server()
    trace = make_trace()
    policy = AlwaysOnPolicy()

    event = trace.events[0]

    with pytest.raises(
        InvalidStateTransitionError,
        match="before every request",
    ):
        policy.before_request(
            server,
            event,
        )


def test_after_workload_stops_server() -> None:
    """Release the simulated server after workload completion."""

    server = make_server()
    trace = make_trace()
    policy = AlwaysOnPolicy()

    policy.before_workload(
        server,
        trace,
    )

    policy.after_workload(
        server,
        trace,
        (),
    )

    assert server.state is ServerState.OFF
    assert server.current_time_seconds == 12.0

    assert server.lifecycle_events[-1] == LifecycleEvent(
        at_seconds=12.0,
        state=ServerState.OFF,
    )