"""Integration tests for running the Fixed Keep-Warm policy."""

from serverless_llm.policies import FixedKeepWarmPolicy
from serverless_llm.policy_runner import run_policy_workload
from serverless_llm.simulator import (
    LifecycleEvent,
    ServerState,
    ServerTiming,
    SimulatedServer,
)
from serverless_llm.workload import generate_sparse_workload


def make_server() -> SimulatedServer:
    """Create a deterministic server for policy integration tests."""

    return SimulatedServer(
        timing=ServerTiming(
            startup_duration_seconds=12.0,
            request_duration_seconds=2.0,
        )
    )


def make_sparse_trace():
    """Create requests at 0, 30, and 60 seconds."""

    return generate_sparse_workload(
        total_requests=3,
        min_interval_seconds=30.0,
        max_interval_seconds=30.0,
        random_seed=42,
    )


def test_requests_inside_timeout_reuse_warm_server() -> None:
    """Keep all later requests warm when each arrives before expiry."""

    server = make_server()
    policy = FixedKeepWarmPolicy(timeout_seconds=30.0)

    results = run_policy_workload(
        server,
        make_sparse_trace(),
        policy,
    )

    assert [result.cold_start for result in results] == [
        True,
        False,
        False,
    ]
    assert [
        result.started_at_seconds
        for result in results
    ] == [12.0, 30.0, 60.0]
    assert server.lifecycle_events == (
        LifecycleEvent(0.0, ServerState.OFF),
        LifecycleEvent(0.0, ServerState.STARTING),
        LifecycleEvent(12.0, ServerState.READY),
        LifecycleEvent(62.0, ServerState.OFF),
    )


def test_request_after_timeout_restarts_server() -> None:
    """Stop at the deadline and cold-start a later request."""

    server = make_server()
    policy = FixedKeepWarmPolicy(timeout_seconds=20.0)

    results = run_policy_workload(
        server,
        make_sparse_trace(),
        policy,
    )

    assert [result.cold_start for result in results] == [
        True,
        False,
        True,
    ]
    assert [
        result.started_at_seconds
        for result in results
    ] == [12.0, 30.0, 72.0]
    assert server.lifecycle_events == (
        LifecycleEvent(0.0, ServerState.OFF),
        LifecycleEvent(0.0, ServerState.STARTING),
        LifecycleEvent(12.0, ServerState.READY),
        LifecycleEvent(52.0, ServerState.OFF),
        LifecycleEvent(60.0, ServerState.STARTING),
        LifecycleEvent(72.0, ServerState.READY),
        LifecycleEvent(74.0, ServerState.OFF),
    )


def test_workload_cleanup_cancels_final_timeout() -> None:
    """Stop immediately when observation ends instead of waiting."""

    server = make_server()
    policy = FixedKeepWarmPolicy(timeout_seconds=30.0)

    results = run_policy_workload(
        server,
        make_sparse_trace(),
        policy,
    )

    assert results[-1].completed_at_seconds == 62.0
    assert server.current_time_seconds == 62.0
    assert server.state is ServerState.OFF
    assert policy.shutdown_deadline_seconds is None
