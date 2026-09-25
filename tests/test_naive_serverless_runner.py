"""Integration tests for running the Naive Serverless policy."""

from serverless_llm.policies import NaiveServerlessPolicy
from serverless_llm.policy_runner import run_policy_workload
from serverless_llm.simulator import (
    LifecycleEvent,
    ServerState,
    ServerTiming,
    SimulatedServer,
)
from serverless_llm.workload import (
    generate_sparse_workload,
    generate_steady_workload,
)


def make_server() -> SimulatedServer:
    """Create a deterministic server for runner tests."""

    return SimulatedServer(
        timing=ServerTiming(
            startup_duration_seconds=12.0,
            request_duration_seconds=2.0,
        )
    )


def test_steady_workload_drains_pending_queue_before_shutdown() -> None:
    """Reuse one startup while already-arrived requests remain queued."""

    server = make_server()
    trace = generate_steady_workload(
        total_requests=3,
        interval_seconds=5.0,
        random_seed=42,
    )

    results = run_policy_workload(
        server,
        trace,
        NaiveServerlessPolicy(),
    )

    assert [
        result.arrival_time_seconds
        for result in results
    ] == [0.0, 5.0, 10.0]
    assert [
        result.started_at_seconds
        for result in results
    ] == [12.0, 14.0, 16.0]
    assert [
        result.completed_at_seconds
        for result in results
    ] == [14.0, 16.0, 18.0]
    assert [result.cold_start for result in results] == [
        True,
        True,
        True,
    ]

    # Pending requests prevent shutdown until the final request completes.
    assert server.lifecycle_events == (
        LifecycleEvent(0.0, ServerState.OFF),
        LifecycleEvent(0.0, ServerState.STARTING),
        LifecycleEvent(12.0, ServerState.READY),
        LifecycleEvent(18.0, ServerState.OFF),
    )


def test_sparse_workload_restarts_server_for_every_request() -> None:
    """Cold-start each request after an empty queue causes shutdown."""

    server = make_server()
    trace = generate_sparse_workload(
        total_requests=3,
        min_interval_seconds=30.0,
        max_interval_seconds=30.0,
        random_seed=42,
    )

    results = run_policy_workload(
        server,
        trace,
        NaiveServerlessPolicy(),
    )

    assert [
        result.started_at_seconds
        for result in results
    ] == [12.0, 42.0, 72.0]
    assert [
        result.completed_at_seconds
        for result in results
    ] == [14.0, 44.0, 74.0]
    assert [result.cold_start for result in results] == [
        True,
        True,
        True,
    ]

    assert server.lifecycle_events == (
        LifecycleEvent(0.0, ServerState.OFF),
        LifecycleEvent(0.0, ServerState.STARTING),
        LifecycleEvent(12.0, ServerState.READY),
        LifecycleEvent(14.0, ServerState.OFF),
        LifecycleEvent(30.0, ServerState.STARTING),
        LifecycleEvent(42.0, ServerState.READY),
        LifecycleEvent(44.0, ServerState.OFF),
        LifecycleEvent(60.0, ServerState.STARTING),
        LifecycleEvent(72.0, ServerState.READY),
        LifecycleEvent(74.0, ServerState.OFF),
    )

    assert server.state is ServerState.OFF

