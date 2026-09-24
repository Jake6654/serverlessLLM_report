"""Tests for running workloads through warm-state policies."""

import pytest

from serverless_llm.policies import AlwaysOnPolicy
from serverless_llm.policy_runner import run_policy_workload
from serverless_llm.simulator import (
    LifecycleEvent,
    ServerState,
    ServerTiming,
    SimulatedServer,
)
from serverless_llm.workload import generate_steady_workload


def make_server() -> SimulatedServer:
    """Create a deterministic server for runner tests."""

    return SimulatedServer(
        timing=ServerTiming(
            startup_duration_seconds=12.0,
            request_duration_seconds=2.0,
        )
    )


def make_trace():
    """Create three requests separated by five seconds."""

    return generate_steady_workload(
        total_requests=3,
        interval_seconds=5.0,
        random_seed=42,
    )


def test_policy_runner_offsets_relative_request_times() -> None:
    """Shift workload-relative times by the pre-warm duration."""

    server = make_server()
    trace = make_trace()
    policy = AlwaysOnPolicy()

    results = run_policy_workload(
        server,
        trace,
        policy,
    )

    assert [
        result.arrival_time_seconds
        for result in results
    ] == [12.0, 17.0, 22.0]

    assert [
        result.started_at_seconds
        for result in results
    ] == [12.0, 17.0, 22.0]

    assert [
        result.completed_at_seconds
        for result in results
    ] == [14.0, 19.0, 24.0]


def test_always_on_requests_are_all_warm() -> None:
    """Ensure every request finds an already-ready server."""

    server = make_server()
    trace = make_trace()

    results = run_policy_workload(
        server,
        trace,
        AlwaysOnPolicy(),
    )

    assert [
        result.cold_start
        for result in results
    ] == [False, False, False]

    assert [
        result.waiting_time_seconds
        for result in results
    ] == [0.0, 0.0, 0.0]

    assert [
        result.total_latency_seconds
        for result in results
    ] == [2.0, 2.0, 2.0]


def test_policy_runner_preserves_original_trace() -> None:
    """Keep the reusable workload trace in relative time."""

    server = make_server()
    trace = make_trace()

    run_policy_workload(
        server,
        trace,
        AlwaysOnPolicy(),
    )

    # The runner creates absolute event copies instead of changing trace.
    assert [
        event.scheduled_at_seconds
        for event in trace.events
    ] == [0.0, 5.0, 10.0]


def test_always_on_records_complete_lifecycle() -> None:
    """Record startup, ready residency, and final shutdown."""

    server = make_server()
    trace = make_trace()

    run_policy_workload(
        server,
        trace,
        AlwaysOnPolicy(),
    )

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
        LifecycleEvent(
            at_seconds=24.0,
            state=ServerState.OFF,
        ),
    )

    assert server.state is ServerState.OFF
    assert server.current_time_seconds == 24.0


def test_policy_runner_returns_immutable_results() -> None:
    """Return completed results as an immutable tuple."""

    server = make_server()
    trace = make_trace()

    results = run_policy_workload(
        server,
        trace,
        AlwaysOnPolicy(),
    )

    assert isinstance(results, tuple)
    assert len(results) == 3


def test_policy_runner_rejects_invalid_server() -> None:
    """Require a SimulatedServer instance."""

    trace = make_trace()

    with pytest.raises(ValueError, match="SimulatedServer"):
        run_policy_workload(
            "not-a-server",
            trace,
            AlwaysOnPolicy(),
        )


def test_policy_runner_rejects_invalid_trace() -> None:
    """Require a WorkloadTrace instance."""

    server = make_server()

    with pytest.raises(ValueError, match="WorkloadTrace"):
        run_policy_workload(
            server,
            "not-a-trace",
            AlwaysOnPolicy(),
        )


def test_policy_runner_rejects_invalid_policy() -> None:
    """Require a WarmPolicy implementation."""

    server = make_server()
    trace = make_trace()

    with pytest.raises(ValueError, match="WarmPolicy"):
        run_policy_workload(
            server,
            trace,
            "not-a-policy",
        )