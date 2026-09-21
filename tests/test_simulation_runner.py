"""Tests for running complete workload traces."""

import pytest

from serverless_llm.simulation_runner import run_workload
from serverless_llm.simulator import (
    ServerState,
    ServerTiming,
    SimulatedServer,
)
from serverless_llm.workload import (
    generate_sparse_workload,
    generate_steady_workload,
)


def make_server() -> SimulatedServer:
    """Create a server with fixed startup and request durations."""

    return SimulatedServer(
        timing=ServerTiming(
            startup_duration_seconds=12.0,
            request_duration_seconds=2.0,
        )
    )


def test_run_workload_returns_one_result_per_request() -> None:
    server = make_server()
    trace = generate_steady_workload(
        total_requests=3,
        interval_seconds=5.0,
        random_seed=42,
    )

    results = run_workload(server, trace)

    assert len(results) == 3
    assert [result.request_id for result in results] == [1, 2, 3]
    assert isinstance(results, tuple)


def test_run_workload_preserves_arrival_and_processing_times() -> None:
    server = make_server()
    trace = generate_steady_workload(
        total_requests=3,
        interval_seconds=5.0,
        random_seed=42,
    )

    results = run_workload(server, trace)

    assert [
        result.arrival_time_seconds for result in results
    ] == [0.0, 5.0, 10.0]

    assert [
        result.started_at_seconds for result in results
    ] == [12.0, 14.0, 16.0]

    assert [
        result.completed_at_seconds for result in results
    ] == [14.0, 16.0, 18.0]


def test_requests_arriving_during_startup_are_marked_as_cold_starts() -> None:
    server = make_server()
    trace = generate_steady_workload(
        total_requests=3,
        interval_seconds=5.0,
        random_seed=42,
    )

    results = run_workload(server, trace)

    assert [result.cold_start for result in results] == [
        True,
        True,
        True,
    ]
    assert server.state is ServerState.READY
    assert server.current_time_seconds == 18.0


def test_sparse_requests_advance_through_idle_time() -> None:
    server = make_server()
    trace = generate_sparse_workload(
        total_requests=3,
        min_interval_seconds=30.0,
        max_interval_seconds=30.0,
        random_seed=42,
    )

    results = run_workload(server, trace)

    assert [
        result.arrival_time_seconds for result in results
    ] == [0.0, 30.0, 60.0]

    assert [
        result.completed_at_seconds for result in results
    ] == [14.0, 32.0, 62.0]


def test_run_workload_rejects_invalid_server() -> None:
    trace = generate_steady_workload(
        total_requests=1,
        interval_seconds=5.0,
        random_seed=42,
    )

    with pytest.raises(ValueError, match="SimulatedServer"):
        run_workload("not-a-server", trace)


def test_run_workload_rejects_invalid_trace() -> None:
    server = make_server()

    with pytest.raises(ValueError, match="WorkloadTrace"):
        run_workload(server, "not-a-trace")
