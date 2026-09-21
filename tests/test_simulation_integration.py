"""End-to-end tests for the complete simulation pipeline."""

import pytest

from serverless_llm.simulation_metrics import (
    SimulationSummary,
    summarize_run,
)
from serverless_llm.simulation_runner import run_workload
from serverless_llm.simulator import (
    RequestResult,
    ServerTiming,
    SimulatedServer,
)
from serverless_llm.workload import (
    WorkloadTrace,
    generate_sparse_workload,
    generate_steady_workload,
)


def make_server() -> SimulatedServer:
    """Create a server with deterministic simulation timing."""

    return SimulatedServer(
        timing=ServerTiming(
            startup_duration_seconds=12.0,
            request_duration_seconds=2.0,
        )
    )


def execute_experiment(
    trace: WorkloadTrace,
) -> tuple[
    SimulatedServer,
    tuple[RequestResult, ...],
    SimulationSummary,
]:
    """Run one complete workload-to-summary experiment."""

    # Start every experiment with a fresh server in the OFF state.
    server = make_server()

    # Convert scheduled requests into completed request results.
    results = run_workload(server, trace)

    # Convert raw request and lifecycle records into comparable metrics.
    summary = summarize_run(
        server,
        results,
        total_latency_slo_seconds=10.0,
    )

    return server, results, summary


def test_steady_workload_runs_through_complete_pipeline() -> None:
    """Connect workload generation, simulation, and metric calculation."""

    trace = generate_steady_workload(
        total_requests=3,
        interval_seconds=5.0,
        random_seed=42,
    )

    server, results, summary = execute_experiment(trace)

    # The generator schedules requests at 0, 5, and 10 seconds.
    assert [
        event.scheduled_at_seconds
        for event in trace.events
    ] == [0.0, 5.0, 10.0]

    # The runner preserves request order and completes every request.
    assert [result.request_id for result in results] == [1, 2, 3]
    assert [
        result.completed_at_seconds
        for result in results
    ] == [14.0, 16.0, 18.0]

    # All requests arrive before the model becomes READY at 12 seconds.
    assert summary.total_requests == 3
    assert summary.cold_start_count == 3
    assert summary.cold_start_rate == pytest.approx(1.0)

    # Request latencies are 14, 11, and 8 seconds.
    assert summary.mean_latency_seconds == pytest.approx(11.0)
    assert summary.p50_latency_seconds == pytest.approx(11.0)
    assert summary.p95_latency_seconds == pytest.approx(13.7)

    # Latencies of 14 and 11 seconds exceed the 10-second SLO.
    assert summary.latency_slo_violation_count == 2
    assert summary.latency_slo_violation_rate == pytest.approx(
        2 / 3
    )

    # The model loads for 12 seconds and processes for 6 seconds.
    assert summary.startup_time_seconds == pytest.approx(12.0)
    assert summary.model_ready_time_seconds == pytest.approx(6.0)
    assert summary.request_processing_time_seconds == pytest.approx(
        6.0
    )
    assert summary.idle_warm_time_seconds == pytest.approx(0.0)
    assert server.current_time_seconds == pytest.approx(18.0)


def test_sparse_workload_exposes_idle_warm_time() -> None:
    """Measure unused READY time between widely separated requests."""

    trace = generate_sparse_workload(
        total_requests=3,
        min_interval_seconds=30.0,
        max_interval_seconds=30.0,
        random_seed=42,
    )

    server, results, summary = execute_experiment(trace)

    # Fixed minimum and maximum intervals create known arrival times.
    assert [
        event.scheduled_at_seconds
        for event in trace.events
    ] == [0.0, 30.0, 60.0]

    assert [
        result.completed_at_seconds
        for result in results
    ] == [14.0, 32.0, 62.0]

    # Only the first request arrives before the model becomes READY.
    assert summary.cold_start_count == 1
    assert summary.cold_start_rate == pytest.approx(1 / 3)

    # Request latencies are 14, 2, and 2 seconds.
    assert summary.mean_latency_seconds == pytest.approx(6.0)
    assert summary.p50_latency_seconds == pytest.approx(2.0)
    assert summary.p95_latency_seconds == pytest.approx(12.8)

    # Only the first request exceeds the 10-second latency SLO.
    assert summary.latency_slo_violation_count == 1
    assert summary.latency_slo_violation_rate == pytest.approx(
        1 / 3
    )

    # READY time 50 minus processing time 6 gives idle warm time 44.
    assert summary.startup_time_seconds == pytest.approx(12.0)
    assert summary.model_ready_time_seconds == pytest.approx(50.0)
    assert summary.request_processing_time_seconds == pytest.approx(
        6.0
    )
    assert summary.idle_warm_time_seconds == pytest.approx(44.0)
    assert server.current_time_seconds == pytest.approx(62.0)


def test_time_accounting_remains_consistent() -> None:
    """Verify relationships between independently calculated metrics."""

    trace = generate_sparse_workload(
        total_requests=3,
        min_interval_seconds=30.0,
        max_interval_seconds=30.0,
        random_seed=42,
    )

    _, _, summary = execute_experiment(trace)

    # Running time consists of model startup time plus READY time.
    assert summary.server_running_time_seconds == pytest.approx(
        summary.startup_time_seconds
        + summary.model_ready_time_seconds
    )

    # READY time consists of processing and unused warm time.
    assert summary.model_ready_time_seconds == pytest.approx(
        summary.request_processing_time_seconds
        + summary.idle_warm_time_seconds
    )

    # Every reported rate must agree with its underlying count.
    assert summary.cold_start_rate == pytest.approx(
        summary.cold_start_count
        / summary.total_requests
    )
    assert summary.latency_slo_violation_rate == pytest.approx(
        summary.latency_slo_violation_count
        / summary.total_requests
    )


def test_same_configuration_produces_identical_results() -> None:
    """Ensure deterministic inputs reproduce the same experiment output."""

    first_trace = generate_sparse_workload(
        total_requests=5,
        min_interval_seconds=20.0,
        max_interval_seconds=40.0,
        random_seed=42,
    )
    second_trace = generate_sparse_workload(
        total_requests=5,
        min_interval_seconds=20.0,
        max_interval_seconds=40.0,
        random_seed=42,
    )

    first_server, first_results, first_summary = (
        execute_experiment(first_trace)
    )
    second_server, second_results, second_summary = (
        execute_experiment(second_trace)
    )

    # The workload, request results, lifecycle, and metrics must match.
    assert first_trace == second_trace
    assert first_results == second_results
    assert (
        first_server.lifecycle_events
        == second_server.lifecycle_events
    )
    assert first_summary == second_summary
