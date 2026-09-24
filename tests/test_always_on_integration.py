"""Integration tests for Always-On experiment metrics."""

import pytest

from serverless_llm.policies import AlwaysOnPolicy
from serverless_llm.policy_runner import run_policy_workload
from serverless_llm.simulation_metrics import summarize_run
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
    """Create a server with deterministic experiment timing."""

    return SimulatedServer(
        timing=ServerTiming(
            startup_duration_seconds=12.0,
            request_duration_seconds=2.0,
        )
    )


def test_always_on_steady_workload_summary() -> None:
    """Produce final metrics for a steady Always-On experiment."""

    server = make_server()
    trace = generate_steady_workload(
        total_requests=3,
        interval_seconds=5.0,
        random_seed=42,
    )

    results = run_policy_workload(
        server,
        trace,
        AlwaysOnPolicy(),
    )

    summary = summarize_run(
        server,
        results,
        total_latency_slo_seconds=10.0,
    )

    # All three requests are included in the final summary.
    assert summary.total_requests == 3

    # The server is ready before workload replay begins.
    assert summary.cold_start_count == 0
    assert summary.cold_start_rate == pytest.approx(0.0)

    # Every request takes only the fixed two-second processing time.
    assert summary.mean_latency_seconds == pytest.approx(2.0)
    assert summary.p50_latency_seconds == pytest.approx(2.0)
    assert summary.p95_latency_seconds == pytest.approx(2.0)
    assert summary.mean_waiting_seconds == pytest.approx(0.0)
    assert summary.mean_processing_seconds == pytest.approx(2.0)

    # A two-second request does not violate the ten-second SLO.
    assert summary.latency_slo_violation_count == 0
    assert summary.latency_slo_violation_rate == pytest.approx(0.0)

    # The model takes twelve seconds to load before replay begins.
    assert summary.startup_time_seconds == pytest.approx(12.0)

    # The server remains READY from simulation time 12 to 24.
    assert summary.model_ready_time_seconds == pytest.approx(12.0)

    # Startup plus READY residency gives total running time.
    assert summary.server_running_time_seconds == pytest.approx(24.0)

    # Three requests each use two seconds of processing.
    assert summary.request_processing_time_seconds == pytest.approx(
        6.0
    )

    # Two three-second gaps occur between the three requests.
    assert summary.idle_warm_time_seconds == pytest.approx(6.0)

    # The policy performs final cleanup after workload completion.
    assert server.state is ServerState.OFF
    assert server.current_time_seconds == pytest.approx(24.0)


def test_always_on_sparse_workload_trades_resources_for_latency() -> None:
    """Expose Always-On idle cost under a sparse request pattern."""

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
        AlwaysOnPolicy(),
    )

    summary = summarize_run(
        server,
        results,
        total_latency_slo_seconds=10.0,
    )

    # Pre-warming prevents cold starts even with long request gaps.
    assert summary.cold_start_count == 0
    assert summary.cold_start_rate == pytest.approx(0.0)

    assert summary.mean_latency_seconds == pytest.approx(2.0)
    assert summary.latency_slo_violation_rate == pytest.approx(0.0)

    # Timeline:
    #   0-12  : model startup
    #   12-14 : request 1
    #   14-42 : idle but READY
    #   42-44 : request 2
    #   44-72 : idle but READY
    #   72-74 : request 3
    assert summary.startup_time_seconds == pytest.approx(12.0)
    assert summary.model_ready_time_seconds == pytest.approx(62.0)
    assert summary.request_processing_time_seconds == pytest.approx(
        6.0
    )
    assert summary.idle_warm_time_seconds == pytest.approx(56.0)
    assert summary.server_running_time_seconds == pytest.approx(74.0)


def test_always_on_summary_preserves_time_accounting() -> None:
    """Verify internal relationships between Always-On metrics."""

    server = make_server()
    trace = generate_steady_workload(
        total_requests=3,
        interval_seconds=5.0,
        random_seed=42,
    )

    results = run_policy_workload(
        server,
        trace,
        AlwaysOnPolicy(),
    )

    summary = summarize_run(
        server,
        results,
        total_latency_slo_seconds=10.0,
    )

    assert summary.server_running_time_seconds == pytest.approx(
        summary.startup_time_seconds
        + summary.model_ready_time_seconds
    )

    assert summary.model_ready_time_seconds == pytest.approx(
        summary.request_processing_time_seconds
        + summary.idle_warm_time_seconds
    )