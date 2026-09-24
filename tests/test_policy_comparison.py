"""Metric comparisons between implemented warm-state policies."""

import pytest

from serverless_llm.policies import (
    AlwaysOnPolicy,
    NaiveServerlessPolicy,
    WarmPolicy,
)
from serverless_llm.policy_runner import run_policy_workload
from serverless_llm.simulation_metrics import (
    SimulationSummary,
    summarize_run,
)
from serverless_llm.simulator import ServerTiming, SimulatedServer
from serverless_llm.workload import WorkloadTrace, generate_sparse_workload


def run_experiment(
    trace: WorkloadTrace,
    policy: WarmPolicy,
) -> SimulationSummary:
    """Run one policy with controlled server timing and summarize it."""

    server = SimulatedServer(
        timing=ServerTiming(
            startup_duration_seconds=12.0,
            request_duration_seconds=2.0,
        )
    )
    results = run_policy_workload(server, trace, policy)
    return summarize_run(
        server,
        results,
        total_latency_slo_seconds=10.0,
    )


def test_naive_sparse_workload_summary() -> None:
    """Calculate the expected resource and latency cost of Naive."""

    trace = generate_sparse_workload(
        total_requests=3,
        min_interval_seconds=30.0,
        max_interval_seconds=30.0,
        random_seed=42,
    )

    summary = run_experiment(
        trace,
        NaiveServerlessPolicy(),
    )

    assert summary.total_requests == 3
    assert summary.cold_start_count == 3
    assert summary.cold_start_rate == pytest.approx(1.0)
    assert summary.mean_latency_seconds == pytest.approx(14.0)
    assert summary.p50_latency_seconds == pytest.approx(14.0)
    assert summary.p95_latency_seconds == pytest.approx(14.0)
    assert summary.mean_waiting_seconds == pytest.approx(12.0)
    assert summary.latency_slo_violation_count == 3
    assert summary.latency_slo_violation_rate == pytest.approx(1.0)
    assert summary.startup_time_seconds == pytest.approx(36.0)
    assert summary.model_ready_time_seconds == pytest.approx(6.0)
    assert summary.request_processing_time_seconds == pytest.approx(6.0)
    assert summary.idle_warm_time_seconds == pytest.approx(0.0)
    assert summary.server_running_time_seconds == pytest.approx(42.0)


def test_always_on_and_naive_expose_expected_tradeoff() -> None:
    """Compare latency protection against unused warm residency."""

    trace = generate_sparse_workload(
        total_requests=3,
        min_interval_seconds=30.0,
        max_interval_seconds=30.0,
        random_seed=42,
    )

    always_on = run_experiment(trace, AlwaysOnPolicy())
    naive = run_experiment(trace, NaiveServerlessPolicy())

    # Always-On protects latency by preparing and retaining the model.
    assert always_on.cold_start_rate < naive.cold_start_rate
    assert always_on.mean_latency_seconds < naive.mean_latency_seconds
    assert (
        always_on.latency_slo_violation_rate
        < naive.latency_slo_violation_rate
    )

    # Naive eliminates unused READY time by shutting down immediately.
    assert naive.idle_warm_time_seconds < always_on.idle_warm_time_seconds

    # Repeated cold starts make Naive spend more cumulative startup time.
    assert naive.startup_time_seconds > always_on.startup_time_seconds

