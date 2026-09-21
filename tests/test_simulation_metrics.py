"""Tests for simulation summary metrics."""

from dataclasses import FrozenInstanceError

import pytest

from serverless_llm.simulation_metrics import (
    SimulationSummary,
    _percentile,
    summarize_run,
)
from serverless_llm.simulation_runner import run_workload
from serverless_llm.simulator import (
    RequestResult,
    ServerState,
    ServerTiming,
    SimulatedServer,
)
from serverless_llm.workload import generate_steady_workload


def make_server() -> SimulatedServer:
    """Create a server with deterministic timing for metric tests."""

    return SimulatedServer(
        timing=ServerTiming(
            startup_duration_seconds=12.0,
            request_duration_seconds=2.0,
        )
    )


def run_steady_simulation() -> tuple[
    SimulatedServer,
    tuple[RequestResult, ...],
]:
    """Run three requests that all arrive during model startup."""

    server = make_server()
    trace = generate_steady_workload(
        total_requests=3,
        interval_seconds=5.0,
        random_seed=42,
    )
    results = run_workload(server, trace)
    return server, results


def test_percentile_uses_linear_interpolation() -> None:
    """Interpolate between neighboring values for non-integer positions."""

    values = [14.0, 8.0, 11.0]

    assert _percentile(values, 0.50) == pytest.approx(11.0)
    assert _percentile(values, 0.95) == pytest.approx(13.7)


def test_summarize_run_calculates_request_metrics() -> None:
    """Summarize latency, cold starts, and SLO violations."""

    server, results = run_steady_simulation()

    summary = summarize_run(
        server,
        results,
        total_latency_slo_seconds=10.0,
    )

    assert summary.total_requests == 3
    assert summary.total_latency_slo_seconds == 10.0
    assert summary.cold_start_count == 3
    assert summary.cold_start_rate == pytest.approx(1.0)
    assert summary.mean_latency_seconds == pytest.approx(11.0)
    assert summary.p50_latency_seconds == pytest.approx(11.0)
    assert summary.p95_latency_seconds == pytest.approx(13.7)
    assert summary.mean_waiting_seconds == pytest.approx(9.0)
    assert summary.mean_processing_seconds == pytest.approx(2.0)
    assert summary.latency_slo_violation_count == 2
    assert summary.latency_slo_violation_rate == pytest.approx(2 / 3)


def test_summarize_run_calculates_server_time_metrics() -> None:
    """Separate startup, processing, and unused warm residency time."""

    server, results = run_steady_simulation()

    # Keep the ready model resident for five seconds after the last request.
    server.advance_to(23.0)

    summary = summarize_run(
        server,
        results,
        total_latency_slo_seconds=10.0,
    )

    assert summary.startup_time_seconds == pytest.approx(12.0)
    assert summary.model_ready_time_seconds == pytest.approx(11.0)
    assert summary.server_running_time_seconds == pytest.approx(23.0)
    assert summary.request_processing_time_seconds == pytest.approx(6.0)
    assert summary.idle_warm_time_seconds == pytest.approx(5.0)


def test_simulation_summary_is_immutable() -> None:
    """Prevent completed experiment results from being modified."""

    server, results = run_steady_simulation()
    summary = summarize_run(
        server,
        results,
        total_latency_slo_seconds=10.0,
    )

    with pytest.raises(FrozenInstanceError):
        summary.total_requests = 99


@pytest.mark.parametrize(
    "server",
    [None, "not-a-server", 123],
)
def test_summarize_run_rejects_invalid_server(
    server: object,
) -> None:
    """Require lifecycle data from a SimulatedServer instance."""

    _, results = run_steady_simulation()

    with pytest.raises(ValueError, match="SimulatedServer"):
        summarize_run(
            server,
            results,
            total_latency_slo_seconds=10.0,
        )

@pytest.mark.parametrize(
    "results",
    [
        (),
        [],
        ("not-a-result",),
    ],
)
def test_summarize_run_rejects_invalid_results(
    results: object,
) -> None:
    """Require a non-empty immutable tuple of request results."""

    server = make_server()

    with pytest.raises(ValueError, match="RequestResult"):
        summarize_run(
            server,
            results,
            total_latency_slo_seconds=10.0,
        )


@pytest.mark.parametrize(
    "slo_seconds",
    [0.0, -1.0, float("inf"), float("nan"), True, "10"],
)
def test_summarize_run_rejects_invalid_slo(
    slo_seconds: object,
) -> None:
    """Require a positive finite numeric SLO threshold."""

    server, results = run_steady_simulation()

    with pytest.raises(ValueError, match="positive and finite"):
        summarize_run(
            server,
            results,
            total_latency_slo_seconds=slo_seconds,
        )


def test_summarize_run_rejects_overlapping_processing() -> None:
    """Reject concurrency that the single-request simulator cannot model."""

    server = SimulatedServer(
        timing=ServerTiming(
            startup_duration_seconds=12.0,
            request_duration_seconds=2.0,
        ),
        state=ServerState.READY,
    )
    server.advance_to(4.0)
    results = (
        RequestResult(1, 0.0, 0.0, 3.0, False),
        RequestResult(2, 1.0, 2.0, 4.0, False),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        summarize_run(
            server,
            results,
            total_latency_slo_seconds=10.0,
        )


def test_summarize_run_requires_processing_while_ready() -> None:
    """Reject results recorded outside a READY lifecycle interval."""

    server = make_server()
    server.advance_to(2.0)
    results = (
        RequestResult(1, 0.0, 0.0, 2.0, True),
    )

    with pytest.raises(ValueError, match="while server is READY"):
        summarize_run(
            server,
            results,
            total_latency_slo_seconds=10.0,
        )
