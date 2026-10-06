"""Unit tests for real experiment timeline normalization."""

from dataclasses import FrozenInstanceError

import pytest

from serverless_llm.docker_runtime import RuntimeStartupResult
from serverless_llm.gpu_monitor import GPUSample
from serverless_llm.real_metrics import (
    NormalizedRunTimeline,
    RelativeGPUSample,
    RelativeRequestTiming,
    RelativeStartupTiming,
    _seconds_since,
    normalize_run_timeline,
)
from serverless_llm.real_policy_runner import (
    RealPolicyRunResult,
    RealRequestResult,
)
from serverless_llm.vllm_client import CompletionResult
from serverless_llm.workload import RequestEvent


def make_run_result() -> RealPolicyRunResult:
    """Build one deterministic real run without Docker, HTTP, or a GPU."""

    request_result = RealRequestResult(
        event=RequestEvent(
            request_id=1,
            scheduled_at_seconds=10.0,
            prompt_id="default",
        ),
        scheduled_for_seconds=110.0,
        handling_started_at_seconds=111.0,
        completion=CompletionResult(
            request_started_at_seconds=120.0,
            first_token_at_seconds=120.5,
            completed_at_seconds=122.0,
            generated_text="generated text",
            finish_reason="length",
        ),
        cold_start=True,
        startup_duration_seconds=8.0,
    )

    return RealPolicyRunResult(
        policy_name="naive_serverless",
        experiment_started_at_seconds=100.0,
        workload_started_at_seconds=110.0,
        experiment_completed_at_seconds=140.0,
        startup_results=(
            RuntimeStartupResult(
                started_at_seconds=111.0,
                ready_at_seconds=119.0,
            ),
        ),
        request_results=(request_result,),
        gpu_samples=(
            GPUSample(
                sampled_at_seconds=105.0,
                gpu_index=0,
                utilization_percent=50.0,
                memory_used_mib=2500.0,
                power_draw_watts=40.0,
                temperature_celsius=48.0,
            ),
        ),
    )


def test_seconds_since_subtracts_origin() -> None:
    assert _seconds_since(115.5, 100.0) == 15.5


def test_seconds_since_returns_zero_at_origin() -> None:
    assert _seconds_since(100.0, 100.0) == 0.0


@pytest.mark.parametrize(
    "timestamp_seconds",
    [-1.0, float("inf"), float("-inf"), float("nan"), True, "101"],
)
def test_seconds_since_rejects_invalid_timestamp(
    timestamp_seconds: object,
) -> None:
    with pytest.raises(ValueError, match="timestamp_seconds must"):
        _seconds_since(
            timestamp_seconds,  # type: ignore[arg-type]
            100.0,
        )


@pytest.mark.parametrize(
    "origin_seconds",
    [-1.0, float("inf"), float("-inf"), float("nan"), True, "100"],
)
def test_seconds_since_rejects_invalid_origin(
    origin_seconds: object,
) -> None:
    with pytest.raises(ValueError, match="origin_seconds must"):
        _seconds_since(
            101.0,
            origin_seconds,  # type: ignore[arg-type]
        )


def test_seconds_since_rejects_timestamp_before_origin() -> None:
    with pytest.raises(ValueError, match="cannot be earlier"):
        _seconds_since(99.0, 100.0)


def test_normalize_run_timeline_converts_all_timestamps() -> None:
    timeline = normalize_run_timeline(make_run_result())

    assert timeline == NormalizedRunTimeline(
        policy_name="naive_serverless",
        workload_started_at_seconds=10.0,
        experiment_completed_at_seconds=40.0,
        startup_timings=(
            RelativeStartupTiming(
                started_at_seconds=11.0,
                ready_at_seconds=19.0,
            ),
        ),
        request_timings=(
            RelativeRequestTiming(
                request_id=1,
                scheduled_at_seconds=10.0,
                handling_started_at_seconds=11.0,
                inference_started_at_seconds=20.0,
                first_token_at_seconds=20.5,
                completed_at_seconds=22.0,
            ),
        ),
        gpu_samples=(
            RelativeGPUSample(
                sampled_at_seconds=5.0,
                gpu_index=0,
                utilization_percent=50.0,
                memory_used_mib=2500.0,
                power_draw_watts=40.0,
                temperature_celsius=48.0,
            ),
        ),
    )


def test_normalize_run_timeline_preserves_gpu_measurements() -> None:
    run_result = make_run_result()

    timeline = normalize_run_timeline(run_result)

    original = run_result.gpu_samples[0]
    normalized = timeline.gpu_samples[0]

    assert normalized.gpu_index == original.gpu_index
    assert normalized.utilization_percent == original.utilization_percent
    assert normalized.memory_used_mib == original.memory_used_mib
    assert normalized.power_draw_watts == original.power_draw_watts
    assert normalized.temperature_celsius == original.temperature_celsius


def test_relative_startup_timing_calculates_duration() -> None:
    timing = RelativeStartupTiming(
        started_at_seconds=11.0,
        ready_at_seconds=19.0,
    )

    assert timing.startup_duration_seconds == 8.0


def test_normalized_timeline_is_immutable() -> None:
    timeline = normalize_run_timeline(make_run_result())

    with pytest.raises(FrozenInstanceError):
        timeline.policy_name = "always_on"  # type: ignore[misc]


def test_normalize_run_timeline_rejects_wrong_object_type() -> None:
    with pytest.raises(ValueError, match="RealPolicyRunResult"):
        normalize_run_timeline(object())  # type: ignore[arg-type]


def test_normalize_run_timeline_rejects_measurement_before_experiment() -> None:
    run_result = make_run_result()
    invalid_run = RealPolicyRunResult(
        policy_name=run_result.policy_name,
        experiment_started_at_seconds=run_result.experiment_started_at_seconds,
        workload_started_at_seconds=run_result.workload_started_at_seconds,
        experiment_completed_at_seconds=(
            run_result.experiment_completed_at_seconds
        ),
        startup_results=(
            RuntimeStartupResult(
                started_at_seconds=99.0,
                ready_at_seconds=101.0,
            ),
        ),
        request_results=run_result.request_results,
        gpu_samples=run_result.gpu_samples,
    )

    with pytest.raises(ValueError, match="cannot be earlier"):
        normalize_run_timeline(invalid_run)

