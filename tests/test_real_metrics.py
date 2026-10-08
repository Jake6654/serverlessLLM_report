"""Unit tests for real experiment timeline normalization."""

from dataclasses import FrozenInstanceError, replace

import pytest

from serverless_llm.docker_runtime import RuntimeStartupResult
from serverless_llm.gpu_monitor import GPUSample
from serverless_llm.real_metrics import (
    NormalizedRunTimeline,
    RealGPUSummary,
    RealPolicySummary,
    RealRequestMetrics,
    RealRequestSummary,
    RealStartupSummary,
    RelativeGPUSample,
    RelativeRequestTiming,
    RelativeStartupTiming,
    _percentile,
    _seconds_since,
    calculate_request_metrics,
    normalize_run_timeline,
    summarize_gpu_usage,
    summarize_policy_run,
    summarize_request_metrics,
    summarize_startups,
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


def make_valid_request_metrics() -> RealRequestMetrics:
    """Build one valid metric object for validation tests."""

    return RealRequestMetrics(
        request_id=1,
        prompt_id="default",
        cold_start=True,
        startup_duration_seconds=8.0,
        scheduling_delay_seconds=1.0,
        pre_inference_delay_seconds=9.0,
        client_ttft_seconds=0.5,
        end_to_end_ttft_seconds=10.5,
        generation_duration_seconds=1.5,
        client_latency_seconds=2.0,
        end_to_end_latency_seconds=12.0,
        finish_reason="length",
    )


def make_warm_request(
    request_id: int = 2,
) -> RealRequestResult:
    """Build one deterministic warm request."""

    return RealRequestResult(
        event=RequestEvent(
            request_id=request_id,
            scheduled_at_seconds=5.0,
            prompt_id="warm-prompt",
        ),
        scheduled_for_seconds=105.0,
        handling_started_at_seconds=105.25,
        completion=CompletionResult(
            request_started_at_seconds=105.5,
            first_token_at_seconds=106.0,
            completed_at_seconds=107.0,
            generated_text="warm response",
            finish_reason="stop",
        ),
        cold_start=False,
        startup_duration_seconds=None,
    )


def make_gpu_sample(
    sampled_at_seconds: float,
    utilization_percent: float,
    memory_used_mib: float,
    power_draw_watts: float,
    temperature_celsius: float,
) -> GPUSample:
    """Build one deterministic GPU sample."""

    return GPUSample(
        sampled_at_seconds=sampled_at_seconds,
        gpu_index=0,
        utilization_percent=utilization_percent,
        memory_used_mib=memory_used_mib,
        power_draw_watts=power_draw_watts,
        temperature_celsius=temperature_celsius,
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


def test_calculate_request_metrics_calculates_cold_request() -> None:
    metrics = calculate_request_metrics(make_run_result())

    assert metrics == (
        RealRequestMetrics(
            request_id=1,
            prompt_id="default",
            cold_start=True,
            startup_duration_seconds=8.0,
            scheduling_delay_seconds=1.0,
            pre_inference_delay_seconds=9.0,
            client_ttft_seconds=0.5,
            end_to_end_ttft_seconds=10.5,
            generation_duration_seconds=1.5,
            client_latency_seconds=2.0,
            end_to_end_latency_seconds=12.0,
            finish_reason="length",
        ),
    )


def test_calculate_request_metrics_calculates_warm_request() -> None:
    warm_request = make_warm_request()
    run_result = RealPolicyRunResult(
        policy_name="always_on",
        experiment_started_at_seconds=100.0,
        workload_started_at_seconds=100.0,
        experiment_completed_at_seconds=108.0,
        startup_results=(RuntimeStartupResult(100.0, 104.0),),
        request_results=(warm_request,),
        gpu_samples=(),
    )

    metrics = calculate_request_metrics(run_result)

    assert metrics == (
        RealRequestMetrics(
            request_id=2,
            prompt_id="warm-prompt",
            cold_start=False,
            startup_duration_seconds=None,
            scheduling_delay_seconds=0.25,
            pre_inference_delay_seconds=0.25,
            client_ttft_seconds=0.5,
            end_to_end_ttft_seconds=1.0,
            generation_duration_seconds=1.0,
            client_latency_seconds=1.5,
            end_to_end_latency_seconds=2.0,
            finish_reason="stop",
        ),
    )


def test_calculate_request_metrics_preserves_request_order() -> None:
    first_run = make_run_result()
    first_request = first_run.request_results[0]
    second_request = RealRequestResult(
        event=RequestEvent(2, 25.0, "second"),
        scheduled_for_seconds=125.0,
        handling_started_at_seconds=125.0,
        completion=CompletionResult(
            request_started_at_seconds=125.0,
            first_token_at_seconds=125.25,
            completed_at_seconds=126.0,
            generated_text="second response",
            finish_reason="stop",
        ),
        cold_start=False,
        startup_duration_seconds=None,
    )
    run_result = RealPolicyRunResult(
        policy_name="fixed_keep_warm",
        experiment_started_at_seconds=100.0,
        workload_started_at_seconds=100.0,
        experiment_completed_at_seconds=127.0,
        startup_results=first_run.startup_results,
        request_results=(first_request, second_request),
        gpu_samples=(),
    )

    metrics = calculate_request_metrics(run_result)

    assert [metric.request_id for metric in metrics] == [1, 2]
    assert [metric.prompt_id for metric in metrics] == ["default", "second"]


def test_calculate_request_metrics_rejects_wrong_object_type() -> None:
    with pytest.raises(ValueError, match="RealPolicyRunResult"):
        calculate_request_metrics(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("request_id", [0, -1, True, 1.5, "1"])
def test_request_metrics_rejects_invalid_request_id(
    request_id: object,
) -> None:
    with pytest.raises(ValueError, match="request_id"):
        replace(
            make_valid_request_metrics(),
            request_id=request_id,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("prompt_id", ["", "   ", None, 42])
def test_request_metrics_rejects_invalid_prompt_id(
    prompt_id: object,
) -> None:
    with pytest.raises(ValueError, match="prompt_id"):
        replace(
            make_valid_request_metrics(),
            prompt_id=prompt_id,  # type: ignore[arg-type]
        )


def test_request_metrics_rejects_non_boolean_cold_start() -> None:
    with pytest.raises(ValueError, match="cold_start"):
        replace(
            make_valid_request_metrics(),
            cold_start="yes",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("cold_start", "startup_duration_seconds", "message"),
    [
        (True, None, "cold-start request"),
        (False, 8.0, "warm request"),
        (True, -1.0, "startup_duration_seconds"),
        (True, float("inf"), "startup_duration_seconds"),
        (True, float("nan"), "startup_duration_seconds"),
    ],
)
def test_request_metrics_validates_startup_relationship(
    cold_start: bool,
    startup_duration_seconds: float | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(
            make_valid_request_metrics(),
            cold_start=cold_start,
            startup_duration_seconds=startup_duration_seconds,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "scheduling_delay_seconds",
        "pre_inference_delay_seconds",
        "client_ttft_seconds",
        "end_to_end_ttft_seconds",
        "generation_duration_seconds",
        "client_latency_seconds",
        "end_to_end_latency_seconds",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [-1.0, float("inf"), float("-inf"), float("nan"), True, "1.0"],
)
def test_request_metrics_rejects_invalid_duration(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError, match="request durations"):
        replace(
            make_valid_request_metrics(),
            **{field_name: invalid_value},
        )


def test_request_metrics_rejects_invalid_finish_reason() -> None:
    with pytest.raises(ValueError, match="finish_reason"):
        replace(
            make_valid_request_metrics(),
            finish_reason=42,  # type: ignore[arg-type]
        )


def test_request_metrics_is_immutable() -> None:
    metrics = make_valid_request_metrics()

    with pytest.raises(FrozenInstanceError):
        metrics.cold_start = False  # type: ignore[misc]


def test_percentile_sorts_and_interpolates_values() -> None:
    values = [30.0, 10.0, 20.0]

    assert _percentile(values, 0.50) == 20.0
    assert _percentile(values, 0.95) == pytest.approx(29.0)


def test_percentile_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _percentile([], 0.50)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), True])
def test_percentile_rejects_invalid_fraction(fraction: object) -> None:
    with pytest.raises(ValueError, match="fraction"):
        _percentile(
            [1.0],
            fraction,  # type: ignore[arg-type]
        )


def test_summarize_startups_counts_cold_request_and_startup() -> None:
    summary = summarize_startups(make_run_result())

    assert summary == RealStartupSummary(
        total_requests=1,
        cold_start_count=1,
        cold_start_rate=1.0,
        startup_count=1,
        total_startup_time_seconds=8.0,
        mean_startup_time_seconds=8.0,
        p50_startup_time_seconds=8.0,
        p95_startup_time_seconds=8.0,
    )


def test_summarize_startups_distinguishes_always_on_startup() -> None:
    run_result = replace(
        make_run_result(),
        policy_name="always_on",
        startup_results=(RuntimeStartupResult(100.0, 104.0),),
        request_results=(make_warm_request(),),
        gpu_samples=(),
    )

    summary = summarize_startups(run_result)

    assert summary.cold_start_count == 0
    assert summary.cold_start_rate == 0.0
    assert summary.startup_count == 1
    assert summary.mean_startup_time_seconds == 4.0


def test_summarize_startups_calculates_mean_and_percentiles() -> None:
    run_result = replace(
        make_run_result(),
        startup_results=(
            RuntimeStartupResult(100.0, 110.0),
            RuntimeStartupResult(120.0, 140.0),
            RuntimeStartupResult(150.0, 180.0),
        ),
    )

    summary = summarize_startups(run_result)

    assert summary.startup_count == 3
    assert summary.total_startup_time_seconds == 60.0
    assert summary.mean_startup_time_seconds == 20.0
    assert summary.p50_startup_time_seconds == 20.0
    assert summary.p95_startup_time_seconds == pytest.approx(29.0)


def test_summarize_startups_handles_no_startups() -> None:
    run_result = replace(
        make_run_result(),
        startup_results=(),
        request_results=(make_warm_request(),),
        gpu_samples=(),
    )

    summary = summarize_startups(run_result)

    assert summary.startup_count == 0
    assert summary.total_startup_time_seconds == 0.0
    assert summary.mean_startup_time_seconds is None
    assert summary.p50_startup_time_seconds is None
    assert summary.p95_startup_time_seconds is None


def test_summarize_startups_rejects_wrong_object_type() -> None:
    with pytest.raises(ValueError, match="RealPolicyRunResult"):
        summarize_startups(object())  # type: ignore[arg-type]


def test_startup_summary_is_immutable() -> None:
    summary = summarize_startups(make_run_result())

    with pytest.raises(FrozenInstanceError):
        summary.startup_count = 2  # type: ignore[misc]


def test_summarize_gpu_usage_handles_no_samples() -> None:
    run_result = replace(make_run_result(), gpu_samples=())

    summary = summarize_gpu_usage(run_result)

    assert summary == RealGPUSummary(
        sample_count=0,
        monitoring_duration_seconds=0.0,
        mean_utilization_percent=None,
        peak_utilization_percent=None,
        mean_memory_used_mib=None,
        peak_memory_used_mib=None,
        mean_power_draw_watts=None,
        peak_power_draw_watts=None,
        mean_temperature_celsius=None,
        peak_temperature_celsius=None,
        estimated_energy_joules=0.0,
        estimated_energy_watt_hours=0.0,
    )


def test_summarize_gpu_usage_handles_one_sample() -> None:
    summary = summarize_gpu_usage(make_run_result())

    assert summary.sample_count == 1
    assert summary.monitoring_duration_seconds == 0.0
    assert summary.mean_utilization_percent == 50.0
    assert summary.peak_memory_used_mib == 2500.0
    assert summary.mean_power_draw_watts == 40.0
    assert summary.peak_temperature_celsius == 48.0
    assert summary.estimated_energy_joules == 0.0
    assert summary.estimated_energy_watt_hours == 0.0


def test_summarize_gpu_usage_calculates_statistics_and_energy() -> None:
    samples = (
        make_gpu_sample(100.0, 10.0, 1000.0, 40.0, 40.0),
        make_gpu_sample(102.0, 50.0, 2000.0, 60.0, 50.0),
        make_gpu_sample(105.0, 90.0, 1500.0, 20.0, 45.0),
    )
    run_result = replace(make_run_result(), gpu_samples=samples)

    summary = summarize_gpu_usage(run_result)

    assert summary.sample_count == 3
    assert summary.monitoring_duration_seconds == 5.0
    assert summary.mean_utilization_percent == 50.0
    assert summary.peak_utilization_percent == 90.0
    assert summary.mean_memory_used_mib == 1500.0
    assert summary.peak_memory_used_mib == 2000.0
    assert summary.mean_power_draw_watts == 40.0
    assert summary.peak_power_draw_watts == 60.0
    assert summary.mean_temperature_celsius == 45.0
    assert summary.peak_temperature_celsius == 50.0
    assert summary.estimated_energy_joules == 220.0
    assert summary.estimated_energy_watt_hours == pytest.approx(
        220.0 / 3600.0
    )


def test_summarize_gpu_usage_rejects_out_of_order_samples() -> None:
    samples = (
        make_gpu_sample(102.0, 50.0, 2000.0, 60.0, 50.0),
        make_gpu_sample(100.0, 10.0, 1000.0, 40.0, 40.0),
    )
    run_result = replace(make_run_result(), gpu_samples=samples)

    with pytest.raises(ValueError, match="must be ordered"):
        summarize_gpu_usage(run_result)


def test_summarize_gpu_usage_rejects_wrong_object_type() -> None:
    with pytest.raises(ValueError, match="RealPolicyRunResult"):
        summarize_gpu_usage(object())  # type: ignore[arg-type]


def test_gpu_summary_is_immutable() -> None:
    summary = summarize_gpu_usage(make_run_result())

    with pytest.raises(FrozenInstanceError):
        summary.sample_count = 2  # type: ignore[misc]


def test_summarize_request_metrics_calculates_single_request() -> None:
    summary = summarize_request_metrics(
        (make_valid_request_metrics(),),
        ttft_slo_seconds=5.0,
    )

    assert summary == RealRequestSummary(
        total_requests=1,
        ttft_slo_seconds=5.0,
        ttft_slo_violation_count=1,
        ttft_slo_violation_rate=1.0,
        mean_scheduling_delay_seconds=1.0,
        mean_pre_inference_delay_seconds=9.0,
        mean_client_ttft_seconds=0.5,
        p50_client_ttft_seconds=0.5,
        p95_client_ttft_seconds=0.5,
        mean_end_to_end_ttft_seconds=10.5,
        p50_end_to_end_ttft_seconds=10.5,
        p95_end_to_end_ttft_seconds=10.5,
        mean_generation_duration_seconds=1.5,
        mean_client_latency_seconds=2.0,
        p50_client_latency_seconds=2.0,
        p95_client_latency_seconds=2.0,
        mean_end_to_end_latency_seconds=12.0,
        p50_end_to_end_latency_seconds=12.0,
        p95_end_to_end_latency_seconds=12.0,
    )


def test_summarize_request_metrics_calculates_distribution() -> None:
    first = make_valid_request_metrics()
    second = replace(
        first,
        request_id=2,
        cold_start=False,
        startup_duration_seconds=None,
        scheduling_delay_seconds=3.0,
        pre_inference_delay_seconds=1.0,
        client_ttft_seconds=1.5,
        end_to_end_ttft_seconds=5.5,
        generation_duration_seconds=0.5,
        client_latency_seconds=2.0,
        end_to_end_latency_seconds=6.0,
        finish_reason="stop",
    )

    summary = summarize_request_metrics(
        (first, second),
        ttft_slo_seconds=8.0,
    )

    assert summary.total_requests == 2
    assert summary.ttft_slo_violation_count == 1
    assert summary.ttft_slo_violation_rate == 0.5
    assert summary.mean_scheduling_delay_seconds == 2.0
    assert summary.mean_pre_inference_delay_seconds == 5.0
    assert summary.mean_client_ttft_seconds == 1.0
    assert summary.p50_client_ttft_seconds == 1.0
    assert summary.p95_client_ttft_seconds == pytest.approx(1.45)
    assert summary.mean_end_to_end_ttft_seconds == 8.0
    assert summary.p50_end_to_end_ttft_seconds == 8.0
    assert summary.p95_end_to_end_ttft_seconds == pytest.approx(10.25)
    assert summary.mean_generation_duration_seconds == 1.0
    assert summary.mean_client_latency_seconds == 2.0
    assert summary.p95_client_latency_seconds == 2.0
    assert summary.mean_end_to_end_latency_seconds == 9.0
    assert summary.p50_end_to_end_latency_seconds == 9.0
    assert summary.p95_end_to_end_latency_seconds == pytest.approx(11.7)


def test_summarize_request_metrics_does_not_violate_equal_slo() -> None:
    summary = summarize_request_metrics(
        (make_valid_request_metrics(),),
        ttft_slo_seconds=10.5,
    )

    assert summary.ttft_slo_violation_count == 0
    assert summary.ttft_slo_violation_rate == 0.0


@pytest.mark.parametrize(
    "request_metrics",
    [(), [], (object(),)],
)
def test_summarize_request_metrics_rejects_invalid_metrics(
    request_metrics: object,
) -> None:
    with pytest.raises(ValueError, match="request_metrics"):
        summarize_request_metrics(
            request_metrics,  # type: ignore[arg-type]
            ttft_slo_seconds=1.0,
        )


@pytest.mark.parametrize(
    "ttft_slo_seconds",
    [0.0, -1.0, float("inf"), float("nan"), True, "1.0"],
)
def test_summarize_request_metrics_rejects_invalid_slo(
    ttft_slo_seconds: object,
) -> None:
    with pytest.raises(ValueError, match="ttft_slo_seconds"):
        summarize_request_metrics(
            (make_valid_request_metrics(),),
            ttft_slo_seconds=ttft_slo_seconds,  # type: ignore[arg-type]
        )


def test_summarize_policy_run_combines_all_metric_groups() -> None:
    summary = summarize_policy_run(
        make_run_result(),
        ttft_slo_seconds=1.0,
    )

    assert isinstance(summary, RealPolicySummary)
    assert summary.policy_name == "naive_serverless"
    assert summary.experiment_duration_seconds == 40.0
    assert summary.request_summary.total_requests == 1
    assert summary.request_summary.ttft_slo_violation_count == 1
    assert summary.startup_summary.cold_start_count == 1
    assert summary.startup_summary.startup_count == 1
    assert summary.gpu_summary.sample_count == 1
    assert summary.gpu_summary.mean_power_draw_watts == 40.0


def test_summarize_policy_run_rejects_wrong_object_type() -> None:
    with pytest.raises(ValueError, match="RealPolicyRunResult"):
        summarize_policy_run(
            object(),  # type: ignore[arg-type]
            ttft_slo_seconds=1.0,
        )


def test_policy_summary_is_immutable() -> None:
    summary = summarize_policy_run(
        make_run_result(),
        ttft_slo_seconds=1.0,
    )

    with pytest.raises(FrozenInstanceError):
        summary.policy_name = "always_on"  # type: ignore[misc]
