"""실험 시각 정규화 실제 실험의 통계 계산을 위한 파일"""


from dataclasses import dataclass
from math import isfinite, floor

from serverless_llm.real_policy_runner import (
    RealPolicyRunResult,
)

@dataclass(frozen=True)
class RelativeStartupTiming:
  """Store one vLLM startup interval relative to experiment start"""

  started_at_seconds: float
  ready_at_seconds: float

  @property
  def startup_duration_seconds(self) -> float:

    return self.ready_at_seconds - self.started_at_seconds

@dataclass(frozen=True)
class RelativeRequestTiming:
    """Store one request timeline relative to experiment start."""

    request_id: int
    scheduled_at_seconds: float
    handling_started_at_seconds: float
    inference_started_at_seconds: float
    first_token_at_seconds: float
    completed_at_seconds: float

@dataclass(frozen=True)
class RelativeGPUSample:
    """Store one GPU sample relative to experiment start."""

    sampled_at_seconds: float
    gpu_index: int
    utilization_percent: float
    memory_used_mib: float
    power_draw_watts: float
    temperature_celsius: float

@dataclass(frozen=True)
class NormalizedRunTimeline:
    """Store a complete policy timeline using experiment-relative time."""

    policy_name: str
    workload_started_at_seconds: float
    experiment_completed_at_seconds: float
    startup_timings: tuple[RelativeStartupTiming, ...]
    request_timings: tuple[RelativeRequestTiming, ...]
    gpu_samples: tuple[RelativeGPUSample, ...]

@dataclass(frozen=True)
class RealRequestMetrics:
    """Store comparable performance metrics for one real request"""

    request_id: int
    prompt_id: str
    cold_start: bool
    startup_duration_seconds: float | None
    scheduling_delay_seconds: float
    pre_inference_delay_seconds: float
    client_ttft_seconds: float
    end_to_end_ttft_seconds: float
    generation_duration_seconds: float
    client_latency_seconds: float
    end_to_end_latency_seconds: float
    finish_reason: str | None

    def __post_init__(self) -> None:
        """Reject invalid request metric values."""

        if type(self.request_id) is not int or self.request_id < 1:
            raise ValueError(
                "request_id must be a positive integer"
            )

        if (
            not isinstance(self.prompt_id, str)
            or not self.prompt_id.strip()
        ):
            raise ValueError(
                "prompt_id must be a non-empty string"
            )

        if type(self.cold_start) is not bool:
            raise ValueError(
                "cold_start must be a boolean"
            )

        if self.startup_duration_seconds is not None:
            if (
                type(self.startup_duration_seconds)
                not in (int, float)
                or not isfinite(self.startup_duration_seconds)
                or self.startup_duration_seconds < 0
            ):
                raise ValueError(
                    "startup_duration_seconds must be null or a "
                    "finite non-negative number"
                )

        if self.cold_start and self.startup_duration_seconds is None:
            raise ValueError(
                "a cold-start request must include startup duration"
            )

        if (
            not self.cold_start
            and self.startup_duration_seconds is not None
        ):
            raise ValueError(
                "a warm request must not include startup duration"
            )

        durations = (
            self.scheduling_delay_seconds,
            self.pre_inference_delay_seconds,
            self.client_ttft_seconds,
            self.end_to_end_ttft_seconds,
            self.generation_duration_seconds,
            self.client_latency_seconds,
            self.end_to_end_latency_seconds,
        )

        if any(
            type(duration) not in (int, float)
            or not isfinite(duration)
            or duration < 0
            for duration in durations
        ):
            raise ValueError(
                "request durations must be finite "
                "non-negative numbers"
            )

        if (
            self.finish_reason is not None
            and not isinstance(self.finish_reason, str)
        ):
            raise ValueError(
                "finish_reason must be a string or null"
            )

@dataclass(frozen=True)
class RealStartupSummary:
    """Store cold-start and server-startup metrics for one real run."""

    total_requests: int
    cold_start_count: int
    cold_start_rate: float
    startup_count: int
    total_startup_time_seconds: float
    mean_startup_time_seconds: float | None
    p50_startup_time_seconds: float | None # 중앙값
    p95_startup_time_seconds: float | None # tail latency
    # 전체 서버 시작의 약 95%가 68초 이내에 완료됐고, 나머지 약 5%는 68초보다 오래 걸렸다.
    # 평균과 무언이 다르냐? 한번의 요청이 심하게 오래 걸릴 가능성이 있기 때문에
    # 그 값을 기록하기 위한 용도

    def __post_init__(self) -> None:
        """Reject invalid startup summary values."""

        if type(self.total_requests) is not int or self.total_requests < 1:
            raise ValueError(
                "total_requests must be a positive integer"
            )

        if (
            type(self.cold_start_count) is not int
            or not 0 <= self.cold_start_count <= self.total_requests
        ):
            raise ValueError(
                "cold_start_count must be between zero "
                "and total_requests"
            )

        if (
            type(self.cold_start_rate) not in (int, float)
            or not isfinite(self.cold_start_rate)
            or not 0 <= self.cold_start_rate <= 1
        ):
            raise ValueError(
                "cold_start_rate must be a finite number "
                "between zero and one"
            )

        if type(self.startup_count) is not int or self.startup_count < 0:
            raise ValueError(
                "startup_count must be a non-negative integer"
            )

        if (
            type(self.total_startup_time_seconds) not in (int, float)
            or not isfinite(self.total_startup_time_seconds)
            or self.total_startup_time_seconds < 0
        ):
            raise ValueError(
                "total_startup_time_seconds must be a finite "
                "non-negative number"
            )

        optional_startup_metrics = (
            self.mean_startup_time_seconds,
            self.p50_startup_time_seconds,
            self.p95_startup_time_seconds,
        )

        for metric in optional_startup_metrics:
            if metric is not None and (
                type(metric) not in (int, float)
                or not isfinite(metric)
                or metric < 0
            ):
                raise ValueError(
                    "startup statistics must be null or finite "
                    "non-negative numbers"
                )

        if self.startup_count == 0:
            if any(
                metric is not None
                for metric in optional_startup_metrics
            ):
                raise ValueError(
                    "startup statistics must be null when "
                    "startup_count is zero"
                )

        if self.startup_count > 0:
            if any(
                metric is None
                for metric in optional_startup_metrics
            ):
                raise ValueError(
                    "startup statistics are required when "
                    "startup_count is positive"
                )


@dataclass(frozen=True)
class RealGPUSummary:
    """Store aggregate GPU and estimated energy metrics for one real run."""

    sample_count: int
    monitoring_duration_seconds: float
    mean_utilization_percent: float | None
    peak_utilization_percent: float | None
    mean_memory_used_mib: float | None
    peak_memory_used_mib: float | None
    mean_power_draw_watts: float | None
    peak_power_draw_watts: float | None
    mean_temperature_celsius: float | None
    peak_temperature_celsius: float | None
    estimated_energy_joules: float
    estimated_energy_watt_hours: float

    def __post_init__(self) -> None:
        """Reject inconsistent GPU summary values."""

        if type(self.sample_count) is not int or self.sample_count < 0:
            raise ValueError(
                "sample_count must be a non-negative integer"
            )

        required_non_negative = (
            self.monitoring_duration_seconds,
            self.estimated_energy_joules,
            self.estimated_energy_watt_hours,
        )

        if any(
            type(value) not in (int, float)
            or not isfinite(value)
            or value < 0
            for value in required_non_negative
        ):
            raise ValueError(
                "GPU duration and energy values must be finite "
                "non-negative numbers"
            )

        optional_metrics = (
            self.mean_utilization_percent,
            self.peak_utilization_percent,
            self.mean_memory_used_mib,
            self.peak_memory_used_mib,
            self.mean_power_draw_watts,
            self.peak_power_draw_watts,
            self.mean_temperature_celsius,
            self.peak_temperature_celsius,
        )

        for metric in optional_metrics:
            if metric is not None and (
                type(metric) not in (int, float)
                or not isfinite(metric)
                or metric < 0
            ):
                raise ValueError(
                    "GPU statistics must be null or finite "
                    "non-negative numbers"
                )

        utilization_metrics = (
            self.mean_utilization_percent,
            self.peak_utilization_percent,
        )

        if any(
            metric is not None and metric > 100
            for metric in utilization_metrics
        ):
            raise ValueError(
                "GPU utilization statistics cannot exceed 100 percent"
            )

        if self.sample_count == 0:
            if any(metric is not None for metric in optional_metrics):
                raise ValueError(
                    "GPU statistics must be null when sample_count is zero"
                )

            if (
                self.monitoring_duration_seconds != 0
                or self.estimated_energy_joules != 0
                or self.estimated_energy_watt_hours != 0
            ):
                raise ValueError(
                    "GPU duration and energy must be zero when "
                    "sample_count is zero"
                )

        if self.sample_count > 0 and any(
            metric is None for metric in optional_metrics
        ):
            raise ValueError(
                "GPU statistics are required when sample_count is positive"
            )


@dataclass(frozen=True)
class RealRequestSummary:
    """Store aggregate request latency and TTFT metrics for one real run."""

    total_requests: int
    ttft_slo_seconds: float
    ttft_slo_violation_count: int
    ttft_slo_violation_rate: float
    mean_scheduling_delay_seconds: float
    mean_pre_inference_delay_seconds: float
    mean_client_ttft_seconds: float
    p50_client_ttft_seconds: float
    p95_client_ttft_seconds: float
    mean_end_to_end_ttft_seconds: float
    p50_end_to_end_ttft_seconds: float
    p95_end_to_end_ttft_seconds: float
    mean_generation_duration_seconds: float
    mean_client_latency_seconds: float
    p50_client_latency_seconds: float
    p95_client_latency_seconds: float
    mean_end_to_end_latency_seconds: float
    p50_end_to_end_latency_seconds: float
    p95_end_to_end_latency_seconds: float

    def __post_init__(self) -> None:
        """Reject invalid aggregate request metrics."""

        if type(self.total_requests) is not int or self.total_requests < 1:
            raise ValueError(
                "total_requests must be a positive integer"
            )

        if (
            type(self.ttft_slo_seconds) not in (int, float)
            or not isfinite(self.ttft_slo_seconds)
            or self.ttft_slo_seconds <= 0
        ):
            raise ValueError(
                "ttft_slo_seconds must be a finite positive number"
            )

        if (
            type(self.ttft_slo_violation_count) is not int
            or not 0
            <= self.ttft_slo_violation_count
            <= self.total_requests
        ):
            raise ValueError(
                "ttft_slo_violation_count must be between zero "
                "and total_requests"
            )

        if (
            type(self.ttft_slo_violation_rate) not in (int, float)
            or not isfinite(self.ttft_slo_violation_rate)
            or not 0 <= self.ttft_slo_violation_rate <= 1
        ):
            raise ValueError(
                "ttft_slo_violation_rate must be a finite number "
                "between zero and one"
            )

        expected_violation_rate = (
            self.ttft_slo_violation_count / self.total_requests
        )

        if abs(
            self.ttft_slo_violation_rate - expected_violation_rate
        ) > 1e-12:
            raise ValueError(
                "ttft_slo_violation_rate must match the violation count"
            )

        latency_metrics = (
            self.mean_scheduling_delay_seconds,
            self.mean_pre_inference_delay_seconds,
            self.mean_client_ttft_seconds,
            self.p50_client_ttft_seconds,
            self.p95_client_ttft_seconds,
            self.mean_end_to_end_ttft_seconds,
            self.p50_end_to_end_ttft_seconds,
            self.p95_end_to_end_ttft_seconds,
            self.mean_generation_duration_seconds,
            self.mean_client_latency_seconds,
            self.p50_client_latency_seconds,
            self.p95_client_latency_seconds,
            self.mean_end_to_end_latency_seconds,
            self.p50_end_to_end_latency_seconds,
            self.p95_end_to_end_latency_seconds,
        )

        if any(
            type(metric) not in (int, float)
            or not isfinite(metric)
            or metric < 0
            for metric in latency_metrics
        ):
            raise ValueError(
                "aggregate request metrics must be finite "
                "non-negative numbers"
            )


@dataclass(frozen=True)
class RealPolicySummary:
    """Combine request, startup, and GPU metrics for one policy run."""

    policy_name: str
    experiment_duration_seconds: float
    request_summary: RealRequestSummary
    startup_summary: RealStartupSummary
    gpu_summary: RealGPUSummary

    def __post_init__(self) -> None:
        """Reject inconsistent policy summary values."""

        if not isinstance(self.policy_name, str) or not self.policy_name.strip():
            raise ValueError(
                "policy_name must be a non-empty string"
            )

        if (
            type(self.experiment_duration_seconds) not in (int, float)
            or not isfinite(self.experiment_duration_seconds)
            or self.experiment_duration_seconds < 0
        ):
            raise ValueError(
                "experiment_duration_seconds must be a finite "
                "non-negative number"
            )

        if not isinstance(self.request_summary, RealRequestSummary):
            raise ValueError(
                "request_summary must be a RealRequestSummary object"
            )

        if not isinstance(self.startup_summary, RealStartupSummary):
            raise ValueError(
                "startup_summary must be a RealStartupSummary object"
            )

        if not isinstance(self.gpu_summary, RealGPUSummary):
            raise ValueError(
                "gpu_summary must be a RealGPUSummary object"
            )

        if (
            self.request_summary.total_requests
            != self.startup_summary.total_requests
        ):
            raise ValueError(
                "request and startup summaries must have the same "
                "total request count"
            )


def _seconds_since(
      timestamp_seconds: float,
      origin_seconds: float,
) -> float:
    """Convert an absolute monotonic timestamp into relative seconds"""

    if (
        type(timestamp_seconds) not in (int, float)
        or not isfinite(timestamp_seconds)
        or timestamp_seconds < 0
    ):
        raise ValueError(
            "timestamp_seconds must be a finite non-negative number"
        )

    if (
        type(origin_seconds) not in (int, float)
        or not isfinite(origin_seconds)
        or origin_seconds < 0
    ):
        raise ValueError(
            "origin_seconds must be a finite non-negative number"
        )

    if timestamp_seconds < origin_seconds:
        raise ValueError(
            "timestamp_seconds cannot be earlier than origin_seconds"
        )

    return float(timestamp_seconds - origin_seconds)

def _percentile(
    values: list[float],
    fraction: float,
) -> float:
    """Calculate one percentile using linear interpolation."""

    if not values:
        raise ValueError(
            "values must not be empty"
        )

    if any(
        type(value) not in (int, float)
        or not isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValueError(
            "values must contain finite non-negative numbers"
        )

    if (
        type(fraction) not in (int, float)
        or not isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError(
            "fraction must be a finite number between zero and one"
        )

    ordered = sorted(values)

    position = (len(ordered) - 1) * fraction

    lower_index = floor(position)
    upper_index = min(
        lower_index + 1,
        len(ordered) - 1,
    )

    weight = position - lower_index

    return (
        ordered[lower_index] * (1 - weight)
        + ordered[upper_index] * weight
    )

def normalize_run_timeline(
        run_result: RealPolicyRunResult,
) -> NormalizedRunTimeline:
    """Convert every timestamp into a real run to experiment-relative time"""

    if not isinstance(run_result, RealPolicyRunResult):
        raise ValueError(
            "run_result must be a RealPolicyRunResult object"
        )

    origin_seconds = run_result.experiment_started_at_seconds

    # Convert every vLLM startup timestamp to experiment-relative time
    startup_timings_list : list[RelativeStartupTiming] = []

    for startup_result in run_result.startup_results:
        relative_startup_timing = RelativeStartupTiming(
            started_at_seconds=_seconds_since(
                startup_result.started_at_seconds,
                origin_seconds,
            ),
            ready_at_seconds=_seconds_since(
                startup_result.ready_at_seconds,
                origin_seconds,
            )
        )

        startup_timings_list.append(relative_startup_timing)
    
    startup_timings = tuple(startup_timings_list)

    # Convert every request timestamp to experiment-relative time.
    request_timings_list: list[RelativeRequestTiming] = []

    for request_result in run_result.request_results:
        relative_request_timing = RelativeRequestTiming(
            request_id=request_result.request_id,
            scheduled_at_seconds=_seconds_since(
                request_result.scheduled_for_seconds,
                origin_seconds,
            ),
            handling_started_at_seconds=_seconds_since(
                request_result.handling_started_at_seconds,
                origin_seconds,
            ),
            inference_started_at_seconds=_seconds_since(
                request_result.completion.request_started_at_seconds,
                origin_seconds,
            ),
            first_token_at_seconds=_seconds_since(
                request_result.completion.first_token_at_seconds,
                origin_seconds,
            ),
            completed_at_seconds=_seconds_since(
                request_result.completion.completed_at_seconds,
                origin_seconds,
            ),
        )

        request_timings_list.append(relative_request_timing)

    request_timings = tuple(request_timings_list)

    # Preserve GPU measurements and normalize only their timestamps.
    gpu_samples_list: list[RelativeGPUSample] = []

    for sample in run_result.gpu_samples:
        relative_gpu_sample = RelativeGPUSample(
            sampled_at_seconds=_seconds_since(
                sample.sampled_at_seconds,
                origin_seconds,
            ),
            gpu_index=sample.gpu_index,
            utilization_percent=sample.utilization_percent,
            memory_used_mib=sample.memory_used_mib,
            power_draw_watts=sample.power_draw_watts,
            temperature_celsius=sample.temperature_celsius,
        )

        gpu_samples_list.append(relative_gpu_sample)

    gpu_samples = tuple(gpu_samples_list)

    # Build one immutable object containing the normalized timeline.
    return NormalizedRunTimeline(
        policy_name=run_result.policy_name,
        workload_started_at_seconds=_seconds_since(
            run_result.workload_started_at_seconds,
            origin_seconds,
        ),
        experiment_completed_at_seconds=_seconds_since(
            run_result.experiment_completed_at_seconds,
            origin_seconds,
        ),
        startup_timings=startup_timings,
        request_timings=request_timings,
        gpu_samples=gpu_samples,
    )

def calculate_request_metrics(
    run_result: RealPolicyRunResult,
) -> tuple[RealRequestMetrics, ...]:

    if not isinstance(run_result, RealPolicyRunResult):
        raise ValueError(
            "run_result must be a RealPolicyRunResult object"
        )

    request_metrics_list: list[RealRequestMetrics] = []

    for request_result in run_result.request_results:
        completion = request_result.completion

        request_metrics = RealRequestMetrics(
            request_id=request_result.request_id,
            prompt_id=request_result.event.prompt_id,
            cold_start=request_result.cold_start,
            startup_duration_seconds=(
                request_result.startup_duration_seconds
            ),
            scheduling_delay_seconds=(
                request_result.scheduling_delay_seconds
            ),
            pre_inference_delay_seconds=(
                request_result.pre_inference_delay_seconds
            ),
            client_ttft_seconds=completion.ttft_seconds,
            end_to_end_ttft_seconds=(
                request_result.end_to_end_ttft_seconds
            ),
            generation_duration_seconds=(
                completion.generation_duration_seconds
            ),
            client_latency_seconds=(
                completion.total_latency_seconds
            ),
            end_to_end_latency_seconds=(
                request_result.end_to_end_latency_seconds
            ),
            finish_reason=completion.finish_reason,
        )

        request_metrics_list.append(request_metrics)

    return tuple(request_metrics_list)

def summarize_startups(
    run_result: RealPolicyRunResult,
) -> RealStartupSummary:
    """Summarize cold starts and vLLM startup durations for one run."""

    if not isinstance(run_result, RealPolicyRunResult):
        raise ValueError(
            "run_result must be a RealPolicyRunResult object"
        )

    total_requests = len(run_result.request_results)

    cold_start_count = sum(
        request_result.cold_start
        for request_result in run_result.request_results
    )

    startup_durations: list[float] = []

    for startup_result in run_result.startup_results:
        startup_durations.append(
            startup_result.startup_duration_seconds
        )

    startup_count = len(startup_durations)

    total_startup_time_seconds = sum(startup_durations)

    if startup_count == 0:
        mean_startup_time_seconds = None
        p50_startup_time_seconds = None
        p95_startup_time_seconds = None
    else:
        mean_startup_time_seconds = (
            total_startup_time_seconds / startup_count
        )
        p50_startup_time_seconds = _percentile(
            startup_durations,
            0.50,
        )
        p95_startup_time_seconds = _percentile(
            startup_durations,
            0.95,
        )

    return RealStartupSummary(
        total_requests=total_requests,
        cold_start_count=cold_start_count,
        cold_start_rate=(
            cold_start_count / total_requests
        ),
        startup_count=startup_count,
        total_startup_time_seconds=(
            total_startup_time_seconds
        ),
        mean_startup_time_seconds=(
            mean_startup_time_seconds
        ),
        p50_startup_time_seconds=(
            p50_startup_time_seconds
        ),
        p95_startup_time_seconds=(
            p95_startup_time_seconds
        ),
    )


def summarize_gpu_usage(
    run_result: RealPolicyRunResult,
) -> RealGPUSummary:
    """Summarize GPU samples and estimate energy with trapezoidal integration."""

    if not isinstance(run_result, RealPolicyRunResult):
        raise ValueError(
            "run_result must be a RealPolicyRunResult object"
        )

    samples = run_result.gpu_samples
    sample_count = len(samples)

    if sample_count == 0:
        return RealGPUSummary(
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

    for current, following in zip(samples, samples[1:]):
        if current.sampled_at_seconds > following.sampled_at_seconds:
            raise ValueError(
                "GPU samples must be ordered by sampled_at_seconds"
            )

    utilization_values = [
        sample.utilization_percent
        for sample in samples
    ]
    memory_values = [
        sample.memory_used_mib
        for sample in samples
    ]
    power_values = [
        sample.power_draw_watts
        for sample in samples
    ]
    temperature_values = [
        sample.temperature_celsius
        for sample in samples
    ]

    monitoring_duration_seconds = (
        samples[-1].sampled_at_seconds
        - samples[0].sampled_at_seconds
    )

    # Estimate energy between each pair of samples. The average of the two
    # power readings approximates the area under the power-versus-time curve.
    estimated_energy_joules = 0.0

    for current, following in zip(samples, samples[1:]):
        interval_seconds = (
            following.sampled_at_seconds
            - current.sampled_at_seconds
        )
        average_interval_power_watts = (
            current.power_draw_watts
            + following.power_draw_watts
        ) / 2
        estimated_energy_joules += (
            average_interval_power_watts * interval_seconds
        )

    return RealGPUSummary(
        sample_count=sample_count,
        monitoring_duration_seconds=monitoring_duration_seconds,
        mean_utilization_percent=(
            sum(utilization_values) / sample_count
        ),
        peak_utilization_percent=max(utilization_values),
        mean_memory_used_mib=sum(memory_values) / sample_count,
        peak_memory_used_mib=max(memory_values),
        mean_power_draw_watts=sum(power_values) / sample_count,
        peak_power_draw_watts=max(power_values),
        mean_temperature_celsius=(
            sum(temperature_values) / sample_count
        ),
        peak_temperature_celsius=max(temperature_values),
        estimated_energy_joules=estimated_energy_joules,
        estimated_energy_watt_hours=(
            estimated_energy_joules / 3600.0
        ),
    )


def summarize_request_metrics(
    request_metrics: tuple[RealRequestMetrics, ...],
    *,
    ttft_slo_seconds: float,
) -> RealRequestSummary:
    """Aggregate request-level metrics into one latency summary."""

    if (
        not isinstance(request_metrics, tuple)
        or not request_metrics
        or not all(
            isinstance(metric, RealRequestMetrics)
            for metric in request_metrics
        )
    ):
        raise ValueError(
            "request_metrics must be a non-empty tuple of "
            "RealRequestMetrics objects"
        )

    if (
        type(ttft_slo_seconds) not in (int, float)
        or not isfinite(ttft_slo_seconds)
        or ttft_slo_seconds <= 0
    ):
        raise ValueError(
            "ttft_slo_seconds must be a finite positive number"
        )

    total_requests = len(request_metrics)

    scheduling_delays = [
        metric.scheduling_delay_seconds
        for metric in request_metrics
    ]
    pre_inference_delays = [
        metric.pre_inference_delay_seconds
        for metric in request_metrics
    ]
    client_ttfts = [
        metric.client_ttft_seconds
        for metric in request_metrics
    ]
    end_to_end_ttfts = [
        metric.end_to_end_ttft_seconds
        for metric in request_metrics
    ]
    generation_durations = [
        metric.generation_duration_seconds
        for metric in request_metrics
    ]
    client_latencies = [
        metric.client_latency_seconds
        for metric in request_metrics
    ]
    end_to_end_latencies = [
        metric.end_to_end_latency_seconds
        for metric in request_metrics
    ]

    # A request violates the TTFT SLO only when it exceeds the threshold.
    ttft_slo_violation_count = sum(
        ttft > ttft_slo_seconds
        for ttft in end_to_end_ttfts
    )

    return RealRequestSummary(
        total_requests=total_requests,
        ttft_slo_seconds=float(ttft_slo_seconds),
        ttft_slo_violation_count=ttft_slo_violation_count,
        ttft_slo_violation_rate=(
            ttft_slo_violation_count / total_requests
        ),
        mean_scheduling_delay_seconds=(
            sum(scheduling_delays) / total_requests
        ),
        mean_pre_inference_delay_seconds=(
            sum(pre_inference_delays) / total_requests
        ),
        mean_client_ttft_seconds=(
            sum(client_ttfts) / total_requests
        ),
        p50_client_ttft_seconds=_percentile(
            client_ttfts,
            0.50,
        ),
        p95_client_ttft_seconds=_percentile(
            client_ttfts,
            0.95,
        ),
        mean_end_to_end_ttft_seconds=(
            sum(end_to_end_ttfts) / total_requests
        ),
        p50_end_to_end_ttft_seconds=_percentile(
            end_to_end_ttfts,
            0.50,
        ),
        p95_end_to_end_ttft_seconds=_percentile(
            end_to_end_ttfts,
            0.95,
        ),
        mean_generation_duration_seconds=(
            sum(generation_durations) / total_requests
        ),
        mean_client_latency_seconds=(
            sum(client_latencies) / total_requests
        ),
        p50_client_latency_seconds=_percentile(
            client_latencies,
            0.50,
        ),
        p95_client_latency_seconds=_percentile(
            client_latencies,
            0.95,
        ),
        mean_end_to_end_latency_seconds=(
            sum(end_to_end_latencies) / total_requests
        ),
        p50_end_to_end_latency_seconds=_percentile(
            end_to_end_latencies,
            0.50,
        ),
        p95_end_to_end_latency_seconds=_percentile(
            end_to_end_latencies,
            0.95,
        ),
    )


def summarize_policy_run(
    run_result: RealPolicyRunResult,
    *,
    ttft_slo_seconds: float,
) -> RealPolicySummary:
    """Build the complete comparable summary for one real policy run."""

    if not isinstance(run_result, RealPolicyRunResult):
        raise ValueError(
            "run_result must be a RealPolicyRunResult object"
        )

    request_metrics = calculate_request_metrics(run_result)
    request_summary = summarize_request_metrics(
        request_metrics,
        ttft_slo_seconds=ttft_slo_seconds,
    )
    startup_summary = summarize_startups(run_result)
    gpu_summary = summarize_gpu_usage(run_result)

    return RealPolicySummary(
        policy_name=run_result.policy_name,
        experiment_duration_seconds=(
            run_result.experiment_duration_seconds
        ),
        request_summary=request_summary,
        startup_summary=startup_summary,
        gpu_summary=gpu_summary,
    )
