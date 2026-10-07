"""실험 시각 정규화 실제 실험의 통계 계산을 위한 파일"""


from dataclasses import dataclass
from math import isfinite

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
    """Calculate comparable metrics for every request in a real run."""

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


