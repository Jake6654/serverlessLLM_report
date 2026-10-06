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
