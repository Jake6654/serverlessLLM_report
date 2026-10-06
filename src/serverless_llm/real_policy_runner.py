"""Run real workloads against a Dockerized vLLM server."""

from dataclasses import dataclass
from math import isfinite
import time

from serverless_llm.vllm_client import CompletionResult
from serverless_llm.workload import RequestEvent

@dataclass(frozen=True)
class RealRequestResult:
    """Store the complete real-world result for one request"""

    event: RequestEvent
    scheduled_for_seconds: float # 요청이 도착하기로 예정된 절대 시각
    handling_started_at_seconds: float # 실제 처리 시작시간
    completion: CompletionResult
    cold_start: bool
    startup_duration_seconds: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.event, RequestEvent):
            raise ValueError(
                "event must be a RequestEvent object"
            )

        if (
            type(self.scheduled_for_seconds) not in (int, float)
            or not isfinite(self.scheduled_for_seconds)
            or self.scheduled_for_seconds < 0
        ):
            raise ValueError(
                "scheduled_for_seconds must be a finite "
                "non-negative number"
            )

        if (
            type(self.handling_started_at_seconds)
            not in (int, float)
            or not isfinite(self.handling_started_at_seconds)
            or self.handling_started_at_seconds < 0
        ):
            raise ValueError(
                "handling_started_at_seconds must be a finite "
                "non-negative number"
            )

        if (
            self.handling_started_at_seconds
            < self.scheduled_for_seconds
        ):
            raise ValueError(
                "handling_started_at_seconds cannot be earlier "
                "than scheduled_for_seconds"
            )

        if not isinstance(self.completion, CompletionResult):
            raise ValueError(
                "completion must be a CompletionResult object"
            )

        if (
            self.completion.request_started_at_seconds
            < self.handling_started_at_seconds
        ):
            raise ValueError(
                "completion request cannot start before request "
                "handling begins"
            )

        if (
            self.completion.first_token_at_seconds
            < self.completion.request_started_at_seconds
        ):
            raise ValueError(
                "first token cannot arrive before the completion "
                "request starts"
            )

        if (
            self.completion.completed_at_seconds
            < self.completion.first_token_at_seconds
        ):
            raise ValueError(
                "completion cannot finish before the first token"
            )

        if type(self.cold_start) is not bool:
            raise ValueError("cold_start must be a boolean")

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
    @property
    def request_id(self) -> int:
        """Return the workload request identifier"""
        return self.event.request_id

    @property
    def scheduling_delay_seconds(self) -> float:
        """Return delay btw scheduled arrival and handling"""

        return(
            self.handling_started_at_seconds - self.scheduled_for_seconds
        )

    @property
    def pre_inference_delay_seconds(self) -> float:

        """Return time from handling start to the HTTP request."""

        return (
            self.completion.request_started_at_seconds # cold start 후 HTTP 요청을 보낸 시각
            - self.handling_started_at_seconds
        )

    @property
    def end_to_end_ttft_seconds(self) -> float:
        """Return TTFT including scheduling and cold-start delay."""

        return (
            self.completion.first_token_at_seconds # 첫 토큰을 받은 시각
            - self.scheduled_for_seconds
        )

    @property
    def end_to_end_latency_seconds(self) -> float:
        """Return complete latency from scheduled arrival to finish."""

        return (
            self.completion.completed_at_seconds # 전체 생성이 끝난 시각
            - self.scheduled_for_seconds
        )

def _scheduled_time_for_event(
        workload_started_at_seconds: float,
        event: RequestEvent,
    ) -> float:
    """Convert a relative workload event into an absolute target time"""


    if (
        type(workload_started_at_seconds) not in (int, float)
        or not isfinite(workload_started_at_seconds)
        or workload_started_at_seconds < 0
    ):
        raise ValueError(
            "workload_started_at_seconds must be a finite "
            "non-negative number"
        )

    if not isinstance(event, RequestEvent):
        raise ValueError(
            "event must be a RequestEvent object"
        )

    scheduled_for_seconds = (
        workload_started_at_seconds
        + event.scheduled_at_seconds
    )

    if not isfinite(scheduled_for_seconds):
        raise ValueError(
            "calculated request schedule must be finite"
        )

    return float(scheduled_for_seconds)

def _wait_until_scheduled(
    scheduled_for_seconds: float,
) -> float:
    """Wait until a target time and return actual handling time."""

    if (
        type(scheduled_for_seconds) not in (int, float)
        or not isfinite(scheduled_for_seconds)
        or scheduled_for_seconds < 0
    ):
        raise ValueError(
            "scheduled_for_seconds must be a finite "
            "non-negative number"
        )

    while True:
        current_time_seconds = time.monotonic()
        remaining_seconds = (
            scheduled_for_seconds - current_time_seconds
        )

        # The request is ready when its target time has arrived.
        if remaining_seconds <= 0:
            return current_time_seconds

        time.sleep(remaining_seconds)
    

