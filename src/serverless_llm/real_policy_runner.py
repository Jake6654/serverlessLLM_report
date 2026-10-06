"""Run real workloads against a Dockerized vLLM server."""

import time
from dataclasses import dataclass
from math import isfinite

from serverless_llm.docker_runtime import (
    DockerVLLMRuntime,
    RuntimeStartupResult,
)
from serverless_llm.gpu_monitor import (
    GPUMonitor,
    GPUSample,
)
from serverless_llm.vllm_client import (
    CompletionResult,
    VLLMClient,
)
from serverless_llm.workload import (
    RequestEvent,
    WorkloadTrace,
)

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

@dataclass(frozen=True)
class RealPolicyRunResult:

    policy_name: str
    experiment_started_at_seconds: float
    workload_started_at_seconds: float
    experiment_completed_at_seconds: float
    startup_results: tuple[RuntimeStartupResult, ...]
    request_results: tuple[RealRequestResult, ...]
    gpu_samples: tuple[GPUSample, ...]


    def __post_init__(self) -> None:
        """Reject invalid policy run results."""

        if (
            not isinstance(self.policy_name, str)
            or not self.policy_name.strip()
        ):
            raise ValueError(
                "policy_name must be a non-empty string"
            )

        timestamps = (
            self.experiment_started_at_seconds,
            self.workload_started_at_seconds,
            self.experiment_completed_at_seconds,
        )

        if any(
            type(timestamp) not in (int, float)
            or not isfinite(timestamp)
            or timestamp < 0
            for timestamp in timestamps
        ):
            raise ValueError(
                "policy run timestamps must be finite "
                "non-negative numbers"
            )

        if (
            self.workload_started_at_seconds
            < self.experiment_started_at_seconds
        ):
            raise ValueError(
                "workload cannot start before the experiment"
            )

        if (
            self.experiment_completed_at_seconds
            < self.workload_started_at_seconds
        ):
            raise ValueError(
                "experiment cannot complete before the workload starts"
            )

        if (
            not isinstance(self.startup_results, tuple)
            or not all(
                isinstance(result, RuntimeStartupResult)
                for result in self.startup_results
            )
        ):
            raise ValueError(
                "startup_results must contain only "
                "RuntimeStartupResult objects"
            )

        if (
            not isinstance(self.request_results, tuple)
            or not self.request_results
            or not all(
                isinstance(result, RealRequestResult)
                for result in self.request_results
            )
        ):
            raise ValueError(
                "request_results must be a non-empty tuple of "
                "RealRequestResult objects"
            )

        if (
            not isinstance(self.gpu_samples, tuple)
            or not all(
                isinstance(sample, GPUSample)
                for sample in self.gpu_samples
            )
        ):
            raise ValueError(
                "gpu_samples must contain only GPUSample objects"
            )

    @property
    def experiment_duration_seconds(self) -> float:

        return(
            self.experiment_completed_at_seconds - self.experiment_started_at_seconds
        )


def run_real_always_on_workload(
        runtime: DockerVLLMRuntime,
        client: VLLMClient,
        monitor: GPUMonitor,
        trace:WorkloadTrace
) -> RealPolicyRunResult:
    """Run one real workload while keeping vLLM continuously ready"""

    if not isinstance(runtime, DockerVLLMRuntime):
        raise ValueError(
            "runtime must be a DockerVLLMRuntime object"
        )

    if not isinstance(client, VLLMClient):
        raise ValueError(
            "client must be a VLLMClient object"
        )

    if not isinstance(monitor, GPUMonitor):
        raise ValueError(
            "monitor must be a GPUMonitor object"
        )

    if not isinstance(trace, WorkloadTrace):
        raise ValueError(
            "trace must be a WorkloadTrace object"
        )

    if runtime.is_running:
        raise RuntimeError(
            "Always-On experiment requires a stopped runtime"
        )

    experiment_started_at_seconds = time.monotonic()
    startup_results: list[RuntimeStartupResult] = []
    request_results: list[RealRequestResult] = []

    monitor.start()

    try:
        # Always-On loads the model before workload timing begins
        startup_result = runtime.start_and_wait()
        startup_results.append(startup_result)

        workload_started_at_seconds = time.monotonic()

        for event in trace.events:
            scheduled_for_seconds = _scheduled_time_for_event(
                workload_started_at_seconds,
                event,
            )

            handling_started_at_seconds = (
                _wait_until_scheduled(scheduled_for_seconds)
            )

            completion = client.complete()

            request_results.append(
                RealRequestResult(
                    event=event,
                    scheduled_for_seconds=scheduled_for_seconds,
                    handling_started_at_seconds=(
                        handling_started_at_seconds
                    ),
                    completion=completion,
                    cold_start=False,
                    startup_duration_seconds=None,
                )
            )

    finally:
        # Stop the monitor even if Docker cleanup fails
        try:
            runtime.stop()
        finally:
            monitor.stop()

    experiment_completed_at_seconds = time.monotonic()

    return RealPolicyRunResult(
        policy_name="always_on",
        experiment_started_at_seconds=(
            experiment_started_at_seconds
        ),
        workload_started_at_seconds= workload_started_at_seconds,
        experiment_completed_at_seconds= experiment_completed_at_seconds,
        startup_results=tuple(startup_results),
        request_results=tuple(request_results),
        gpu_samples=monitor.samples,
    )

def run_real_naive_serverless_workload(
    runtime: DockerVLLMRuntime,
    client: VLLMClient,
    monitor: GPUMonitor,
    trace: WorkloadTrace,
) -> RealPolicyRunResult:
    """Run one real workload by stopping vLLM after every request."""

    if not isinstance(runtime, DockerVLLMRuntime):
        raise ValueError(
            "runtime must be a DockerVLLMRuntime object"
        )

    if not isinstance(client, VLLMClient):
        raise ValueError(
            "client must be a VLLMClient object"
        )

    if not isinstance(monitor, GPUMonitor):
        raise ValueError(
            "monitor must be a GPUMonitor object"
        )

    if not isinstance(trace, WorkloadTrace):
        raise ValueError(
            "trace must be a WorkloadTrace object"
        )

    if runtime.is_running:
        raise RuntimeError(
            "Naive Serverless experiment requires a stopped runtime"
        )

    experiment_started_at_seconds = time.monotonic()
    startup_results: list[RuntimeStartupResult] = []
    request_results: list[RealRequestResult] = []

    monitor.start()

    try:
        # Unlike Always-on, workload timing begins while vLLM is off.
        workload_started_at_seconds = time.monotonic()

        for event in trace.events:
            scheduled_for_seconds = _scheduled_time_for_event(
                workload_started_at_seconds,
                event,
            )

            handling_started_at_seconds = (
                _wait_until_scheduled(scheduled_for_seconds)
            )

        # Every request starts from an off server.
            startup_result = runtime.start_and_wait()
            startup_results.append(startup_result)

            try:
                completion = client.complete()
            finally:
                # Naive Serverless stops immediately after each request.
                runtime.stop()

            request_results.append(
                RealRequestResult(
                    event=event,
                    scheduled_for_seconds=scheduled_for_seconds,
                    handling_started_at_seconds=(
                        handling_started_at_seconds
                    ),
                    completion=completion,
                    cold_start=True,
                    startup_duration_seconds=(
                        startup_result.startup_duration_seconds
                    ),
                )
            )

    finally:
        # The extra stop is safe because runtime.stop() is idempotent.
        try:
            runtime.stop()
        finally:
            monitor.stop()

    experiment_completed_at_seconds = time.monotonic()

    return RealPolicyRunResult(
        policy_name="naive_serverless",
        experiment_started_at_seconds=(
            experiment_started_at_seconds
        ),
        workload_started_at_seconds=(
            workload_started_at_seconds
        ),
        experiment_completed_at_seconds=(
            experiment_completed_at_seconds
        ),
        startup_results=tuple(startup_results),
        request_results=tuple(request_results),
        gpu_samples=monitor.samples,
    )

def run_real_fixed_keep_warm_workload(
    runtime: DockerVLLMRuntime,
    client: VLLMClient,
    monitor: GPUMonitor,
    trace: WorkloadTrace,
    timeout_seconds: float,
) -> RealPolicyRunResult:
    """Run a workload with a fixed post-request warm timeout."""

    if not isinstance(runtime, DockerVLLMRuntime):
        raise ValueError(
            "runtime must be a DockerVLLMRuntime object"
        )

    if not isinstance(client, VLLMClient):
        raise ValueError(
            "client must be a VLLMClient object"
        )

    if not isinstance(monitor, GPUMonitor):
        raise ValueError(
            "monitor must be a GPUMonitor object"
        )

    if not isinstance(trace, WorkloadTrace):
        raise ValueError(
            "trace must be a WorkloadTrace object"
        )

    if (
        type(timeout_seconds) not in (int, float)
        or not isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError(
            "timeout_seconds must be positive and finite"
        )

    if runtime.is_running:
        raise RuntimeError(
            "Fixed Keep-Warm experiment requires a stopped runtime"
        )

    normalized_timeout_seconds = float(timeout_seconds)


    experiment_started_at_seconds = time.monotonic()
    startup_results: list[RuntimeStartupResult] = []
    request_results: list[RealRequestResult] = []

    # None means that no warm server currently has a shutdown timer.
    shutdown_deadline_seconds: float | None = None

    monitor.start()

    try:
        # Fixed Keep-Warm starts the workload while vLLM is off
        workload_started_at_seconds = time.monotonic()

        for event in trace.events:
            scheduled_for_seconds = _scheduled_time_for_event(
                workload_started_at_seconds,
                event,
            )

            server_running = runtime.is_running

            # Stop at the previous request's deadline when the next
            # request is schduled outside the keep-warm window
            if (
                server_running
                and shutdown_deadline_seconds is not None
                and scheduled_for_seconds > shutdown_deadline_seconds
            ):

                _wait_until_scheduled(
                    shutdown_deadline_seconds
                )
                runtime.stop()
                server_running = False
                shutdown_deadline_seconds = None

            handling_started_at_seconds = (
                _wait_until_scheduled(scheduled_for_seconds)
            )

            cold_start = not server_running
            startup_duration_seconds: float | None = None

            if cold_start:
                startup_result = runtime.start_and_wait()
                startup_results.append(startup_result)

                startup_duration_seconds = (
                    startup_result.startup_duration_seconds
                )

            completion = client.complete()

            request_results.append(
                RealRequestResult(
                    event=event,
                    scheduled_for_seconds=scheduled_for_seconds,
                    handling_started_at_seconds=(
                        handling_started_at_seconds
                    ),
                    completion=completion,
                    cold_start=cold_start,
                    startup_duration_seconds=(
                        startup_duration_seconds
                    ),
                )
            )

            # Start or refresh the fixed keep-warm window after every completed request.
            shutdown_deadline_seconds = (
                completion.completed_at_seconds
                + normalized_timeout_seconds
            )

    finally:
        # Do not wait for the final timeout after workload completion.
        try:
            runtime.stop()
        finally:
            monitor.stop()

    experiment_completed_at_seconds = time.monotonic()

    return RealPolicyRunResult(
        policy_name="fixed_keep_warm",
        experiment_started_at_seconds=(
            experiment_started_at_seconds
        ),
        workload_started_at_seconds=(
            workload_started_at_seconds
        ),
        experiment_completed_at_seconds=(
            experiment_completed_at_seconds
        ),
        startup_results=tuple(startup_results),
        request_results=tuple(request_results),
        gpu_samples=monitor.samples,
    )

