"""Unit tests for real workload scheduling and policy execution."""

from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock

import pytest

import serverless_llm.real_policy_runner as runner_module
from serverless_llm.docker_runtime import (
    DockerVLLMRuntime,
    RuntimeStartupResult,
)
from serverless_llm.gpu_monitor import GPUMonitor, GPUSample
from serverless_llm.real_policy_runner import (
    RealPolicyRunResult,
    RealRequestResult,
    _scheduled_time_for_event,
    _wait_until_scheduled,
    run_real_always_on_workload,
    run_real_fixed_keep_warm_workload,
    run_real_naive_serverless_workload,
)
from serverless_llm.vllm_client import CompletionResult, VLLMClient
from serverless_llm.workload import RequestEvent, WorkloadTrace


class FakeRuntime(DockerVLLMRuntime):
    """Provide deterministic Docker lifecycle behavior without Docker."""

    def __init__(
        self,
        startup_results: list[RuntimeStartupResult],
        event_log: list[str],
    ) -> None:
        self._fake_running = False
        self._startup_results = iter(startup_results)
        self.event_log = event_log
        self.start_calls = 0
        self.stop_calls = 0

    @property
    def is_running(self) -> bool:
        return self._fake_running

    def start_and_wait(self) -> RuntimeStartupResult:
        if self._fake_running:
            raise RuntimeError("fake runtime is already running")

        self.start_calls += 1
        self.event_log.append("start")
        self._fake_running = True
        return next(self._startup_results)

    def stop(self) -> None:
        if not self._fake_running:
            return

        self.stop_calls += 1
        self.event_log.append("stop")
        self._fake_running = False


class FakeClient(VLLMClient):
    """Return prepared completions without sending HTTP requests."""

    def __init__(
        self,
        completions: list[CompletionResult | Exception],
        event_log: list[str],
    ) -> None:
        self._completions = iter(completions)
        self.event_log = event_log
        self.complete_calls = 0

    def complete(self, prompt: str | None = None) -> CompletionResult:
        self.complete_calls += 1
        self.event_log.append("complete")
        result = next(self._completions)

        if isinstance(result, Exception):
            raise result

        return result


class FakeMonitor(GPUMonitor):
    """Record monitor lifecycle calls without querying a real GPU."""

    def __init__(
        self,
        samples: tuple[GPUSample, ...],
        event_log: list[str],
    ) -> None:
        self._fake_samples = samples
        self.event_log = event_log
        self.start_calls = 0
        self.stop_calls = 0

    @property
    def samples(self) -> tuple[GPUSample, ...]:
        return self._fake_samples

    def start(self) -> None:
        self.start_calls += 1
        self.event_log.append("monitor_start")

    def stop(self) -> None:
        self.stop_calls += 1
        self.event_log.append("monitor_stop")


@pytest.fixture
def gpu_sample() -> GPUSample:
    return GPUSample(
        sampled_at_seconds=101.0,
        gpu_index=0,
        utilization_percent=50.0,
        memory_used_mib=2500.0,
        power_draw_watts=40.0,
        temperature_celsius=48.0,
    )


def make_trace(*scheduled_times: float) -> WorkloadTrace:
    """Build a deterministic trace from relative arrival times."""

    return WorkloadTrace(
        name="real-runner-test",
        pattern="steady",
        random_seed=42,
        events=tuple(
            RequestEvent(
                request_id=index + 1,
                scheduled_at_seconds=scheduled_time,
                prompt_id="default",
            )
            for index, scheduled_time in enumerate(scheduled_times)
        ),
    )


def make_completion(
    request_started_at_seconds: float,
    first_token_at_seconds: float,
    completed_at_seconds: float,
) -> CompletionResult:
    """Build one deterministic streamed completion result."""

    return CompletionResult(
        request_started_at_seconds=request_started_at_seconds,
        first_token_at_seconds=first_token_at_seconds,
        completed_at_seconds=completed_at_seconds,
        generated_text="generated text",
        finish_reason="length",
    )


def test_real_request_result_calculates_end_to_end_metrics() -> None:
    result = RealRequestResult(
        event=RequestEvent(1, 5.0, "default"),
        scheduled_for_seconds=100.0,
        handling_started_at_seconds=102.0,
        completion=make_completion(110.0, 110.5, 112.0),
        cold_start=True,
        startup_duration_seconds=8.0,
    )

    assert result.request_id == 1
    assert result.scheduling_delay_seconds == 2.0
    assert result.pre_inference_delay_seconds == 8.0
    assert result.end_to_end_ttft_seconds == 10.5
    assert result.end_to_end_latency_seconds == 12.0


def test_real_request_result_is_immutable() -> None:
    result = RealRequestResult(
        event=RequestEvent(1, 0.0, "default"),
        scheduled_for_seconds=100.0,
        handling_started_at_seconds=100.0,
        completion=make_completion(100.0, 100.5, 101.0),
        cold_start=False,
        startup_duration_seconds=None,
    )

    with pytest.raises(FrozenInstanceError):
        result.cold_start = True  # type: ignore[misc]


@pytest.mark.parametrize(
    ("cold_start", "startup_duration_seconds", "message"),
    [
        (True, None, "cold-start request"),
        (False, 2.0, "warm request"),
        (True, -1.0, "startup_duration_seconds"),
        (True, float("nan"), "startup_duration_seconds"),
    ],
)
def test_real_request_result_validates_startup_relationship(
    cold_start: bool,
    startup_duration_seconds: float | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RealRequestResult(
            event=RequestEvent(1, 0.0, "default"),
            scheduled_for_seconds=100.0,
            handling_started_at_seconds=100.0,
            completion=make_completion(101.0, 101.5, 102.0),
            cold_start=cold_start,
            startup_duration_seconds=startup_duration_seconds,
        )


def test_real_request_result_rejects_invalid_timing_order() -> None:
    event = RequestEvent(1, 0.0, "default")

    with pytest.raises(ValueError, match="handling_started"):
        RealRequestResult(
            event=event,
            scheduled_for_seconds=100.0,
            handling_started_at_seconds=99.0,
            completion=make_completion(101.0, 102.0, 103.0),
            cold_start=False,
            startup_duration_seconds=None,
        )

    with pytest.raises(ValueError, match="cannot start before"):
        RealRequestResult(
            event=event,
            scheduled_for_seconds=100.0,
            handling_started_at_seconds=101.0,
            completion=make_completion(100.5, 102.0, 103.0),
            cold_start=False,
            startup_duration_seconds=None,
        )


def test_scheduled_time_converts_relative_event_time() -> None:
    event = RequestEvent(1, 5.0, "default")

    assert _scheduled_time_for_event(1000.0, event) == 1005.0


@pytest.mark.parametrize(
    "workload_started_at_seconds",
    [-1.0, float("inf"), float("nan"), True, "100"],
)
def test_scheduled_time_rejects_invalid_workload_start(
    workload_started_at_seconds: object,
) -> None:
    with pytest.raises(ValueError, match="workload_started_at_seconds"):
        _scheduled_time_for_event(
            workload_started_at_seconds,  # type: ignore[arg-type]
            RequestEvent(1, 0.0, "default"),
        )


def test_wait_until_scheduled_returns_immediately_for_past_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sleep = Mock()
    monkeypatch.setattr(
        runner_module.time,
        "monotonic",
        Mock(return_value=12.0),
    )
    monkeypatch.setattr(runner_module.time, "sleep", fake_sleep)

    assert _wait_until_scheduled(10.0) == 12.0
    fake_sleep.assert_not_called()


def test_wait_until_scheduled_sleeps_until_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sleep = Mock()
    monkeypatch.setattr(
        runner_module.time,
        "monotonic",
        Mock(side_effect=[8.0, 10.0]),
    )
    monkeypatch.setattr(runner_module.time, "sleep", fake_sleep)

    assert _wait_until_scheduled(10.0) == 10.0
    fake_sleep.assert_called_once_with(2.0)


def configure_runner_clock(
    monkeypatch: pytest.MonkeyPatch,
    *,
    experiment_started: float,
    workload_started: float,
    experiment_completed: float,
    handled_at: list[float],
) -> list[float]:
    """Replace wall-clock waiting with deterministic timestamps."""

    waited_targets: list[float] = []
    handling_times = iter(handled_at)

    monkeypatch.setattr(
        runner_module.time,
        "monotonic",
        Mock(
            side_effect=[
                experiment_started,
                workload_started,
                experiment_completed,
            ]
        ),
    )

    def fake_wait(target: float) -> float:
        waited_targets.append(target)
        return next(handling_times)

    monkeypatch.setattr(
        runner_module,
        "_wait_until_scheduled",
        fake_wait,
    )
    return waited_targets


def test_always_on_starts_once_and_keeps_requests_warm(
    monkeypatch: pytest.MonkeyPatch,
    gpu_sample: GPUSample,
) -> None:
    event_log: list[str] = []
    runtime = FakeRuntime(
        [RuntimeStartupResult(100.0, 109.0)],
        event_log,
    )
    client = FakeClient(
        [
            make_completion(110.0, 110.5, 111.0),
            make_completion(115.0, 115.5, 116.0),
        ],
        event_log,
    )
    monitor = FakeMonitor((gpu_sample,), event_log)
    waited_targets = configure_runner_clock(
        monkeypatch,
        experiment_started=100.0,
        workload_started=110.0,
        experiment_completed=117.0,
        handled_at=[110.0, 115.0],
    )

    result = run_real_always_on_workload(
        runtime,
        client,
        monitor,
        make_trace(0.0, 5.0),
    )

    assert result.policy_name == "always_on"
    assert result.experiment_duration_seconds == 17.0
    assert [item.cold_start for item in result.request_results] == [
        False,
        False,
    ]
    assert result.startup_results == (
        RuntimeStartupResult(100.0, 109.0),
    )
    assert result.gpu_samples == (gpu_sample,)
    assert waited_targets == [110.0, 115.0]
    assert runtime.start_calls == 1
    assert runtime.stop_calls == 1
    assert event_log == [
        "monitor_start",
        "start",
        "complete",
        "complete",
        "stop",
        "monitor_stop",
    ]


def test_naive_serverless_restarts_for_every_request(
    monkeypatch: pytest.MonkeyPatch,
    gpu_sample: GPUSample,
) -> None:
    event_log: list[str] = []
    startup_results = [
        RuntimeStartupResult(100.0, 110.0),
        RuntimeStartupResult(112.0, 122.0),
    ]
    runtime = FakeRuntime(startup_results.copy(), event_log)
    client = FakeClient(
        [
            make_completion(110.0, 110.5, 112.0),
            make_completion(122.0, 122.5, 124.0),
        ],
        event_log,
    )
    monitor = FakeMonitor((gpu_sample,), event_log)
    waited_targets = configure_runner_clock(
        monkeypatch,
        experiment_started=100.0,
        workload_started=100.0,
        experiment_completed=125.0,
        handled_at=[100.0, 112.0],
    )

    result = run_real_naive_serverless_workload(
        runtime,
        client,
        monitor,
        make_trace(0.0, 5.0),
    )

    assert result.policy_name == "naive_serverless"
    assert [item.cold_start for item in result.request_results] == [
        True,
        True,
    ]
    assert [
        item.startup_duration_seconds
        for item in result.request_results
    ] == [10.0, 10.0]
    assert result.startup_results == tuple(startup_results)
    assert waited_targets == [100.0, 105.0]
    assert runtime.start_calls == 2
    assert runtime.stop_calls == 2
    assert event_log == [
        "monitor_start",
        "start",
        "complete",
        "stop",
        "start",
        "complete",
        "stop",
        "monitor_stop",
    ]


def test_fixed_keep_warm_reuses_server_inside_timeout(
    monkeypatch: pytest.MonkeyPatch,
    gpu_sample: GPUSample,
) -> None:
    event_log: list[str] = []
    runtime = FakeRuntime(
        [RuntimeStartupResult(100.0, 102.0)],
        event_log,
    )
    client = FakeClient(
        [
            make_completion(102.0, 102.5, 103.0),
            make_completion(105.0, 105.5, 106.0),
        ],
        event_log,
    )
    monitor = FakeMonitor((gpu_sample,), event_log)
    waited_targets = configure_runner_clock(
        monkeypatch,
        experiment_started=100.0,
        workload_started=100.0,
        experiment_completed=107.0,
        handled_at=[100.0, 105.0],
    )

    result = run_real_fixed_keep_warm_workload(
        runtime,
        client,
        monitor,
        make_trace(0.0, 5.0),
        timeout_seconds=10.0,
    )

    assert [item.cold_start for item in result.request_results] == [
        True,
        False,
    ]
    assert runtime.start_calls == 1
    assert runtime.stop_calls == 1
    assert waited_targets == [100.0, 105.0]


def test_fixed_keep_warm_stops_at_deadline_and_restarts(
    monkeypatch: pytest.MonkeyPatch,
    gpu_sample: GPUSample,
) -> None:
    event_log: list[str] = []
    startup_results = [
        RuntimeStartupResult(100.0, 102.0),
        RuntimeStartupResult(120.0, 122.0),
    ]
    runtime = FakeRuntime(startup_results.copy(), event_log)
    client = FakeClient(
        [
            make_completion(102.0, 102.5, 103.0),
            make_completion(122.0, 122.5, 123.0),
        ],
        event_log,
    )
    monitor = FakeMonitor((gpu_sample,), event_log)
    waited_targets = configure_runner_clock(
        monkeypatch,
        experiment_started=100.0,
        workload_started=100.0,
        experiment_completed=124.0,
        handled_at=[100.0, 113.0, 120.0],
    )

    result = run_real_fixed_keep_warm_workload(
        runtime,
        client,
        monitor,
        make_trace(0.0, 20.0),
        timeout_seconds=10.0,
    )

    assert [item.cold_start for item in result.request_results] == [
        True,
        True,
    ]
    assert result.startup_results == tuple(startup_results)
    assert waited_targets == [100.0, 113.0, 120.0]
    assert runtime.start_calls == 2
    assert runtime.stop_calls == 2


def test_fixed_keep_warm_treats_exact_deadline_as_warm(
    monkeypatch: pytest.MonkeyPatch,
    gpu_sample: GPUSample,
) -> None:
    event_log: list[str] = []
    runtime = FakeRuntime(
        [RuntimeStartupResult(100.0, 102.0)],
        event_log,
    )
    client = FakeClient(
        [
            make_completion(102.0, 102.5, 103.0),
            make_completion(113.0, 113.5, 114.0),
        ],
        event_log,
    )
    monitor = FakeMonitor((gpu_sample,), event_log)
    waited_targets = configure_runner_clock(
        monkeypatch,
        experiment_started=100.0,
        workload_started=100.0,
        experiment_completed=115.0,
        handled_at=[100.0, 113.0],
    )

    result = run_real_fixed_keep_warm_workload(
        runtime,
        client,
        monitor,
        make_trace(0.0, 13.0),
        timeout_seconds=10.0,
    )

    assert [item.cold_start for item in result.request_results] == [
        True,
        False,
    ]
    assert waited_targets == [100.0, 113.0]
    assert runtime.start_calls == 1


@pytest.mark.parametrize(
    "timeout_seconds",
    [0.0, -1.0, float("inf"), float("nan"), True, "10"],
)
def test_fixed_keep_warm_rejects_invalid_timeout(
    timeout_seconds: object,
    gpu_sample: GPUSample,
) -> None:
    event_log: list[str] = []
    runtime = FakeRuntime([], event_log)
    client = FakeClient([], event_log)
    monitor = FakeMonitor((gpu_sample,), event_log)

    with pytest.raises(ValueError, match="timeout_seconds"):
        run_real_fixed_keep_warm_workload(
            runtime,
            client,
            monitor,
            make_trace(0.0),
            timeout_seconds=timeout_seconds,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "runner",
    [run_real_always_on_workload, run_real_naive_serverless_workload],
)
def test_runner_cleans_up_after_completion_failure(
    runner: object,
    monkeypatch: pytest.MonkeyPatch,
    gpu_sample: GPUSample,
) -> None:
    event_log: list[str] = []
    runtime = FakeRuntime(
        [RuntimeStartupResult(100.0, 102.0)],
        event_log,
    )
    client = FakeClient([RuntimeError("request failed")], event_log)
    monitor = FakeMonitor((gpu_sample,), event_log)
    monkeypatch.setattr(
        runner_module.time,
        "monotonic",
        Mock(side_effect=[100.0, 100.0]),
    )
    monkeypatch.setattr(
        runner_module,
        "_wait_until_scheduled",
        Mock(return_value=100.0),
    )

    with pytest.raises(RuntimeError, match="request failed"):
        runner(  # type: ignore[operator]
            runtime,
            client,
            monitor,
            make_trace(0.0),
        )

    assert runtime.is_running is False
    assert monitor.stop_calls == 1
    assert event_log[-2:] == ["stop", "monitor_stop"]


def test_policy_run_result_is_immutable(
    gpu_sample: GPUSample,
) -> None:
    request_result = RealRequestResult(
        event=RequestEvent(1, 0.0, "default"),
        scheduled_for_seconds=100.0,
        handling_started_at_seconds=100.0,
        completion=make_completion(100.0, 100.5, 101.0),
        cold_start=False,
        startup_duration_seconds=None,
    )
    result = RealPolicyRunResult(
        policy_name="always_on",
        experiment_started_at_seconds=90.0,
        workload_started_at_seconds=100.0,
        experiment_completed_at_seconds=102.0,
        startup_results=(RuntimeStartupResult(90.0, 99.0),),
        request_results=(request_result,),
        gpu_samples=(gpu_sample,),
    )

    assert result.experiment_duration_seconds == 12.0

    with pytest.raises(FrozenInstanceError):
        result.policy_name = "changed"  # type: ignore[misc]


def test_policy_run_result_rejects_invalid_timestamp_order(
    gpu_sample: GPUSample,
) -> None:
    request_result = RealRequestResult(
        event=RequestEvent(1, 0.0, "default"),
        scheduled_for_seconds=100.0,
        handling_started_at_seconds=100.0,
        completion=make_completion(100.0, 100.5, 101.0),
        cold_start=False,
        startup_duration_seconds=None,
    )
    valid = RealPolicyRunResult(
        policy_name="always_on",
        experiment_started_at_seconds=90.0,
        workload_started_at_seconds=100.0,
        experiment_completed_at_seconds=102.0,
        startup_results=(RuntimeStartupResult(90.0, 99.0),),
        request_results=(request_result,),
        gpu_samples=(gpu_sample,),
    )

    with pytest.raises(ValueError, match="workload cannot start"):
        replace(valid, workload_started_at_seconds=89.0)
