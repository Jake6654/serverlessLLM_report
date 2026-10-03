"""Unit tests for NVIDIA GPU sampling and background monitoring."""

from dataclasses import FrozenInstanceError
from subprocess import CompletedProcess
from threading import Event
from unittest.mock import Mock

import pytest

import serverless_llm.gpu_monitor as gpu_monitor_module
from serverless_llm.config import MonitoringConfig
from serverless_llm.gpu_monitor import (
    GPUMonitor,
    GPUMonitorError,
    GPUSample,
    _parse_nvidia_smi_output,
    collect_gpu_sample,
)


@pytest.fixture
def monitoring_config() -> MonitoringConfig:
    """Return a short sample interval suitable for thread tests."""

    return MonitoringConfig(gpu_sample_interval_ms=20)


@pytest.fixture
def sample() -> GPUSample:
    """Return one deterministic GPU measurement."""

    return GPUSample(
        sampled_at_seconds=100.0,
        gpu_index=0,
        utilization_percent=74.0,
        memory_used_mib=18342.0,
        power_draw_watts=61.5,
        temperature_celsius=57.0,
    )


def completed_process(
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> CompletedProcess[str]:
    """Build a fake result returned by subprocess.run."""

    return CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_gpu_sample_stores_valid_measurements(
    sample: GPUSample,
) -> None:
    assert sample.sampled_at_seconds == 100.0
    assert sample.gpu_index == 0
    assert sample.utilization_percent == 74.0
    assert sample.memory_used_mib == 18342.0
    assert sample.power_draw_watts == 61.5
    assert sample.temperature_celsius == 57.0


def test_gpu_sample_is_immutable(sample: GPUSample) -> None:
    with pytest.raises(FrozenInstanceError):
        sample.utilization_percent = 50.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "sampled_at_seconds",
    [-1.0, float("inf"), float("-inf"), float("nan"), True, "1.0"],
)
def test_gpu_sample_rejects_invalid_timestamp(
    sampled_at_seconds: object,
) -> None:
    with pytest.raises(ValueError, match="sampled_at_seconds"):
        GPUSample(
            sampled_at_seconds=sampled_at_seconds,  # type: ignore[arg-type]
            gpu_index=0,
            utilization_percent=0.0,
            memory_used_mib=0.0,
            power_draw_watts=0.0,
            temperature_celsius=0.0,
        )


@pytest.mark.parametrize("gpu_index", [-1, 1.5, True, "0"])
def test_gpu_sample_rejects_invalid_gpu_index(
    gpu_index: object,
) -> None:
    with pytest.raises(ValueError, match="gpu_index"):
        GPUSample(
            sampled_at_seconds=1.0,
            gpu_index=gpu_index,  # type: ignore[arg-type]
            utilization_percent=0.0,
            memory_used_mib=0.0,
            power_draw_watts=0.0,
            temperature_celsius=0.0,
        )


@pytest.mark.parametrize(
    "utilization_percent",
    [-0.1, 100.1, float("inf"), float("nan"), True, "50"],
)
def test_gpu_sample_rejects_invalid_utilization(
    utilization_percent: object,
) -> None:
    with pytest.raises(ValueError, match="utilization_percent"):
        GPUSample(
            sampled_at_seconds=1.0,
            gpu_index=0,
            utilization_percent=utilization_percent,  # type: ignore[arg-type]
            memory_used_mib=0.0,
            power_draw_watts=0.0,
            temperature_celsius=0.0,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("memory_used_mib", -1.0, "memory_used_mib"),
        ("memory_used_mib", float("nan"), "memory_used_mib"),
        ("power_draw_watts", -1.0, "power_draw_watts"),
        ("power_draw_watts", float("inf"), "power_draw_watts"),
        ("temperature_celsius", -1.0, "temperature_celsius"),
        ("temperature_celsius", "hot", "temperature_celsius"),
    ],
)
def test_gpu_sample_rejects_invalid_non_negative_measurements(
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "sampled_at_seconds": 1.0,
        "gpu_index": 0,
        "utilization_percent": 0.0,
        "memory_used_mib": 0.0,
        "power_draw_watts": 0.0,
        "temperature_celsius": 0.0,
    }
    values[field_name] = invalid_value

    with pytest.raises(ValueError, match=message):
        GPUSample(**values)  # type: ignore[arg-type]


def test_parse_nvidia_smi_output_builds_sample() -> None:
    result = _parse_nvidia_smi_output(
        "\n  0, 74, 18342, 61.5, 57  \n\n",
        sampled_at_seconds=100.0,
    )

    assert result == GPUSample(
        sampled_at_seconds=100.0,
        gpu_index=0,
        utilization_percent=74.0,
        memory_used_mib=18342.0,
        power_draw_watts=61.5,
        temperature_celsius=57.0,
    )


def test_parse_nvidia_smi_output_rejects_non_string() -> None:
    with pytest.raises(GPUMonitorError, match="must be a string"):
        _parse_nvidia_smi_output(123, 100.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "output",
    [
        "",
        "\n \n",
        "0, 10, 100, 20, 30\n1, 20, 200, 30, 40\n",
    ],
)
def test_parse_nvidia_smi_output_requires_one_row(
    output: str,
) -> None:
    with pytest.raises(GPUMonitorError, match="exactly one"):
        _parse_nvidia_smi_output(output, 100.0)


def test_parse_nvidia_smi_output_requires_five_fields() -> None:
    with pytest.raises(
        GPUMonitorError,
        match="unexpected number of fields",
    ):
        _parse_nvidia_smi_output("0, 74, 18342, 61.5", 100.0)


def test_parse_nvidia_smi_output_rejects_non_numeric_value() -> None:
    with pytest.raises(GPUMonitorError, match="non-numeric"):
        _parse_nvidia_smi_output(
            "0, N/A, 18342, 61.5, 57",
            100.0,
        )


@pytest.mark.parametrize("gpu_index", [-1, 1.5, True, "0"])
def test_collect_gpu_sample_rejects_invalid_gpu_index(
    gpu_index: object,
) -> None:
    with pytest.raises(ValueError, match="gpu_index"):
        collect_gpu_sample(gpu_index)  # type: ignore[arg-type]


def test_collect_gpu_sample_executes_expected_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run = Mock(
        return_value=completed_process(
            stdout="0, 74, 18342, 61.5, 57\n"
        )
    )
    monkeypatch.setattr(
        gpu_monitor_module.subprocess,
        "run",
        fake_run,
    )
    monkeypatch.setattr(
        gpu_monitor_module.time,
        "monotonic",
        Mock(return_value=123.5),
    )

    result = collect_gpu_sample(gpu_index=0)

    assert result.sampled_at_seconds == 123.5
    assert result.gpu_index == 0
    assert result.utilization_percent == 74.0
    fake_run.assert_called_once_with(
        [
            "nvidia-smi",
            "--id=0",
            (
                "--query-gpu=index,utilization.gpu,memory.used,"
                "power.draw,temperature.gpu"
            ),
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=5.0,
    )


def test_collect_gpu_sample_reports_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_monitor_module.subprocess,
        "run",
        Mock(
            side_effect=gpu_monitor_module.subprocess.TimeoutExpired(
                cmd=["nvidia-smi"],
                timeout=5.0,
            )
        ),
    )

    with pytest.raises(GPUMonitorError, match="within 5 seconds"):
        collect_gpu_sample()


def test_collect_gpu_sample_reports_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_monitor_module.subprocess,
        "run",
        Mock(side_effect=FileNotFoundError("nvidia-smi not found")),
    )

    with pytest.raises(GPUMonitorError, match="Could not execute"):
        collect_gpu_sample()


@pytest.mark.parametrize(
    ("stderr", "message"),
    [
        ("GPU access denied\n", "GPU access denied"),
        ("", "unknown nvidia-smi error"),
    ],
)
def test_collect_gpu_sample_reports_command_failure(
    monkeypatch: pytest.MonkeyPatch,
    stderr: str,
    message: str,
) -> None:
    monkeypatch.setattr(
        gpu_monitor_module.subprocess,
        "run",
        Mock(
            return_value=completed_process(
                returncode=1,
                stderr=stderr,
            )
        ),
    )

    with pytest.raises(GPUMonitorError, match=message):
        collect_gpu_sample()


def test_gpu_monitor_rejects_invalid_constructor_arguments(
    monitoring_config: MonitoringConfig,
) -> None:
    with pytest.raises(ValueError, match="monitoring_config"):
        GPUMonitor(object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="gpu_index"):
        GPUMonitor(monitoring_config, gpu_index=-1)


def test_gpu_monitor_initial_state(
    monitoring_config: MonitoringConfig,
) -> None:
    monitor = GPUMonitor(monitoring_config)

    assert monitor.is_running is False
    assert monitor.samples == ()
    assert monitor.stop() is None


def test_gpu_monitor_collects_on_background_thread(
    monitoring_config: MonitoringConfig,
    sample: GPUSample,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_started = Event()
    allow_collection_to_finish = Event()

    def fake_collect_gpu_sample(gpu_index: int = 0) -> GPUSample:
        assert gpu_index == 0
        collection_started.set()
        assert allow_collection_to_finish.wait(timeout=1.0)
        return sample

    monkeypatch.setattr(
        gpu_monitor_module,
        "collect_gpu_sample",
        fake_collect_gpu_sample,
    )
    monitor = GPUMonitor(monitoring_config, gpu_index=0)

    monitor.start()
    assert collection_started.wait(timeout=1.0)
    assert monitor.is_running is True

    allow_collection_to_finish.set()
    monitor.stop()

    assert monitor.is_running is False
    assert monitor.samples == (sample,)
    assert isinstance(monitor.samples, tuple)


def test_gpu_monitor_rejects_second_start(
    monitoring_config: MonitoringConfig,
    sample: GPUSample,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_started = Event()
    allow_collection_to_finish = Event()

    def fake_collect_gpu_sample(gpu_index: int = 0) -> GPUSample:
        collection_started.set()
        assert allow_collection_to_finish.wait(timeout=1.0)
        return sample

    monkeypatch.setattr(
        gpu_monitor_module,
        "collect_gpu_sample",
        fake_collect_gpu_sample,
    )
    monitor = GPUMonitor(monitoring_config)

    monitor.start()
    assert collection_started.wait(timeout=1.0)

    with pytest.raises(GPUMonitorError, match="already been started"):
        monitor.start()

    allow_collection_to_finish.set()
    monitor.stop()


def test_gpu_monitor_reports_background_failure(
    monitoring_config: MonitoringConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_failed = Event()

    def fail_collection(gpu_index: int = 0) -> GPUSample:
        collection_failed.set()
        raise GPUMonitorError("query failed")

    monkeypatch.setattr(
        gpu_monitor_module,
        "collect_gpu_sample",
        fail_collection,
    )
    monitor = GPUMonitor(monitoring_config)

    monitor.start()
    assert collection_failed.wait(timeout=1.0)

    with pytest.raises(
        GPUMonitorError,
        match="GPU monitoring failed: query failed",
    ):
        monitor.stop()

    assert monitor.is_running is False
    assert monitor.samples == ()
