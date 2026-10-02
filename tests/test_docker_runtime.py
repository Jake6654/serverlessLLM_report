"""Unit tests for the Dockerized vLLM runtime."""

from subprocess import CompletedProcess
from unittest.mock import MagicMock, Mock, PropertyMock
from urllib.error import URLError

import pytest

import serverless_llm.docker_runtime as docker_runtime_module
from serverless_llm.config import DockerConfig, ServerConfig
from serverless_llm.docker_runtime import (
    DockerRuntimeError,
    DockerVLLMRuntime,
    RuntimeStartupResult,
    VLLMReadinessTimeoutError,
)


@pytest.fixture
def docker_config() -> DockerConfig:
    """Return deterministic Docker settings for unit tests."""

    return DockerConfig(
        image="serverless-llm-vllm:0.19.0",
        container_name="serverless-llm-vllm",
        huggingface_cache_volume="serverless-llm-hf-cache",
        container_port=8000,
    )


@pytest.fixture
def server_config() -> ServerConfig:
    """Return short server timings suitable for mocked tests."""

    return ServerConfig(
        host="127.0.0.1",
        port=8000,
        startup_timeout_seconds=10,
        request_timeout_seconds=30,
        readiness_poll_interval_seconds=1.0,
    )


@pytest.fixture
def runtime(
    docker_config: DockerConfig,
    server_config: ServerConfig,
) -> DockerVLLMRuntime:
    """Create a runtime without starting a real Docker container."""

    return DockerVLLMRuntime(
        docker_config=docker_config,
        server_config=server_config,
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


def test_startup_result_calculates_duration() -> None:
    result = RuntimeStartupResult(
        started_at_seconds=10.0,
        ready_at_seconds=22.5,
    )

    assert result.startup_duration_seconds == 12.5


def test_runtime_rejects_invalid_configuration_objects(
    docker_config: DockerConfig,
    server_config: ServerConfig,
) -> None:
    with pytest.raises(ValueError, match="docker_config"):
        DockerVLLMRuntime(object(), server_config)

    with pytest.raises(ValueError, match="server_config"):
        DockerVLLMRuntime(docker_config, object())


def test_health_url_uses_server_address(
    runtime: DockerVLLMRuntime,
) -> None:
    assert runtime.health_url == "http://127.0.0.1:8000/health"


def test_build_run_command_contains_required_docker_options(
    runtime: DockerVLLMRuntime,
) -> None:
    assert runtime._build_run_command() == [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--name",
        "serverless-llm-vllm",
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--publish",
        "127.0.0.1:8000:8000",
        "--volume",
        "serverless-llm-hf-cache:/root/.cache/huggingface",
        "serverless-llm-vllm:0.19.0",
    ]


def test_start_executes_docker_run(
    runtime: DockerVLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run = Mock(
        return_value=completed_process(stdout="container-id\n")
    )
    monkeypatch.setattr(
        docker_runtime_module.subprocess,
        "run",
        fake_run,
    )

    assert runtime.start() is None
    fake_run.assert_called_once_with(
        runtime._build_run_command(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_start_reports_docker_failure(
    runtime: DockerVLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        docker_runtime_module.subprocess,
        "run",
        Mock(
            return_value=completed_process(
                returncode=125,
                stderr="container name is already in use\n",
            )
        ),
    )

    with pytest.raises(
        DockerRuntimeError,
        match="container name is already in use",
    ):
        runtime.start()


def test_missing_docker_executable_becomes_runtime_error(
    runtime: DockerVLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        docker_runtime_module.subprocess,
        "run",
        Mock(side_effect=FileNotFoundError("docker not found")),
    )

    with pytest.raises(
        DockerRuntimeError,
        match="Could not execute Docker",
    ):
        runtime.start()


@pytest.mark.parametrize(
    ("docker_output", "expected"),
    [
        ("serverless-llm-vllm\n", True),
        ("", False),
    ],
)
def test_is_running_uses_exact_container_name(
    runtime: DockerVLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
    docker_output: str,
    expected: bool,
) -> None:
    fake_run = Mock(
        return_value=completed_process(stdout=docker_output)
    )
    monkeypatch.setattr(
        docker_runtime_module.subprocess,
        "run",
        fake_run,
    )

    assert runtime.is_running is expected
    command = fake_run.call_args.args[0]
    assert command == [
        "docker",
        "ps",
        "--filter",
        "name=^/serverless-llm-vllm$",
        "--format",
        "{{.Names}}",
    ]


def test_is_running_reports_inspection_failure(
    runtime: DockerVLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        docker_runtime_module.subprocess,
        "run",
        Mock(
            return_value=completed_process(
                returncode=1,
                stderr="Docker daemon unavailable",
            )
        ),
    )

    with pytest.raises(
        DockerRuntimeError,
        match="Docker daemon unavailable",
    ):
        _ = runtime.is_running


def test_stop_stops_a_running_container(
    runtime: DockerVLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run = Mock(
        side_effect=[
            completed_process(stdout="serverless-llm-vllm\n"),
            completed_process(stdout="serverless-llm-vllm\n"),
        ]
    )
    monkeypatch.setattr(
        docker_runtime_module.subprocess,
        "run",
        fake_run,
    )

    assert runtime.stop() is None
    assert fake_run.call_count == 2
    assert fake_run.call_args_list[1].args[0] == [
        "docker",
        "stop",
        "serverless-llm-vllm",
    ]


def test_stop_is_no_op_when_container_is_not_running(
    runtime: DockerVLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run = Mock(return_value=completed_process(stdout=""))
    monkeypatch.setattr(
        docker_runtime_module.subprocess,
        "run",
        fake_run,
    )

    assert runtime.stop() is None
    fake_run.assert_called_once()


def test_is_ready_returns_true_for_http_200(
    runtime: DockerVLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = MagicMock()
    response.__enter__.return_value.status = 200
    fake_urlopen = Mock(return_value=response)
    monkeypatch.setattr(
        docker_runtime_module,
        "urlopen",
        fake_urlopen,
    )

    assert runtime._is_ready() is True
    fake_urlopen.assert_called_once_with(
        "http://127.0.0.1:8000/health",
        timeout=1.0,
    )


def test_is_ready_returns_false_for_connection_failure(
    runtime: DockerVLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        docker_runtime_module,
        "urlopen",
        Mock(side_effect=URLError("connection refused")),
    )

    assert runtime._is_ready() is False


def test_wait_until_ready_retries_until_health_succeeds(
    runtime: DockerVLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        DockerVLLMRuntime,
        "is_running",
        PropertyMock(return_value=True),
    )
    readiness = Mock(side_effect=[False, True])
    monkeypatch.setattr(runtime, "_is_ready", readiness)
    monotonic = Mock(side_effect=[0.0, 1.0, 2.0, 3.0])
    sleep = Mock()
    monkeypatch.setattr(
        docker_runtime_module.time,
        "monotonic",
        monotonic,
    )
    monkeypatch.setattr(
        docker_runtime_module.time,
        "sleep",
        sleep,
    )

    assert runtime.wait_until_ready() == 3.0
    assert readiness.call_count == 2
    sleep.assert_called_once_with(1.0)


def test_wait_until_ready_rejects_stopped_container(
    runtime: DockerVLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        DockerVLLMRuntime,
        "is_running",
        PropertyMock(return_value=False),
    )
    monkeypatch.setattr(
        docker_runtime_module.time,
        "monotonic",
        Mock(side_effect=[0.0, 1.0]),
    )

    with pytest.raises(
        DockerRuntimeError,
        match="stopped before becoming ready",
    ):
        runtime.wait_until_ready()


def test_wait_until_ready_reports_timeout(
    runtime: DockerVLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        DockerVLLMRuntime,
        "is_running",
        PropertyMock(return_value=True),
    )
    monkeypatch.setattr(runtime, "_is_ready", Mock(return_value=False))
    monkeypatch.setattr(
        docker_runtime_module.time,
        "monotonic",
        Mock(side_effect=[0.0, 1.0, 10.0]),
    )
    sleep = Mock()
    monkeypatch.setattr(
        docker_runtime_module.time,
        "sleep",
        sleep,
    )

    with pytest.raises(
        VLLMReadinessTimeoutError,
        match="within 10 seconds",
    ):
        runtime.wait_until_ready()

    sleep.assert_called_once_with(1.0)


def test_start_and_wait_returns_measured_startup_result(
    runtime: DockerVLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = Mock()
    wait_until_ready = Mock(return_value=15.5)
    monkeypatch.setattr(runtime, "start", start)
    monkeypatch.setattr(runtime, "wait_until_ready", wait_until_ready)
    monkeypatch.setattr(
        docker_runtime_module.time,
        "monotonic",
        Mock(return_value=3.0),
    )

    result = runtime.start_and_wait()

    assert result == RuntimeStartupResult(
        started_at_seconds=3.0,
        ready_at_seconds=15.5,
    )
    assert result.startup_duration_seconds == 12.5
    start.assert_called_once_with()
    wait_until_ready.assert_called_once_with()


def test_start_and_wait_stops_container_after_readiness_failure(
    runtime: DockerVLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = Mock()
    stop = Mock()
    monkeypatch.setattr(runtime, "start", start)
    monkeypatch.setattr(
        runtime,
        "wait_until_ready",
        Mock(side_effect=VLLMReadinessTimeoutError("timed out")),
    )
    monkeypatch.setattr(runtime, "stop", stop)
    monkeypatch.setattr(
        docker_runtime_module.time,
        "monotonic",
        Mock(return_value=0.0),
    )

    with pytest.raises(
        VLLMReadinessTimeoutError,
        match="timed out",
    ):
        runtime.start_and_wait()

    start.assert_called_once_with()
    stop.assert_called_once_with()

@pytest.mark.parametrize(
    "connection_error",
    [
        URLError("connection refused"),
        TimeoutError("health check timed out"),
        ConnectionResetError(
            104,
            "Connection reset by peer",
        ),
    ],
)
def test_is_ready_returns_false_for_transient_connection_errors(
    runtime: DockerVLLMRuntime,
    monkeypatch: pytest.MonkeyPatch,
    connection_error: Exception,
) -> None:
    monkeypatch.setattr(
        docker_runtime_module,
        "urlopen",
        Mock(side_effect=connection_error),
    )

    assert runtime._is_ready() is False
