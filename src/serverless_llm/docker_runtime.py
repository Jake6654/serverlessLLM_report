"""Control the lifecycle of a Dockerized vLLM server."""

import subprocess
import time
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import urlopen

from serverless_llm.config import DockerConfig, ServerConfig


class DockerRuntimeError(RuntimeError):
    """Raised when a Docker lifecycle operation fails."""


class VLLMReadinessTimeoutError(DockerRuntimeError):
    """Raised when vLLM does not become ready before the timeout."""


@dataclass(frozen=True)
class RuntimeStartupResult:
    """Store timestamps measured while starting a real vLLM server."""

    started_at_seconds: float
    ready_at_seconds: float

    @property
    def startup_duration_seconds(self) -> float:
        """Return the elapsed time from Docker start to vLLM readiness."""

        return self.ready_at_seconds - self.started_at_seconds


class DockerVLLMRuntime:
    """Start, inspect, and stop one Dockerized vLLM server."""

    # Store validated configuration objects so every lifecycle operation uses
    # the same image, container name, network address, and timeout settings.
    def __init__(
        self,
        docker_config: DockerConfig,
        server_config: ServerConfig,
    ) -> None:
        """Initialize the runtime with Docker and server configuration."""

        if not isinstance(docker_config, DockerConfig):
            raise ValueError("docker_config must be a DockerConfig object")
        if not isinstance(server_config, ServerConfig):
            raise ValueError("server_config must be a ServerConfig object")

        self._docker_config = docker_config
        self._server_config = server_config

    # Combine the configured host and port with vLLM's standard health path.
    @property
    def health_url(self) -> str:
        """Return the HTTP endpoint used to check vLLM readiness."""

        return (
            f"http://{self._server_config.host}:"
            f"{self._server_config.port}/health"
        )

    # Build a list of arguments instead of a shell string. This prevents shell
    # parsing and makes every Docker argument independently testable.
    def _build_run_command(self) -> list[str]:
        """Return the complete Docker command used to launch vLLM."""

        published_port = (
            f"{self._server_config.host}:"
            f"{self._server_config.port}:"
            f"{self._docker_config.container_port}"
        )
        cache_mount = (
            f"{self._docker_config.huggingface_cache_volume}:"
            "/root/.cache/huggingface"
        )

        return [
            # docker CLI 의 컨테이너 생성 명령
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            self._docker_config.container_name,
            "--gpus",
            "all",
            "--ipc",
            "host",
            "--publish",
            published_port,
            "--volume",
            cache_mount,
            self._docker_config.image,
        ]

  
    @staticmethod
    def _run_docker_command(
        command: list[str],
    ) -> subprocess.CompletedProcess[str]:
        """Execute one Docker command and return the completed process."""

        try:
            #subprocess.run 은 전달받은 리스트의 첫 번째 문자열을 실행할 프로그램 이름으로 해석한다
            # subprocess.run 은 Python이 외부 프로그램을 실행하는 함수이다
            return subprocess.run(
                command, # -> 여기서는 docker
                capture_output=True,
                text=True, # 출력을 byte 가 아닌 문자열로 변환
                check=False, # 명령이 실패해도 자동으로 예외를 발생시키지 않고 아래에서 직접확인
            )
        except OSError as error:
            raise DockerRuntimeError(
                f"Could not execute Docker: {error}"
            ) from error

    # Start only the container here. A successful detached Docker command does
    # not mean vLLM is ready because CUDA and model loading happen afterward.
    def start(self) -> None:
        """Start the container or raise DockerRuntimeError on failure."""

        completed = self._run_docker_command(
            self._build_run_command()
        )

        if completed.returncode != 0:
            raise DockerRuntimeError(
                "Could not start the vLLM container: "
                f"{completed.stderr.strip()}"
            )

    # Ask Docker for running containers and use an anchored name filter so a
    # similarly named container cannot be mistaken for the experiment server.
    @property
    def is_running(self) -> bool:
        """Return True when the configured container is currently running."""

        completed = self._run_docker_command(
            [
                "docker",
                "ps",
                "--filter",
                f"name=^/{self._docker_config.container_name}$",
                "--format",
                "{{.Names}}",
            ]
        )

        if completed.returncode != 0:
            raise DockerRuntimeError(
                "Could not inspect the vLLM container: "
                f"{completed.stderr.strip()}"
            )

        return (
            completed.stdout.strip()
            == self._docker_config.container_name
        )

    # Make cleanup idempotent. Calling stop more than once is safe because an
    # already stopped container returns immediately without another command.
    def stop(self) -> None:
        """Stop the container if running and otherwise return immediately."""

        if not self.is_running:
            return

        completed = self._run_docker_command(
            # -> subprocess.run(["docker", "stop", "serverless-llm-vllm"])
            # 실행 프로그램 docker, docker subcommand: stop, argument: serverless-llm-vllm
            [
                "docker",
                "stop",
                self._docker_config.container_name,
            ]
        )

        if completed.returncode != 0:
            raise DockerRuntimeError(
                "Could not stop the vLLM container: "
                f"{completed.stderr.strip()}"
            )

    # Connection failures are expected while vLLM loads. Treat them as "not
    # ready yet" and let wait_until_ready retry until the startup deadline.
    def _is_ready(self) -> bool:
        """Return True when the vLLM health endpoint responds with HTTP 200."""

        try:
            with urlopen(self.health_url, timeout=1.0) as response:
                return response.status == 200
        except (URLError, TimeoutError, ConnectionError):
            return False

    # Poll three conditions in order: the deadline has not expired, Docker is
    # still running, and the health endpoint is ready. Return the monotonic
    # readiness timestamp so startup duration can be calculated accurately.
    def wait_until_ready(self) -> float:
        """Wait for readiness and return its monotonic timestamp."""

        deadline = (
            time.monotonic()
            + self._server_config.startup_timeout_seconds
        )

        while time.monotonic() < deadline:
            if not self.is_running:
                raise DockerRuntimeError(
                    "The vLLM container stopped before becoming ready"
                )

            if self._is_ready():
                return time.monotonic()

            time.sleep(
                self._server_config.readiness_poll_interval_seconds
            )

        raise VLLMReadinessTimeoutError(
            "vLLM did not become ready within "
            f"{self._server_config.startup_timeout_seconds} seconds"
        )

    # Measure the complete startup operation. If readiness fails after Docker
    # starts, stop the partial container before re-raising the original error.
    def start_and_wait(self) -> RuntimeStartupResult:
        """Start vLLM, wait for readiness, and return startup measurements."""

        started_at_seconds = time.monotonic()
        self.start()

        try:
            ready_at_seconds = self.wait_until_ready()
        except Exception as startup_error:
            try:
                self.stop()
            except DockerRuntimeError as cleanup_error:
                raise DockerRuntimeError(
                    "vLLM startup failed and the container could not be "
                    f"cleaned up: {cleanup_error}"
                ) from startup_error
            raise

        return RuntimeStartupResult(
            started_at_seconds=started_at_seconds,
            ready_at_seconds=ready_at_seconds,
        )
