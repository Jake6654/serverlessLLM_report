"""Collect and represent GPU measurements from NVIDIA hardware."""

import subprocess
import time
from dataclasses import dataclass
from math import isfinite
from threading import Event, Lock, Thread

from serverless_llm.config import MonitoringConfig

class GPUMonitorError(RuntimeError):
    """Raised when an NVIDIA GPU measurment cannot be collected"""

@dataclass(frozen=True)
class GPUSample:
    """Store one GPU measurement captured at a monotonic timestamp"""

    sampled_at_seconds: float
    gpu_index: int
    utilization_percent: float
    memory_used_mib: float
    power_draw_watts: float
    temperature_celsius: float

    def __post_init__(self) -> None:
        """Reject invalid GPU measurement values immediately"""

        if (
            type(self.sampled_at_seconds) not in (int, float)
            or not isfinite(self.sampled_at_seconds)
            or self.sampled_at_seconds < 0
        ):
            raise ValueError(
                "sampled_at_seconds must be a finite "
                "non-negative number"
            )

        # GPU indexed start at zero and must be integers
        if type(self.gpu_index) is not int or self.gpu_index < 0:
            raise ValueError(
                "gpu_index must be a non-negative integer"
            )

        #GPU utilization is reported as a percentage from 0 to 100.
        if (
            type(self.utilization_percent) not in (int, float)
            or not isfinite(self.utilization_percent)
            or not 0 <= self.utilization_percent <= 100
        ):
            raise ValueError(
                "utilization_percent must be a finite number "
                "between 0 and 100"
            )
        # Used GPU memory cannot be negative.
        if (
            type(self.memory_used_mib) not in (int, float)
            or not isfinite(self.memory_used_mib)
            or self.memory_used_mib < 0
        ):
            raise ValueError(
                "memory_used_mib must be a finite "
                "non-negative number"
            )

        # Power draw is measured in watts and cannot be negative.
        if (
            type(self.power_draw_watts) not in (int, float)
            or not isfinite(self.power_draw_watts)
            or self.power_draw_watts < 0
        ):
            raise ValueError(
                "power_draw_watts must be a finite "
                "non-negative number"
            )

        # NVIDIA reports GPU temperature in degrees Celsius.
        if (
            type(self.temperature_celsius) not in (int, float)
            or not isfinite(self.temperature_celsius)
            or self.temperature_celsius < 0
        ):
            raise ValueError(
                "temperature_celsius must be a finite "
                "non-negative number"
            )

def _parse_nvidia_smi_output(
        output: str,
        sampled_at_seconds: float,
) -> GPUSample:
    """Parse one CSV row from nvidia-smi into a GPUSample"""

    if not isinstance(output, str):
        raise GPUMonitorError(
            "nvidia-smi output must be a string"
        )

    # Remove surrounding whitespace and ignore empty lines
    rows = [
        row.strip() # 반환값
        for row in output.splitlines()
        if row.strip() # 빈 문자열이면  False 
    ]

    # One query targets exactly one GPU, so exactly one row is expected.
    if len(rows) != 1:
        raise GPUMonitorError(
            "nividia-smi must return exactly one row is expected"
        )

    fields = [
        field.strip()
        for field in rows[0].split(",")
    ]

    # The query requests five values in a fixed order.
    if len(fields) != 5:
        raise GPUMonitorError(
            "nvidia-smi returned an unexpected number of fields"
        )

    try:
        # 문열을 숫자열로 반환
        gpu_index= int(fields[0])
        utilization_percent = float(fields[1])
        memory_used_mib = float(fields[2])
        power_draw_watts = float(fields[3])
        temperature_celsius = float(fields[4])
    except ValueError as error:
        raise GPUMonitorError(
            "nvidia-smi returned a non-numeric GPU value"
        ) from error

    return GPUSample(
        sampled_at_seconds=sampled_at_seconds,
        gpu_index=gpu_index,
        utilization_percent=utilization_percent,
        memory_used_mib=memory_used_mib,
        power_draw_watts=power_draw_watts,
        temperature_celsius=temperature_celsius,
    )

def collect_gpu_sample(gpu_index: int = 0) -> GPUSample:
    """Collect and return one measurement from an NVIDIA GPU."""

    if type(gpu_index) is not int or gpu_index < 0:
        raise ValueError(
            "gpu_index must be a non-negative integer"
        )

    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        (
            "--query-gpu=index,utilization.gpu,memory.used,"
            "power.draw,temperature.gpu"
        ),
        "--format=csv,noheader,nounits",
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except subprocess.TimeoutExpired as error:
        raise GPUMonitorError(
            "nvidia-smi did not finish within 5 seconds"
        ) from error
    except OSError as error:
        raise GPUMonitorError(
            f"Could not execute nvidia-smi: {error}"
        ) from error

    if completed.returncode != 0:
        error_message = (
            completed.stderr.strip()
            or "unknown nvidia-smi error"
        )
        raise GPUMonitorError(
            f"Could not collect GPU measurement: {error_message}"
        )

    sampled_at_seconds = time.monotonic()

    return _parse_nvidia_smi_output(
        completed.stdout,
        sampled_at_seconds,
    )


class GPUMonitor:
    """Collect NVIDIA GPU samples on a background thread"""

    def __init__(
        self,
        monitoring_config: MonitoringConfig,
        gpu_index: int = 0,
    ) -> None:
        """Initialize one reusable configuration for a monitoring run."""

        if not isinstance(monitoring_config, MonitoringConfig):
            raise ValueError(
                "monitoring_config must be a MonitoringConfig object"
            )

        if type(gpu_index) is not int or gpu_index < 0:
            raise ValueError(
                "gpu_index must be a non-negative integer"
            )

        self._gpu_index = gpu_index
        self._sample_interval_seconds = (
            monitoring_config.gpu_sample_interval_ms / 1000.0
        )

        # _속성들은 클래스 내부 구현을 위한 것이므로 외부에서 직ㅈ버 사용하거나 변경하지 않는 것이 좋다
        # stop 신호가 있으면 측정 종료
        self._stop_event = Event() 

        # Lock protects state shared by the main and worker threads.
        self._lock = Lock()

        # Samples are appended by the worker and read by the main thread.
        self._samples: list[GPUSample] = []

        # Store a background failure so stop() can report it to the caller.
        self._worker_error: Exception | None = None

        # A monitor is started at most once.
        self._thread: Thread | None = None

    @property
    def is_running(self) -> bool:
        """Retrun True while the background worker thread is alive"""

        return(
            self._thread is not None
            and self._thread.is_alive()
        )
    @property
    def samples(self) -> tuple[GPUSample, ...]:
        """Return an immutable sample of all collected gpu smaples"""

        with self._lock:
            return tuple(self._samples)


    def _run(self) -> None:
        """Collect samples until the main thread requests a stop."""

        try:
            while not self._stop_event.is_set():
                iteration_started_at = time.monotonic()

                sample = collect_gpu_sample(
                    gpu_index=self._gpu_index
                )

                with self._lock:
                    self._samples.append(sample)

                # Subtract command execution time so sample starts remain
                # close to the configured monitoring interval.
                iteration_duration = (
                    time.monotonic() - iteration_started_at
                )

                wait_seconds = max(
                    0.0,
                    self._sample_interval_seconds - iteration_duration
                )

                # Event.wait() returns early when stop() sets the event.
                self._stop_event.wait(wait_seconds)

        except Exception as error:
            # Exceptions cannot automatically cross thread boundaries.
            # Save the error so stop() can raise it in the main thread.
            with self._lock:
                self._worker_error = error

            self._stop_event.set()

    def start(self) -> None:
        """Start collecting GPU samples on a background thread."""

        if self._thread is not None:
            raise GPUMonitorError(
                "GPU monitor has already been started"
            )

        self._stop_event.clear()

        self._thread = Thread(
            target=self._run, # 새 스레드가 실행할 메서드
            name="gpu-monitor",
            daemon=True, #
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and report any background worker failure."""

        thread = self._thread

        # Stopping a monitor that was never started is safe.
        if thread is None:
            return

        self._stop_event.set()
        thread.join()

        with self._lock:
            worker_error = self._worker_error

        if worker_error is not None:
            raise GPUMonitorError(
                f"GPU monitoring failed: {worker_error}"
            ) from worker_error


