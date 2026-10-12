"""Orchestrate complete real serverless LLM experiments."""
from dataclasses import dataclass
from pathlib import Path


from serverless_llm.config import ExperimentConfig
from serverless_llm.docker_runtime import DockerVLLMRuntime
from serverless_llm.gpu_monitor import GPUMonitor
from serverless_llm.real_policy_runner import (
    RealPolicyRunResult,
    run_real_always_on_workload,
    run_real_fixed_keep_warm_workload,
    run_real_naive_serverless_workload,
)
from serverless_llm.vllm_client import VLLMClient
from serverless_llm.workload import (
    WorkloadTrace,
    generate_steady_workload,
)

from serverless_llm.real_metrics import (
    RealPolicySummary,
    summarize_policy_run,
)
from serverless_llm.result_writer import (
    PolicyResultPaths,
    write_policy_results,
)

SUPPORTED_REAL_POLICIES = frozenset(
    {
        "always_on",
        "naive_serverless",
        "fixed_keep_warm",
    }
)

class RealExperimentError(RuntimeError):
    """Raised when a real experiment cannot be constructed or executed."""

@dataclass(frozen=True)
class RealExperimentComponents:
    """Store the runtime services needed by one real policy run."""

    runtime: DockerVLLMRuntime
    client: VLLMClient
    monitor: GPUMonitor

@dataclass(frozen=True)
class RealExperimentResult:
    """Store outputs produced by one complete real policy experiment."""

    run_result: RealPolicyRunResult
    summary: RealPolicySummary
    paths: PolicyResultPaths

def build_workload(
        config: ExperimentConfig,
) -> WorkloadTrace:
    """Build a deterministic workload from experiment configuration."""

    if not isinstance(config, ExperimentConfig):
        raise ValueError(
            "config must be an ExperimentConfig object"
        )

    # The current real policy runners process one request at a time.
    if config.workload.concurrency != 1:
        raise RealExperimentError(
            "real experiments currently require concurrency=1"
        )

    # Phase G7 initially supports the steady workload configuration.
    if config.workload.type != "steady":
        raise RealExperimentError(
            "real experiments currently support only "
            "workload.type=steady"
        )

    return generate_steady_workload(
        total_requests=config.workload.total_requests,
        interval_seconds=(
            config.workload.request_interval_seconds
        ),
        random_seed=config.experiment.random_seed,
        prompt_id="default",
    )

def build_components(
        config: ExperimentConfig,
) -> RealExperimentComponents:
    """Build fresh runtime services for one real policy run."""

    if not isinstance(config, ExperimentConfig):
        raise ValueError(
            "config must be an ExperimentConfig object"
        )

    runtime = DockerVLLMRuntime(
        docker_config=config.docker,
        server_config=config.server,
    )

    client = VLLMClient(
        model_config=config.model,
        server_config=config.server,
        request_config=config.request,
    )

    monitor = GPUMonitor(
        monitoring_config=config.monitoring,
    )

    return RealExperimentComponents(
        runtime=runtime,
        client=client,
        monitor=monitor,
    )


def run_real_policy(
    policy_name: str,
    config: ExperimentConfig,
    components: RealExperimentComponents,
    trace: WorkloadTrace,
) -> RealPolicyRunResult:
    """전달받은 정책 이름에 맞는 기존 runner 을 선택하는 dispatcher이다"""

    if (
        not isinstance(policy_name, str)
        or not policy_name.strip()
    ):
        raise ValueError(
            "policy_name must be a non-empty string"
        )

    if not isinstance(config, ExperimentConfig):
        raise ValueError(
            "config must be an ExperimentConfig object"
        )

    if not isinstance(
        components,
        RealExperimentComponents,
    ):
        raise ValueError(
            "components must be a RealExperimentComponents object"
        )

    if not isinstance(trace, WorkloadTrace):
        raise ValueError(
            "trace must be a WorkloadTrace object"
        )

    if policy_name == "always_on":
        return run_real_always_on_workload(
            runtime=components.runtime,
            client=components.client,
            monitor=components.monitor,
            trace=trace,
        )

    if policy_name == "naive_serverless":
        return run_real_naive_serverless_workload(
            runtime=components.runtime,
            client=components.client,
            monitor=components.monitor,
            trace=trace,
        )

    if policy_name == "fixed_keep_warm":
        return run_real_fixed_keep_warm_workload(
            runtime=components.runtime,
            client=components.client,
            monitor=components.monitor,
            trace=trace,
            timeout_seconds=(
                config.policies.fixed_keep_warm.timeout_seconds
            ),
        )

    supported = ", ".join(
        sorted(SUPPORTED_REAL_POLICIES)
    )

    raise RealExperimentError(
        f"unsupported real policy: {policy_name}. "
        f"Supported policies: {supported}"
    )

def build_repetition_output_directory(
        config: ExperimentConfig,
        policy_name: str,
        repetition: int,
) -> Path:
    """Return the result directory for one policy repetition."""

    if not isinstance(config, ExperimentConfig):
        raise ValueError(
            "config must be an ExperimentConfig object"
        )

    if (
        not isinstance(policy_name, str)
        or not policy_name.strip()
    ):
        raise ValueError(
            "policy_name must be a non-empty string"
        )

    if policy_name not in SUPPORTED_REAL_POLICIES:
        supported = ", ".join(
            sorted(SUPPORTED_REAL_POLICIES)
        )
        raise RealExperimentError(
            f"unsupported real policy: {policy_name}. "
            f"Supported policies: {supported}"
        )

    if type(repetition) is not int:
        raise ValueError(
            "repetition must be an integer"
        )

    if (
        repetition < 1
        or repetition > config.experiment.repetitions
    ):
        raise ValueError(
            "repetition must be between 1 and "
            f"{config.experiment.repetitions}"
        )

    return (
        # Path 객체에서 / 는 실제 나눗셈이 아니라 경로를 연결하는 연산자이다
        config.output.results_directory
        / config.experiment.name
        / policy_name
        / f"repetition-{repetition:03d}" # d 는 정수형, 03 -> 001 세 자리까지 앞을 0
    )
    
def execute_policy_experiment(
    policy_name: str,
    config: ExperimentConfig,
    output_directory: Path,
) -> RealExperimentResult:
    """Execute, summarize, and save one complete real policy run."""

    if not isinstance(config, ExperimentConfig):
        raise ValueError(
            "config must be an ExperimentConfig object"
        )

    if not isinstance(output_directory, Path):
        raise ValueError(
            "output_directory must be a Path object"
        )

    if (
        output_directory.exists()
        and not output_directory.is_dir()
    ):
        raise ValueError(
            "output_directory must be a directory path"
        )

    trace = build_workload(config)
    components = build_components(config)

    run_result = run_real_policy(
        policy_name=policy_name,
        config=config,
        components=components,
        trace=trace,
    )

    summary = summarize_policy_run(
        run_result,
        ttft_slo_seconds=(
            config.metrics.ttft_slo_seconds
        ),
    )

    paths = write_policy_results(
        output_directory=output_directory,
        run_result=run_result,
        policy_summary=summary,
    )

    return RealExperimentResult(
        run_result=run_result,
        summary=summary,
        paths=paths,
    )

def execute_policy_repetitions(
        policy_name: str,
        config: ExperimentConfig,
) -> tuple[RealExperimentResult,...]:
    """Execute every configured repetition for one real policy"""

    if not isinstance(config, ExperimentConfig):
        raise ValueError(
            "config must be an ExperimentConfig object"
        )

    output_directories: list[Path] = []

    for repetition in range(
        1, config.experiment.repetitions +1,
    ):
        output_directory = (
            build_repetition_output_directory(
                config=config,
                policy_name=policy_name,
                repetition=repetition
            )
        )
        output_directories.append(output_directory)

    existing_directories = [
        output_directory
        for output_directory in output_directories
        if output_directory.exists()
    ]

    if existing_directories:
        formatted_directories = ", ".join(
            str(directory)
            for directory in existing_directories
        )

        raise RealExperimentError(
            "experiment result directories already exist: "
            f"{formatted_directories}"
        )

    results: list[RealExperimentResult] = []

    for output_directory in output_directories:
        result = execute_policy_experiment(
            policy_name=policy_name,
            config=config,
            output_directory=output_directory,
        )
        results.append(result)

    return tuple(results)
