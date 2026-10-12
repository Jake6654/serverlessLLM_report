"""Orchestrate complete real serverless LLM experiments."""
from dataclasses import dataclass


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
    """Run one configured real policy against one workload trace."""

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
    
