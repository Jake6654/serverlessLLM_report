"""Unit tests for real experiment orchestration."""

from dataclasses import replace
from pathlib import Path

import pytest

import serverless_llm.real_experiment as experiment_module
from serverless_llm.config import ExperimentConfig, load_config
from serverless_llm.docker_runtime import DockerVLLMRuntime
from serverless_llm.gpu_monitor import GPUMonitor
from serverless_llm.real_experiment import (
    RealExperimentError,
    RealExperimentComponents,
    build_components,
    build_workload,
    run_real_policy,
)
from serverless_llm.vllm_client import VLLMClient
from serverless_llm.workload import WorkloadTrace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


@pytest.fixture
def experiment_config() -> ExperimentConfig:
    """Load the valid project configuration used by orchestration tests."""

    return load_config(CONFIG_PATH)


@pytest.fixture
def workload_trace(
    experiment_config: ExperimentConfig,
) -> WorkloadTrace:
    """Build the valid deterministic trace used by dispatch tests."""

    return build_workload(experiment_config)


@pytest.fixture
def experiment_components(
    experiment_config: ExperimentConfig,
) -> RealExperimentComponents:
    """Build services without starting Docker, HTTP, or GPU sampling."""

    return build_components(experiment_config)


def test_build_workload_creates_steady_trace(
    experiment_config: ExperimentConfig,
) -> None:
    """Convert steady workload configuration into a request trace."""

    trace = build_workload(experiment_config)

    assert isinstance(trace, WorkloadTrace)
    assert trace.pattern == "steady"
    assert trace.total_requests == 20
    assert trace.random_seed == 42
    assert trace.duration_seconds == 95.0


def test_build_workload_uses_configured_request_interval(
    experiment_config: ExperimentConfig,
) -> None:
    """Schedule requests using the interval stored in configuration."""

    trace = build_workload(experiment_config)

    scheduled_times = [
        event.scheduled_at_seconds
        for event in trace.events[:3]
    ]

    assert scheduled_times == [0.0, 5.0, 10.0]


def test_build_workload_is_deterministic(
    experiment_config: ExperimentConfig,
) -> None:
    """Produce identical traces from identical configuration."""

    first_trace = build_workload(experiment_config)
    second_trace = build_workload(experiment_config)

    assert first_trace == second_trace


def test_build_workload_rejects_non_config_object() -> None:
    """Reject values that are not ExperimentConfig objects."""

    with pytest.raises(
        ValueError,
        match="config must be an ExperimentConfig object",
    ):
        build_workload("configs/experiment.yaml")


def test_build_workload_rejects_concurrent_execution(
    experiment_config: ExperimentConfig,
) -> None:
    """Reject concurrency unsupported by sequential real runners."""

    concurrent_workload = replace(
        experiment_config.workload,
        concurrency=2,
    )
    concurrent_config = replace(
        experiment_config,
        workload=concurrent_workload,
    )

    with pytest.raises(
        RealExperimentError,
        match="currently require concurrency=1",
    ):
        build_workload(concurrent_config)


def test_build_workload_rejects_unsupported_pattern(
    experiment_config: ExperimentConfig,
) -> None:
    """Reject workload patterns lacking complete real-run settings."""

    bursty_workload = replace(
        experiment_config.workload,
        type="bursty",
    )
    bursty_config = replace(
        experiment_config,
        workload=bursty_workload,
    )

    with pytest.raises(
        RealExperimentError,
        match="currently support only workload.type=steady",
    ):
        build_workload(bursty_config)


def test_build_components_creates_required_services(
    experiment_config: ExperimentConfig,
) -> None:
    """Construct every service required by one real policy run."""

    components = build_components(experiment_config)

    assert isinstance(components, RealExperimentComponents)
    assert isinstance(components.runtime, DockerVLLMRuntime)
    assert isinstance(components.client, VLLMClient)
    assert isinstance(components.monitor, GPUMonitor)


def test_build_components_returns_fresh_services(
    experiment_config: ExperimentConfig,
) -> None:
    """Prevent runtime and monitor state from leaking between runs."""

    first = build_components(experiment_config)
    second = build_components(experiment_config)

    assert first is not second
    assert first.runtime is not second.runtime
    assert first.client is not second.client
    assert first.monitor is not second.monitor


def test_build_components_rejects_non_config_object() -> None:
    """Reject values that are not complete experiment configurations."""

    with pytest.raises(
        ValueError,
        match="config must be an ExperimentConfig object",
    ):
        build_components("configs/experiment.yaml")


def test_run_real_policy_dispatches_always_on(
    monkeypatch: pytest.MonkeyPatch,
    experiment_config: ExperimentConfig,
    experiment_components: RealExperimentComponents,
    workload_trace: WorkloadTrace,
) -> None:
    """Forward Always-On runs to the Always-On runner."""

    expected_result = object()
    received: dict[str, object] = {}

    def fake_runner(**kwargs: object) -> object:
        received.update(kwargs)
        return expected_result

    monkeypatch.setattr(
        experiment_module,
        "run_real_always_on_workload",
        fake_runner,
    )

    result = run_real_policy(
        "always_on",
        experiment_config,
        experiment_components,
        workload_trace,
    )

    assert result is expected_result
    assert received == {
        "runtime": experiment_components.runtime,
        "client": experiment_components.client,
        "monitor": experiment_components.monitor,
        "trace": workload_trace,
    }


def test_run_real_policy_dispatches_naive_serverless(
    monkeypatch: pytest.MonkeyPatch,
    experiment_config: ExperimentConfig,
    experiment_components: RealExperimentComponents,
    workload_trace: WorkloadTrace,
) -> None:
    """Forward Naive Serverless runs to its dedicated runner."""

    expected_result = object()
    received: dict[str, object] = {}

    def fake_runner(**kwargs: object) -> object:
        received.update(kwargs)
        return expected_result

    monkeypatch.setattr(
        experiment_module,
        "run_real_naive_serverless_workload",
        fake_runner,
    )

    result = run_real_policy(
        "naive_serverless",
        experiment_config,
        experiment_components,
        workload_trace,
    )

    assert result is expected_result
    assert received == {
        "runtime": experiment_components.runtime,
        "client": experiment_components.client,
        "monitor": experiment_components.monitor,
        "trace": workload_trace,
    }


def test_run_real_policy_dispatches_fixed_keep_warm(
    monkeypatch: pytest.MonkeyPatch,
    experiment_config: ExperimentConfig,
    experiment_components: RealExperimentComponents,
    workload_trace: WorkloadTrace,
) -> None:
    """Pass the configured timeout to the Fixed Keep-Warm runner."""

    expected_result = object()
    received: dict[str, object] = {}

    def fake_runner(**kwargs: object) -> object:
        received.update(kwargs)
        return expected_result

    monkeypatch.setattr(
        experiment_module,
        "run_real_fixed_keep_warm_workload",
        fake_runner,
    )

    result = run_real_policy(
        "fixed_keep_warm",
        experiment_config,
        experiment_components,
        workload_trace,
    )

    assert result is expected_result
    assert received == {
        "runtime": experiment_components.runtime,
        "client": experiment_components.client,
        "monitor": experiment_components.monitor,
        "trace": workload_trace,
        "timeout_seconds": 30,
    }


def test_run_real_policy_rejects_unsupported_policy(
    experiment_config: ExperimentConfig,
    experiment_components: RealExperimentComponents,
    workload_trace: WorkloadTrace,
) -> None:
    """Reject policies that have no real runner implementation."""

    with pytest.raises(
        RealExperimentError,
        match="unsupported real policy: ml_based",
    ):
        run_real_policy(
            "ml_based",
            experiment_config,
            experiment_components,
            workload_trace,
        )


@pytest.mark.parametrize("policy_name", ["", "   ", None])
def test_run_real_policy_rejects_invalid_policy_name(
    policy_name: object,
    experiment_config: ExperimentConfig,
    experiment_components: RealExperimentComponents,
    workload_trace: WorkloadTrace,
) -> None:
    """Require a non-empty string policy identifier."""

    with pytest.raises(
        ValueError,
        match="policy_name must be a non-empty string",
    ):
        run_real_policy(
            policy_name,
            experiment_config,
            experiment_components,
            workload_trace,
        )
