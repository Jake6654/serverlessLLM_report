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
    RealExperimentResult,
    build_components,
    build_repetition_output_directory,
    build_workload,
    execute_policy_experiment,
    execute_policy_repetitions,
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


def test_execute_policy_experiment_runs_complete_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    experiment_config: ExperimentConfig,
) -> None:
    """Build, execute, summarize, and save one policy run in order."""

    trace = object()
    components = object()
    run_result = object()
    summary = object()
    paths = object()
    calls: list[tuple[str, object]] = []

    def fake_build_workload(config: object) -> object:
        calls.append(("build_workload", config))
        return trace

    def fake_build_components(config: object) -> object:
        calls.append(("build_components", config))
        return components

    def fake_run_real_policy(**kwargs: object) -> object:
        calls.append(("run_real_policy", kwargs))
        return run_result

    def fake_summarize_policy_run(
        received_run_result: object,
        *,
        ttft_slo_seconds: float,
    ) -> object:
        calls.append(
            (
                "summarize_policy_run",
                (received_run_result, ttft_slo_seconds),
            )
        )
        return summary

    def fake_write_policy_results(**kwargs: object) -> object:
        calls.append(("write_policy_results", kwargs))
        return paths

    monkeypatch.setattr(
        experiment_module,
        "build_workload",
        fake_build_workload,
    )
    monkeypatch.setattr(
        experiment_module,
        "build_components",
        fake_build_components,
    )
    monkeypatch.setattr(
        experiment_module,
        "run_real_policy",
        fake_run_real_policy,
    )
    monkeypatch.setattr(
        experiment_module,
        "summarize_policy_run",
        fake_summarize_policy_run,
    )
    monkeypatch.setattr(
        experiment_module,
        "write_policy_results",
        fake_write_policy_results,
    )

    output_directory = tmp_path / "always_on" / "repetition-001"

    result = execute_policy_experiment(
        policy_name="always_on",
        config=experiment_config,
        output_directory=output_directory,
    )

    assert isinstance(result, RealExperimentResult)
    assert result.run_result is run_result
    assert result.summary is summary
    assert result.paths is paths
    assert calls == [
        ("build_workload", experiment_config),
        ("build_components", experiment_config),
        (
            "run_real_policy",
            {
                "policy_name": "always_on",
                "config": experiment_config,
                "components": components,
                "trace": trace,
            },
        ),
        (
            "summarize_policy_run",
            (run_result, 1.0),
        ),
        (
            "write_policy_results",
            {
                "output_directory": output_directory,
                "run_result": run_result,
                "policy_summary": summary,
            },
        ),
    ]


def test_execute_policy_experiment_rejects_non_config_object(
    tmp_path: Path,
) -> None:
    """Reject invalid configuration before starting an experiment."""

    with pytest.raises(
        ValueError,
        match="config must be an ExperimentConfig object",
    ):
        execute_policy_experiment(
            policy_name="always_on",
            config="configs/experiment.yaml",
            output_directory=tmp_path,
        )


def test_execute_policy_experiment_rejects_non_path_output(
    experiment_config: ExperimentConfig,
) -> None:
    """Require a Path object for the result directory."""

    with pytest.raises(
        ValueError,
        match="output_directory must be a Path object",
    ):
        execute_policy_experiment(
            policy_name="always_on",
            config=experiment_config,
            output_directory="results/always_on",
        )


def test_execute_policy_experiment_rejects_existing_file_path(
    tmp_path: Path,
    experiment_config: ExperimentConfig,
) -> None:
    """Reject a regular file where a result directory is required."""

    output_file = tmp_path / "not-a-directory"
    output_file.write_text("existing file", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="output_directory must be a directory path",
    ):
        execute_policy_experiment(
            policy_name="always_on",
            config=experiment_config,
            output_directory=output_file,
        )


def test_build_repetition_output_directory_formats_path(
    tmp_path: Path,
    experiment_config: ExperimentConfig,
) -> None:
    """Build a stable experiment, policy, and repetition hierarchy."""

    output_config = replace(
        experiment_config.output,
        results_directory=tmp_path / "results",
    )
    config = replace(
        experiment_config,
        output=output_config,
    )

    output_directory = build_repetition_output_directory(
        config=config,
        policy_name="always_on",
        repetition=2,
    )

    assert output_directory == (
        tmp_path
        / "results"
        / "traffic-aware-serverless-llm"
        / "always_on"
        / "repetition-002"
    )


@pytest.mark.parametrize("repetition", [0, 4])
def test_build_repetition_output_directory_rejects_out_of_range_value(
    repetition: int,
    experiment_config: ExperimentConfig,
) -> None:
    """Require repetitions to remain within configured run bounds."""

    with pytest.raises(
        ValueError,
        match="repetition must be between 1 and 3",
    ):
        build_repetition_output_directory(
            config=experiment_config,
            policy_name="always_on",
            repetition=repetition,
        )


@pytest.mark.parametrize("repetition", [1.0, True, "1"])
def test_build_repetition_output_directory_rejects_non_integer(
    repetition: object,
    experiment_config: ExperimentConfig,
) -> None:
    """Reject non-integer repetition identifiers, including booleans."""

    with pytest.raises(
        ValueError,
        match="repetition must be an integer",
    ):
        build_repetition_output_directory(
            config=experiment_config,
            policy_name="always_on",
            repetition=repetition,
        )


def test_build_repetition_output_directory_rejects_unknown_policy(
    experiment_config: ExperimentConfig,
) -> None:
    """Reject result paths for policies without real implementations."""

    with pytest.raises(
        RealExperimentError,
        match="unsupported real policy: ml_based",
    ):
        build_repetition_output_directory(
            config=experiment_config,
            policy_name="ml_based",
            repetition=1,
        )


def test_execute_policy_repetitions_runs_every_configured_repetition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    experiment_config: ExperimentConfig,
) -> None:
    """Execute one isolated policy run for every configured repetition."""

    output_config = replace(
        experiment_config.output,
        results_directory=tmp_path / "results",
    )
    config = replace(
        experiment_config,
        output=output_config,
    )
    produced_results = [object(), object(), object()]
    received_calls: list[dict[str, object]] = []

    def fake_execute_policy_experiment(
        **kwargs: object,
    ) -> object:
        received_calls.append(kwargs)
        return produced_results[len(received_calls) - 1]

    monkeypatch.setattr(
        experiment_module,
        "execute_policy_experiment",
        fake_execute_policy_experiment,
    )

    results = execute_policy_repetitions(
        policy_name="always_on",
        config=config,
    )

    policy_directory = (
        tmp_path
        / "results"
        / "traffic-aware-serverless-llm"
        / "always_on"
    )

    assert results == tuple(produced_results)
    assert received_calls == [
        {
            "policy_name": "always_on",
            "config": config,
            "output_directory": (
                policy_directory / "repetition-001"
            ),
        },
        {
            "policy_name": "always_on",
            "config": config,
            "output_directory": (
                policy_directory / "repetition-002"
            ),
        },
        {
            "policy_name": "always_on",
            "config": config,
            "output_directory": (
                policy_directory / "repetition-003"
            ),
        },
    ]


def test_execute_policy_repetitions_rejects_existing_directory_before_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    experiment_config: ExperimentConfig,
) -> None:
    """Protect existing results before starting any expensive GPU run."""

    output_config = replace(
        experiment_config.output,
        results_directory=tmp_path / "results",
    )
    config = replace(
        experiment_config,
        output=output_config,
    )
    existing_directory = build_repetition_output_directory(
        config=config,
        policy_name="always_on",
        repetition=1,
    )
    existing_directory.mkdir(parents=True)
    execution_count = 0

    def fake_execute_policy_experiment(
        **kwargs: object,
    ) -> object:
        nonlocal execution_count
        execution_count += 1
        return object()

    monkeypatch.setattr(
        experiment_module,
        "execute_policy_experiment",
        fake_execute_policy_experiment,
    )

    with pytest.raises(
        RealExperimentError,
        match="experiment result directories already exist",
    ):
        execute_policy_repetitions(
            policy_name="always_on",
            config=config,
        )

    assert execution_count == 0
