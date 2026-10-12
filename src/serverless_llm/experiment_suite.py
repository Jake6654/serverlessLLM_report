"""Run a balanced, reproducible suite of real policy experiments."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from serverless_llm.config import (
    ConfigLoadError,
    ExperimentConfig,
    load_config,
    validate_config,
)
from serverless_llm.experiment_metadata import (
    build_run_manifest,
    collect_environment_metadata,
    write_run_metadata,
)
from serverless_llm.real_experiment import (
    SUPPORTED_REAL_POLICIES,
    RealExperimentError,
    RealExperimentResult,
    execute_policy_experiment,
)
from serverless_llm.result_writer import write_json


@dataclass(frozen=True)
class SuiteWorkload:
    """Describe one steady workload in the report matrix."""

    name: str
    request_interval_seconds: float
    total_requests: int


@dataclass(frozen=True)
class ExperimentSuiteConfig:
    """Store settings shared by a complete comparison suite."""

    name: str
    base_config_path: Path
    repetitions: int
    random_seed: int
    policies: tuple[str, ...]
    workloads: tuple[SuiteWorkload, ...]


@dataclass(frozen=True)
class SuiteRunSpec:
    """Describe one policy/workload/repetition execution."""

    workload_name: str
    policy_name: str
    repetition: int
    execution_order: int
    output_directory: Path
    config: ExperimentConfig


@dataclass(frozen=True)
class SuiteRunResult:
    """Pair one completed run specification with its measurements."""

    spec: SuiteRunSpec
    experiment_result: RealExperimentResult


def load_suite_config(path: str | Path) -> ExperimentSuiteConfig:
    """Load and validate a report-suite YAML file."""

    suite_path = Path(path)
    try:
        raw = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigLoadError(
            f"Could not read suite configuration {suite_path}: {error}"
        ) from error
    except yaml.YAMLError as error:
        raise ConfigLoadError(
            f"Invalid YAML in {suite_path}: {error}"
        ) from error

    try:
        suite = raw["suite"]
        base_path = Path(raw["base_config"])
        if not base_path.is_absolute():
            base_path = suite_path.parent / base_path

        workloads = tuple(
            SuiteWorkload(
                name=item["name"],
                request_interval_seconds=item[
                    "request_interval_seconds"
                ],
                total_requests=item["total_requests"],
            )
            for item in raw["workloads"]
        )
        result = ExperimentSuiteConfig(
            name=suite["name"],
            base_config_path=base_path,
            repetitions=suite["repetitions"],
            random_seed=suite["random_seed"],
            policies=tuple(suite["policies"]),
            workloads=workloads,
        )
    except (KeyError, TypeError) as error:
        raise ConfigLoadError(
            f"Invalid suite configuration structure in {suite_path}: {error}"
        ) from error

    errors: list[str] = []
    if not isinstance(result.name, str) or not result.name.strip():
        errors.append("suite.name must be a non-empty string")
    if type(result.repetitions) is not int or result.repetitions < 1:
        errors.append("suite.repetitions must be a positive integer")
    if type(result.random_seed) is not int:
        errors.append("suite.random_seed must be an integer")
    if not result.policies:
        errors.append("suite.policies must not be empty")
    if len(result.policies) != len(set(result.policies)):
        errors.append("suite.policies must not contain duplicates")
    unsupported = set(result.policies) - SUPPORTED_REAL_POLICIES
    if unsupported:
        errors.append(
            "unsupported suite policies: " + ", ".join(sorted(unsupported))
        )
    if not result.workloads:
        errors.append("workloads must not be empty")
    workload_names = [workload.name for workload in result.workloads]
    if len(workload_names) != len(set(workload_names)):
        errors.append("workload names must not contain duplicates")
    for workload in result.workloads:
        if not isinstance(workload.name, str) or not workload.name.strip():
            errors.append("every workload name must be non-empty")
        if (
            type(workload.request_interval_seconds) not in (int, float)
            or workload.request_interval_seconds <= 0
        ):
            errors.append(
                f"workload {workload.name!r} interval must be positive"
            )
        if type(workload.total_requests) is not int or workload.total_requests < 1:
            errors.append(
                f"workload {workload.name!r} total_requests must be positive"
            )

    if errors:
        raise ConfigLoadError(
            "Invalid suite configuration:\n"
            + "\n".join(f"- {error}" for error in errors)
        )

    # Load now so a bad referenced base configuration fails before AWS work.
    load_config(result.base_config_path)
    return result


def _rotated_policies(
    policies: tuple[str, ...],
    offset: int,
) -> tuple[str, ...]:
    """Rotate policy order to reduce first/last-run thermal bias."""

    rotation = offset % len(policies)
    return policies[rotation:] + policies[:rotation]


def build_suite_schedule(
    suite: ExperimentSuiteConfig,
) -> tuple[SuiteRunSpec, ...]:
    """Build every run with deterministic balanced policy ordering."""

    if not isinstance(suite, ExperimentSuiteConfig):
        raise ValueError("suite must be an ExperimentSuiteConfig object")

    base = load_config(suite.base_config_path)
    root = base.output.results_directory / suite.name
    specs: list[SuiteRunSpec] = []
    execution_order = 1

    for workload_index, workload in enumerate(suite.workloads):
        for repetition in range(1, suite.repetitions + 1):
            policies = _rotated_policies(
                suite.policies,
                workload_index + repetition - 1,
            )
            effective_config = replace(
                base,
                experiment=replace(
                    base.experiment,
                    name=suite.name,
                    repetitions=suite.repetitions,
                    random_seed=suite.random_seed + repetition - 1,
                ),
                workload=replace(
                    base.workload,
                    type="steady",
                    request_interval_seconds=(
                        workload.request_interval_seconds
                    ),
                    total_requests=workload.total_requests,
                    concurrency=1,
                ),
            )
            validate_config(effective_config)

            for policy_name in policies:
                specs.append(
                    SuiteRunSpec(
                        workload_name=workload.name,
                        policy_name=policy_name,
                        repetition=repetition,
                        execution_order=execution_order,
                        output_directory=(
                            root
                            / workload.name
                            / policy_name
                            / f"repetition-{repetition:03d}"
                        ),
                        config=effective_config,
                    )
                )
                execution_order += 1

    return tuple(specs)


def schedule_as_dicts(
    specs: tuple[SuiteRunSpec, ...],
) -> list[dict[str, object]]:
    """Convert a schedule into JSON-safe rows for inspection."""

    return [
        {
            "execution_order": spec.execution_order,
            "workload_name": spec.workload_name,
            "policy_name": spec.policy_name,
            "repetition": spec.repetition,
            "random_seed": spec.config.experiment.random_seed,
            "request_interval_seconds": (
                spec.config.workload.request_interval_seconds
            ),
            "total_requests": spec.config.workload.total_requests,
            "output_directory": str(spec.output_directory),
        }
        for spec in specs
    ]


def execute_experiment_suite(
    suite: ExperimentSuiteConfig,
) -> tuple[SuiteRunResult, ...]:
    """Execute all scheduled runs and save provenance beside every result."""

    specs = build_suite_schedule(suite)
    existing = [spec.output_directory for spec in specs if spec.output_directory.exists()]
    if existing:
        raise RealExperimentError(
            "suite result directories already exist: "
            + ", ".join(str(path) for path in existing)
        )

    suite_root = specs[0].output_directory.parents[2]
    schedule_path = suite_root / "schedule.json"
    if schedule_path.exists():
        raise RealExperimentError(
            f"suite schedule already exists: {schedule_path}"
        )

    write_json(
        schedule_path,
        {
            "schema_version": 1,
            "suite_name": suite.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repetitions": suite.repetitions,
            "runs": schedule_as_dicts(specs),
        },
    )

    environment = collect_environment_metadata(specs[0].config)
    completed: list[SuiteRunResult] = []

    for spec in specs:
        try:
            experiment_result = execute_policy_experiment(
                policy_name=spec.policy_name,
                config=spec.config,
                output_directory=spec.output_directory,
            )
            manifest = build_run_manifest(
                suite_name=suite.name,
                workload_name=spec.workload_name,
                policy_name=spec.policy_name,
                repetition=spec.repetition,
                execution_order=spec.execution_order,
                config=spec.config,
                environment=environment,
            )
            write_run_metadata(
                spec.output_directory,
                manifest,
                spec.config,
            )
            completed.append(
                SuiteRunResult(
                    spec=spec,
                    experiment_result=experiment_result,
                )
            )
        except Exception as error:
            failure_path = suite_root / "failure.json"
            if not failure_path.exists():
                write_json(
                    failure_path,
                    {
                        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "execution_order": spec.execution_order,
                        "workload_name": spec.workload_name,
                        "policy_name": spec.policy_name,
                        "repetition": spec.repetition,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
            raise

    return tuple(completed)
