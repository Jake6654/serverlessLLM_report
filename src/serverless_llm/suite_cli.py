"""Command-line interface for report-ready experiment suites."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from serverless_llm.config import ConfigLoadError, ConfigValidationError
from serverless_llm.docker_runtime import DockerRuntimeError
from serverless_llm.experiment_suite import (
    build_suite_schedule,
    execute_experiment_suite,
    load_suite_config,
)
from serverless_llm.gpu_monitor import GPUMonitorError
from serverless_llm.real_experiment import RealExperimentError
from serverless_llm.result_writer import ResultWriterError
from serverless_llm.vllm_client import VLLMClientError


def build_parser() -> argparse.ArgumentParser:
    """Create the suite command-line parser."""

    parser = argparse.ArgumentParser(
        prog="serverless-llm-suite",
        description="Run a balanced multi-workload policy comparison.",
    )
    parser.add_argument("config", type=Path, help="Suite YAML file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the schedule without starting Docker or using the GPU.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate, preview, or execute a complete experiment suite."""

    args = build_parser().parse_args(argv)
    try:
        suite = load_suite_config(args.config)
        schedule = build_suite_schedule(suite)
    except (ConfigLoadError, ConfigValidationError, ValueError) as error:
        print(f"Configuration error:\n{error}", file=sys.stderr)
        return 1

    print(
        f"Suite {suite.name!r}: {len(schedule)} runs "
        f"({len(suite.workloads)} workloads × {len(suite.policies)} policies "
        f"× {suite.repetitions} repetitions)"
    )
    for spec in schedule:
        print(
            f"{spec.execution_order:03d}  "
            f"{spec.workload_name:<18}  "
            f"rep={spec.repetition:02d}  "
            f"policy={spec.policy_name}"
        )

    if args.dry_run:
        print("Dry run complete; no containers were started.")
        return 0

    try:
        results = execute_experiment_suite(suite)
    except (
        DockerRuntimeError,
        GPUMonitorError,
        RealExperimentError,
        ResultWriterError,
        VLLMClientError,
        ValueError,
    ) as error:
        print(f"Suite failed:\n{error}", file=sys.stderr)
        return 1

    print(f"Completed all {len(results)} runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
