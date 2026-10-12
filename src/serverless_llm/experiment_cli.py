"""Command-line interface for real serverless LLM experiments"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from serverless_llm.config import (
    ConfigLoadError,
    ConfigValidationError,
    load_config,
)
from serverless_llm.docker_runtime import DockerRuntimeError
from serverless_llm.gpu_monitor import GPUMonitorError
from serverless_llm.real_experiment import (
    SUPPORTED_REAL_POLICIES,
    RealExperimentError,
    RealExperimentResult,
    execute_policy_repetitions,
)
from serverless_llm.result_writer import ResultWriterError
from serverless_llm.vllm_client import VLLMClientError

def build_parser() -> argparse.ArgumentParser:
    """Create the real experiment command-line parser."""

    parser = argparse.ArgumentParser(
        prog="serverless-llm-run",
        description=(
            "Run repeated real serverless LLM policy experiments."
        ),
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to the experiment YAML configuration file.",
    )

    parser.add_argument(
        "--policy",
        required=True,
        choices=sorted(SUPPORTED_REAL_POLICIES),
        help="Real server lifecycle policy to execute.",
    )

    return parser


def format_result_summary(
    repetition: int,
    result: RealExperimentResult,
) -> str:
    """Return a human-readable summary for one completed repetition."""

    summary = result.summary
    requests = summary.request_summary
    startups = summary.startup_summary
    gpu = summary.gpu_summary

    lines = [
        f"Repetition {repetition} completed",
        f"  Policy: {summary.policy_name}",
        f"  Requests: {requests.total_requests}",
        (
            "  Cold starts: "
            f"{startups.cold_start_count}/"
            f"{startups.total_requests} "
            f"({startups.cold_start_rate:.1%})"
        ),
        (
            "  Mean end-to-end TTFT: "
            f"{requests.mean_end_to_end_ttft_seconds:.3f} seconds"
        ),
        (
            "  P95 end-to-end TTFT: "
            f"{requests.p95_end_to_end_ttft_seconds:.3f} seconds"
        ),
        (
            "  TTFT SLO violations: "
            f"{requests.ttft_slo_violation_count}/"
            f"{requests.total_requests} "
            f"({requests.ttft_slo_violation_rate:.1%})"
        ),
        (
            "  Estimated GPU energy: "
            f"{gpu.estimated_energy_watt_hours:.3f} Wh"
        ),
        (
            "  Result directory: "
            f"{result.paths.output_directory}"
        ),
    ]

    return "\n".join(lines)


def main(
        argv: Sequence[str] | None = None,
) -> int:
    """Load configuration and execute one policy's repetitions."""

    parser = build_parser()
    args = parser.parse_args(argv)


    try:
        config = load_config(args.config)
    except (ConfigLoadError, ConfigValidationError) as error:
        print(
            f"Configuration error:\n{error}",
            file=sys.stderr,
        )

        return 1

    print(
        f"Starting {args.policy} experiment "
        f"with {config.experiment.repetitions} repetition(s)...",
        flush=True,
    )

    try:
        results = execute_policy_repetitions(
            policy_name=args.policy,
            config=config,
        )
    except (
        DockerRuntimeError,
        GPUMonitorError,
        RealExperimentError,
        ResultWriterError,
        VLLMClientError,
        ValueError,
    ) as error:
        print(
            f"Experiment failed:\n{error}",
            file=sys.stderr,
        )
        return 1

    print()

    for repetition, result in enumerate(
        results,
        start=1,
    ):
        print(
            format_result_summary(
                repetition,
                result,
            )
        )
        print()

    print(
        f"Completed {len(results)} "
        f"{args.policy} repetition(s)."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())