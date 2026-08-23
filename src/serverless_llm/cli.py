"""Command-line interface for experiment configuration checks."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from serverless_llm.config import (
    ConfigLoadError,
    ConfigValidationError,
    ExperimentConfig,
    load_config,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="serverless-llm-check",
        description=(
            "Validate and display an experiment configuration."
        ),
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to an experiment YAML configuration file.",
    )
    return parser


def format_config_summary(
    config: ExperimentConfig,
    config_path: Path,
) -> str:
    """Return a human-readable configuration summary."""

    lines = [
        "Configuration valid",
        "",
        f"Path: {config_path}",
        f"Experiment: {config.experiment.name}",
        f"Model: {config.model.name}",
        f"Model revision: {config.model.revision}",
        f"Workload: {config.workload.type}",
        f"Requests: {config.workload.total_requests}",
        f"Concurrency: {config.workload.concurrency}",
        f"Repetitions: {config.experiment.repetitions}",
        f"Results directory: {config.output.results_directory}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configuration-check command."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (ConfigLoadError, ConfigValidationError) as error:
        print(f"Configuration error:\n{error}", file=sys.stderr)
        return 1

    print(format_config_summary(config, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())