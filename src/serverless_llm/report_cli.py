"""Command-line interface for report artifact generation."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from serverless_llm.reporting import generate_report
from serverless_llm.result_writer import ResultWriterError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serverless-llm-report",
        description="Aggregate a completed experiment suite.",
    )
    parser.add_argument(
        "suite_directory",
        type=Path,
        help="Result directory containing workload/policy repetitions.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = generate_report(args.suite_directory)
    except (ResultWriterError, ValueError, KeyError) as error:
        print(f"Report generation failed:\n{error}", file=sys.stderr)
        return 1

    print(f"Run data: {paths.runs_csv}")
    print(f"Comparison table: {paths.comparison_csv}")
    print(f"Markdown table: {paths.report_table_markdown}")
    print(f"Figures: {len(paths.figure_paths)} SVG files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
