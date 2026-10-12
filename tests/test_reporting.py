"""Tests for report-ready aggregation artifacts."""

import json
from pathlib import Path

from serverless_llm.reporting import (
    aggregate_run_rows,
    collect_run_rows,
    generate_report,
)


def _write_fake_run(
    root: Path,
    *,
    workload: str,
    policy: str,
    repetition: int,
    ttft: float,
    cold_rate: float,
    energy: float,
) -> None:
    run = root / workload / policy / f"repetition-{repetition:03d}"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "workload_name": workload,
                "policy_name": policy,
                "repetition": repetition,
                "execution_order": repetition,
            }
        ),
        encoding="utf-8",
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "policy_name": policy,
                "experiment_duration_seconds": 100.0,
                "request_summary": {
                    "total_requests": 10,
                    "mean_end_to_end_ttft_seconds": ttft - 0.1,
                    "p95_end_to_end_ttft_seconds": ttft,
                    "p95_end_to_end_latency_seconds": ttft + 0.5,
                    "ttft_slo_violation_rate": 0.2,
                },
                "startup_summary": {
                    "cold_start_rate": cold_rate,
                    "total_startup_time_seconds": cold_rate * 100,
                },
                "gpu_summary": {
                    "estimated_energy_watt_hours": energy,
                },
            }
        ),
        encoding="utf-8",
    )


def test_collect_and_aggregate_repetitions(tmp_path) -> None:
    _write_fake_run(
        tmp_path,
        workload="dense",
        policy="always_on",
        repetition=1,
        ttft=1.0,
        cold_rate=0.1,
        energy=2.0,
    )
    _write_fake_run(
        tmp_path,
        workload="dense",
        policy="always_on",
        repetition=2,
        ttft=3.0,
        cold_rate=0.1,
        energy=4.0,
    )

    rows = collect_run_rows(tmp_path)
    aggregated = aggregate_run_rows(rows)

    assert len(rows) == 2
    assert aggregated[0]["repetitions"] == 2
    assert aggregated[0]["p95_end_to_end_ttft_seconds_mean"] == 2.0
    assert aggregated[0]["p95_end_to_end_ttft_seconds_ci95"] is not None
    assert rows[0]["energy_per_request_watt_hours"] == 0.2


def test_generate_report_writes_tables_and_figures(tmp_path) -> None:
    _write_fake_run(
        tmp_path,
        workload="dense",
        policy="naive_serverless",
        repetition=1,
        ttft=50.0,
        cold_rate=1.0,
        energy=3.0,
    )

    paths = generate_report(tmp_path)

    assert paths.runs_csv.exists()
    assert paths.comparison_csv.exists()
    assert "naive_serverless" in paths.report_table_markdown.read_text()
    assert len(paths.figure_paths) == 4
    assert all(path.read_text().startswith("<svg") for path in paths.figure_paths)
