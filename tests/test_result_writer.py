"""Unit tests for real experiment result artifact writing."""

import csv
import json
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest

from serverless_llm.docker_runtime import RuntimeStartupResult
from serverless_llm.gpu_monitor import GPUSample
from serverless_llm.real_metrics import summarize_policy_run
from serverless_llm.real_policy_runner import (
    RealPolicyRunResult,
    RealRequestResult,
)
from serverless_llm.result_writer import (
    GPU_SAMPLE_FIELDNAMES,
    REQUEST_FIELDNAMES,
    PolicyResultPaths,
    ResultWriterError,
    write_csv,
    write_json,
    write_policy_results,
)
from serverless_llm.vllm_client import CompletionResult
from serverless_llm.workload import RequestEvent


def make_run_result() -> RealPolicyRunResult:
    """Build a deterministic policy result without Docker or a GPU."""

    request_result = RealRequestResult(
        event=RequestEvent(
            request_id=1,
            scheduled_at_seconds=0.0,
            prompt_id="기본-프롬프트",
        ),
        scheduled_for_seconds=110.0,
        handling_started_at_seconds=110.0,
        completion=CompletionResult(
            request_started_at_seconds=110.0,
            first_token_at_seconds=110.5,
            completed_at_seconds=111.0,
            generated_text="generated text",
            finish_reason="length",
        ),
        cold_start=False,
        startup_duration_seconds=None,
    )

    return RealPolicyRunResult(
        policy_name="always_on",
        experiment_started_at_seconds=100.0,
        workload_started_at_seconds=110.0,
        experiment_completed_at_seconds=112.0,
        startup_results=(RuntimeStartupResult(100.0, 105.0),),
        request_results=(request_result,),
        gpu_samples=(
            GPUSample(100.0, 0, 0.0, 0.0, 25.0, 40.0),
            GPUSample(111.0, 0, 80.0, 2500.0, 55.0, 50.0),
        ),
    )


def test_write_json_creates_parent_directories_and_utf8_file(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "nested" / "summary.json"
    data = {
        "policy_name": "always_on",
        "description": "기본 실험",
        "total_requests": 1,
    }

    result = write_json(output_path, data)

    assert result is None
    assert json.loads(output_path.read_text(encoding="utf-8")) == data
    assert "기본 실험" in output_path.read_text(encoding="utf-8")
    assert output_path.read_text(encoding="utf-8").endswith("\n")


def test_write_json_rejects_existing_file(tmp_path: Path) -> None:
    output_path = tmp_path / "summary.json"
    write_json(output_path, {"first": True})

    with pytest.raises(ResultWriterError, match="already exists"):
        write_json(output_path, {"second": True})

    assert json.loads(output_path.read_text()) == {"first": True}


@pytest.mark.parametrize(
    ("output_path", "data", "message"),
    [
        ("summary.json", {}, "Path object"),
        (Path("summary.csv"), {}, ".json extension"),
        (Path("summary.json"), [], "dictionary"),
    ],
)
def test_write_json_rejects_invalid_arguments(
    output_path: object,
    data: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        write_json(
            output_path,  # type: ignore[arg-type]
            data,  # type: ignore[arg-type]
        )


def test_write_json_rejects_unserializable_data(tmp_path: Path) -> None:
    output_path = tmp_path / "summary.json"

    with pytest.raises(ResultWriterError, match="serialize JSON"):
        write_json(output_path, {"invalid": object()})

    assert not output_path.exists()


def test_write_csv_creates_expected_header_and_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "requests.csv"
    fieldnames = ("request_id", "cold_start", "prompt_id")
    rows = [
        {
            "request_id": 1,
            "cold_start": True,
            "prompt_id": "기본-프롬프트",
        },
        {
            "request_id": 2,
            "cold_start": False,
            "prompt_id": "default",
        },
    ]

    result = write_csv(output_path, fieldnames, rows)

    assert result is None
    with output_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        assert tuple(reader.fieldnames or ()) == fieldnames
        assert list(reader) == [
            {
                "request_id": "1",
                "cold_start": "True",
                "prompt_id": "기본-프롬프트",
            },
            {
                "request_id": "2",
                "cold_start": "False",
                "prompt_id": "default",
            },
        ]


def test_write_csv_writes_header_for_empty_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "gpu_samples.csv"

    write_csv(output_path, ("sampled_at_seconds", "gpu_index"), [])

    assert output_path.read_text(encoding="utf-8") == (
        "sampled_at_seconds,gpu_index\n"
    )


def test_write_csv_rejects_existing_file(tmp_path: Path) -> None:
    output_path = tmp_path / "requests.csv"
    write_csv(output_path, ("request_id",), [{"request_id": 1}])

    with pytest.raises(ResultWriterError, match="already exists"):
        write_csv(output_path, ("request_id",), [{"request_id": 2}])


@pytest.mark.parametrize(
    "fieldnames",
    [(), [], ("",), ("request_id", "request_id"), ("request_id", 1)],
)
def test_write_csv_rejects_invalid_fieldnames(
    tmp_path: Path,
    fieldnames: object,
) -> None:
    with pytest.raises(ValueError, match="fieldnames"):
        write_csv(
            tmp_path / "requests.csv",
            fieldnames,  # type: ignore[arg-type]
            [],
        )


@pytest.mark.parametrize(
    "rows",
    [(), {}, ["not-a-row"]],
)
def test_write_csv_rejects_invalid_rows(
    tmp_path: Path,
    rows: object,
) -> None:
    with pytest.raises(ValueError, match="rows"):
        write_csv(
            tmp_path / "requests.csv",
            ("request_id",),
            rows,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "row",
    [
        {"request_id": 1},
        {"request_id": 1, "cold_start": False, "extra": "value"},
    ],
)
def test_write_csv_rejects_inexact_row_schema(
    tmp_path: Path,
    row: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="exactly"):
        write_csv(
            tmp_path / "requests.csv",
            ("request_id", "cold_start"),
            [row],
        )


def test_write_policy_results_creates_all_artifacts(tmp_path: Path) -> None:
    run_result = make_run_result()
    summary = summarize_policy_run(
        run_result,
        ttft_slo_seconds=1.0,
    )
    output_directory = (
        tmp_path / "run-001" / "always_on" / "repetition-001"
    )

    paths = write_policy_results(
        output_directory,
        run_result,
        summary,
    )

    assert paths == PolicyResultPaths(
        output_directory=output_directory,
        requests_csv=output_directory / "requests.csv",
        gpu_samples_csv=output_directory / "gpu_samples.csv",
        timeline_json=output_directory / "timeline.json",
        summary_json=output_directory / "summary.json",
    )
    assert all(
        path.exists()
        for path in (
            paths.requests_csv,
            paths.gpu_samples_csv,
            paths.timeline_json,
            paths.summary_json,
        )
    )

    with paths.requests_csv.open(
        encoding="utf-8",
        newline="",
    ) as requests_file:
        request_reader = csv.DictReader(requests_file)
        request_rows = list(request_reader)

    assert tuple(request_reader.fieldnames or ()) == REQUEST_FIELDNAMES
    assert request_rows[0]["request_id"] == "1"
    assert request_rows[0]["prompt_id"] == "기본-프롬프트"
    assert request_rows[0]["cold_start"] == "False"

    with paths.gpu_samples_csv.open(
        encoding="utf-8",
        newline="",
    ) as gpu_file:
        gpu_reader = csv.DictReader(gpu_file)
        gpu_rows = list(gpu_reader)

    assert tuple(gpu_reader.fieldnames or ()) == GPU_SAMPLE_FIELDNAMES
    assert [row["sampled_at_seconds"] for row in gpu_rows] == [
        "0.0",
        "11.0",
    ]

    timeline_data = json.loads(paths.timeline_json.read_text())
    summary_data = json.loads(paths.summary_json.read_text())

    assert timeline_data["policy_name"] == "always_on"
    assert timeline_data["experiment_completed_at_seconds"] == 12.0
    assert summary_data == asdict(summary)


def test_write_policy_results_rejects_existing_artifact(
    tmp_path: Path,
) -> None:
    run_result = make_run_result()
    summary = summarize_policy_run(run_result, ttft_slo_seconds=1.0)
    output_directory = tmp_path / "repetition-001"
    output_directory.mkdir()
    existing_summary = output_directory / "summary.json"
    existing_summary.write_text("existing", encoding="utf-8")

    with pytest.raises(ResultWriterError, match="already exist"):
        write_policy_results(output_directory, run_result, summary)

    assert existing_summary.read_text(encoding="utf-8") == "existing"
    assert not (output_directory / "requests.csv").exists()


def test_write_policy_results_rejects_policy_mismatch(
    tmp_path: Path,
) -> None:
    run_result = make_run_result()
    summary = summarize_policy_run(run_result, ttft_slo_seconds=1.0)
    mismatched_summary = replace(
        summary,
        policy_name="naive_serverless",
    )

    with pytest.raises(ValueError, match="same policy"):
        write_policy_results(
            tmp_path / "repetition-001",
            run_result,
            mismatched_summary,
        )


def test_write_policy_results_rejects_file_as_output_directory(
    tmp_path: Path,
) -> None:
    run_result = make_run_result()
    summary = summarize_policy_run(run_result, ttft_slo_seconds=1.0)
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("file", encoding="utf-8")

    with pytest.raises(ValueError, match="directory path"):
        write_policy_results(output_file, run_result, summary)


def test_policy_result_paths_is_immutable(tmp_path: Path) -> None:
    run_result = make_run_result()
    summary = summarize_policy_run(run_result, ttft_slo_seconds=1.0)
    paths = write_policy_results(
        tmp_path / "repetition-001",
        run_result,
        summary,
    )

    with pytest.raises(FrozenInstanceError):
        paths.summary_json = tmp_path / "other.json"  # type: ignore[misc]
