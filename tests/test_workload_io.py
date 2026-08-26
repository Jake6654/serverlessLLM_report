"""Tests for workload CSV serialization."""

from pathlib import Path

import pytest

from serverless_llm.workload import (
    WorkloadTrace,
    generate_steady_workload,
)
from serverless_llm.workload_io import (
    WORKLOAD_CSV_FIELDS,
    WorkloadFileError,
    load_workload_trace,
    save_workload_trace,
)


@pytest.fixture
def steady_trace() -> WorkloadTrace:
    """Create a small deterministic trace for file tests."""

    return generate_steady_workload(
        total_requests=4,
        interval_seconds=5.0,
        random_seed=42,
    )


def test_workload_csv_round_trip(
    tmp_path: Path,
    steady_trace: WorkloadTrace,
) -> None:
    path = tmp_path / "steady.csv"

    saved_path = save_workload_trace(steady_trace, path)
    loaded_trace = load_workload_trace(saved_path)

    assert saved_path == path
    assert loaded_trace == steady_trace


def test_save_creates_parent_directories(
    tmp_path: Path,
    steady_trace: WorkloadTrace,
) -> None:
    path = tmp_path / "nested" / "workloads" / "steady.csv"

    save_workload_trace(steady_trace, path)

    assert path.exists()


def test_save_writes_expected_header(
    tmp_path: Path,
    steady_trace: WorkloadTrace,
) -> None:
    path = tmp_path / "steady.csv"

    save_workload_trace(steady_trace, path)

    first_line = path.read_text(encoding="utf-8").splitlines()[0]

    assert first_line == ",".join(WORKLOAD_CSV_FIELDS)


def test_save_refuses_to_overwrite_existing_file(
    tmp_path: Path,
    steady_trace: WorkloadTrace,
) -> None:
    path = tmp_path / "steady.csv"

    save_workload_trace(steady_trace, path)

    with pytest.raises(WorkloadFileError, match="already exists"):
        save_workload_trace(steady_trace, path)


def test_save_can_overwrite_when_explicitly_enabled(
    tmp_path: Path,
    steady_trace: WorkloadTrace,
) -> None:
    path = tmp_path / "steady.csv"

    save_workload_trace(steady_trace, path)
    save_workload_trace(steady_trace, path, overwrite=True)

    assert load_workload_trace(path) == steady_trace


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"

    with pytest.raises(WorkloadFileError, match="could not read"):
        load_workload_trace(path)


def test_load_rejects_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "missing-column.csv"
    path.write_text(
        "trace_name,pattern,random_seed,request_id,prompt_id\n"
        "steady,steady,42,1,default\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkloadFileError, match="missing fields"):
        load_workload_trace(path)


def test_load_rejects_empty_trace_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text(
        ",".join(WORKLOAD_CSV_FIELDS) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkloadFileError, match="no request events"):
        load_workload_trace(path)


def test_load_rejects_inconsistent_metadata(tmp_path: Path) -> None:
    path = tmp_path / "inconsistent.csv"
    path.write_text(
        ",".join(WORKLOAD_CSV_FIELDS)
        + "\n"
        + "steady-example,steady,42,1,0.0,default\n"
        + "steady-example,sparse,42,2,5.0,default\n",
        encoding="utf-8",
    )

    with pytest.raises(
        WorkloadFileError,
        match="inconsistent trace metadata",
    ):
        load_workload_trace(path)


def test_load_rejects_invalid_numeric_data(tmp_path: Path) -> None:
    path = tmp_path / "invalid-number.csv"
    path.write_text(
        ",".join(WORKLOAD_CSV_FIELDS)
        + "\n"
        + "steady-example,steady,42,not-an-id,0.0,default\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkloadFileError, match="invalid workload data"):
        load_workload_trace(path)
