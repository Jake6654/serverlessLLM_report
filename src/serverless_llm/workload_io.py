"""CSV serialization for workload traces."""

import csv
from pathlib import Path

from serverless_llm.workload import RequestEvent, WorkloadTrace


WORKLOAD_CSV_FIELDS = (
    "trace_name",
    "pattern",
    "random_seed",
    "request_id",
    "scheduled_at_seconds",
    "prompt_id",
)


class WorkloadFileError(ValueError):
    """Raised when a workload trace cannot be saved or loaded."""


def save_workload_trace(
    trace: WorkloadTrace,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a workload trace as a self-contained CSV file.

    Args:
        trace: Valid workload trace to serialize.
        path: Destination CSV path.
        overwrite: Whether an existing file may be replaced.

    Returns:
        The destination path as a Path object.

    Raises:
        WorkloadFileError: If the path is invalid or cannot be written.
    """

    csv_path = Path(path)

    # 확장자 검사
    if csv_path.suffix.lower() != ".csv":
        raise WorkloadFileError(
            f"workload path must use the .csv extension: {csv_path}"
        )

    if csv_path.exists() and not overwrite:
        raise WorkloadFileError(
            f"workload file already exists: {csv_path}"
        )

    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open(
            mode="w", # 쓰기 모드
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=WORKLOAD_CSV_FIELDS,
            )

            # Dictionary key matches each CSV column
            writer.writeheader()

            for event in trace.events:
                writer.writerow(
                    {
                        "trace_name": trace.name,
                        "pattern": trace.pattern,
                        "random_seed": trace.random_seed,
                        "request_id": event.request_id,
                        "scheduled_at_seconds": (
                            event.scheduled_at_seconds
                        ),
                        "prompt_id": event.prompt_id,
                    }
                )

    except (OSError, csv.Error) as error:
        raise WorkloadFileError(
            f"could not write workload file {csv_path}: {error}"
        ) from error

    return csv_path


# CSV를 읽고 doamin object 로 복원한다
def load_workload_trace(path: str | Path) -> WorkloadTrace:
    """Load a workload trace from a self-contained CSV file.

    Args:
        path: Source workload CSV path.

    Returns:
        A validated immutable WorkloadTrace.

    Raises:
        WorkloadFileError: If the file is missing, malformed, or inconsistent.
    """

    csv_path = Path(path)

    try:
        with csv_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise WorkloadFileError(
                    f"workload file has no header: {csv_path}"
                )

            actual_fields = set(reader.fieldnames)
            required_fields = set(WORKLOAD_CSV_FIELDS)

            missing_fields = required_fields - actual_fields
            unexpected_fields = actual_fields - required_fields

            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise WorkloadFileError(
                    f"workload file is missing fields: {missing}"
                )

            if unexpected_fields:
                unexpected = ", ".join(sorted(unexpected_fields))
                raise WorkloadFileError(
                    f"workload file has unexpected fields: {unexpected}"
                )

            rows = list(reader)

    except WorkloadFileError:
        raise
    except (OSError, csv.Error) as error:
        raise WorkloadFileError(
            f"could not read workload file {csv_path}: {error}"
        ) from error

    if not rows:
        raise WorkloadFileError(
            f"workload file contains no request events: {csv_path}"
        )

    first_row = rows[0]
    expected_metadata = (
        first_row["trace_name"],
        first_row["pattern"],
        first_row["random_seed"],
    )

    for line_number, row in enumerate(rows, start=2):
        row_metadata = (
            row["trace_name"],
            row["pattern"],
            row["random_seed"],
        )

        if row_metadata != expected_metadata:
            raise WorkloadFileError(
                "inconsistent trace metadata "
                f"at CSV line {line_number}"
            )

    try:
        events = tuple(
            RequestEvent(
                request_id=int(row["request_id"]),
                scheduled_at_seconds=float(
                    row["scheduled_at_seconds"]
                ),
                prompt_id=row["prompt_id"],
            )
            for row in rows
        )

        return WorkloadTrace(
            name=first_row["trace_name"],
            pattern=first_row["pattern"],
            random_seed=int(first_row["random_seed"]),
            events=events,
        )

    except (TypeError, ValueError) as error:
        raise WorkloadFileError(
            f"invalid workload data in {csv_path}: {error}"
        ) from error