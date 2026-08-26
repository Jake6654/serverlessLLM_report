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

    # Normalize both string paths and Path objects into one Path type.
    csv_path = Path(path)

    # Use one predictable file format for every generated workload.
    if csv_path.suffix.lower() != ".csv":
        raise WorkloadFileError(
            f"workload path must use the .csv extension: {csv_path}"
        )

    # Protect previous experiment inputs unless replacement is explicit.
    if csv_path.exists() and not overwrite:
        raise WorkloadFileError(
            f"workload file already exists: {csv_path}"
        )

    try:
        # Create nested output folders so the caller does not need to.
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open(
            mode="w",
            encoding="utf-8",
            newline="",
        ) as file:
            # DictWriter keeps each value matched to its named CSV column.
            writer = csv.DictWriter(
                file,
                fieldnames=WORKLOAD_CSV_FIELDS,
            )
            writer.writeheader()

            # Repeat trace metadata in every row to keep the CSV self-contained.
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

    # Convert low-level file errors into one project-specific error type.
    except (OSError, csv.Error) as error:
        raise WorkloadFileError(
            f"could not write workload file {csv_path}: {error}"
        ) from error

    return csv_path


def load_workload_trace(path: str | Path) -> WorkloadTrace:
    """Load a workload trace from a self-contained CSV file.

    Args:
        path: Source workload CSV path.

    Returns:
        A validated immutable WorkloadTrace.

    Raises:
        WorkloadFileError: If the file is missing, malformed, or inconsistent.
    """

    # Normalize the input before opening and reporting path errors.
    csv_path = Path(path)

    try:
        with csv_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as file:
            # DictReader maps every CSV row to {column_name: value}.
            reader = csv.DictReader(file)

            # A header is required to know what each value represents.
            if reader.fieldnames is None:
                raise WorkloadFileError(
                    f"workload file has no header: {csv_path}"
                )

            actual_fields = set(reader.fieldnames)
            required_fields = set(WORKLOAD_CSV_FIELDS)

            # Set differences reveal missing or unknown columns clearly.
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

            # Materialize rows because metadata and events need separate checks.
            rows = list(reader)

    except WorkloadFileError:
        raise
    except (OSError, csv.Error) as error:
        raise WorkloadFileError(
            f"could not read workload file {csv_path}: {error}"
        ) from error

    # WorkloadTrace requires at least one request event.
    if not rows:
        raise WorkloadFileError(
            f"workload file contains no request events: {csv_path}"
        )

    # The first row defines metadata that every later row must repeat.
    first_row = rows[0]
    expected_metadata = (
        first_row["trace_name"],
        first_row["pattern"],
        first_row["random_seed"],
    )

    # CSV line 1 is the header, so data line numbers begin at 2.
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
        # CSV values are strings, so restore their Python numeric types.
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

        # Rebuild the validated domain model from the parsed rows.
        return WorkloadTrace(
            name=first_row["trace_name"],
            pattern=first_row["pattern"],
            random_seed=int(first_row["random_seed"]),
            events=events,
        )

    # Include both parsing failures and model validation failures.
    except (TypeError, ValueError) as error:
        raise WorkloadFileError(
            f"invalid workload data in {csv_path}: {error}"
        ) from error
