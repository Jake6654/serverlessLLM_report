"""Write real experiment results to JSON and CSV artifacts."""

import json # convert pythone dict into json
import csv
from io import StringIO
from pathlib import Path # 문자열 대신 파일 경로를 나타내는 객체

class ResultWriterError(RuntimeError):
  """Rasied when an experiment result cannot be written"""


# 입력값
def write_json( 
    output_path: Path,  # 저장할 json 파일 경로
    data: dict[str, object], # json 으로 변환할 dict
) -> None:
    """Serialize a dictionary and write it to a new JSON file"""


    if not isinstance(output_path, Path):
        raise ValueError(
            "output_path must be a Path object"
        )

    # 확장자 검사
    if output_path.suffix.lower() != ".json":
        raise ValueError(
            "output_path must use the .json extension"
        )

    if not isinstance(data, dict):
        raise ValueError(
            "data must be a dictionary"
        )

    try:
        # dumps 을 사용하면 python dict 을 json 으로 만들어줌
        json_text = json.dumps(
            data,
            indent=2, # 사람이 읽기 쉽게 직렬화를 해줌
            sort_keys=True,
            ensure_ascii=False, # 한글일 경우 그대로 저장
        )
    except (TypeError, ValueError) as error:
        raise ResultWriterError(
            f"Could not serialize JSON data: {error}"
        ) from error

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "x",
            encoding="utf-8",
        ) as output_file:
            output_file.write(json_text)
            output_file.write("\n")

    except FileExistsError as error:
        raise ResultWriterError(
            f"Result file already exists: {output_path}"
        ) from error

    except OSError as error:
        raise ResultWriterError(
            f"Could not write JSON result: {error}"
        ) from error

def write_csv(
      output_path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    """Write dictionary rows to a new CSV file."""

    if not isinstance(output_path, Path):
        raise ValueError(
            "output_path must be a Path object"
        )

    if output_path.suffix.lower() != ".csv":
        raise ValueError(
            "output_path must use the .csv extension"
        )

    if (
        not isinstance(fieldnames, tuple)
        or not fieldnames
        or not all(
            isinstance(fieldname, str)
            and bool(fieldname.strip())
            for fieldname in fieldnames
        )
    ):
        raise ValueError(
            "fieldnames must be a non-empty tuple of "
            "non-empty strings"
        )

    if len(fieldnames) != len(set(fieldnames)):
        raise ValueError(
            "fieldnames must not contain duplicates"
        )

    if (
        not isinstance(rows, list)
        or not all(
            isinstance(row, dict)
            for row in rows
        )
    ):
        raise ValueError(
            "rows must be a list of dictionaries"
        )

    expected_fields = set(fieldnames)

    for row in rows:
        if set(row) != expected_fields:
            raise ValueError(
                "every row must contain exactly the configured fields"
            )

    try:
        csv_buffer = StringIO(newline="")

        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

        csv_text = csv_buffer.getvalue()

    except (csv.Error, TypeError, ValueError) as error:
        raise ResultWriterError(
            f"Could not serialize CSV data: {error}"
        ) from error

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "x",
            encoding="utf-8",
            newline="",
        ) as output_file:
            output_file.write(csv_text)

    except FileExistsError as error:
        raise ResultWriterError(
            f"Result file already exists: {output_path}"
        ) from error

    except OSError as error:
        raise ResultWriterError(
            f"Could not write CSV result: {error}"
        ) from error
