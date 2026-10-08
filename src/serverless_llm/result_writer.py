"""Write real experiment results to JSON and CSV artifacts."""

import json # convert pythone dict into json
import csv
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path # 문자열 대신 파일 경로를 나타내는 객체

from serverless_llm.real_metrics import (
    RealPolicySummary,
    calculate_request_metrics,
    normalize_run_timeline,
)
from serverless_llm.real_policy_runner import (
    RealPolicyRunResult,
)

REQUEST_FIELDNAMES = (
    "request_id",
    "prompt_id",
    "cold_start",
    "startup_duration_seconds",
    "scheduling_delay_seconds",
    "pre_inference_delay_seconds",
    "client_ttft_seconds",
    "end_to_end_ttft_seconds",
    "generation_duration_seconds",
    "client_latency_seconds",
    "end_to_end_latency_seconds",
    "finish_reason",
)

GPU_SAMPLE_FIELDNAMES = (
    "sampled_at_seconds",
    "gpu_index",
    "utilization_percent",
    "memory_used_mib",
    "power_draw_watts",
    "temperature_celsius",
)

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

@dataclass(frozen=True)
class PolicyResultPaths:
    """Store every artifact path created for one policy run."""

    output_directory: Path
    requests_csv: Path
    gpu_samples_csv: Path
    timeline_json: Path
    summary_json: Path


def write_csv(
      output_path: Path,
    fieldnames: tuple[str, ...], # CSV Column
    rows: list[dict[str, object]], # 한 딕셔너리가 csv 한 행이된다
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

    # 겹치는 key 값이 있으면 지움 set 은 중복허용 x
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

    # Require every row to contain exactly the configured CSV fields.
    for row in rows:
        if set(row) != expected_fields:
            raise ValueError(
                "every row must contain exactly the configured fields"
            )


    try:
        csv_buffer = StringIO(newline="")

        # connect keys in dict to CSV column
        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=fieldnames,
        )

        # 첫 줄을 작성 request_id,cold_start,end_to_end_ttft_seconds
        writer.writeheader()
        writer.writerows(rows)

        csv_text = csv_buffer.getvalue()

    except (csv.Error, TypeError, ValueError) as error:
        raise ResultWriterError(
            f"Could not serialize CSV data: {error}"
        ) from error

    try:
        # output_path 가 "results/run-001/always_on/repetition-001/requests.csv" 일때
        # .parent 을 하면 results/run-001/always_on/repetition-001 이 부분을 뜻함
        # parent.parent 할쑤록 한 단계씩 올라간다
        output_path.parent.mkdir(
            parents=True, # 필요하면 상위 디렉터리까지 생성
            exist_ok=True, # 생성하려는 디렉토리가 있어도 오류 발생 x
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

def write_policy_results(
    output_directory: Path,
    run_result: RealPolicyRunResult,
    policy_summary: RealPolicySummary,
) -> PolicyResultPaths:
    """Write every artifact belonging to one real policy run."""

    if not isinstance(output_directory, Path):
        raise ValueError(
            "output_directory must be a Path object"
        )

    if (
        output_directory.exists()
        and not output_directory.is_dir()
    ):
        raise ValueError(
            "output_directory must be a directory path"
        )

    if not isinstance(run_result, RealPolicyRunResult):
        raise ValueError(
            "run_result must be a RealPolicyRunResult object"
        )

    if not isinstance(policy_summary, RealPolicySummary):
        raise ValueError(
            "policy_summary must be a RealPolicySummary object"
        )

    if run_result.policy_name != policy_summary.policy_name:
        raise ValueError(
            "run_result and policy_summary must use the same policy"
        )

    if (
        len(run_result.request_results)
        != policy_summary.request_summary.total_requests
    ):
        raise ValueError(
            "run_result and policy_summary must contain the same "
            "number of requests"
        )

    paths = PolicyResultPaths(
        output_directory=output_directory,
        requests_csv=output_directory / "requests.csv",
        gpu_samples_csv=output_directory / "gpu_samples.csv",
        timeline_json=output_directory / "timeline.json",
        summary_json=output_directory / "summary.json",
    )

    artifact_paths = (
        paths.requests_csv,
        paths.gpu_samples_csv,
        paths.timeline_json,
        paths.summary_json,
    )

    existing_paths = [
        path
        for path in artifact_paths
        if path.exists()
    ]

    if existing_paths:
        formatted_paths = ", ".join(
            str(path)
            for path in existing_paths
        )

        raise ResultWriterError(
            f"Result artifacts already exist: {formatted_paths}"
        )

    timeline = normalize_run_timeline(run_result)

    request_metrics = calculate_request_metrics(
        run_result
    )

    request_rows = [
        asdict(metric)
        for metric in request_metrics
    ]

    gpu_sample_rows = [
        asdict(sample)
        for sample in timeline.gpu_samples
    ]

    timeline_data = asdict(timeline)
    summary_data = asdict(policy_summary)

    write_csv(
        paths.requests_csv,
        REQUEST_FIELDNAMES,
        request_rows,
    )

    write_csv(
        paths.gpu_samples_csv,
        GPU_SAMPLE_FIELDNAMES,
        gpu_sample_rows,
    )

    write_json(
        paths.timeline_json,
        timeline_data,
    )

    write_json(
        paths.summary_json,
        summary_data,
    )

    return paths
