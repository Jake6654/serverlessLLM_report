"""Capture reproducibility metadata for real experiments."""

import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from serverless_llm.config import ExperimentConfig
from serverless_llm.result_writer import ResultWriterError, write_json


def _json_safe(value: Any) -> Any:
    """Convert dataclass output into JSON/YAML-safe built-in values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def config_as_dict(config: ExperimentConfig) -> dict[str, object]:
    """Return the complete experiment configuration as plain data."""

    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be an ExperimentConfig object")

    return _json_safe(asdict(config))


def _run_optional(command: list[str]) -> dict[str, object]:
    """Run a metadata command without making collection itself fatal."""

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        return {"available": False, "error": str(error)}

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    result: dict[str, object] = {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
    }
    if stdout:
        result["output"] = stdout
    if stderr:
        result["error"] = stderr
    return result


def collect_environment_metadata(
    config: ExperimentConfig,
) -> dict[str, object]:
    """Collect software, source-control, container, and GPU identity."""

    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be an ExperimentConfig object")

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": sys.version.split()[0],
        },
        "git": {
            "commit": _run_optional(["git", "rev-parse", "HEAD"]),
            "branch": _run_optional(
                ["git", "branch", "--show-current"]
            ),
            "status": _run_optional(
                ["git", "status", "--short"]
            ),
        },
        "docker": {
            "version": _run_optional(
                ["docker", "version", "--format", "{{.Server.Version}}"]
            ),
            "image": config.docker.image,
            "image_id": _run_optional(
                [
                    "docker",
                    "image",
                    "inspect",
                    config.docker.image,
                    "--format",
                    "{{.Id}}",
                ]
            ),
        },
        "gpu": _run_optional(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


def build_run_manifest(
    *,
    suite_name: str,
    workload_name: str,
    policy_name: str,
    repetition: int,
    execution_order: int,
    config: ExperimentConfig,
    environment: dict[str, object],
) -> dict[str, object]:
    """Build the immutable provenance record for one successful run."""

    return {
        "schema_version": 1,
        "suite_name": suite_name,
        "workload_name": workload_name,
        "policy_name": policy_name,
        "repetition": repetition,
        "execution_order": execution_order,
        "model_name": config.model.name,
        "model_revision": config.model.revision,
        "random_seed": config.experiment.random_seed,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
    }


def write_config_snapshot(
    output_path: Path,
    config: ExperimentConfig,
) -> None:
    """Write the exact effective configuration used by one run."""

    if output_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("output_path must use a YAML extension")

    yaml_text = yaml.safe_dump(
        config_as_dict(config),
        sort_keys=False,
        allow_unicode=True,
    )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as output_file:
            output_file.write(yaml_text)
    except FileExistsError as error:
        raise ResultWriterError(
            f"Configuration snapshot already exists: {output_path}"
        ) from error
    except OSError as error:
        raise ResultWriterError(
            f"Could not write configuration snapshot: {error}"
        ) from error


def write_run_metadata(
    output_directory: Path,
    manifest: dict[str, object],
    config: ExperimentConfig,
) -> None:
    """Write manifest.json and effective-config.yaml beside run results."""

    write_json(output_directory / "manifest.json", manifest)
    write_config_snapshot(
        output_directory / "effective-config.yaml",
        config,
    )
