"""Tests for report reproducibility metadata."""

import json
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import Mock

import yaml

import serverless_llm.experiment_metadata as metadata_module
from serverless_llm.config import load_config
from serverless_llm.experiment_metadata import (
    collect_environment_metadata,
    config_as_dict,
    write_run_metadata,
)


def test_config_as_dict_converts_paths() -> None:
    config = load_config(Path("configs/experiment.yaml"))

    data = config_as_dict(config)

    assert data["output"]["results_directory"] == "results"
    assert data["model"]["revision"] == (
        "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    )


def test_collect_environment_metadata_records_commands(
    monkeypatch,
) -> None:
    fake_run = Mock(
        return_value=CompletedProcess(
            args=[], returncode=0, stdout="value\n", stderr=""
        )
    )
    monkeypatch.setattr(metadata_module.subprocess, "run", fake_run)
    config = load_config(Path("configs/experiment.yaml"))

    metadata = collect_environment_metadata(config)

    assert metadata["git"]["commit"]["output"] == "value"
    assert metadata["docker"]["image"] == "serverless-llm-vllm:0.19.0"
    assert metadata["gpu"]["available"] is True
    assert fake_run.call_count == 6


def test_write_run_metadata_creates_manifest_and_config(tmp_path) -> None:
    config = load_config(Path("configs/experiment.yaml"))
    manifest = {"suite_name": "test-suite", "repetition": 1}

    write_run_metadata(tmp_path / "run", manifest, config)

    assert json.loads((tmp_path / "run/manifest.json").read_text()) == manifest
    snapshot = yaml.safe_load(
        (tmp_path / "run/effective-config.yaml").read_text()
    )
    assert snapshot["workload"]["request_interval_seconds"] == 5.0

