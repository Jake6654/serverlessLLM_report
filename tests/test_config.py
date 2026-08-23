"""Tests for experiment configuration loading and validation."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from serverless_llm.config import (
    ConfigLoadError,
    ConfigValidationError,
    ExperimentConfig,
    load_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


@pytest.fixture
def valid_config() -> ExperimentConfig:
    """Load the project's valid default experiment configuration."""

    return load_config(CONFIG_PATH)


@pytest.fixture
def valid_config_data() -> dict:
    """Return a mutable copy of the default YAML data."""

    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def write_yaml(tmp_path: Path, data: object) -> Path:
    """Write test data to a temporary YAML file."""

    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_valid_config_loads_as_experiment_config(
    valid_config: ExperimentConfig,
) -> None:
    assert isinstance(valid_config, ExperimentConfig)
    assert valid_config.experiment.name == "traffic-aware-serverless-llm"
    assert valid_config.model.name == (
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )


def test_loader_converts_special_types(
    valid_config: ExperimentConfig,
) -> None:
    assert isinstance(valid_config.output.results_directory, Path)
    assert isinstance(
        valid_config.policies.fixed_keep_warm.timeout_candidates_seconds,
        tuple,
    )


def test_loaded_config_is_immutable(
    valid_config: ExperimentConfig,
) -> None:
    with pytest.raises(FrozenInstanceError):
        valid_config.model.name = "another-model"


def test_missing_file_raises_config_load_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(
        ConfigLoadError,
        match="Could not read configuration file",
    ):
        load_config(missing_path)


def test_malformed_yaml_raises_config_load_error(tmp_path: Path) -> None:
    path = tmp_path / "malformed.yaml"
    path.write_text("model: [\n", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="Invalid YAML"):
        load_config(path)


def test_non_mapping_root_raises_config_load_error(tmp_path: Path) -> None:
    path = tmp_path / "not-a-mapping.yaml"
    path.write_text("- first\n- second\n", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="must be a mapping"):
        load_config(path)


def test_missing_required_section_raises_config_load_error(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    valid_config_data.pop("model")
    path = write_yaml(tmp_path, valid_config_data)

    with pytest.raises(
        ConfigLoadError,
        match="Invalid configuration structure",
    ):
        load_config(path)


def test_invalid_gpu_memory_utilization_is_rejected(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    valid_config_data["model"]["gpu_memory_utilization"] = 5.0
    path = write_yaml(tmp_path, valid_config_data)

    with pytest.raises(
        ConfigValidationError,
        match="model.gpu_memory_utilization",
    ):
        load_config(path)


def test_invalid_ml_threshold_is_rejected(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    valid_config_data["policies"]["ml_based"]["decision_threshold"] = 2.0
    path = write_yaml(tmp_path, valid_config_data)

    with pytest.raises(
        ConfigValidationError,
        match="ML decision threshold",
    ):
        load_config(path)


def test_invalid_timeout_relationship_is_rejected(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    rule_config = valid_config_data["policies"]["rule_based"]
    rule_config["minimum_timeout_seconds"] = 120
    rule_config["maximum_timeout_seconds"] = 10
    path = write_yaml(tmp_path, valid_config_data)

    with pytest.raises(
        ConfigValidationError,
        match="maximum timeout cannot be smaller",
    ):
        load_config(path)


def test_multiple_validation_errors_are_reported(
    tmp_path: Path,
    valid_config_data: dict,
) -> None:
    valid_config_data["model"]["max_model_len"] = 0
    valid_config_data["model"]["gpu_memory_utilization"] = 5.0
    valid_config_data["server"]["port"] = -1
    path = write_yaml(tmp_path, valid_config_data)

    with pytest.raises(ConfigValidationError) as captured:
        load_config(path)

    message = str(captured.value)

    assert "model.max_model_len" in message
    assert "model.gpu_memory_utilization" in message
    assert "server.port" in message
