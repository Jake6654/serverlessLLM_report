"""Tests for the configuration command-line interface."""

from pathlib import Path

import pytest

from serverless_llm.cli import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


def test_cli_reports_valid_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main([str(CONFIG_PATH)])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Configuration valid" in captured.out
    assert "traffic-aware-serverless-llm" in captured.out
    assert captured.err == ""


def test_cli_reports_missing_configuration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_path = tmp_path / "missing.yaml"

    exit_code = main([str(missing_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "Configuration error" in captured.err
    assert "Could not read configuration file" in captured.err