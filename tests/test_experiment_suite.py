"""Tests for balanced report-suite scheduling."""

from pathlib import Path

from serverless_llm.experiment_suite import (
    build_suite_schedule,
    load_suite_config,
    schedule_as_dicts,
)


def test_pilot_suite_loads_relative_base_config() -> None:
    suite = load_suite_config(Path("configs/pilot_suite.yaml"))

    assert suite.base_config_path == Path("configs/experiment.yaml")
    assert suite.policies == (
        "always_on",
        "naive_serverless",
        "fixed_keep_warm",
    )


def test_schedule_rotates_policy_order_between_repetitions(tmp_path) -> None:
    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(
        """
suite:
  name: balanced-test
  repetitions: 3
  random_seed: 100
  policies: [always_on, naive_serverless, fixed_keep_warm]
base_config: BASE_CONFIG
workloads:
  - name: dense
    request_interval_seconds: 5.0
    total_requests: 2
""".replace(
            "BASE_CONFIG",
            str(Path("configs/experiment.yaml").resolve()),
        ),
        encoding="utf-8",
    )
    suite = load_suite_config(suite_file)

    schedule = build_suite_schedule(suite)

    assert len(schedule) == 9
    assert [spec.policy_name for spec in schedule[0:3]] == [
        "always_on",
        "naive_serverless",
        "fixed_keep_warm",
    ]
    assert [spec.policy_name for spec in schedule[3:6]] == [
        "naive_serverless",
        "fixed_keep_warm",
        "always_on",
    ]
    assert [spec.config.experiment.random_seed for spec in schedule[0::3]] == [
        100,
        101,
        102,
    ]


def test_schedule_rows_include_effective_workload() -> None:
    suite = load_suite_config(Path("configs/pilot_suite.yaml"))

    rows = schedule_as_dicts(build_suite_schedule(suite))

    assert len(rows) == 3
    assert rows[0]["request_interval_seconds"] == 5.0
    assert rows[0]["total_requests"] == 3

