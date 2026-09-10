"""Tests for the core server simulator data models."""

from dataclasses import FrozenInstanceError

import pytest

from serverless_llm.simulator import (
    InvalidStateTransitionError,
    ServerState,
    ServerTiming,
    SimulatedServer,
)


def make_timing() -> ServerTiming:
    """Create valid timing assumptions shared by simulator tests."""

    return ServerTiming(
        startup_duration_seconds=12.0,
        request_duration_seconds=2.0,
    )


def test_server_state_has_expected_serialized_values() -> None:
    assert ServerState.OFF.value == "off"
    assert ServerState.STARTING.value == "starting"
    assert ServerState.READY.value == "ready"


def test_server_timing_stores_duration_values() -> None:
    timing = make_timing()

    assert timing.startup_duration_seconds == 12.0
    assert timing.request_duration_seconds == 2.0


def test_server_timing_is_immutable() -> None:
    timing = make_timing()

    with pytest.raises(FrozenInstanceError):
        timing.startup_duration_seconds = 30.0


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("startup_duration_seconds", 0),
        ("startup_duration_seconds", -1.0),
        ("startup_duration_seconds", "12"),
        ("startup_duration_seconds", True),
        ("request_duration_seconds", 0),
        ("request_duration_seconds", -1.0),
        ("request_duration_seconds", "2"),
        ("request_duration_seconds", True),
    ],
)
def test_server_timing_rejects_invalid_values(
    field_name: str,
    invalid_value: object,
) -> None:
    values = {
        "startup_duration_seconds": 12.0,
        "request_duration_seconds": 2.0,
    }
    values[field_name] = invalid_value

    with pytest.raises(ValueError, match=field_name):
        ServerTiming(**values)


def test_simulated_server_starts_off_at_zero_seconds() -> None:
    server = SimulatedServer(timing=make_timing())

    assert server.state is ServerState.OFF
    assert server.current_time_seconds == 0.0
    assert server.is_off
    assert not server.is_ready
    assert not server.is_running


def test_starting_server_is_running_but_not_ready() -> None:
    server = SimulatedServer(
        timing=make_timing(),
        state=ServerState.STARTING,
        current_time_seconds=5.0,
    )

    assert not server.is_off
    assert not server.is_ready
    assert server.is_running


def test_ready_server_is_running_and_ready() -> None:
    server = SimulatedServer(
        timing=make_timing(),
        state=ServerState.READY,
        current_time_seconds=12.0,
    )

    assert not server.is_off
    assert server.is_ready
    assert server.is_running


@pytest.mark.parametrize("current_time_seconds", [-1.0, "0", True])
def test_simulated_server_rejects_invalid_current_time(
    current_time_seconds: object,
) -> None:
    with pytest.raises(ValueError, match="current_time_seconds"):
        SimulatedServer(
            timing=make_timing(),
            current_time_seconds=current_time_seconds,
        )


def test_simulated_server_rejects_string_state() -> None:
    with pytest.raises(ValueError, match="state"):
        SimulatedServer(
            timing=make_timing(),
            state="off",
        )
def test_start_moves_off_server_to_starting() -> None:
    server = SimulatedServer(timing=make_timing())

    server.start()

    assert server.state is ServerState.STARTING
    assert server.current_time_seconds == 0.0
    assert server.is_running
    assert not server.is_ready


def test_mark_ready_finishes_startup() -> None:
    server = SimulatedServer(timing=make_timing())
    server.start()

    server.mark_ready()

    assert server.state is ServerState.READY
    assert server.current_time_seconds == 12.0
    assert server.is_ready


def test_stop_moves_ready_server_to_off() -> None:
    server = SimulatedServer(
        timing=make_timing(),
        state=ServerState.READY,
        current_time_seconds=12.0,
    )

    server.stop()

    assert server.state is ServerState.OFF
    assert server.current_time_seconds == 12.0
    assert server.is_off


def test_server_can_complete_one_lifecycle() -> None:
    server = SimulatedServer(timing=make_timing())

    server.start()
    server.mark_ready()
    server.stop()

    assert server.state is ServerState.OFF
    assert server.current_time_seconds == 12.0


def test_start_rejects_server_that_is_already_running() -> None:
    server = SimulatedServer(
        timing=make_timing(),
        state=ServerState.READY,
    )

    with pytest.raises(
        InvalidStateTransitionError,
        match="cannot start",
    ):
        server.start()


def test_mark_ready_rejects_off_server() -> None:
    server = SimulatedServer(timing=make_timing())

    with pytest.raises(
        InvalidStateTransitionError,
        match="cannot mark ready",
    ):
        server.mark_ready()


def test_stop_rejects_starting_server() -> None:
    server = SimulatedServer(
        timing=make_timing(),
        state=ServerState.STARTING,
    )

    with pytest.raises(
        InvalidStateTransitionError,
        match="cannot stop",
    ):
        server.stop()