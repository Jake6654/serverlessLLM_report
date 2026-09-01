"""Tests for request event and workload trace data models."""

from dataclasses import FrozenInstanceError

import pytest

from serverless_llm.workload import (
    RequestEvent,
    WorkloadTrace,
    generate_steady_workload,
)


def make_event(request_id: int, scheduled_at_seconds: float) -> RequestEvent:
    """Create a request event with the default prompt identifier."""

    return RequestEvent(
        request_id=request_id,
        scheduled_at_seconds=scheduled_at_seconds,
        prompt_id="default",
    )


def test_request_event_stores_relative_schedule() -> None:
    event = make_event(request_id=1, scheduled_at_seconds=5.0)

    assert event.request_id == 1
    assert event.scheduled_at_seconds == 5.0
    assert event.prompt_id == "default"


@pytest.mark.parametrize(
    ("request_id", "scheduled_at_seconds", "prompt_id", "message"),
    [
        (0, 0.0, "default", "request_id"),
        (1, -0.1, "default", "scheduled_at_seconds"),
        (1, 0.0, "   ", "prompt_id"),
    ],
)
def test_request_event_rejects_invalid_values(
    request_id: int,
    scheduled_at_seconds: float,
    prompt_id: str,
    message: str,
) -> None:
    # ValueError 가 발생하야 성공
    with pytest.raises(ValueError, match=message):
        RequestEvent(
            request_id=request_id,
            scheduled_at_seconds=scheduled_at_seconds,
            prompt_id=prompt_id,
        )


def test_request_event_is_immutable() -> None:
    event = make_event(request_id=1, scheduled_at_seconds=0.0)

    # Since event is imuutable, when user tries to change the value
    # it raises the error 
    with pytest.raises(FrozenInstanceError):
        event.scheduled_at_seconds = 10.0


def test_workload_trace_reports_size_and_duration() -> None:
    trace = WorkloadTrace(
        name="steady-example",
        pattern="steady",
        random_seed=42,
        events=(
            make_event(1, 0.0),
            make_event(2, 5.0),
            make_event(3, 10.0),
        ),
    )

    assert trace.total_requests == 3
    assert trace.duration_seconds == 10.0


def test_workload_trace_allows_simultaneous_requests() -> None:
    trace = WorkloadTrace(
        name="burst-example",
        pattern="bursty",
        random_seed=42,
        events=(
            make_event(1, 0.0),
            make_event(2, 0.0),
        ),
    )

    assert trace.total_requests == 2


def test_workload_trace_rejects_duplicate_request_ids() -> None:
    with pytest.raises(ValueError, match="request_id values must be unique"):
        WorkloadTrace(
            name="duplicate-ids",
            pattern="steady",
            random_seed=42,
            events=(
                make_event(1, 0.0),
                make_event(1, 5.0),
            ),
        )


def test_workload_trace_rejects_out_of_order_events() -> None:
    with pytest.raises(ValueError, match="events must be ordered"):
        WorkloadTrace(
            name="out-of-order",
            pattern="steady",
            random_seed=42,
            events=(
                make_event(1, 5.0),
                make_event(2, 0.0),
            ),
        )


def test_workload_trace_rejects_unknown_pattern() -> None:
    with pytest.raises(ValueError, match="pattern must be one of"):
        WorkloadTrace(
            name="unknown-pattern",
            pattern="random",
            random_seed=42,
            events=(make_event(1, 0.0),),
        )

def test_generate_steady_workload_creates_fixed_intervals() -> None:
    trace = generate_steady_workload(
        total_requests=4,
        interval_seconds=5.0,
        random_seed=42,
    )

    timestamps = [
        event.scheduled_at_seconds for event in trace.events
    ]

    assert timestamps == [0.0, 5.0, 10.0, 15.0]
    assert trace.total_requests == 4
    assert trace.duration_seconds == 15.0


def test_generate_steady_workload_assigns_sequential_ids() -> None:
    trace = generate_steady_workload(
        total_requests=4,
        interval_seconds=5.0,
        random_seed=42,
    )

    request_ids = [event.request_id for event in trace.events]

    assert request_ids == [1, 2, 3, 4]

# field 가 가 모두 같으면 두 객체가 같다고 판단한다
def test_generate_steady_workload_is_deterministic() -> None:
    first_trace = generate_steady_workload(
        total_requests=4,
        interval_seconds=5.0,
        random_seed=42,
    )
    second_trace = generate_steady_workload(
        total_requests=4,
        interval_seconds=5.0,
        random_seed=42,
    )

    assert first_trace == second_trace


def test_generate_steady_workload_uses_prompt_id() -> None:
    trace = generate_steady_workload(
        total_requests=3,
        interval_seconds=5.0,
        random_seed=42,
        prompt_id="short",
    )

    assert all(event.prompt_id == "short" for event in trace.events)


@pytest.mark.parametrize("total_requests", [0, -1, 1.5, True])
def test_generate_steady_workload_rejects_invalid_request_count(
    total_requests: object,
) -> None:
    with pytest.raises(ValueError, match="total_requests"):
        generate_steady_workload(
            total_requests=total_requests,
            interval_seconds=5.0,
            random_seed=42,
        )


@pytest.mark.parametrize("interval_seconds", [0, -1.0, "5", True])
def test_generate_steady_workload_rejects_invalid_interval(
    interval_seconds: object,
) -> None:
    with pytest.raises(ValueError, match="interval_seconds"):
        generate_steady_workload(
            total_requests=4,
            interval_seconds=interval_seconds,
            random_seed=42,
        )