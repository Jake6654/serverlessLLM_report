"""Tests for request event and workload trace data models."""

from dataclasses import FrozenInstanceError

import pytest

from serverless_llm.workload import RequestEvent, WorkloadTrace


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
    with pytest.raises(ValueError, match=message):
        RequestEvent(
            request_id=request_id,
            scheduled_at_seconds=scheduled_at_seconds,
            prompt_id=prompt_id,
        )


def test_request_event_is_immutable() -> None:
    event = make_event(request_id=1, scheduled_at_seconds=0.0)

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