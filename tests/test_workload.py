"""Tests for request event and workload trace data models."""

from dataclasses import FrozenInstanceError

import pytest

from serverless_llm.workload import (
    RequestEvent,
    WorkloadTrace,
    generate_steady_workload,
    generate_bursty_workload,
    generate_sparse_workload,
    generate_mixed_workload
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


def test_generate_bursty_workload_creates_bursts() -> None:
    trace = generate_bursty_workload(
        burst_count=2,
        requests_per_burst=3,
        request_interval_seconds=1.0,
        idle_seconds_between_bursts=10.0,
        random_seed=42,
    )

    timestamps = [
        event.scheduled_at_seconds for event in trace.events
    ]

    assert timestamps == [0.0, 1.0, 2.0, 12.0, 13.0, 14.0]
    assert trace.pattern == "bursty"
    assert trace.total_requests == 6
    assert trace.duration_seconds == 14.0


def test_generate_bursty_workload_assigns_sequential_ids() -> None:
    trace = generate_bursty_workload(
        burst_count=2,
        requests_per_burst=3,
        request_interval_seconds=1.0,
        idle_seconds_between_bursts=10.0,
        random_seed=42,
    )

    request_ids = [event.request_id for event in trace.events]

    assert request_ids == [1, 2, 3, 4, 5, 6]


def test_generate_bursty_workload_is_deterministic() -> None:
    arguments = {
        "burst_count": 2,
        "requests_per_burst": 3,
        "request_interval_seconds": 1.0,
        "idle_seconds_between_bursts": 10.0,
        "random_seed": 42,
    }

    first_trace = generate_bursty_workload(**arguments)
    second_trace = generate_bursty_workload(**arguments)

    assert first_trace == second_trace


@pytest.mark.parametrize("burst_count", [0, -1, 1.5, True])
def test_generate_bursty_workload_rejects_invalid_burst_count(
    burst_count: object,
) -> None:
    with pytest.raises(ValueError, match="burst_count"):
        generate_bursty_workload(
            burst_count=burst_count,
            requests_per_burst=3,
            request_interval_seconds=1.0,
            idle_seconds_between_bursts=10.0,
            random_seed=42,
        )


@pytest.mark.parametrize("requests_per_burst", [0, -1, 1.5, True])
def test_generate_bursty_workload_rejects_invalid_burst_size(
    requests_per_burst: object,
) -> None:
    with pytest.raises(ValueError, match="requests_per_burst"):
        generate_bursty_workload(
            burst_count=2,
            requests_per_burst=requests_per_burst,
            request_interval_seconds=1.0,
            idle_seconds_between_bursts=10.0,
            random_seed=42,
        )


def test_generate_bursty_workload_rejects_short_idle_period() -> None:
    with pytest.raises(
        ValueError,
        match="idle_seconds_between_bursts must be greater",
    ):
        generate_bursty_workload(
            burst_count=2,
            requests_per_burst=3,
            request_interval_seconds=5.0,
            idle_seconds_between_bursts=5.0,
            random_seed=42,
        )


@pytest.mark.parametrize(
    "request_interval_seconds",
    [0, -1.0, "1", True],
)
def test_generate_bursty_workload_rejects_invalid_request_interval(
    request_interval_seconds: object,
) -> None:
    with pytest.raises(ValueError, match="request_interval_seconds"):
        generate_bursty_workload(
            burst_count=2,
            requests_per_burst=3,
            request_interval_seconds=request_interval_seconds,
            idle_seconds_between_bursts=10.0,
            random_seed=42,
        )


@pytest.mark.parametrize(
    "idle_seconds_between_bursts",
    [0, -1.0, "10", True],
)
def test_generate_bursty_workload_rejects_invalid_idle_period(
    idle_seconds_between_bursts: object,
) -> None:
    with pytest.raises(ValueError, match="idle_seconds_between_bursts"):
        generate_bursty_workload(
            burst_count=2,
            requests_per_burst=3,
            request_interval_seconds=1.0,
            idle_seconds_between_bursts=idle_seconds_between_bursts,
            random_seed=42,
        )
def test_generate_sparse_workload_creates_long_intervals() -> None:
    trace = generate_sparse_workload(
        total_requests=4,
        min_interval_seconds=10.0,
        max_interval_seconds=20.0,
        random_seed=42,
    )

    timestamps = [
        event.scheduled_at_seconds for event in trace.events
    ]
    intervals = [
        following - current
        for current, following in zip(
            timestamps,
            timestamps[1:],
        )
    ]

    assert timestamps[0] == 0.0
    assert all(10.0 <= interval <= 20.0 for interval in intervals)
    assert trace.pattern == "sparse"
    assert trace.total_requests == 4


def test_generate_sparse_workload_assigns_sequential_ids() -> None:
    trace = generate_sparse_workload(
        total_requests=4,
        min_interval_seconds=10.0,
        max_interval_seconds=20.0,
        random_seed=42,
    )

    request_ids = [event.request_id for event in trace.events]

    assert request_ids == [1, 2, 3, 4]


def test_generate_sparse_workload_is_deterministic() -> None:
    arguments = {
        "total_requests": 4,
        "min_interval_seconds": 10.0,
        "max_interval_seconds": 20.0,
        "random_seed": 42,
    }

    first_trace = generate_sparse_workload(**arguments)
    second_trace = generate_sparse_workload(**arguments)

    assert first_trace == second_trace


def test_generate_sparse_workload_changes_with_seed() -> None:
    first_trace = generate_sparse_workload(
        total_requests=4,
        min_interval_seconds=10.0,
        max_interval_seconds=20.0,
        random_seed=42,
    )
    second_trace = generate_sparse_workload(
        total_requests=4,
        min_interval_seconds=10.0,
        max_interval_seconds=20.0,
        random_seed=100,
    )

    assert first_trace.events != second_trace.events


@pytest.mark.parametrize("total_requests", [0, -1, 1.5, True])
def test_generate_sparse_workload_rejects_invalid_request_count(
    total_requests: object,
) -> None:
    with pytest.raises(ValueError, match="total_requests"):
        generate_sparse_workload(
            total_requests=total_requests,
            min_interval_seconds=10.0,
            max_interval_seconds=20.0,
            random_seed=42,
        )


@pytest.mark.parametrize("min_interval_seconds", [0, -1.0, "10", True])
def test_generate_sparse_workload_rejects_invalid_minimum_interval(
    min_interval_seconds: object,
) -> None:
    with pytest.raises(ValueError, match="min_interval_seconds"):
        generate_sparse_workload(
            total_requests=4,
            min_interval_seconds=min_interval_seconds,
            max_interval_seconds=20.0,
            random_seed=42,
        )


@pytest.mark.parametrize("max_interval_seconds", [0, -1.0, "20", True])
def test_generate_sparse_workload_rejects_invalid_maximum_interval(
    max_interval_seconds: object,
) -> None:
    with pytest.raises(ValueError, match="max_interval_seconds"):
        generate_sparse_workload(
            total_requests=4,
            min_interval_seconds=10.0,
            max_interval_seconds=max_interval_seconds,
            random_seed=42,
        )


def test_generate_sparse_workload_rejects_reversed_interval_range() -> None:
    with pytest.raises(
        ValueError,
        match="max_interval_seconds must be greater",
    ):
        generate_sparse_workload(
            total_requests=4,
            min_interval_seconds=20.0,
            max_interval_seconds=10.0,
            random_seed=42,
        )

def test_generate_mixed_workload_combines_all_segments() -> None:
    trace = generate_mixed_workload(
        steady_total_requests=3,
        steady_interval_seconds=5.0,
        burst_count=2,
        burst_requests_per_burst=2,
        burst_request_interval_seconds=1.0,
        burst_idle_seconds=10.0,
        sparse_total_requests=3,
        sparse_min_interval_seconds=15.0,
        sparse_max_interval_seconds=15.0,
        transition_interval_seconds=20.0,
        random_seed=42,
    )

    timestamps = [
        event.scheduled_at_seconds for event in trace.events
    ]

    assert timestamps == [
        0.0,
        5.0,
        10.0,
        30.0,
        31.0,
        41.0,
        42.0,
        62.0,
        77.0,
        92.0,
    ]
    assert trace.pattern == "mixed"
    assert trace.total_requests == 10
    assert trace.duration_seconds == 92.0


def test_generate_mixed_workload_assigns_unique_sequential_ids() -> None:
    trace = generate_mixed_workload(
        steady_total_requests=3,
        steady_interval_seconds=5.0,
        burst_count=2,
        burst_requests_per_burst=2,
        burst_request_interval_seconds=1.0,
        burst_idle_seconds=10.0,
        sparse_total_requests=3,
        sparse_min_interval_seconds=15.0,
        sparse_max_interval_seconds=15.0,
        transition_interval_seconds=20.0,
        random_seed=42,
    )

    request_ids = [event.request_id for event in trace.events]

    assert request_ids == list(range(1, 11))


def test_generate_mixed_workload_is_deterministic() -> None:
    arguments = {
        "steady_total_requests": 3,
        "steady_interval_seconds": 5.0,
        "burst_count": 2,
        "burst_requests_per_burst": 2,
        "burst_request_interval_seconds": 1.0,
        "burst_idle_seconds": 10.0,
        "sparse_total_requests": 3,
        "sparse_min_interval_seconds": 10.0,
        "sparse_max_interval_seconds": 20.0,
        "transition_interval_seconds": 20.0,
        "random_seed": 42,
    }

    first_trace = generate_mixed_workload(**arguments)
    second_trace = generate_mixed_workload(**arguments)

    assert first_trace == second_trace


def test_generate_mixed_workload_uses_prompt_id() -> None:
    trace = generate_mixed_workload(
        steady_total_requests=2,
        steady_interval_seconds=5.0,
        burst_count=1,
        burst_requests_per_burst=2,
        burst_request_interval_seconds=1.0,
        burst_idle_seconds=10.0,
        sparse_total_requests=2,
        sparse_min_interval_seconds=10.0,
        sparse_max_interval_seconds=20.0,
        transition_interval_seconds=20.0,
        random_seed=42,
        prompt_id="short",
    )

    assert all(
        event.prompt_id == "short"
        for event in trace.events
    )


@pytest.mark.parametrize(
    "transition_interval_seconds",
    [0, -1.0, "20", True],
)
def test_generate_mixed_workload_rejects_invalid_transition(
    transition_interval_seconds: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="transition_interval_seconds",
    ):
        generate_mixed_workload(
            steady_total_requests=3,
            steady_interval_seconds=5.0,
            burst_count=2,
            burst_requests_per_burst=2,
            burst_request_interval_seconds=1.0,
            burst_idle_seconds=10.0,
            sparse_total_requests=3,
            sparse_min_interval_seconds=10.0,
            sparse_max_interval_seconds=20.0,
            transition_interval_seconds=transition_interval_seconds,
            random_seed=42,
        )



