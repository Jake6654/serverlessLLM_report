"""Unit tests for the Naive Serverless policy."""

import pytest

from serverless_llm.policies import NaiveServerlessPolicy
from serverless_llm.simulator import (
    InvalidStateTransitionError,
    RequestResult,
    ServerState,
    ServerTiming,
    SimulatedServer,
)
from serverless_llm.workload import (
    RequestEvent,
    generate_steady_workload,
)


def make_server() -> SimulatedServer:
    """Create a deterministic server for policy tests."""

    return SimulatedServer(
        timing=ServerTiming(
            startup_duration_seconds=12.0,
            request_duration_seconds=2.0,
        )
    )


def make_ready_server() -> SimulatedServer:
    """Create a server that has completed startup at 12 seconds."""

    server = make_server()
    server.start()
    server.mark_ready()
    return server


def make_event() -> RequestEvent:
    """Create one request event for direct policy calls."""

    return RequestEvent(
        request_id=1,
        scheduled_at_seconds=0.0,
        prompt_id="default",
    )


def make_result() -> RequestResult:
    """Create a completed cold-start request result."""

    return RequestResult(
        request_id=1,
        arrival_time_seconds=0.0,
        started_at_seconds=12.0,
        completed_at_seconds=14.0,
        cold_start=True,
    )


def make_trace():
    """Create a minimal valid workload trace."""

    return generate_steady_workload(
        total_requests=1,
        interval_seconds=5.0,
        random_seed=42,
    )


def test_naive_serverless_policy_has_stable_name() -> None:
    """Expose a stable identifier for experiment results."""

    assert NaiveServerlessPolicy().name == "naive_serverless"


def test_before_workload_leaves_off_server_stopped() -> None:
    """Do not pre-warm the server before workload replay."""

    server = make_server()

    NaiveServerlessPolicy().before_workload(
        server,
        make_trace(),
    )

    assert server.state is ServerState.OFF
    assert server.current_time_seconds == 0.0


def test_before_workload_rejects_ready_server() -> None:
    """Require a clean OFF state at experiment start."""

    server = make_ready_server()

    with pytest.raises(
        InvalidStateTransitionError,
        match="OFF before the workload",
    ):
        NaiveServerlessPolicy().before_workload(
            server,
            make_trace(),
        )


def test_before_request_allows_off_server() -> None:
    """Allow an incoming request to trigger a cold start."""

    server = make_server()

    NaiveServerlessPolicy().before_request(
        server,
        make_event(),
    )

    assert server.state is ServerState.OFF


def test_before_request_allows_ready_server() -> None:
    """Allow an already-waiting request to reuse a ready server."""

    server = make_ready_server()

    NaiveServerlessPolicy().before_request(
        server,
        make_event(),
    )

    assert server.state is ServerState.READY


def test_before_request_rejects_starting_server() -> None:
    """Reject request processing while startup is incomplete."""

    server = make_server()
    server.start()

    with pytest.raises(
        InvalidStateTransitionError,
        match="OFF or READY",
    ):
        NaiveServerlessPolicy().before_request(
            server,
            make_event(),
        )


def test_after_request_keeps_server_for_pending_requests() -> None:
    """Keep READY while draining requests that already arrived."""

    server = make_ready_server()
    server.advance_to(14.0)

    NaiveServerlessPolicy().after_request(
        server,
        make_event(),
        make_result(),
        has_pending_requests=True,
    )

    assert server.state is ServerState.READY
    assert server.current_time_seconds == 14.0


def test_after_request_stops_server_when_queue_is_empty() -> None:
    """Stop immediately after the final pending request completes."""

    server = make_ready_server()
    server.advance_to(14.0)

    NaiveServerlessPolicy().after_request(
        server,
        make_event(),
        make_result(),
        has_pending_requests=False,
    )

    assert server.state is ServerState.OFF
    assert server.current_time_seconds == 14.0


def test_after_request_requires_ready_server() -> None:
    """Reject a completed request when the server is not READY."""

    with pytest.raises(
        InvalidStateTransitionError,
        match="READY after request",
    ):
        NaiveServerlessPolicy().after_request(
            make_server(),
            make_event(),
            make_result(),
            has_pending_requests=False,
        )


def test_after_workload_stops_remaining_ready_server() -> None:
    """Guarantee cleanup even if the server remains READY."""

    server = make_ready_server()

    NaiveServerlessPolicy().after_workload(
        server,
        make_trace(),
        (),
    )

    assert server.state is ServerState.OFF

