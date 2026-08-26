"""Data models and generators for deterministic request workloads."""

from dataclasses import dataclass

# A frozenset prevents the supported pattern names from being mutated.
SUPPORTED_WORKLOAD_PATTERNS = frozenset(
    {"steady", "bursty", "sparse", "mixed"}
)


@dataclass(frozen=True)
class RequestEvent:
    """One request scheduled relative to the start of an experiment."""

    request_id: int
    scheduled_at_seconds: float
    prompt_id: str

    # dataclass __init__이 끝난 직후 자동으로 실행되는 특별한 method
    def __post_init__(self) -> None:
        """Reject invalid request event values immediately."""

        if type(self.request_id) is not int or self.request_id < 1:
            raise ValueError("request_id must be a positive integer")

        if (
            type(self.scheduled_at_seconds) not in (int, float)
            or self.scheduled_at_seconds < 0
        ):
            raise ValueError(
                "scheduled_at_seconds must be a non-negative number"
            )

        if not isinstance(self.prompt_id, str) or not self.prompt_id.strip():
            raise ValueError("prompt_id must be a non-empty string")


@dataclass(frozen=True)
class WorkloadTrace:
    """An immutable, time-ordered collection of request events."""

    name: str
    pattern: str
    random_seed: int
    events: tuple[RequestEvent, ...]

    def __post_init__(self) -> None:
        """Validate trace metadata, event identity, and event ordering."""

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")

        if self.pattern not in SUPPORTED_WORKLOAD_PATTERNS:
            supported = ", ".join(sorted(SUPPORTED_WORKLOAD_PATTERNS))
            raise ValueError(
                f"pattern must be one of: {supported}"
            )

        # seed must be integer
        if type(self.random_seed) is not int:
            raise ValueError("random_seed must be an integer")

        if not isinstance(self.events, tuple) or not self.events:
            raise ValueError("events must be a non-empty tuple")

        if not all(isinstance(event, RequestEvent) for event in self.events):
            raise ValueError("events must contain only RequestEvent objects")

        request_ids = [event.request_id for event in self.events]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request_id values must be unique")

        timestamps = [
            event.scheduled_at_seconds for event in self.events
        ]

        # Validate timestamp order.
        # 0.0 > 10.0 → False
        # 10.0 > 5.0 → True
        # Using >= would incorrectly reject simultaneous requests.
        if any(
            current > following
            for current, following in zip(timestamps, timestamps[1:])
        ):
            raise ValueError(
                "events must be ordered by scheduled_at_seconds"
            )

    # A property makes this derived value readable like a field.
    @property
    def total_requests(self) -> int:
        """Return the number of requests in this trace."""

        return len(self.events)

    @property
    def duration_seconds(self) -> float:
        """Return the scheduled time of the final request."""

        return float(self.events[-1].scheduled_at_seconds)


def generate_steady_workload(
    # Parameters after * must be passed by name for readability.
    *,
    total_requests: int,
    interval_seconds: float,
    random_seed: int,
    prompt_id: str = "default",
) -> WorkloadTrace:
    """Generate requests separated by a fixed time interval.

    The first request is scheduled at zero seconds. Every following
    request is scheduled exactly ``interval_seconds`` after the
    previous request.

    Args:
        total_requests: Total number of request events to generate.
        interval_seconds: Time between consecutive request arrivals.
        random_seed: Seed recorded in the generated trace metadata.
        prompt_id: Identifier of the prompt used by every request.

    Returns:
        An immutable steady WorkloadTrace.

    Raises:
        ValueError: If any generator argument is invalid.
    """
    if type(total_requests) is not int or total_requests < 1:
        raise ValueError("total_requests must be a positive integer")

    if (
        type(interval_seconds) not in (int, float)
        or interval_seconds <= 0
    ):
        raise ValueError("interval_seconds must be a positive number")

    if type(random_seed) is not int:
        raise ValueError("random_seed must be an integer")

    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError("prompt_id must be a non-empty string")

    # index 0 → float(0 × 5)  → 0.0
    # index 1 → float(1 × 5)  → 5.0
    # index 2 → float(2 × 5)  → 10.0
    # index 3 → float(3 × 5)  → 15.0
    events = tuple(
        RequestEvent(
            request_id=index + 1,
            scheduled_at_seconds=float(index * interval_seconds),
            prompt_id=prompt_id,
        )
        for index in range(total_requests)
    )

    trace_name = (
        f"steady-{float(interval_seconds):g}s-"
        f"{total_requests}-requests"
    )

    return WorkloadTrace(
        name=trace_name,
        pattern="steady",
        random_seed=random_seed,
        events=events,
    )
