"""Data models for deterministic request workload traces."""

from dataclasses import dataclass


SUPPORTED_WORKLOAD_PATTERNS = frozenset(
    {"steady", "bursty", "sparse", "mixed"}
)


@dataclass(frozen=True)
class RequestEvent:
    """One request scheduled relative to the start of an experiment."""

    request_id: int
    scheduled_at_seconds: float
    prompt_id: str

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
        if any(
            current > following
            for current, following in zip(timestamps, timestamps[1:])
        ):
            raise ValueError(
                "events must be ordered by scheduled_at_seconds"
            )

    @property
    def total_requests(self) -> int:
        """Return the number of requests in this trace."""

        return len(self.events)

    @property
    def duration_seconds(self) -> float:
        """Return the scheduled time of the final request."""

        return float(self.events[-1].scheduled_at_seconds)