"""Common interface for simulated warm-state management policies"""

from abc import ABC, abstractmethod

from serverless_llm.simulator import(
RequestResult,
SimulatedServer
)

from serverless_llm.workload import(
RequestEvent,
WorkloadTrace,
)


class WarmPolicy(ABC):
    """Define lifecycle hooks shared by all warm-state policies"""

    @property
    @abstractmethod #
    def name(self) -> str:
        """Return the stable name used in experiment results."""

        raise NotImplementedError # 이 메서드의 본문이 의도적으로 구현되지 않았다는 것을 코드에 보여줌

    @abstractmethod
    def before_workload(
        self,
        server: SimulatedServer,
        trace: WorkloadTrace,
    ) -> None:
        """Prepare the server before workload request timing begins."""

        raise NotImplementedError
    @abstractmethod
    def before_request(
        self,
        server: SimulatedServer,
        event: RequestEvent,
    ) -> None:
        """Apply pending policy actions before a request is processed."""

        raise NotImplementedError

    @abstractmethod
    def after_request(
        self,
        server: SimulatedServer,
        event: RequestEvent,
        result: RequestResult,
    ) -> None:
        """Update policy state after one request completes."""

        raise NotImplementedError

    @abstractmethod
    def after_workload(
        self,
        server: SimulatedServer,
        trace: WorkloadTrace,
        results: tuple[RequestResult, ...],
    ) -> None:
        """Finalize policy state after every request has completed."""

        raise NotImplementedError