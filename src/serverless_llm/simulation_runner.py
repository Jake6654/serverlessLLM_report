"""Run an entire workload through one simulated server."""

from serverless_llm.simulator import RequestResult, SimulatedServer
from serverless_llm.workload import WorkloadTrace

def run_workload(
    server: SimulatedServer,
    trace: WorkloadTrace,
) -> tuple[RequestResult, ...]:
  """Process every scheduled request and return its results."""

  if not isinstance(server, SimulatedServer):
    raise ValueError("server must be a SimulatedServer object")
  
  if not isinstance(trace, WorkloadTrace):
        raise ValueError("trace must be a WorkloadTrace object")
  
  #실행 중 결과를 계속 추가해야 하므로 list 을 사용한다
  results: list[RequestResult] = []

  # WorkloadTrace already guarantees that events are time-ordered
  for event in trace.events:
      result = server.process_request(event)
      results.append(result)

  # Return immutable results after the run has finished
  return tuple(results)
