"""Summarize request latency and simulated server residency"""

from dataclasses import dataclass
from math import floor, isfinite

from serverless_llm.simulator import (
  RequestResult,
  ServerState,
  SimulatedServer
)

@dataclass(frozen=True)
class SimulationSummary:
  """Comparable metrics for one completed simulation run."""

  total_requests: int # Num of requests processed during the simulation

  # Latency threshold used to decide whether a request violated the SLO.
  total_latency_slo_seconds: float

  # Number and proportion of requests affected by a cold start.
  cold_start_count: int
  cold_start_rate: float

  mean_latency_seconds: float
  p50_latency_seconds: float
  p95_latency_seconds: float

  # Average queueing/startup wait and request processing duration.
  mean_waiting_seconds: float
  mean_processing_seconds: float

  # Number and proportion of requests exceeding the latency SLO.
  latency_slo_violation_count: int
  latency_slo_violation_rate: float

  # Total time spent loading the model in the STARTING state.
  startup_time_seconds: float

  # Total time the model remained loaded in the READY state.
  model_ready_time_seconds: float

  # Total time the server occupied resources in STARTING or READY.
  server_running_time_seconds: float

  # READY time actually spent processing requests.
  request_processing_time_seconds: float

  # READY time where the model was loaded but no request was processed.
  idle_warm_time_seconds: float

def _percentile(
    values: list[float],
    fraction: float,
) -> float:
  """Calculate a percentile using linear interpolation"""

  #Percentiles depend on value order, so sort a copy first
  ordered = sorted(values)

  # percentile 에 해당하는 상대적 index 를 계산한다
  #
  # Example:
  #   3 values and p95:
  #   position = (3 - 1) * 0.95 = 1.9
  position = (len(ordered) - 1) * fraction

  # Find the values immediately below and above the position.
  lower = floor(position)
  upper = min(lower + 1, len(ordered) - 1)

  # 아래 index에서 위 index 방향으로 이동한 비율을 계산한다
  weight = position - lower

  # Linearly interpolate(선형보간) between the lower and upper values.
  # 정확환 위치에 실제 데이터가 없을 때, 양쪽 데이터 사이의 비율을 이용해 값을 추정하는 방법이다
  return (
      ordered[lower] * (1 - weight)
      + ordered[upper] * weight
  )

def summarize_run(
    server: SimulatedServer,
    results: tuple[RequestResult, ...],
    *,
    total_latency_slo_seconds:float
) -> SimulationSummary:
  """Calculate request and server-time metrics from one simulation run"""

  
  # Validate the server because lifecycle metrics require a real server.
  if not isinstance(server, SimulatedServer):
      raise ValueError(
          "server must be a SimulatedServer object"
      )

  # A non-empty tuple prevents division by zero and ensures that every
  # element contains the timing properties used below.
  if (
      not isinstance(results, tuple)
      or not results
      or not all(
          isinstance(result, RequestResult)
          for result in results
      )
  ):
      raise ValueError(
          "results must be a non-empty RequestResult tuple"
      )
  if (
        type(total_latency_slo_seconds) not in (int, float)
        or not isfinite(total_latency_slo_seconds)
        or total_latency_slo_seconds <= 0
    ):
        raise ValueError(
            "total_latency_slo_seconds must be positive and finite"
        )

  # ----- Request-level metrics -----

  count = len(results)

  latencies = [
    result.total_latency_seconds
    for result in results
  ]

  # True is trated as 1 and False as 0 by sum()
  cold_count = sum(
    result.cold_start
    for result in results
  )

  # A request violates the SLO only when it exceeds the threshold.
  violation_count = sum(
      latency > total_latency_slo_seconds
      for latency in latencies
  )

  # ----- Server lifecycle metrics -----

  events = server.lifecycle_events
  startup_time = 0.0
  ready_time = 0.0

  # Store READY intervals so request processing can later be verified
  ready_intervals: list[tuple[float, float]] = []
  
  for index, event in enumerate(events):
    # A lifecycle state remains active until the next event.
      # For the final event, it remains active until simulation end.
      if index + 1 < len(events):
          end = events[index + 1].at_seconds
      else:
          end = server.current_time_seconds

      duration = end - event.at_seconds

      # Negative duration means lifecycle events are not chronological.
      if duration < 0:
          raise ValueError(
              "lifecycle events must be time-ordered"
          )

      if event.state is ServerState.STARTING:
          # STARTING time represents model loading/cold-start time.
          startup_time += duration

      elif event.state is ServerState.READY:
          # READY time represents time during which the model is loaded.
          ready_time += duration
          ready_intervals.append(
              (event.at_seconds, end)
          )

  # ----- Consistency checks for processing intervals -----

  # Sort by actual processing start time rather than request ID.
  ordered_results = sorted(
      results,
      key=lambda result: result.started_at_seconds,
  )

  previous_completion = -1.0

  for result in ordered_results:
      # The current simulator processes only one request at a time.
      # Therefore, request processing intervals must not overlap.
      if result.started_at_seconds < previous_completion:
          raise ValueError(
              "request processing intervals must not overlap"
          )

      previous_completion = result.completed_at_seconds

      # Every request must be processed entirely within a READY interval.
      processed_while_ready = any(
          begin <= result.started_at_seconds
          and result.completed_at_seconds <= end
          for begin, end in ready_intervals
      )

      if not processed_while_ready:
          raise ValueError(
              "request processing must occur while server is READY"
          )

  # Add together the actual inference duration of every request.
  processing_time = sum(
      result.processing_time_seconds
      for result in results
  )

  # READY time consists of processing time plus unused warm time.
  idle_warm_time = ready_time - processing_time

  # A small tolerance protects against floating-point rounding noise.
  if idle_warm_time < -1e-9:
      raise ValueError(
          "processing time cannot exceed READY time"
      )

  # ----- Build an immutable experiment summary -----

  return SimulationSummary(
      total_requests=count,
      total_latency_slo_seconds=float(
          total_latency_slo_seconds
      ),
      cold_start_count=cold_count,
      cold_start_rate=cold_count / count,
      mean_latency_seconds=sum(latencies) / count,
      p50_latency_seconds=_percentile(
          latencies,
          0.50,
      ),
      p95_latency_seconds=_percentile(
          latencies,
          0.95,
      ),
      mean_waiting_seconds=(
          sum(
              result.waiting_time_seconds
              for result in results
          )
          / count
      ),
      mean_processing_seconds=processing_time / count,
      latency_slo_violation_count=violation_count,
      latency_slo_violation_rate=(
          violation_count / count
      ),
      startup_time_seconds=startup_time,
      model_ready_time_seconds=ready_time,
      server_running_time_seconds=(
          startup_time + ready_time
      ),
      request_processing_time_seconds=processing_time,
      idle_warm_time_seconds=max(
          0.0,
          idle_warm_time,
      ),
  )




