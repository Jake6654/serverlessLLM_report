# Traffic-Aware Serverless LLM Inference

[English](README.md) | [한국어](README.ko.md)

An experimental framework for measuring the latency, cold-start behavior,
GPU energy use, and service-level objective (SLO) compliance of serverless LLM
warm-state policies on a real GPU instance.

The completed AWS baseline consists of **48 runs** across four traffic
patterns, three policies, and four repetitions. It uses vLLM serving
TinyLlama on an NVIDIA L4 GPU and records request-, startup-, GPU-, and
energy-level metrics in reproducible artifacts.

## Research question

How does GPU warm-state management change the latency–energy trade-off of
serverless LLM inference under different request arrival intervals?

The current study evaluates three implemented baseline policies:

| Policy | Behavior |
|---|---|
| `always_on` | Starts vLLM before the workload and keeps it running. |
| `naive_serverless` | Starts vLLM for every request and stops it immediately afterward. |
| `fixed_keep_warm` | Keeps vLLM alive for 30 seconds after each request. |

Rule-based and ML-based adaptive policies remain planned follow-up work; they
are not included in the results below.

## System overview

```text
suite configuration
        |
        v
policy runner ---> Docker/vLLM ---> streamed completion response
      |                  |                    |
      |                  v                    v
      +------------> nvidia-smi          request timing
                         |                    |
                         +---------+----------+
                                   v
                      per-run artifacts and report
```

The runtime starts and stops the vLLM container, waits for HTTP readiness,
sends streamed completion requests, and samples the GPU in a background
thread. The suite runner rotates policy order, preserves the effective
configuration and environment metadata, and aggregates completed runs into
tables and SVG figures.

## Experimental setup

### Hardware and software

| Component | Configuration |
|---|---|
| Cloud instance | AWS EC2 `g6.xlarge` |
| GPU | NVIDIA L4, 23,034 MiB |
| Operating system | Ubuntu 22.04.5 LTS, x86_64 |
| Python | 3.10.12 |
| NVIDIA driver / CUDA | 595.91.07 / 13.2 |
| Docker | 29.8.1 |
| Inference server | vLLM 0.19.0 |
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Model revision | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` |

### Controlled request and measurement settings

| Setting | Value |
|---|---:|
| Endpoint | `/v1/completions` |
| Prompt | `Serverless computing is` |
| Maximum generated tokens | 32 |
| Temperature | 0.0 |
| Streaming | enabled |
| Concurrent requests | 1 |
| GPU sampling interval | 200 ms |
| TTFT SLO | 1.0 s |
| Fixed keep-warm timeout | 30 s |

The Docker image, model revision, random seed, and effective configuration
are pinned or captured with every run. See
[`configs/experiment.yaml`](configs/experiment.yaml) and
[`docker/Dockerfile.vllm`](docker/Dockerfile.vllm) for the full settings.

## Experiment design and process

The report experiment uses four steady workloads designed to cover traffic
well below, near, and far beyond the 30-second keep-warm timeout:

| Workload | Arrival interval | Requests per run | Purpose |
|---|---:|---:|---|
| `dense-5s` | 5 s | 20 | Requests arrive much faster than the timeout. |
| `boundary-25s` | 25 s | 8 | Requests arrive just before the timeout. |
| `boundary-35s` | 35 s | 8 | Requests arrive just after the timeout. |
| `sparse-90s` | 90 s | 6 | Every request arrives far beyond the timeout. |

Each workload–policy combination is repeated four times:

```text
4 workloads x 3 policies x 4 repetitions = 48 runs
```

For every run, the framework:

1. Resolves the base configuration and workload/policy override.
2. Records Git, Docker, model, GPU, OS, and Python provenance.
3. Starts GPU monitoring and executes the selected policy.
4. Measures startup time, TTFT, generation time, end-to-end TTFT, cold starts,
   SLO violations, GPU utilization, memory, power, and estimated energy.
5. Stops the container and waits until Docker releases its name.
6. Writes raw and normalized run artifacts.
7. Aggregates repetitions into means, sample standard deviations, and
   normal-approximation 95% confidence intervals.

Policy order rotates across repetitions and workloads to reduce systematic
thermal or time-order bias. The exact completed schedule is defined in
[`configs/report_suite_48.yaml`](configs/report_suite_48.yaml).

## Reproducing the experiment

### 1. Install and validate the project

Install [`uv`](https://docs.astral.sh/uv/), clone the repository, and run:

```bash
UV_CACHE_DIR=.cache/uv uv sync --extra dev
UV_CACHE_DIR=.cache/uv uv run pytest -q
```

The unit tests mock Docker, HTTP, and `nvidia-smi` where appropriate, so most
of the framework can be validated without a GPU.

### 2. Build the vLLM image on the GPU host

```bash
docker build \
  --file docker/Dockerfile.vllm \
  --tag serverless-llm-vllm:0.19.0 \
  .

docker volume create serverless-llm-hf-cache
```

The named volume persists downloaded Hugging Face model files between
containers, avoiding a fresh download on every cold start.

### 3. Run preflight checks and a pilot

```bash
UV_CACHE_DIR=.cache/uv uv run serverless-llm-check configs/experiment.yaml

UV_CACHE_DIR=.cache/uv uv run serverless-llm-suite \
  configs/pilot_suite.yaml \
  --dry-run

UV_CACHE_DIR=.cache/uv uv run serverless-llm-suite \
  configs/pilot_suite.yaml
```

The pilot verifies the complete Docker, vLLM, request, monitoring, artifact,
and reporting pipeline before the full GPU experiment.

### 4. Run the 48-run suite

Preview the execution schedule:

```bash
UV_CACHE_DIR=.cache/uv uv run serverless-llm-suite \
  configs/report_suite_48.yaml \
  --dry-run
```

Run inside `tmux` so an SSH disconnect does not terminate the experiment:

```bash
tmux new -s serverless-llm-report-48
set -o pipefail
PYTHONUNBUFFERED=1 UV_CACHE_DIR=.cache/uv \
  uv run serverless-llm-suite configs/report_suite_48.yaml \
  2>&1 | tee report-suite-48.log
```

Detach with `Ctrl-b`, then `d`; reattach with:

```bash
tmux attach -t serverless-llm-report-48
```

### 5. Generate the report

```bash
UV_CACHE_DIR=.cache/uv uv run serverless-llm-report \
  results/traffic-aware-serverless-llm-report-48-v1
```

## Results

Values are **mean ± 95% confidence-interval half-width** across four
repetitions. P95 E2E TTFT includes queueing and cold-start delay. Energy is
estimated by integrating sampled GPU power over each policy's observation
window.

| Workload | Policy | P95 E2E TTFT (s) | Cold-start rate | SLO violation rate | Energy/request (Wh) |
|---|---|---:|---:|---:|---:|
| dense-5s | Always-On | 0.0531 ± 0.0005 | 0.000 | 0.000 | 0.0674 ± 0.0005 |
| dense-5s | Fixed Keep-Warm | 50.9361 ± 0.4989 | 0.050 | 0.600 | 0.0446 ± 0.0001 |
| dense-5s | Naive Serverless | 996.2840 ± 1.1215 | 1.000 | 1.000 | 0.4976 ± 0.0019 |
| boundary-25s | Always-On | 0.4205 ± 0.0015 | 0.000 | 0.000 | 0.2436 ± 0.0012 |
| boundary-25s | Fixed Keep-Warm | 46.7528 ± 0.4947 | 0.125 | 0.375 | 0.1877 ± 0.0003 |
| boundary-25s | Naive Serverless | 269.7498 ± 2.2465 | 1.000 | 1.000 | 0.4969 ± 0.0039 |
| boundary-35s | Always-On | 0.4234 ± 0.0034 | 0.000 | 0.000 | 0.3137 ± 0.0005 |
| boundary-35s | Fixed Keep-Warm | 55.8160 ± 0.4733 | 0.375 | 0.750 | 0.2863 ± 0.0004 |
| boundary-35s | Naive Serverless | 201.8690 ± 2.3440 | 1.000 | 1.000 | 0.4938 ± 0.0024 |
| sparse-90s | Always-On | 0.4911 ± 0.0017 | 0.000 | 0.000 | 0.6944 ± 0.0035 |
| sparse-90s | Fixed Keep-Warm | 56.1430 ± 0.0378 | 1.000 | 1.000 | 0.7174 ± 0.0031 |
| sparse-90s | Naive Serverless | 56.1372 ± 0.1039 | 1.000 | 1.000 | 0.7224 ± 0.0031 |

Across the 168 requests executed by each policy, the measured total GPU
energy was 39.895 Wh for Always-On, 35.956 Wh for Fixed Keep-Warm, and
88.846 Wh for Naive Serverless. These totals are descriptive rather than a
strict energy ranking because policy observation windows are not identical;
see the limitations below.

## Key findings

1. **Always-On provides the strongest latency and reliability.** It produced
   no cold starts or TTFT SLO violations in any workload.
2. **Naive Serverless is unstable under frequent arrivals.** A cold start took
   roughly 55 seconds, so requests arriving every 5 seconds accumulated in a
   queue and pushed P95 E2E TTFT close to 1,000 seconds.
3. **Avoiding a cold start is not sufficient to meet the SLO.** In the dense
   workload, Fixed Keep-Warm had only a 5% cold-start rate but a 60% SLO
   violation rate because the initial cold start delayed later requests.
4. **The 30-second timeout is highly workload-dependent.** It saved energy
   relative to Always-On in the 5-, 25-, and 35-second workloads, but at 90
   seconds it expired before every request and behaved almost identically to
   Naive Serverless.
5. **Startup time must influence policy design.** A timeout shorter than the
   roughly 55-second startup cost cannot protect sparse traffic from repeated
   cold starts; adaptive policies should consider arrival history, startup
   cost, and queue state together.

## Limitations

- The study uses one EC2 instance type, one NVIDIA L4 GPU, and one small model.
- Workloads are deterministic and steady; bursty, mixed, and production traces
  remain to be evaluated.
- Concurrency is fixed at one, and each run contains only 6–20 requests.
- Four repetitions provide limited uncertainty estimates. The report currently
  uses a normal 1.96 multiplier; a small-sample Student's *t* interval would
  be wider.
- Energy is integrated over each policy's own runtime. Always-On prewarms
  before the trace, while on-demand startup can overlap the trace horizon, so
  energy values are not fully apples-to-apples. A follow-up should use a common
  observation window or report startup and serving energy separately.
- P95 estimates from short traces should be interpreted as comparative signals,
  not production tail-latency guarantees.

## Result artifacts

Each run contains:

```text
requests.csv          per-request timing and cold-start measurements
gpu_samples.csv       raw GPU utilization, memory, power, and temperature
timeline.json         normalized startup, request, and GPU timestamps
summary.json          run-level latency, startup, GPU, and energy metrics
manifest.json         Git, Docker, model, hardware, and software provenance
effective-config.yaml exact resolved settings for that run
```

The generated `aggregate/` directory contains:

```text
runs.csv
policy-comparison.csv
report-table.md
report-manifest.json
latency-by-policy.svg
cold-start-rate.svg
slo-violation-rate.svg
energy-per-request.svg
```

Raw experiment results are intentionally excluded from Git because GPU sample
files can be large. Preserve them separately before terminating the EC2
instance.

## Repository layout

```text
configs/                  base, pilot, and report suite configurations
docker/                   pinned vLLM container image
src/serverless_llm/       simulator, real runtime, metrics, and reporting code
tests/                    unit and integration-style local tests
workloads/                reusable workload inputs
README.md                 English project documentation
README.ko.md              Korean project documentation
```

## Next steps

- Add rule-based adaptive keep-warm using observed inter-arrival time.
- Add an ML-based policy and compare it against the three baselines.
- Repeat the experiment with bursty and real request traces.
- Equalize energy observation windows and increase repetitions.
- Evaluate larger models, higher concurrency, and additional GPU types.
