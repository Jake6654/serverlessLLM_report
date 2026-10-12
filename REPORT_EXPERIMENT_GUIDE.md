# Report-ready AWS experiment guide

This guide runs the three implemented policies—Always-On, Naive Serverless,
and Fixed Keep-Warm—under the same workloads and produces report tables.

## 1. Update and validate the EC2 checkout

```bash
cd ~/serverlessLLM_report
git pull --ff-only
UV_CACHE_DIR=.cache/uv uv sync --extra dev
UV_CACHE_DIR=.cache/uv uv run pytest -q
```

`git pull --ff-only` updates the checkout only when Git can move it forward
without creating an unexpected merge commit. `uv sync` makes the virtual
environment match `pyproject.toml` and `uv.lock`.

## 2. Build and inspect the vLLM image

```bash
docker build \
  --file docker/Dockerfile.vllm \
  --tag serverless-llm-vllm:0.19.0 \
  .

docker image inspect serverless-llm-vllm:0.19.0 \
  --format '{{.Id}}'
```

The image ID is also captured automatically in every run manifest.

## 3. Remove an old experiment container, if one exists

```bash
docker rm -f serverless-llm-vllm 2>/dev/null || true
docker ps -a --filter 'name=^/serverless-llm-vllm$'
```

The second command should show only the table header. The runtime now waits
for Docker to release this name after every stop, which prevents the rapid
stop/start race previously seen with Naive Serverless.

## 4. Preview and run the pilot suite

```bash
UV_CACHE_DIR=.cache/uv uv run serverless-llm-suite \
  configs/pilot_suite.yaml \
  --dry-run

UV_CACHE_DIR=.cache/uv uv run serverless-llm-suite \
  configs/pilot_suite.yaml
```

The pilot contains three GPU runs: one workload, three policies, and one
repetition. Use it to validate the complete pipeline before paying for the
full experiment.

## 5. Inspect pilot artifacts

```bash
find results/traffic-aware-serverless-llm-pilot-v2 \
  -maxdepth 5 -type f | sort
```

Each repetition contains:

- `requests.csv`: per-request latency and cold-start measurements.
- `gpu_samples.csv`: raw GPU utilization, memory, power, and temperature.
- `timeline.json`: normalized startup/request/GPU timestamps.
- `summary.json`: run-level request, startup, GPU, and energy metrics.
- `manifest.json`: policy order, seed, model revision, Git, Docker, GPU, OS,
  and Python provenance.
- `effective-config.yaml`: the exact resolved settings used by that run.

## 6. Generate the pilot report

```bash
UV_CACHE_DIR=.cache/uv uv run serverless-llm-report \
  results/traffic-aware-serverless-llm-pilot-v2
```

The `aggregate/` directory then contains run-level CSV data, policy-level
means, sample standard deviations, 95% confidence intervals, a Markdown
table, and four SVG figures. A one-repetition pilot cannot estimate variance,
so its standard deviation and confidence interval are blank by design.

## 7. Run the full baseline suite

First inspect the 75-run order:

```bash
UV_CACHE_DIR=.cache/uv uv run serverless-llm-suite \
  configs/report_suite.yaml \
  --dry-run
```

Then run it inside `tmux` so an SSH disconnect does not terminate the job:

```bash
tmux new -s serverless-llm-report
UV_CACHE_DIR=.cache/uv uv run serverless-llm-suite \
  configs/report_suite.yaml 2>&1 | tee report-suite.log
```

Detach with `Ctrl-b`, then `d`. Reattach later with:

```bash
tmux attach -t serverless-llm-report
```

The suite compares five traffic regimes: 5 seconds, the 25/30/35-second
boundary around the Fixed Keep-Warm timeout, and a sparse 90-second interval.
Each combination is repeated five times. Policy order rotates between
repetitions/workloads to reduce systematic thermal or temporal ordering bias.

## 8. Generate final report artifacts

```bash
UV_CACHE_DIR=.cache/uv uv run serverless-llm-report \
  results/traffic-aware-serverless-llm-report-v1
```

The main files for analysis are:

```text
aggregate/runs.csv
aggregate/policy-comparison.csv
aggregate/report-table.md
aggregate/report-manifest.json
aggregate/*.svg
```

## 9. Preserve results outside the EC2 instance

`results/` is intentionally ignored by Git because raw GPU samples can grow
large and experiment data should not be mixed with source commits. From the
Mac, copy the result directory before stopping or terminating EC2:

```bash
scp -i ~/Downloads/serverless-llm-virginia.pem \
  -r ubuntu@YOUR_EC2_HOST:~/serverlessLLM_report/results/traffic-aware-serverless-llm-report-v1 \
  ~/Downloads/
```

Stopping EC2 preserves EBS data; terminating EC2 may delete the root EBS
volume when “Delete on termination” is enabled.
