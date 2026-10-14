# 트래픽 인지형 서버리스 LLM 추론

[English](README.md) | [한국어](README.ko.md)

실제 GPU 인스턴스에서 서버리스 LLM의 warm-state 관리 정책에 따른 지연시간,
cold start, GPU 에너지 사용량, SLO(Service-Level Objective) 준수 여부를 측정하는
실험 프레임워크입니다.

완료된 AWS baseline 실험은 **4개 트래픽 패턴 × 3개 정책 × 4회 반복 = 총 48회**로
구성됩니다. NVIDIA L4 GPU에서 vLLM으로 TinyLlama를 실행하고 요청, 서버 시작,
GPU, 에너지 지표를 재현 가능한 결과 파일로 저장합니다.

## 연구 질문

요청 도착 간격이 달라질 때 GPU warm-state 관리 방식은 서버리스 LLM 추론의
지연시간과 에너지 사용량 사이의 trade-off에 어떤 영향을 주는가?

현재 완료된 실험에서는 다음 세 가지 baseline 정책을 비교합니다.

| 정책 | 동작 방식 |
|---|---|
| `always_on` | workload 시작 전에 vLLM을 실행하고 실험이 끝날 때까지 유지합니다. |
| `naive_serverless` | 요청마다 vLLM을 시작하고 요청 처리가 끝나면 즉시 종료합니다. |
| `fixed_keep_warm` | 요청 처리 후 vLLM을 30초 동안 유지합니다. |

Rule-Based와 ML-Based adaptive 정책은 후속 연구 대상으로, 아래 결과에는 포함되지
않았습니다.

## 시스템 구조

```text
실험 suite 설정
        |
        v
정책 실행기 ---> Docker/vLLM ---> streaming completion 응답
    |                  |                       |
    |                  v                       v
    +------------> nvidia-smi              요청 시간 측정
                       |                       |
                       +-----------+-----------+
                                   v
                          개별 결과 및 종합 보고서
```

Runtime은 vLLM 컨테이너를 시작·종료하고 HTTP readiness를 확인한 뒤 streaming
요청을 보냅니다. 별도 background thread는 GPU 상태를 주기적으로 측정합니다.
Suite runner는 정책 실행 순서를 순환시키고, 실제 적용된 설정과 실행 환경 정보를
보존한 뒤 여러 반복 실험을 표와 SVG 그래프로 집계합니다.

## 실험 환경

### 하드웨어 및 소프트웨어

| 구성 요소 | 설정 |
|---|---|
| Cloud instance | AWS EC2 `g6.xlarge` |
| GPU | NVIDIA L4, 23,034 MiB |
| 운영체제 | Ubuntu 22.04.5 LTS, x86_64 |
| Python | 3.10.12 |
| NVIDIA driver / CUDA | 595.91.07 / 13.2 |
| Docker | 29.8.1 |
| 추론 서버 | vLLM 0.19.0 |
| 모델 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| 모델 revision | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` |

### 요청 및 측정 통제 변수

| 설정 | 값 |
|---|---:|
| Endpoint | `/v1/completions` |
| Prompt | `Serverless computing is` |
| 최대 생성 token | 32 |
| Temperature | 0.0 |
| Streaming | 활성화 |
| 동시 요청 수 | 1 |
| GPU 측정 간격 | 200 ms |
| TTFT SLO | 1.0 s |
| Fixed keep-warm timeout | 30 s |

Docker image와 모델 revision은 고정되어 있으며, random seed와 실제 적용된 설정은
각 실행 결과에 함께 기록됩니다. 전체 설정은
[`configs/experiment.yaml`](configs/experiment.yaml)과
[`docker/Dockerfile.vllm`](docker/Dockerfile.vllm)에서 확인할 수 있습니다.

## 실험 설계 및 과정

30초 keep-warm timeout보다 훨씬 짧은 요청부터 훨씬 긴 요청까지 비교하도록 네
가지 steady workload를 사용했습니다.

| Workload | 요청 간격 | 실행당 요청 수 | 목적 |
|---|---:|---:|---|
| `dense-5s` | 5초 | 20 | Timeout보다 훨씬 자주 요청이 도착하는 상황입니다. |
| `boundary-25s` | 25초 | 8 | Timeout 직전에 다음 요청이 도착하는 상황입니다. |
| `boundary-35s` | 35초 | 8 | Timeout 직후에 다음 요청이 도착하는 상황입니다. |
| `sparse-90s` | 90초 | 6 | Timeout보다 훨씬 드물게 요청이 도착하는 상황입니다. |

각 workload와 정책의 조합을 네 번씩 반복했습니다.

```text
4 workloads × 3 policies × 4 repetitions = 48 runs
```

각 실행은 다음 순서로 진행됩니다.

1. 기본 설정에 workload와 정책별 설정을 적용합니다.
2. Git, Docker, 모델, GPU, 운영체제, Python 환경 정보를 기록합니다.
3. GPU monitoring을 시작하고 선택된 정책을 실행합니다.
4. Startup time, TTFT, generation time, end-to-end TTFT, cold start, SLO
   위반, GPU utilization, memory, power, energy를 측정합니다.
5. 컨테이너를 종료하고 Docker가 컨테이너 이름을 완전히 해제할 때까지 기다립니다.
6. 원본 측정값과 정규화한 결과 파일을 저장합니다.
7. 반복 실험의 평균, 표본 표준편차, 정규근사 95% 신뢰구간을 계산합니다.

정책 실행 순서는 반복 및 workload마다 순환하여 온도나 실행 시점에 따른 편향을
줄였습니다. 완료된 48회 실행 순서는
[`configs/report_suite_48.yaml`](configs/report_suite_48.yaml)에 정의되어 있습니다.

## 실험 재현 방법

### 1. 프로젝트 설치 및 로컬 검증

[`uv`](https://docs.astral.sh/uv/)를 설치하고 저장소를 clone한 뒤 실행합니다.

```bash
UV_CACHE_DIR=.cache/uv uv sync --extra dev
UV_CACHE_DIR=.cache/uv uv run pytest -q
```

단위 테스트에서는 필요한 Docker, HTTP, `nvidia-smi` 동작을 mock 처리하므로 GPU가
없는 로컬 환경에서도 프레임워크 대부분을 검증할 수 있습니다.

### 2. GPU 서버에서 vLLM image 생성

```bash
docker build \
  --file docker/Dockerfile.vllm \
  --tag serverless-llm-vllm:0.19.0 \
  .

docker volume create serverless-llm-hf-cache
```

Docker named volume은 다운로드한 Hugging Face 모델을 컨테이너가 종료된 뒤에도
보존합니다. 따라서 cold start마다 모델을 다시 다운로드하지 않습니다.

### 3. 설정 검사 및 pilot 실험

```bash
UV_CACHE_DIR=.cache/uv uv run serverless-llm-check configs/experiment.yaml

UV_CACHE_DIR=.cache/uv uv run serverless-llm-suite \
  configs/pilot_suite.yaml \
  --dry-run

UV_CACHE_DIR=.cache/uv uv run serverless-llm-suite \
  configs/pilot_suite.yaml
```

Pilot 실험으로 전체 GPU 실험 전에 Docker, vLLM, 요청 전송, GPU monitoring, 결과
저장, 보고서 생성 과정이 정상인지 확인할 수 있습니다.

### 4. 48회 본 실험

먼저 실제 실행 순서를 확인합니다.

```bash
UV_CACHE_DIR=.cache/uv uv run serverless-llm-suite \
  configs/report_suite_48.yaml \
  --dry-run
```

SSH 연결이 끊겨도 실험이 종료되지 않도록 `tmux` 안에서 실행합니다.

```bash
tmux new -s serverless-llm-report-48
set -o pipefail
PYTHONUNBUFFERED=1 UV_CACHE_DIR=.cache/uv \
  uv run serverless-llm-suite configs/report_suite_48.yaml \
  2>&1 | tee report-suite-48.log
```

`Ctrl-b`를 누른 다음 `d`를 누르면 실험을 중단하지 않고 tmux에서 빠져나옵니다.
다시 접속하려면 다음 명령을 사용합니다.

```bash
tmux attach -t serverless-llm-report-48
```

### 5. 종합 보고서 생성

```bash
UV_CACHE_DIR=.cache/uv uv run serverless-llm-report \
  results/traffic-aware-serverless-llm-report-48-v1
```

## 실험 결과

아래 값은 네 번의 반복 실험에 대한 **평균 ± 95% 신뢰구간 반폭**입니다. P95 E2E
TTFT에는 queueing과 cold-start 지연이 포함됩니다. 요청당 에너지는 각 정책의
관측 구간에서 측정한 GPU power를 적분하여 계산했습니다.

| Workload | 정책 | P95 E2E TTFT (s) | Cold-start 비율 | SLO 위반율 | 요청당 에너지 (Wh) |
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

### 통계값 읽는 방법

`TTFT`(Time To First Token)는 요청이 도착한 순간부터 첫 번째 생성 token이
client에 도착할 때까지 걸린 시간입니다. `E2E`는 사용자가 체감하는 전체 경로를
측정한다는 뜻이며 다음 시간이 모두 포함됩니다.

```text
E2E TTFT
= queue 대기 시간
+ container 시작 및 model loading
+ 앞선 요청의 처리 대기
+ 현재 요청의 첫 token 생성 시간
```

`P95`는 95번째 백분위수입니다. P95 E2E TTFT가 2초라면 요청의 약 95%가 2초
이내에 첫 token을 받았고, 가장 느린 약 5%는 2초보다 오래 걸렸다는 뜻입니다.
P95는 느린 쪽 요청의 tail latency를 나타내며 산술평균과는 다른 값입니다.

결과 표에는 네 번의 반복 실험에서 얻은 **P95의 평균**과 정규근사 95%
신뢰구간 반폭을 다음 형식으로 표시했습니다.

```text
평균 ± 95% 신뢰구간 반폭
```

예를 들어 `boundary-25s / fixed_keep_warm`에서 측정한 네 개의 P95는 다음과
같습니다.

```text
46.5655, 47.5014, 46.5489, 46.3956초
```

먼저 산술평균을 계산합니다.

```text
(46.5655 + 47.5014 + 46.5489 + 46.3956) / 4
= 46.7528초
```

신뢰구간 반폭은 다음 순서로 계산합니다.

```text
표본 표준편차 = 0.504847초
표준오차 = 0.504847 / √4 = 0.252423초
95% 신뢰구간 반폭 = 1.96 × 0.252423 = 0.494750초
```

따라서 최종 결과는 다음과 같습니다.

```text
46.7528 ± 0.4947초
95% 신뢰구간: [46.2581, 47.2476]초
```

P95와 95% 신뢰구간은 서로 다른 질문에 답합니다. P95는 한 번의 실행 안에서
느린 쪽 요청이 얼마나 오래 걸렸는지를 나타내고, 신뢰구간은 여러 번 측정한 P95
평균의 불확실성을 나타냅니다. 반폭이 작을수록 반복 결과가 더 일관적이었다는
뜻입니다. 이번 연구는 반복 횟수가 네 번으로 적기 때문에 신뢰구간을 모집단의
확정적인 범위가 아니라 불확실성의 참고 지표로 해석해야 합니다. 소표본에 적합한
Student's *t* 신뢰구간을 사용하면 현재의 정규근사 구간보다 더 넓어집니다.

정책별로 총 168개 요청을 처리했으며, 측정된 전체 GPU 에너지는 Always-On
39.895 Wh, Fixed Keep-Warm 35.956 Wh, Naive Serverless 88.846 Wh였습니다. 다만
정책별 관측 시간이 동일하지 않으므로 이 값은 엄격한 에너지 효율 순위가 아닌
기술적 비교로 해석해야 합니다.

## 핵심 발견

1. **Always-On의 지연시간과 안정성이 가장 우수했습니다.** 모든 workload에서
   cold start와 TTFT SLO 위반이 발생하지 않았습니다.
2. **Naive Serverless는 요청이 잦을 때 불안정했습니다.** 약 55초가 걸리는 cold
   start보다 5초 요청 간격이 훨씬 짧아 요청이 queue에 누적되었고, P95 E2E
   TTFT가 약 1,000초까지 증가했습니다.
3. **Cold start를 피했다고 반드시 SLO를 만족하는 것은 아닙니다.** Dense
   workload에서 Fixed Keep-Warm의 cold-start 비율은 5%였지만, 첫 cold start가
   뒤따르는 요청까지 지연시켜 SLO 위반율은 60%였습니다.
4. **30초 timeout의 효과는 workload에 크게 좌우됩니다.** 요청 간격이 5초,
   25초, 35초일 때는 Always-On보다 에너지 사용량이 감소했지만, 90초에서는 모든
   요청 전에 timeout이 만료되어 Naive Serverless와 거의 같아졌습니다.
5. **정책은 startup cost를 함께 고려해야 합니다.** 약 55초인 startup time보다
   짧은 timeout은 sparse traffic의 반복적인 cold start를 막지 못합니다. Adaptive
   정책은 요청 도착 기록뿐만 아니라 startup cost와 queue 상태도 고려해야 합니다.

## 실험의 한계

- 하나의 EC2 instance, NVIDIA L4 GPU, 소형 모델만 사용했습니다.
- 결정적인 steady workload만 사용했으며 burst, mixed, 실제 traffic trace는 아직
  평가하지 않았습니다.
- 동시 요청 수는 1이고, 실행당 요청 수는 6–20개입니다.
- 반복 횟수가 4회로 적습니다. 현재 보고서는 정규분포의 1.96 multiplier를
  사용하며, 소표본 Student's *t* 신뢰구간을 적용하면 구간이 더 넓어집니다.
- 에너지는 정책별 runtime을 기준으로 적분했습니다. Always-On은 trace 시작 전에
  prewarm하지만 on-demand startup은 trace 시간과 겹칠 수 있어 완전히 동일한
  조건의 에너지 비교는 아닙니다. 후속 실험에서는 동일 관측 구간을 사용하거나
  startup 에너지와 serving 에너지를 분리해야 합니다.
- 짧은 trace에서 계산한 P95는 정책 비교 지표이며 production tail-latency를
  보장하는 값은 아닙니다.

## 결과 파일

각 실행 폴더에는 다음 파일이 저장됩니다.

```text
requests.csv          요청별 지연시간 및 cold-start 측정값
gpu_samples.csv       GPU utilization, memory, power, temperature 원본 값
timeline.json         정규화한 startup, request, GPU timestamp
summary.json          실행 단위 latency, startup, GPU, energy 지표
manifest.json         Git, Docker, 모델, 하드웨어, 소프트웨어 실행 환경
effective-config.yaml 실제 실행에 적용된 전체 설정
```

생성된 `aggregate/` 폴더에는 다음 파일이 포함됩니다.

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

GPU sample 파일의 크기가 커질 수 있어 원본 실험 결과는 Git에서 제외합니다.
EC2 instance를 terminate하기 전에 별도 위치에 결과를 보관해야 합니다.

## 저장소 구조

```text
configs/                  기본 설정, pilot 및 본 실험 suite
docker/                   고정된 vLLM container image
src/serverless_llm/       simulator, 실제 runtime, metric, report 코드
tests/                    단위 및 integration-style 로컬 테스트
workloads/                재사용 가능한 workload 입력
README.md                 영문 프로젝트 문서
README.ko.md              한글 프로젝트 문서
```

## 다음 연구 단계

- 최근 요청 간격을 이용하는 Rule-Based adaptive keep-warm 정책을 추가합니다.
- ML-Based 정책을 구현하고 세 baseline과 비교합니다.
- Burst workload와 실제 요청 trace를 사용해 반복 실험합니다.
- 정책별 에너지 관측 구간을 통일하고 반복 횟수를 늘립니다.
- 더 큰 모델, 높은 concurrency, 다른 GPU에서 실험합니다.
