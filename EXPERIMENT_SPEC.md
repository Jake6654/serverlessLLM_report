# Traffic-Aware Serverless LLM Experiment Specification

## 1. Experiment Goal

이 프로젝트는 과거 실험의 수치를 재현하는 것을 목적으로 하지 않는다. 동일한
LLM, 하드웨어, 요청 workload와 측정 방법을 사용하여 다음 다섯 가지 GPU
warm-state 관리 전략을 새로 구현하고 비교한다.

1. Always-On
2. Naive Serverless
3. Fixed Keep-Warm
4. Rule-Based Adaptive Keep-Warm
5. ML-Based Adaptive Keep-Warm

핵심 목표는 각 정책이 요청 지연시간과 GPU 자원 사용량 사이의 trade-off에
어떤 영향을 주는지 측정하는 것이다.

## 2. Research Question

Can traffic-aware warm-state policies reduce unnecessary GPU active time while
maintaining acceptable LLM request latency compared with Always-On, Naive
Serverless, and Fixed Keep-Warm policies?

트래픽을 고려하는 Rule-Based 및 ML-Based 정책이 Always-On, Naive Serverless,
Fixed Keep-Warm과 비교하여 허용 가능한 요청 지연시간을 유지하면서 불필요한
GPU active time을 줄일 수 있는가?

## 3. Hypotheses

### Always-On

- 서버와 모델이 항상 준비되어 있으므로 가장 낮은 TTFT를 보일 것으로 예상한다.
- 요청이 없는 동안에도 GPU를 점유하므로 GPU active time이 가장 클 것으로
  예상한다.

### Naive Serverless

- 요청 처리가 끝나면 서버를 종료하므로 유휴 GPU active time이 가장 작을
  것으로 예상한다.
- 서버가 꺼진 상태에서 요청이 도착하면 cold start가 발생하므로 TTFT와 SLO
  위반율이 높을 것으로 예상한다.

### Fixed Keep-Warm

- 마지막 요청 후 고정된 timeout 동안 서버를 유지하므로 Naive Serverless보다
  cold start가 감소할 것으로 예상한다.
- 트래픽이 드문 구간에도 동일한 timeout을 사용하므로 불필요한 GPU active
  time이 발생할 수 있다.

### Rule-Based Adaptive Keep-Warm

- 최근 요청 간격을 이용해 timeout을 조정하므로 Fixed Keep-Warm보다 트래픽
  변화에 잘 대응할 것으로 예상한다.
- 단순한 규칙이므로 구현과 설명은 쉽지만 복잡한 트래픽 패턴에는 한계가 있을
  수 있다.

### ML-Based Adaptive Keep-Warm

- 최근 요청 패턴으로 가까운 미래의 요청 도착 확률을 예측하여 GPU 유지 여부를
  결정한다.
- 예측이 정확하다면 cold start를 억제하면서 GPU active time을 줄일 수 있지만,
  false positive와 false negative가 각각 자원 낭비와 cold start를 발생시킨다.

## 4. Experiment Strategies

### 4.1 Always-On

Workload 실행 전에 vLLM 서버를 시작하고 readiness check가 성공할 때까지
기다린다. 요청이 없는 구간에도 서버와 GPU 모델을 유지하며, 전체 workload가
끝난 후에만 서버를 종료한다.

### 4.2 Naive Serverless

요청이 도착했을 때 서버가 꺼져 있으면 vLLM 서버를 시작한다. 요청은 서버가
준비될 때까지 기다린다. 대기 중인 요청이 모두 처리되고 queue가 비면 서버를
종료한다.

### 4.3 Fixed Keep-Warm

마지막 요청 처리가 끝난 후 고정된 timeout 동안 서버를 유지한다. Timeout 안에
새 요청이 도착하면 타이머를 초기화한다. Timeout 동안 요청이 없으면 서버를
종료한다.

초기 timeout 후보는 10초, 30초, 60초이다. Validation workload에서 가장 좋은
결과를 보인 값을 최종 비교에 사용한다.

### 4.4 Rule-Based Adaptive Keep-Warm

최근 요청 간격의 exponential moving average(EMA)를 계산하고 이를 이용해
warm timeout을 동적으로 정한다.

```text
EMA = alpha * latest_interval + (1 - alpha) * previous_EMA
timeout = clip(k * EMA, minimum_timeout, maximum_timeout)
```

요청 간격이 짧아지면 서버를 더 오래 유지하고, 요청 간격이 길어지면 서버를
더 빨리 종료한다. `alpha`, `k`, 최소 및 최대 timeout은 validation workload에서
결정한다.

### 4.5 ML-Based Adaptive Keep-Warm

ML classifier가 앞으로 N초 안에 요청이 도착할 확률을 예측한다.

```text
if predicted_probability >= decision_threshold:
    KEEP_WARM
else:
    SHUTDOWN
```

첫 모델은 Logistic Regression으로 시작한다. 이후 필요하면 Random Forest와
XGBoost를 비교한다. 정확도만이 아니라 cold-start rate, GPU active time, TTFT,
SLO 위반율을 기준으로 최종 모델과 threshold를 선택한다.

## 5. Environment and Selection Rationale

| Component | Selected value | Why this is used |
|---|---|---|
| Cloud | Google Cloud GPU VM | NVIDIA L4를 사용할 수 있고 VM 환경을 반복해서 구성하기 쉽다. |
| Operating system | Ubuntu 22.04 LTS | NVIDIA driver, CUDA, Python 및 vLLM 지원이 안정적인 Linux 환경이다. |
| Python | 3.10.12 | vLLM 0.19.0과 주요 ML 라이브러리가 지원하며 기존보다 최신 문법에 과도하게 의존하지 않는 안정적인 버전이다. |
| GPU | NVIDIA L4 24GB | LLM 추론용 데이터센터 GPU이며 BF16을 지원하고 메모리, utilization, 전력을 `nvidia-smi`로 측정할 수 있다. |
| CUDA | 12.8 | vLLM 0.19.0의 사전 빌드 CUDA 환경과 맞추어 설치 및 호환성 문제를 줄인다. |
| Inference engine | vLLM 0.19.0 | OpenAI 호환 API, continuous batching, KV cache 관리와 streaming 응답을 제공해 LLM serving 실험에 적합하다. |
| GPU monitoring | `nvidia-smi` | 추가 하드웨어 없이 GPU utilization, memory, power를 일정 간격으로 수집할 수 있다. |

각 실행은 실제 환경을 `environment.txt`에 저장한다. 같은 실험 안에서는 위
버전을 변경하지 않는다. 버전을 변경하면 정책 효과와 소프트웨어 버전 효과를
구분하기 어렵기 때문이다.

## 6. Model Configuration and Selection Rationale

| Setting | Selected value | Why this is used |
|---|---|---|
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 초기 개발과 반복 실험 비용이 낮고 단일 L4에서 빠르게 실행할 수 있다. 전체 파이프라인이 완성된 후 더 큰 모델로 대표성을 검증한다. |
| Architecture | `LlamaForCausalLM` | 일반적인 decoder-only LLM 구조로 token 생성과 KV cache 동작을 실험할 수 있다. |
| Data type | `bfloat16` | L4가 지원하며 FP32보다 GPU 메모리 사용량을 줄이면서 별도 양자화 없이 추론할 수 있다. |
| Maximum model length | 2,048 tokens | TinyLlama의 기본 context 범위 안에서 메모리 사용량과 요청 조건을 통제한다. |
| Tensor parallel size | 1 | 단일 GPU 실험이므로 여러 GPU에 모델을 분할하지 않는다. |
| Pipeline parallel size | 1 | 단일 GPU에서 불필요한 분산 실행 변수를 제거한다. |
| Model revision | TODO: commit hash | 실험 중 모델 파일이 바뀌는 것을 방지하기 위해 정확한 revision을 고정한다. |
| GPU memory utilization | TODO | vLLM이 KV cache 등에 예약할 GPU 메모리 비율을 모든 정책에서 동일하게 유지한다. |
| Maximum concurrent sequences | TODO | 동시에 처리할 수 있는 요청 수를 통제하여 정책 간 비교를 공정하게 한다. |
| KV-cache data type | `auto` | 첫 실험에서는 모델 dtype과 vLLM 기본 동작을 사용하고, 별도 KV-cache 최적화는 독립 실험으로 분리한다. |

TinyLlama는 시스템을 개발하고 정책을 비교하기 위한 시작 모델이다. 이 모델의
결과만으로 모든 대형 LLM의 cold-start 특성을 일반화하지 않는다. 파이프라인이
안정화되면 L4 메모리에 맞는 7B 또는 8B 모델로 핵심 실험 일부를 반복한다.

## 7. Request Configuration and Selection Rationale

| Setting | Selected value | Why this is used |
|---|---|---|
| Endpoint | `/v1/completions` | 단순한 고정 prompt를 반복하여 정책 외의 변수를 줄일 수 있다. |
| Prompt | `Explain serverless computing in one sentence.` | 짧고 일정한 입력을 사용해 요청마다 비슷한 계산량을 유지한다. |
| Maximum output tokens | 32 | 출력 길이를 제한하여 정책 비교 중 생성 작업량이 크게 달라지는 것을 막는다. |
| Temperature | 0 | 출력 무작위성을 줄여 반복 실행의 차이를 줄인다. |
| Streaming | `true` | 전체 응답 완료 전 첫 token 도착 시각을 기록하여 TTFT를 측정한다. |
| Request timeout | TODO | Cold start를 포함할 만큼 길게 설정하되 무한 대기를 방지한다. |

모든 정책에는 동일한 request payload를 사용한다.

## 8. Workloads

정책의 특성은 요청 패턴에 따라 달라질 수 있으므로 한 가지 workload만으로
결론을 내리지 않는다.

| Workload | Initial definition | Why this is used |
|---|---|---|
| Steady | 5초마다 요청 1개 | 일정한 트래픽에서 각 정책의 기본 동작을 확인한다. |
| Bursty | 유휴 구간과 짧은 대량 요청 구간 반복 | 갑작스러운 요청 증가에 대한 cold start와 queue 처리를 확인한다. |
| Sparse | 30~180초 사이의 요청 간격 | 요청이 드문 상황에서 GPU 자원 절감 효과를 확인한다. |
| Mixed | Steady, bursty, sparse 구간 결합 | 트래픽 분포가 변할 때 adaptive 정책의 대응 능력을 확인한다. |

Workload generator는 고정된 random seed를 사용하고 생성된 요청 시각을 trace
파일로 저장한다. 동일한 trace를 다섯 정책에 replay한다.

## 9. Controlled and Independent Variables

### Independent variable

- Warm-state management policy

### Controlled variables

- GPU와 VM 사양
- 모델과 model revision
- Python, CUDA, vLLM 버전
- Request payload
- Workload trace
- vLLM memory 및 concurrency 설정
- GPU 측정 간격
- 실험 반복 횟수

### Dependent variables

- Startup time
- TTFT p50, p95, p99
- Total request latency
- Throughput
- Cold-start rate
- SLO violation rate
- GPU active time
- GPU memory와 utilization
- 평균 및 최대 power draw
- 추정 energy와 cost

## 10. Metric Definitions

### Startup Time

```text
startup_time = first_ready_time - server_process_start_time
```

### Time To First Token (TTFT)

```text
TTFT = first_token_received_time - request_arrival_time
```

Cold start를 경험한 요청에서는 서버 시작을 기다린 시간도 TTFT에 포함한다.

### Total Request Latency

```text
total_latency = last_token_received_time - request_arrival_time
```

### Cold Start

요청 도착 시 LLM 서버가 `READY` 상태가 아니어서 요청이 서버 시작을 기다린
경우로 정의한다.

### Cold-Start Rate

```text
cold_start_rate = requests_with_cold_start / total_requests
```

### GPU Active Time

vLLM worker가 GPU 모델 메모리를 점유한 누적 시간으로 정의한다. 구현 전에
GPU memory threshold를 정해 판정 규칙을 고정한다.

### Throughput

```text
throughput = successful_requests / elapsed_seconds
```

### SLO Violation Rate

```text
SLO threshold: TODO
SLO_violation_rate = requests_exceeding_SLO / total_requests
```

### Approximate GPU Energy

GPU power samples를 시간에 대해 적분하여 근사한다.

```text
energy_Wh = sum(power_W * sample_interval_seconds) / 3600
```

## 11. Cache Policy

첫 번째 비교는 모델 파일과 compilation artifact가 로컬에 존재하지만 vLLM
프로세스와 GPU 모델은 새로 시작하는 `process-cold` 조건을 사용한다. 완전히
비어 있는 다운로드 cache는 네트워크 속도 영향을 크게 받으므로 주 정책 비교와
분리한다.

각 실행에서 다음을 기록한다.

- Hugging Face model cache 존재 여부
- vLLM/PyTorch compilation cache 존재 여부
- 실행 전 vLLM 프로세스 상태
- 실행 전 GPU memory 사용량

## 12. Experiment Procedure

1. 사용할 config와 workload trace를 선택한다.
2. 고유한 run ID를 가진 결과 디렉터리를 생성한다.
3. 하드웨어, 소프트웨어, Git commit과 cache 상태를 기록한다.
4. GPU 모니터링을 시작한다.
5. 선택한 policy controller를 시작한다.
6. 동일한 workload trace를 replay한다.
7. 요청, 서버 상태 전이, 정책 결정과 GPU metric을 기록한다.
8. Workload가 끝나면 controller와 vLLM 서버를 안전하게 종료한다.
9. GPU 모니터링을 종료한다.
10. 실행별 summary를 생성한다.
11. 각 policy-workload 조합을 최소 3회 반복한다.

## 13. Expected Output Files

```text
results/<policy>/<workload>/<run_id>/
├── config.yaml
├── environment.txt
├── workload_trace.csv
├── server.log
├── request_metrics.csv
├── gpu_metrics.csv
├── policy_decisions.csv
└── summary.json
```

## 14. Known Limitations

- TinyLlama 1.1B 결과는 훨씬 큰 모델의 시작 동작을 완전히 대표하지 않는다.
- vLLM 프로세스 종료는 GPU VM 자체의 종료가 아니므로 클라우드 GPU 비용이
  실제로 0이 되지 않는다.
- 이 실험은 managed serverless GPU 서비스가 아니라 process-level warm-state
  management를 평가한다.
- 클라우드 VM 성능과 background workload가 실행마다 달라질 수 있다.
- Synthetic workload는 실제 production traffic을 완전히 대표하지 않는다.
- ML 성능은 학습에 사용한 traffic distribution에 의존한다.

## 15. Decisions Required Before Implementation

- [ ] Model revision 확정
- [ ] `gpu_memory_utilization` 확정
- [ ] `max_num_seqs` 확정
- [ ] Request timeout 확정
- [ ] Workload별 정확한 시간과 요청 수 확정
- [ ] GPU active 판정 threshold 확정
- [ ] TTFT SLO 확정
- [ ] Rule-Based EMA parameter 탐색 범위 확정
- [ ] ML prediction horizon 확정
- [ ] ML decision threshold 탐색 범위 확정
- [ ] 실제 GPU 실행 전 simulator의 검증 기준 확정
