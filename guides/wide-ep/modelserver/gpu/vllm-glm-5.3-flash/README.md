# GLM-5.3-Flash Wide-EP P/D Disaggregated (`vllm-glm-5.3-flash`)

## Overview

This guide deploys [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (`Glm5NextForConditionalGeneration`, 320B total / 18B active parameters, 288 routed experts) using vLLM's **Prefill/Decode (P/D) disaggregation** with NIXL (`NixlConnector`) in a wide expert parallel (`Wide-EP`) topology managed by a single `DisaggregatedSet`.

* **Prefill role (`size: 2` — 2 pods / 16 GPUs):** Runs `DP=16, EP=16, TP=1` across 2 × 8-GPU nodes (18 routed experts per GPU) using vLLM's DP Supervisor (`--data-parallel-multi-port-external-lb`, rank ports `8000`–`8007`, supervisor port `8008`).
* **Decode role (`size: 2` — 2 pods / 16 GPUs):** Runs `DP=16, EP=16, TP=1` across 2 × 8-GPU nodes (18 routed experts per GPU) with the `llm-d-routing-sidecar` (`routing-proxy` on ports `8000`–`8007` forwarding to vLLM decode worker ports `8200`–`8207`, supervisor port `8208`).
* **Hybrid KDA + Sparse NoPE MLA KV Transfer (`NixlConnector`):** `GLM-5.3-Flash` interleaves Kimi Delta Attention (`KDA` linear attention backed by `MambaSpec` conv + recurrent state) with sparse NoPE Multi-head Latent Attention (`MLA`). Both prefill and decode set `VLLM_SSM_CONV_STATE_LAYOUT=DS` so NIXL's 3-read Mamba/KDA conv state transfer decomposes and transfers the KDA conv/recurrent states alongside the FP8 MLA KV cache over RDMA.

## Default Configuration

| Parameter | Value |
| --- | --- |
| Model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| Architecture | `Glm5NextForConditionalGeneration` (Hybrid KDA Linear Attention + Sparse NoPE MLA, 288 routed experts) |
| Workload Topology | P/D Disaggregated (`DisaggregatedSet`: 2 Prefill pods + 2 Decode pods) |
| Prefill Parallelism | `DP=16, EP=16, TP=1` (2 × 8-GPU nodes = 16 GPUs, 18 routed experts/GPU) |
| Decode Parallelism | `DP=16, EP=16, TP=1` (2 × 8-GPU nodes = 16 GPUs, 18 routed experts/GPU) |
| Total GPUs | **32** (4 × 8-GPU nodes, e.g., GKE `a4-highgpu-8g` with 32 × NVIDIA B200 GPUs) |
| DP Mode | vLLM DP Supervisor (`--data-parallel-multi-port-external-lb`) |
| KV Cache Dtype | `fp8` (`--kv-cache-dtype fp8`) |
| KV Transfer | `NixlConnector` (`VLLM_SSM_CONV_STATE_LAYOUT=DS`, `UCX_IB_ROCE_REACHABILITY_MODE=all` on GKE RoCE) |
| All-to-All Backend | `allgather_reducescatter` |
| MoE Backend | `deep_gemm` |
| Max Model Length | `131072` (`MAX_MODEL_LEN`) |
| Reasoning / Tool Parsers | `glm47` / `glm47` (`--enable-auto-tool-choice`) |

### Supported Hardware Backends

| Backend | Directory | Notes |
| --- | --- | --- |
| NVIDIA GPU (Base) | `modelserver/gpu/vllm-glm-5.3-flash/base/` | Base `DisaggregatedSet` (2 prefill + 2 decode pods) |
| NVIDIA GPU (GKE) | `modelserver/gpu/vllm-glm-5.3-flash/gke/` | GKE A4 (`a4-highgpu-8g`, B200) / A3 Ultra (`a3-ultragpu-8g`, H200) with DRANET RoCE RDMA |

## Prerequisites

* Complete the shared [Wide-EP prerequisites](../../../README.md#prerequisites):
  * A Kubernetes cluster with **4 × 8-GPU RDMA-capable nodes** (32 GPUs total, e.g., 4 × GKE `a4-highgpu-8g` nodes with NVIDIA B200 GPUs and GKE managed DRANET enabled).
  * On GKE, label the GPU nodes with `cloud.google.com/gke-networking-dra-driver=true` so `gke-managed-networking-dra-driver` publishes `dra.net` `ResourceSlices` for all 4 nodes:
    ```bash
    kubectl label nodes -l cloud.google.com/gke-accelerator=nvidia-b200 \
      cloud.google.com/gke-networking-dra-driver=true --overwrite
    kubectl get resourceslices
    ```
  * LeaderWorkerSet controller `v0.10.0+` installed with `DisaggregatedSet` enabled (`--set enableDisaggregatedSet=true`).
  * Gateway API Inference Extension CRDs installed.
* Set the environment variables:
  ```bash
  export REPO_ROOT=$(realpath $(git rev-parse --show-toplevel))
  source ${REPO_ROOT}/guides/env.sh
  export GUIDE_NAME="wide-ep"
  export NAMESPACE="llm-d-wide-ep"
  export MODEL="zai-org/GLM-5.3-Flash"
  ```
* Create the namespace and `llm-d-hf-token` secret:
  ```bash
  kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
  kubectl create secret generic llm-d-hf-token \
    --from-literal=HF_TOKEN="${HF_TOKEN}" \
    -n ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
  ```

## Installation

### 1. Deploy the `llm-d` Router (Disaggregated P/D Mode)

#### Standalone Mode

```bash
helm upgrade --install ${GUIDE_NAME} \
    ${ROUTER_STANDALONE_CHART} \
    -f ${REPO_ROOT}/guides/recipes/router/base.values.yaml \
    -f ${REPO_ROOT}/guides/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml \
    -n ${NAMESPACE} --version ${ROUTER_CHART_VERSION}
```

<details>
<summary><b>Gateway Mode</b></summary>

```bash
export PROVIDER_NAME=gke # options: none, gke, agentgateway, istio
helm upgrade --install ${GUIDE_NAME} \
    ${ROUTER_GATEWAY_CHART} \
    -f ${REPO_ROOT}/guides/recipes/router/base.values.yaml \
    -f ${REPO_ROOT}/guides/recipes/router/features/httproute-flags.yaml \
    -f ${REPO_ROOT}/guides/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml \
    --set provider.name=${PROVIDER_NAME} \
    -n ${NAMESPACE} --version ${ROUTER_CHART_VERSION}
```

</details>

### 2. Deploy the Disaggregated `GLM-5.3-Flash` Model Server (2 Prefill + 2 Decode Pods)

```bash
export INFRA_PROVIDER=gke # options: base, gke
kubectl apply -n ${NAMESPACE} -k ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/vllm-glm-5.3-flash/${INFRA_PROVIDER}
```

Wait for both `prefill` (`0`, `0-1`) and `decode` (`0`, `0-1`) pods to reach `Ready`:

```bash
kubectl get disaggregatedsets,lws,pods -n ${NAMESPACE} -l llm-d.ai/model=GLM-5.3-Flash -w
```

Expected pods (2 prefill + 2 decode across 4 × 8-GPU nodes):

```text
NAME                                               READY   STATUS    RESTARTS
pod/wide-ep-nvidia-gpu-vllm-decode-xxxx-0          2/2     Running   0
pod/wide-ep-nvidia-gpu-vllm-decode-xxxx-0-1        2/2     Running   0
pod/wide-ep-nvidia-gpu-vllm-prefill-xxxx-0         1/1     Running   0
pod/wide-ep-nvidia-gpu-vllm-prefill-xxxx-0-1       1/1     Running   0
```

## Verification

### 1. Get the Router / Gateway IP

#### Standalone Mode

```bash
export IP=$(kubectl get service ${GUIDE_NAME}-epp -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
```

<details>
<summary><b>Gateway Mode</b></summary>

```bash
export IP=$(kubectl get gateway llm-d-inference-gateway -n ${NAMESPACE} -o jsonpath='{.status.addresses[0].value}')
```

</details>

### 2. Send Test Requests

```bash
curl -s http://${IP}/v1/models | jq
```

```bash
curl -s -X POST http://${IP}/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{
        \"model\": \"${MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Explain Prefill/Decode disaggregation in two sentences.\"}],
        \"max_tokens\": 256
    }" | jq
```

## Benchmarking

This guide includes [`inference-perf.yaml`](inference-perf.yaml), configured for [`inference-perf`](https://github.com/kubernetes-sigs/inference-perf) with `concurrency_level=2048`, `num_requests=8192`, and `2000 ISL / 2000 OSL` random tokens against `zai-org/GLM-5.3-Flash`.

> [!IMPORTANT]
> At `concurrency_level=2048` with 2,000-token prompts, ensure the `wide-ep-epp` router pod is scheduled on a node with sufficient CPU (e.g., `16` vCPUs / `16Gi`–`32Gi` memory) so prefix-cache hashing does not saturate smaller CPU nodes.

```bash
kubectl apply -n ${NAMESPACE} -f ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/vllm-glm-5.3-flash/inference-perf.yaml
```

### Benchmarking Results — GKE (4× `a4-highgpu-8g`, 32× NVIDIA B200 GPUs, RoCEv2 RDMA)

Full Stage 0 JSON output is saved in [`benchmark-results.json`](benchmark-results.json).

**Benchmark:** `2048_concurrent_2k_isl_2k_osl` (`concurrency_level=2048`, `num_requests=8192`, 2K input / 2K output tokens, 2 prefill + 2 decode pods, `DP=16, EP=16, TP=1` per role)

| Metric | Value |
| --- | --- |
| **Output tokens/s** | **37,884.6** |
| **Input tokens/s** | **37,568.0** |
| **Total tokens/s** | **75,452.6** |
| **Requests/s** | **18.8** |
| **Output tokens/s per decode GPU (16 GPUs)** | **~2,368** |
| **Completed requests** | **8,192 / 8,192 (0.0% error rate)** |
| **Benchmark duration** | **436.1 s** |

#### Latency & Token Generation Speed (`8,192 / 8,192` Requests)

| Metric | Min | P25 | Median (P50) | Mean | P75 | P90 | P95 | P99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Request Latency (s)** | 82.50 s | 99.91 s | **100.82 s** | **103.96 s** | 101.51 s | **124.38 s** | 137.52 s | 143.89 s | 145.65 s |
| **Normalized TPOT (ms/tok)** | 21.4 ms | 49.8 ms | **50.4 ms** | **52.0 ms** | 50.7 ms | **62.3 ms** | 68.7 ms | 72.0 ms | 1,027.8 ms |
| **Prompt Length (tokens)** | 2,000.0 | 2,000.0 | 2,000.0 | 2,000.0 | 2,000.0 | 2,000.0 | 2,000.0 | 2,000.0 | 2,000.0 |
| **Output Length (tokens)** | 98.0 | 2,000.0 | 2,001.0 | 2,016.9 | 2,004.0 | 2,011.0 | 2,018.0 | 2,084.2 | 4,026.0 |

### 2. Multi-Turn Agentic Prefix-Caching Benchmark (`conversation_replay` up to 100K Input Tokens)

[`inference-perf-agentic.yaml`](inference-perf-agentic.yaml) uses `inference-perf`'s `conversation_replay` generator (`streaming: true`, `max_model_len: 100000`) to simulate **32 concurrent multi-turn coding sessions** (`320` total requests). Each session has a `3,000`-token shared system prompt, a `10K–75K` dynamic codebase context (mean `40K`), and `5–25` turns (mean `12`) accumulating `500–6,000` input tokens and `100–1,000` output tokens per turn with `1–5s` tool execution round-trip sleeps.

```bash
kubectl apply -n ${NAMESPACE} -f ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/vllm-glm-5.3-flash/inference-perf-agentic.yaml
```

Full Stage 0 JSON output is saved in [`benchmark-results-agentic.json`](benchmark-results-agentic.json).

| Metric | Value |
| --- | --- |
| **Input tokens/s** | **80,360.4** |
| **Output tokens/s** | **666.9** |
| **Total tokens/s** | **81,027.3** |
| **Requests/s** | **1.42** (`32` concurrent sessions with `1–5s` tool-call intervals) |
| **Prefill GPU Prefix Cache Hit Rate** | **78.1%** (`14,133,248 / 18,107,443` prompt tokens served from GPU cache) |
| **Completed requests** | **320 / 320 (0.0% error rate)** |
| **Benchmark duration** | **225.3 s** |

#### Latency, TTFT, ITL & Token Length (`320 / 320` Streamed Requests)

| Metric | Min | P25 | Median (P50) | Mean | P75 | P90 | P95 | P99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **TTFT (s)** | 0.69 s | 1.41 s | **1.87 s** | **4.61 s** | 3.16 s | **12.54 s** | 26.74 s | 36.83 s | 41.53 s |
| **Inter-Token Latency (ms)** | 0.0 ms | 0.0 ms | **24.0 ms** | **19.6 ms** | 24.6 ms | **26.8 ms** | 31.0 ms | 157.6 ms | 951.6 ms |
| **TPOT (ms/tok)** | 0.4 ms | 25.9 ms | **27.4 ms** | **27.0 ms** | 28.6 ms | **29.5 ms** | 29.8 ms | 31.0 ms | 31.8 ms |
| **Request Latency (s)** | 5.06 s | 9.95 s | **12.79 s** | **16.06 s** | 18.55 s | **28.06 s** | 35.39 s | 52.76 s | 68.49 s |
| **Prompt Length (tokens)** | 23,082 | 41,589 | **54,266** | **56,586** | 69,536 | **86,484** | 92,748 | 98,050 | **98,687** |
| **Output Length (tokens)** | 99 | 207 | **282** | **470** | 409 | **607** | 811 | 993 | 43,183 |

## Cleanup

```bash
kubectl delete -n ${NAMESPACE} -k ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/vllm-glm-5.3-flash/${INFRA_PROVIDER}
helm uninstall ${GUIDE_NAME} -n ${NAMESPACE}
```
