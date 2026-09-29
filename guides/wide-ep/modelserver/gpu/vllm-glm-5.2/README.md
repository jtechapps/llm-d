# GLM-5.2-FP8 on H200 / B200

## Overview

This guide deploys [GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) (753B MoE) on NVIDIA
H200 or B200 GPUs using a P/D-disaggregated DisaggregatedSet with NIXL for KV transfer. Prefill runs
DEP8 (TP=1, DP=8) on 1 node or DEP16 (TP=1, DP=16) across 2 nodes; decode runs DEP16 (TP=1, DP=16)
across 2 nodes (wide EP). DeepEP high-throughput all-to-all for prefill, low-latency for decode.

DeepGemm MoE backend, tool calling (`glm47`) and reasoning (`glm45`) parsers.
MTP speculative decoding is on by default (3 tokens).

Tested on CoreWeave (CKS) with InfiniBand networking and Google Kubernetes Engine (GKE) on
`a4-highgpu-8g` (NVIDIA B200) with RoCE RDMA networking (GKE-managed DRANET). This recipe reuses the
[wide-ep guide](../../../README.md) for the router/gateway and shared prerequisites
(namespace, HF token secret, LeaderWorkerSet controller with DisaggregatedSet enabled).

## Default Configuration

| Parameter               | Value                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------- |
| Model                   | [zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8)                |
| Accelerator             | NVIDIA H200 or NVIDIA B200 (8 GPUs per node)                                       |
| DP model                | Supervisor (`--data-parallel-multi-port-external-lb`)                              |
| Prefill parallelism     | TP=1, DP=8, EP=8 (DEP8, 1 node) or TP=1, DP=16, EP=16 (DEP16, 2 nodes)            |
| Decode parallelism      | TP=1, DP=16, EP=16 (DEP16, wide) — 2 nodes                                        |
| All-to-all (prefill)    | `deepep_high_throughput`                                                           |
| All-to-all (decode)     | `deepep_low_latency` (IBGDA + NVSHMEM)                                            |
| MoE backend             | DeepGemm                                                                           |
| KV transfer             | NixlConnector                                                                      |
| KV cache offloading     | Off (opt-in via components)                                                        |
| MTP speculative decoding | On (3 tokens; opt-out via `no-mtp` component)                                     |
| Prefill `gpu-memory-utilization` | 0.935 (CoreWeave single-node) / 0.85 (GKE) / 0.80 (multi-node)           |
| Decode `gpu-memory-utilization`  | 0.95 (CoreWeave single-node) / 0.85 (GKE) / 0.80 (multi-node)            |
| Reasoning / tool-call   | glm45 / glm47                                                                     |

### P/D Deployment Options

| Deployment | Prefill                    | Decode                        | Nodes / GPUs | CoreWeave Path | GKE Path |
| ---------- | -------------------------- | ----------------------------- | ------------ | -------------- | -------- |
| `p1w1d1w1` | 1 replica, 1 node, DEP8    | 1 replica, 1 node, DEP8      | 2 / 16       | `deployments/p1w1d1w1` | `deployments/gke/p1w1d1w1` |
| `p1w1d1w2` | 1 replica, 1 node, DEP8    | 1 replica, 2 nodes, DEP16    | 3 / 24       | `deployments/p1w1d1w2` | `deployments/gke/p1w1d1w2` |
| `p1w2d1w2` | 1 replica, 2 nodes, DEP16  | 1 replica, 2 nodes, DEP16    | 4 / 32       | `deployments/p1w2d1w2` | `deployments/gke/p1w2d1w2` (or `gke/`) |
| `p2w1d1w1` | 2 replicas, 1 node, DEP8   | 1 replica, 1 node, DEP8      | 3 / 24       | `deployments/p2w1d1w1` | `deployments/gke/p2w1d1w1` |
| `p2w1d1w1-precise` | 2 replicas, 1 node, DEP8 | 1 replica, 1 node, DEP8  | 3 / 24       | `deployments/p2w1d1w1-precise` | `deployments/gke/p2w1d1w1-precise` |
| `p2w1d1w2` | 2 replicas, 1 node, DEP8   | 1 replica, 2 nodes, DEP16    | 4 / 32       | `deployments/p2w1d1w2` | `deployments/gke/p2w1d1w2` |
| `p2w2d1w2` | 2 replicas, 2 nodes, DEP16 | 1 replica, 2 nodes, DEP16    | 6 / 48       | `deployments/p2w2d1w2` | `deployments/gke/p2w2d1w2` |
| `p2w2d2w2` | 2 replicas, 2 nodes, DEP16 | 2 replicas, 2 nodes, DEP16   | 8 / 64       | `deployments/p2w2d2w2` | `deployments/gke/p2w2d2w2` |
| `p3w2d1w2` | 3 replicas, 2 nodes, DEP16 | 1 replica, 2 nodes, DEP16    | 8 / 64       | `deployments/p3w2d1w2` | `deployments/gke/p3w2d1w2` |
| `p3w2d2w2` | 3 replicas, 2 nodes, DEP16 | 2 replicas, 2 nodes, DEP16   | 10 / 80      | `deployments/p3w2d2w2` | `deployments/gke/p3w2d2w2` |

### Supported Hardware Backends

| Backend | Provider Overlay | Deployments | Notes |
| --- | --- | --- | --- |
| NVIDIA GPU (CoreWeave) | `providers/coreweave` | `deployments/<deployment>` | H200, InfiniBand (`rdma/ib`), P/D disaggregated |
| NVIDIA GPU (GKE) | [`providers/gke`](providers/gke/README.md) | `deployments/gke/<deployment>` or [`gke/`](gke/README.md) | B200 (`a4-highgpu-8g`) or H200 (`a3-ultragpu-8g`), RoCE DRANET (`mrdma.google.com`), P/D disaggregated |

## Components

Add [kustomize Components](https://kubectl.docs.kubernetes.io/guides/config_management/components/)
to a deployment's `kustomization.yaml` under `components:`.

| Component | Targets | Effect |
| --------- | ------- | ------ |
| `no-mtp` | prefill + decode | Disables MTP speculative decoding (`ENABLE_MTP=0`) |
| `offloading-cpu` | prefill only | CPU-only KV cache offloading (`OFFLOADING_MODE=cpu`) |
| `offloading-tiered` | prefill only | CPU + NVMe tiered KV cache offloading (`OFFLOADING_MODE=tiered`) |

Component env entries merge by name, so their values replace the base defaults.

## Prerequisites

In addition to the [wide-ep prerequisites](../../../README.md#prerequisites):

```bash
export KUBECONFIG=~/.kube/config
export NAMESPACE=<your-namespace>
export MODEL=zai-org/GLM-5.2-FP8
```

When installing the `wide-ep` router Helm release (Step 1 of the [wide-ep guide](../../../README.md#1-deploy-the-llm-d-router)), include `-f ${REPO_ROOT}/guides/wide-ep/router/glm-5.2-overrides.values.yaml` after `wide-ep.values.yaml`:

```bash
helm upgrade --install ${GUIDE_NAME} \
    ${ROUTER_STANDALONE_CHART} \
    -f ${REPO_ROOT}/guides/recipes/router/base.values.yaml \
    -f ${REPO_ROOT}/guides/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml \
    -f ${REPO_ROOT}/guides/${GUIDE_NAME}/router/glm-5.2-overrides.values.yaml \
    -n ${NAMESPACE} --version ${ROUTER_CHART_VERSION}
```

## Deploy the Model Server

### P/D Disaggregated

Pick a deployment from the [P/D Deployment Options](#pd-deployment-options) table and apply:

**CoreWeave (InfiniBand):**

```bash
kubectl apply -n ${NAMESPACE} -k deployments/<deployment>
```

**GKE (RoCE DRANET):**

```bash
# Default 4-node / 32-GPU topology (p1w2d1w2):
kubectl apply -n ${NAMESPACE} -k gke

# Or pick any GKE topology from deployments/gke/<deployment>:
kubectl apply -n ${NAMESPACE} -k deployments/gke/<deployment>
```

Wait for pods to become ready (model load takes time; the startup probe allows up to 45 minutes):

```bash
kubectl get pods -n ${NAMESPACE} -l llm-d.ai/model=GLM-5.2-FP8 -w
```

## Verification

### 1. Get the IP of the Proxy

```bash
export IP=$(kubectl get service wide-ep-epp -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
```

### 2. Send Test Requests

Open a temporary shell inside the cluster:

```bash
kubectl run curl-debug --rm -it \
    --image=cfmanteiga/alpine-bash-curl-jq \
    --env="IP=$IP" \
    --env="NAMESPACE=$NAMESPACE" \
    -- /bin/bash
```

Send a completion request:

```bash
curl -X POST http://${IP}/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "zai-org/GLM-5.2-FP8",
        "prompt": "How are you today?"
    }' | jq
```

## Benchmarking

These manifests back the [agentic-serving GLM-5.2 guide](../../../../agentic-serving/glm-5-2-h200.md);
benchmark results, the workload description, and key takeaways are in its
[Benchmark Results](../../../../agentic-serving/glm-5-2-h200.md#benchmark-results) section, with
the full analysis and figures in the
[blog post](https://llm-d.ai/blog/serving-glm-5-2-agentic-workloads-on-llm-d).

### Benchmark Overlays

Pre-built overlays under `deployments/benchmark/<config>/<topology>/` combine components with
topology patches. Each matches a tested configuration on CoreWeave H200:

| Configuration | Directory | Components |
| ------------- | --------- | ---------- |
| Baseline | `benchmark/baseline/` | `no-mtp` |
| MTP + Offloading | `benchmark/mtp-offloading/` | `offloading-tiered` |
| Offloading | `benchmark/offloading/` | `no-mtp` + `offloading-tiered` |
| Full ISL + MTP + Offloading | `benchmark/full-isl-mtp-offloading/` | `offloading-tiered` |

Deploy a benchmark config:

```bash
kubectl apply -n ${NAMESPACE} -k deployments/benchmark/<config>/<topology>
```

Example overlay (`mtp-offloading/p1w1d1w1`):

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../../../providers/coreweave
components:
  - ../../../../components/offloading-tiered
patches:
  - target:
      group: disaggregatedset.x-k8s.io
      kind: DisaggregatedSet
    patch: |-
      # Roles are ordered [prefill, decode] in the base DisaggregatedSet.
      - op: replace
        path: /spec/roles/0/spec/replicas
        value: 1
      - op: replace
        path: /spec/roles/0/spec/leaderWorkerTemplate/size
        value: 1
      - op: replace
        path: /spec/roles/1/spec/replicas
        value: 1
      - op: replace
        path: /spec/roles/1/spec/leaderWorkerTemplate/size
        value: 1
```

### aiperf Command

Every reported number comes from the same [aiperf](https://github.com/ai-dynamo/aiperf) `profile`
invocation, run from inside the cluster against the Kubernetes Gateway Service
(`llm-d-inference-gateway-istio`, from [Gateway Mode](../../../README.md#gateway-mode) with
`PROVIDER_NAME=istio`) and swept across concurrency. The dataset used is the
`semianalysis_cc_traces_weka_with_subagents` aiperf preset, backed by
[`semianalysisai/cc-traces-weka-062126`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)
on HuggingFace.

```bash
aiperf profile \
    --scenario 'inferencex-agentx-mvp' \
    --url 'http://llm-d-inference-gateway-istio:80/v1' \
    --model 'zai-org/GLM-5.2-FP8' \
    --max-context-length <142000|10000000> \
    --endpoint-type 'chat' \
    --streaming \
    --use-server-token-count \
    --public-dataset 'semianalysis_cc_traces_weka_with_subagents' \
    --concurrency <16|32|64|128|256|512> \
    --random-seed 42 \
    --benchmark-duration 900 \
    --server-metrics 'http://llm-d-inference-gateway-istio:80/metrics' \
    --no-gpu-telemetry \
    --output-artifact-dir <path> \
    --ui 'simple'
```

Only `--max-context-length` and the `--concurrency` sweep differ across the four reported
configurations:

| Configuration | `--max-context-length` | `--concurrency` sweep |
| --------------------------- | ----------------------- | ------------------------- |
| Baseline | `142000` | 16, 32, 64, 128 |
| MTP + Offloading | `142000` | 16, 32, 64, 128 |
| Offloading | `142000` | 16, 32, 64, 128 |
| Full ISL + MTP + Offloading | `10000000` | 16, 32, 64, 128, 256, 512 |

`142000` truncates the trace dataset to requests that fit an EP8 (1-node) prefill deployment.
`10000000` is effectively unbounded and replays full traces (up to ~1M input tokens) — this is
the "Full ISL" config. At concurrency 256/512, some Full ISL runs on smaller topologies exceeded
server capacity (warmup failures) and are excluded from the reported results.

### GKE `inference-perf` Benchmarks (4× `a4-highgpu-8g`, 32× NVIDIA B200 GPUs, RoCEv2 RDMA)

This guide also includes two [`inference-perf`](https://github.com/kubernetes-sigs/inference-perf) benchmark manifests tested against the default GKE `p1w2d1w2` deployment (`2` prefill pods + `2` decode pods, `DP=16, EP=16, TP=1` per role, MTP speculative decoding `3` tokens):

#### 1. 2048-Concurrency 2K ISL / 2K OSL Random Tokens (`inference-perf.yaml`)

[`inference-perf.yaml`](inference-perf.yaml) runs [`inference-perf`](https://github.com/kubernetes-sigs/inference-perf) with `concurrency_level=2048`, `num_requests=8192`, and `2000 ISL / 2000 OSL` random tokens against `zai-org/GLM-5.2-FP8`.

> [!IMPORTANT]
> At `concurrency_level=2048`, route requests through the Kubernetes Gateway (`http://${GATEWAY_IP}`) or ensure the `wide-ep-epp` router pod is scheduled on a node with sufficient CPU (`16` vCPUs / `16Gi`–`32Gi` memory) and Envoy `max_concurrent_streams >= 2048`. In the base [`disaggregatedset.yaml`](base/disaggregatedset.yaml), decode pods set `MAX_TOKENS_PER_NODE=32` (`MAX_TOKENS=64` per GPU rank with `MTP_NUM_TOKENS=3`), capping each decode GPU to `16` concurrent active sequences (`256` active across `16` decode GPUs).

```bash
kubectl apply -n ${NAMESPACE} -f ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/vllm-glm-5.2/inference-perf.yaml
```

Full Stage 0 JSON output is saved in [`benchmark-results.json`](benchmark-results.json).

| Metric | Value |
| --- | --- |
| **Output tokens/s** | **6,187.2** |
| **Input tokens/s** | **6,096.5** |
| **Total tokens/s** | **12,283.6** |
| **Requests/s** | **3.05** |
| **Output tokens/s per decode GPU (16 GPUs)** | **~386.7** |
| **Completed requests** | **8,192 / 8,192 (0.0% error rate)** |
| **Benchmark duration** | **2,687.5 s** |

##### Latency & Token Generation Speed (`8,192 / 8,192` Requests)

| Metric | Min | P25 | Median (P50) | Mean | P75 | P90 | P95 | P99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Request Latency (s)** | 34.33 s | 610.39 s | **653.77 s** | **586.47 s** | 674.06 s | **686.45 s** | 693.29 s | 705.73 s | 742.30 s |
| **Normalized TPOT (ms/tok)** | 17.2 ms | 303.6 ms | **324.2 ms** | **289.9 ms** | 333.6 ms | **339.8 ms** | 343.2 ms | 349.8 ms | 390.1 ms |
| **Prompt Length (tokens)** | 2,000.0 | 2,000.0 | 2,000.0 | 2,000.0 | 2,000.0 | 2,000.0 | 2,000.0 | 2,000.0 | 2,000.0 |
| **Output Length (tokens)** | 1,550.0 | 2,007.0 | 2,016.0 | 2,029.8 | 2,023.0 | 2,028.0 | 2,031.0 | 2,040.1 | 4,036.0 |

#### 2. Multi-Turn Agentic Prefix-Caching Benchmark (`conversation_replay` up to 100K Input Tokens)

[`inference-perf-agentic.yaml`](inference-perf-agentic.yaml) uses `inference-perf`'s `conversation_replay` generator (`streaming: true`, `max_model_len: 100000`) to simulate **32 concurrent multi-turn coding sessions** (`320` total requests). Each session has a `3,000`-token shared system prompt, a `10K–75K` dynamic codebase context (mean `40K`), and `5–25` turns (mean `12`) accumulating `500–6,000` input tokens and `100–1,000` output tokens per turn with `1–5s` tool execution round-trip sleeps.

```bash
kubectl apply -n ${NAMESPACE} -f ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/vllm-glm-5.2/inference-perf-agentic.yaml
```

Full Stage 0 JSON output is saved in [`benchmark-results-agentic.json`](benchmark-results-agentic.json).

| Metric | Value |
| --- | --- |
| **Input tokens/s** | **91,429.0** |
| **Output tokens/s** | **759.6** |
| **Total tokens/s** | **92,188.6** |
| **Requests/s** | **1.62** (`32` concurrent sessions with `1–5s` tool-call intervals) |
| **Prefill GPU Prefix Cache Hit Rate** | **86.2%** (`15,606,080 / 18,108,104` prompt tokens served from GPU cache) |
| **Completed requests** | **320 / 320 (0.0% error rate)** |
| **Benchmark duration** | **198.1 s** |

##### Latency, TTFT, ITL & Token Length (`320 / 320` Streamed Requests)

| Metric | Min | P25 | Median (P50) | Mean | P75 | P90 | P95 | P99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **TTFT (s)** | 0.67 s | 1.25 s | **1.56 s** | **3.30 s** | 1.96 s | **7.47 s** | 19.83 s | 25.55 s | 27.75 s |
| **Inter-Token Latency (ms)** | 0.0 ms | 0.0 ms | **25.3 ms** | **19.9 ms** | 26.2 ms | **28.3 ms** | 30.7 ms | 83.1 ms | 3,672.8 ms |
| **TPOT (ms/tok)** | 0.4 ms | 25.3 ms | **25.8 ms** | **29.6 ms** | 26.1 ms | **26.9 ms** | 46.5 ms | 147.4 ms | 191.2 ms |
| **Request Latency (s)** | 5.43 s | 9.13 s | **11.42 s** | **14.90 s** | 15.93 s | **31.16 s** | 36.12 s | 43.75 s | 51.70 s |
| **Prompt Length (tokens)** | 23,082 | 41,612 | **54,262** | **56,588** | 69,542 | **86,482** | 92,750 | 98,047 | **98,678** |
| **Output Length (tokens)** | 100 | 207 | **282** | **470** | 409 | **608** | 803 | 996 | 43,196 |

## Optional Features

### MTP Speculative Decoding

On by default (3 tokens) for both prefill and decode. Disable with the `no-mtp`
component or `ENABLE_MTP=0`. Token count: `MTP_NUM_TOKENS` (default `3`).

### EPP Routing

The GLM-5.2 EPP overrides (`router/glm-5.2-overrides.values.yaml`) replace the wide-ep
prefix-cache scorer with dual prefix-cache scoring for P/D routing — include the file after
`wide-ep.values.yaml` when installing the router:

- **GPU prefix-cache scorer** (weight 5) — auto-tuned, tracks GPU-resident prefix blocks
- **CPU prefix-cache scorer** (weight 2) — fixed LRU capacity (200k entries per server),
  tracks CPU-offloaded prefix blocks
- **Active-request scorer** (weight 1 prefill, 3 decode) — load balancing

All 8 DP rank ports (8000-8007) are exposed as `targetPorts` for per-rank routing.

### KV Cache Offloading (Prefill)

Off by default. Enable via the `offloading-cpu` or `offloading-tiered` component.

- **`offloading-cpu`** — CPU-only offloading via `OffloadingConnector`. Uses mmap in
  `/dev/shm`. The pod allocates 1500Gi memory and 1500Gi `dshm` to accommodate 8 DP
  ranks' mmap regions. `cpu_bytes_to_use` is per-rank — total CPU KV cache = value x 8.
- **`offloading-tiered`** — CPU + NVMe tiered offloading via `TieringOffloadingSpec`.
  Same CPU tier as above, plus NVMe as a secondary eviction target. Host-path volume at
  `/mnt/local/kv-cache` mounted as `/mnt/nvme-cache`.

Decode pods do not use offloading (256Gi dshm, 512Gi memory).

### InfiniBand Networking

Both prefill and decode configure IB for multi-node communication:

| Variable                  | Value  | Purpose                                          |
| ------------------------- | ------ | ------------------------------------------------ |
| `NCCL_IB_HCA`            | `ibp`  | Filter IB HCAs for NCCL collectives              |
| `NVSHMEM_HCA_PREFIX`      | `ibp`  | Filter IB HCAs for NVSHMEM (decode low-latency)  |
| `NVSHMEM_REMOTE_TRANSPORT` | `ibgda` | GPUDirect Async for NVSHMEM                     |
| `rdma/ib`                 | `8`    | Request 8 RDMA/IB devices per pod                |

Multi-node deployments (`LWS_GROUP_SIZE > 1`) automatically set `NVSHMEM_SYMMETRIC_SIZE=16G`
and reduce `gpu-memory-utilization` to 0.80 to reserve VRAM for the NVSHMEM heap.

### KV Cache Evictor

`base/kv-cache-evictor.yaml` deploys a DaemonSet that evicts stale KV cache data from NVMe
when utilization exceeds 90%, targeting 70%.

### Monitoring

Node-exporter sidecars on each pod collect InfiniBand, CPU, memory pressure, and network
retransmission metrics. Apply Prometheus scrape configs:

```bash
NAMESPACE=${NAMESPACE} bash guides/wide-ep/monitoring/apply-scrape-configs.sh
```

DCGM custom metrics: `base/dcgm-custom-metrics.yaml`.

## Cleanup

```bash
kubectl delete -n ${NAMESPACE} -k deployments/<deployment>
```
