# GKE Overlay (`zai-org/GLM-5.3-Flash` Disaggregated Wide-EP: 2 Prefill + 2 Decode)

This overlay configures GKE-specific settings for DP-aware Wide Expert Parallelism (`DP=16, EP=16, TP=1` per role) serving [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B total / 18B active parameters, 288 routed experts = 18 routed experts per GPU) in **P/D disaggregated mode** (`2 prefill + 2 decode` pods) across **4 × 8-GPU nodes** (e.g., `a4-highgpu-8g` with 32 × NVIDIA B200 GPUs).

## Summary of GKE-Specific Patches

| Patch | Description |
|---|---|
| DRANET RDMA NIC claims | Requests eight `gke-rdma-nic-template` (`mrdma.google.com`) claims per pod (`gpu0rdma0`–`gpu7rdma0`), while GPUs (`nvidia.com/gpu: "8"`) are allocated via the GKE NVIDIA GPU device plugin. |
| Node tolerations | Tolerates `nvidia.com/gpu` and `sandbox.gke.io/runtime` taints on GPU node pools. |
| Privileged container | Required for GPU-initiated RDMA on GKE. |
| Topology affinity | Prefers the same GCE topology block and subblock (`cloud.google.com/gce-topology-block` / `cloud.google.com/gce-topology-subblock`) within each prefill and decode replica group. |
| `UCX_IB_ROCE_REACHABILITY_MODE=all` | Enables cross-rail RoCEv2 reachability for NIXL/UCX KV transfers across GKE DRANET RDMA NIC subnets. |
| `DEEP_EP_DEVICE_TO_HCA_MAPPING` | Maps each GPU index (`0..7`) to its paired Mellanox HCA (`mlx5_0..mlx5_7`). |
| `NVSHMEM_DISABLED_GDRCOPY` | Disables GDRCopy in favor of GPU-initiated IBGDA RDMA on GKE. |
| `HF_HUB_DOWNLOAD_TIMEOUT` / `HF_HUB_DISABLE_XET` | Avoids Hugging Face Xet CDN 429 rate-limiting when 4 pods pull weights concurrently. |
| Host volumes | Uses `/mnt/stateful_partition/kube-ephemeral-ssd/shared_disk/` for Hugging Face (`hf-cache`) and JIT (`jit-cache`) caches. |
| `NCCL_TUNER_PLUGIN` / `NCCL_NET_PLUGIN` | Disables GKE's built-in NCCL tuner and net plugin. |

## Cluster Prerequisites

Label all 4 GPU nodes so `gke-managed-networking-dra-driver` publishes `dra.net` `ResourceSlices`:

```bash
kubectl label nodes -l cloud.google.com/gke-accelerator=nvidia-b200 \
  cloud.google.com/gke-networking-dra-driver=true --overwrite
kubectl get resourceslices
```
