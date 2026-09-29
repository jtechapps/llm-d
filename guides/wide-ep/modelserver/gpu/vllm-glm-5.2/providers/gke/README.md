# GKE Provider Overlay (`GLM-5.2-FP8`)

This provider overlay configures GKE-specific settings for DP-aware WideEP scheduling of `zai-org/GLM-5.2-FP8` on NVIDIA B200 (`a4-highgpu-8g`) or H200 (`a3-ultragpu-8g`) nodes with RoCE RDMA networking allocated via GKE-managed DRANET (`mrdma.google.com`).

## Summary of GKE-Specific Configuration

| Setting | Description |
| --- | --- |
| `ResourceClaimTemplate/gke-rdma-nic-template` | Allocates 8 RoCE RDMA NICs (`mrdma.google.com`) per pod (`gpu0rdma0`..`gpu7rdma0`). GPUs (`nvidia.com/gpu: "8"`) are allocated via the GKE NVIDIA device plugin. |
| GPU & Sandbox node tolerations | Tolerates `nvidia.com/gpu: NoSchedule` and `sandbox.gke.io/runtime: NoSchedule`. |
| Privileged container | `securityContext.privileged: true` on `vllm` container, required for GPU-initiated RDMA (IBGDA / NVSHMEM) on GKE. |
| GCE topology affinity | Prefers scheduling prefill and decode pods within the same `cloud.google.com/gce-topology-block` and `cloud.google.com/gce-topology-subblock`. |
| `TRITON_LIBCUDA_PATH` / `LD_LIBRARY_PATH` | Includes `/usr/local/nvidia/lib64` so Triton/DeepGemm JIT compilation and CUDA runtime locate GKE's host-mounted `libcuda.so.1`. |
| `NCCL_TUNER_PLUGIN` / `NCCL_NET_PLUGIN` | Disables GKE's built-in NCCL tuner (`none`) and net plugin (`""`) to use native RoCE IB verbs with `mlx5_0`..`mlx5_7`. |
| `UCX_IB_ROCE_REACHABILITY_MODE=all` | Enables cross-rail RoCE reachability in UCX/NIXL when prefill and decode ranks communicate across different NIC rails on GKE's full-mesh RoCE fabric. |
| `NVSHMEM_DISABLED_GDRCOPY=true` | Disables GDRCopy on GKE where `gdrdrv` is not loaded. |
| `BASH_ENV` RoCE / DeepEP startup hook | Applies an in-place startup fix before `vllm serve` launches: (1) zeroes `ibv_ah_attr.static_rate` in `nvshmem_transport_ibgda.so.3` (`nvidia-nvshmem-cu13 3.4.5`) so `ibv_create_ah` succeeds on GKE RoCE without `EINVAL`, (2) sets `NVSHMEM_ENABLE_NIC_PE_MAPPING=1` and `NVSHMEM_HCA_LIST=mlx5_<local_device_id>:1` in `deep_ep` so each GPU rank `0..7` binds 1-to-1 to its PCIe-aligned `mlx5_0`..`mlx5_7` NIC in both `deepep_high_throughput` and `deepep_low_latency` modes, and (3) sets `self.cudagraph_mode = CUDAGraphMode.NONE` in `vllm/config/compilation.py` when `all2all_backend == 'deepep_high_throughput'` and `data_parallel_size > 1` before the `CompilationMode.VLLM_COMPILE` early return so breakable CUDA graphs are disabled on prefill. |
| Local SSD hostPath caches | Mounts `hf-cache`, `jit-cache`, and `nvme-cache` under `/mnt/stateful_partition/kube-ephemeral-ssd/shared_disk/`. |
| `wide-ep-kv-cache-evictor` DaemonSet | Restricts the NVMe cache evictor DaemonSet to GPU nodes (`cloud.google.com/gke-gpu: "true"`) with matching tolerations and GKE local SSD path. |
