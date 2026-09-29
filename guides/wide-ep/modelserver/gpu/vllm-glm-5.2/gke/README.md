# GKE Overlay (`GLM-5.2-FP8`)

This directory provides a convenience entrypoint (`-k guides/wide-ep/modelserver/gpu/vllm-glm-5.2/gke`) that deploys the 4-node / 32-GPU WideEP configuration ([`deployments/gke/p1w2d1w2`](../deployments/gke/p1w2d1w2/kustomization.yaml)):

- **Prefill:** 1 replica, 2 nodes, DEP16 (`TP=1, DP=16, EP=16`, `deepep_high_throughput`)
- **Decode:** 1 replica, 2 nodes, DEP16 (`TP=1, DP=16, EP=16`, `deepep_low_latency`)

For other GKE topologies (`p1w1d1w1`, `p1w1d1w2`, `p2w1d1w1`, `p2w1d1w1-precise`, `p2w1d1w2`, `p2w2d1w2`, `p2w2d2w2`, `p3w2d1w2`, `p3w2d2w2`), apply `-k deployments/gke/<deployment>`.

See [`providers/gke/README.md`](../providers/gke/README.md) for details on the GKE provider patches and [`../README.md`](../README.md) for full deployment instructions.
