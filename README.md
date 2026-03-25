# PyTorch DDP Baseline (Distributed Training)

## Why this artifact matters

DistributedDataParallel (DDP) is the practical default for multi-GPU training in PyTorch.
For ML Engineer / MLOps roles, interviewers expect you to understand not just model code, but process topology, synchronization costs, and reliability/debug patterns.

This note focuses on an implementation you can run locally and explain confidently.

---

## 1) DDP mental model

Core terms:

- **world size**: total number of training processes
- **rank**: unique process id in `[0, world_size-1]`
- **local rank**: GPU index within one machine
- **global batch size**: `per_device_batch * world_size`

How DDP works:

1. each rank has a full model replica
2. each rank processes a different mini-batch shard
3. gradients are all-reduced across ranks during backward
4. optimizer step runs locally but stays in sync due to reduced grads

---

## 2) Correct baseline setup

Must-have steps:

1. initialize process group (`torch.distributed.init_process_group`)
2. set device by local rank
3. wrap model with `DistributedDataParallel`
4. use `DistributedSampler` for dataset sharding
5. call `sampler.set_epoch(epoch)` every epoch
6. only rank 0 handles checkpointing/logging outputs
7. cleanly destroy process group on exit

Launch pattern:

- `torchrun --standalone --nproc_per_node=2 distributed-training/train_ddp.py`

---

## 3) Common failure/debug paths

### A) Hangs/deadlocks

Typical causes:

- one rank exits early while others wait in collective op
- inconsistent batch counts across ranks
- missing `sampler.set_epoch` or bad dataloader settings

Debug approach:

- set `TORCH_DISTRIBUTED_DEBUG=DETAIL`
- reduce workers/batch size and run minimal config
- ensure every rank executes same training loop shape

### B) NCCL init/runtime errors

Typical causes:

- incompatible CUDA/NCCL versions
- wrong network interface
- GPU visibility mismatch

Useful env flags:

- `NCCL_DEBUG=INFO`
- `NCCL_IB_DISABLE=1` (when RDMA path is broken)
- `NCCL_SOCKET_IFNAME=eth0` (or your correct interface)

### C) Uneven performance

Typical causes:

- CPU dataloader bottleneck
- too-small per-rank batch
- heavy host-device sync points in loop

---

## 4) Performance checklist (high ROI)

1. Increase per-rank batch until memory limit
2. Enable AMP (`torch.cuda.amp.autocast` + `GradScaler`)
3. Tune dataloader (`num_workers`, `pin_memory`, `persistent_workers`)
4. Minimize Python-side logging/sync in hot loop
5. Profile dataloader vs compute time before model-level tuning
6. Save checkpoints less frequently and only on rank 0

---

## 5) Reliability checklist

- deterministic seed strategy per rank
- checkpoint contains:
- model state
- optimizer state
- scaler state (if AMP)
- epoch/step counters
- rank 0 writes checkpoints atomically (tmp file + rename)
- fail-fast on NaN/Inf loss
- preserve run metadata (commit hash, config, world size)

---

## 6) What to measure and report

System metrics:

- samples/sec per rank
- global samples/sec
- GPU utilization
- dataloader wait time
- step time variance across ranks

Training metrics:

- loss curve consistency vs single-GPU baseline
- convergence per wall-clock hour
- final accuracy/performance parity check

For portfolio quality, include before/after numbers for at least one tuning change.

---

## 7) Scope and next upgrades

This baseline is intentionally minimal and local-first.

Good follow-ups:

- multi-node DDP with explicit rendezvous config
- experiment tracking integration (MLflow/W&B)
- fault-tolerant checkpoint resume in orchestrated jobs
- FSDP/ZeRO comparison for larger models