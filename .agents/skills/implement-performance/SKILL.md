---
name: implement-performance
description: Design and implement EmbodiChain computation when task analysis identifies algorithmic complexity, data layout, batching, memory use or synchronization costs as material. Use for new computational capabilities and performance-sensitive changes to existing code.
---

# Implement performance-sensitive computation

Use a systems and HPC engineering perspective once task analysis identifies
material computational costs. Scale the work to the affected execution path;
ordinary configuration and documentation edits do not need performance analysis.

## Establish the workload and cost model

Before choosing an implementation, briefly record the relevant workload,
expected bottleneck and design tradeoff in the task update or PR:

- Identify the caller, call frequency, typical and largest supported shapes,
  dtype/device, and target hardware. Distinguish latency, throughput and memory
  goals; account for single-item or small batches when callers use them.
- For existing behavior, establish a baseline and use focused measurements or
  profiling to locate the dominant cost. For new behavior, estimate time/space
  complexity and intermediate storage first, then validate the estimate.
- Trace the execution path from input preparation to result consumption.
  Consider compute, memory traffic, allocations, transfers, dispatch and waits.
  State assumptions when workload or hardware evidence is unavailable; keep
  performance conclusions limited to what was measured.

## Choose the implementation

Prefer the simplest approach that meets the workload and correctness contracts.
Consider algorithmic improvements before backend-specific tuning. Choose CPU,
Torch, Warp or another existing backend from measured or estimated costs,
including dispatch and transfers for the actual batch sizes.

- Define batch axes, strides and ownership at boundaries. Evaluate contiguous
  access and broadcast views; account for copies introduced by indexing,
  reshaping or making inputs contiguous. Keep dtype/device conversion at owned
  boundaries and avoid repeated host/device round trips.
- Bound intermediate storage as well as arithmetic. Vectorization can create
  quadratic pairwise or candidate tensors; use bounded chunks, reductions or
  direct selection when full materialization is unnecessary. Preserve ordering,
  masks and selection semantics across chunk boundaries.
- Inspect loops for repeated allocation, conversion and invariant work. Reuse
  buffers or cache results when useful, with explicit capacity, invalidation
  and lifetime. Returned results must retain their documented ownership across
  later calls; protect reuse across callers, threads and streams.
- Check device-to-host reads such as `.item()`, `.cpu()`, `.numpy()`, tensor
  truth tests and data-dependent indexing for transfers or synchronization.
  Keep device work on the intended stream; establish dependencies and storage
  lifetimes explicitly when interoperating with Warp or other backends.
- Account for CPU thread pools and native library threads before adding
  parallel workers. Bound queues and in-flight work; avoid oversubscription.
  Introduce fusion, compilation, graph capture or custom kernels when expected
  gains justify startup cost, shape constraints and maintenance.

Shared numerical algorithms and private Warp kernels follow the ownership in
`AGENTS.md`. Runtime adapters own simulation state, configuration and resource
lifecycles. For solver buffer changes, read the
[IK overview](../../../agent_context/topics/ik-solvers/ik-solvers.md) and its
linked buffer lifecycle before changing reuse or stream behavior.

## Verify behavior and performance

Use [add-test](../add-test/SKILL.md) for focused production-contract tests.
Choose relevant checks for numerical tolerance, failure masks, empty/singleton
and partial batches, noncontiguous inputs, dtype/device, determinism and
gradients where supported. Buffer or concurrency changes need checks that old
results remain valid and intended stream/thread use is safe. Do not change
precision, convergence, candidate ordering or success criteria merely to claim
a speedup.

Use [benchmark](../benchmark/SKILL.md) for measurement methodology. Compare the
baseline and proposed implementation on equivalent inputs and quality settings;
include typical sizes and memory stress cases within the supported budget.
Separate device computation from transfers and end-to-end execution. Evaluate
startup/steady-state or tail latency when the caller depends on them.

Report the chosen design, tested workload/hardware, correctness evidence,
latency or throughput and memory tradeoffs, including regressions. Hardware
that was unavailable remains unvalidated. Stop adding complexity when the
workload's needs are met or measurements do not support further optimization.
