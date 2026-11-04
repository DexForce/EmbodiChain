# Contact-loss acquisition cost

The benchmark compares the main sensor at `7cb0c97c2`, the initial PR #944 fix
at `db7515d3b`, and the optimized working-tree implementation. The subsequent
main commit `203b86832` changes Task Program contracts, not these sensor kernels.
Candidate file hashes are recorded alongside the pre-commit HEAD: the hashes,
not that HEAD alone, identify the measured implementation.

## Workload and method

- AMD EPYC 9K65; NVIDIA RTX PRO 5000 72GB Blackwell, driver 580.126.20.
- GPU 4 was idle when selected. CPU Torch threads: 1; seed: 0.
- Real ContactSensor, ContactHistory, Warp kernels and ContactEvidenceSampler
  over fixed SDK buffers; **no native physics, rendering or policy inference**.
- 1, 64 and 2048 environments; four valid contacts/environment; capacity 64.
- Three scopes: four-substep control interval, loss-counter read, full evidence
  refresh including source read, scatter, validation and owned channel copies.
- 10 warmups; 15 paired trials of 60 calls, rotating implementation order.
  Warp and Torch synchronize before/after timed windows. Values below are
  median microseconds/call; raw samples and dependency versions are in JSON.
- Checks outside timing compare normal evidence values/masks exactly and verify
  overflow is unavailable after both fixes. Independent GPU tests cover graph
  replay, empty-contact query loss, cross-stream reads and reset semantics.

## GPU results

| Environments | Scope | Main | Initial fix | Optimized |
|---:|---|---:|---:|---:|
| 1 | Four-substep control | 137.34 | 135.62 | 135.34 |
| 64 | Four-substep control | 139.42 | 136.40 | 136.74 |
| 2048 | Four-substep control | 137.25 | 134.62 | 134.35 |
| 1 | Loss read | 28.69 | 59.96 | 13.59 |
| 64 | Loss read | 28.70 | 59.96 | 13.78 |
| 2048 | Loss read | 29.02 | 60.53 | 16.76 |
| 1 | Evidence refresh | 705.06 | 750.32 | 678.34 |
| 64 | Evidence refresh | 1010.88 | 1059.43 | 984.00 |
| 2048 | Evidence refresh | 10164.66 | 10255.22 | 10164.67 |

The CUDA fused kernel records query loss without a separate copy launch. CPU
keeps a scalar copy to avoid adding a branch to every scatter row. Packed
interval/latest counters require one host readback, followed by int64 reduction,
instead of two sets of GPU reductions and scalar reads. A zero-loss fast path
avoids unnecessary host reductions. Interval statistics and
latest-sample statistics remain independently available; their maximum, not sum,
controls evidence validity. Existing standalone consumers retain their fallback.

## Limits

The optimized GPU evidence path is 3.79%/2.66% faster than the measured main at
1/64 environments, and effectively equal at 2048 (less than 0.001% difference).
CPU evidence refresh is 0.15%, 0.09%, and 0.01% slower at those sizes. CPU
four-substep sampling differs by -0.13% to +0.21%. These results do **not**
establish a universal zero-regression guarantee. Small total-path differences
should not be interpreted as reliable speedups on a shared host. Large-batch
channel copying and validation dominate counter work. Earlier all-backend fused
prototypes had a roughly 2% CPU sampling regression at 2048 environments; the
final implementation retains the original CPU scatter kernel. No end-to-end
simulator FPS claim is made.

## Reproduce

```bash
PYTHONPATH="$PWD:$PWD/extensions:$PWD/extensions/embodichain_tasks:$PWD/extensions/embodichain_models" \
python -m scripts.benchmark.sensors.contact_loss --device cpu --output /tmp/contact-loss-cpu.json

CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH="$PWD:$PWD/extensions:$PWD/extensions/embodichain_tasks:$PWD/extensions/embodichain_models" \
python -m scripts.benchmark.sensors.contact_loss --device cuda:0 --output /tmp/contact-loss-cuda.json
```

Only use trusted local Git revisions as reference implementations. Reference
loading verifies original scatter/history dependencies; the added fused entry
point is separate from the original reference kernel.

Results: [CPU](results/contact_loss_cpu.json), [CUDA](results/contact_loss_cuda.json).
