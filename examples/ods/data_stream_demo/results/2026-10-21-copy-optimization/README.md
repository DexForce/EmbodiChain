# ODS sampling-copy optimization

This CPU component comparison measures the real Engine and Source against
baseline commit `6d86b81c3`. It does not rerun or replace the
[historical delivery study](../2026-10-21/RESULTS.md), and it contains no
generation, decoding or real-training end-to-end result.

The implementation keeps the indexed copy inside the producer lock and
removes the subsequent redundant clone. The Source trusts the Engine's
padding and repads only custom transformed observations. The showcase also
combines CPU RGB conversion with division, preserves existing zero image
padding, and copies recorded RGB episodes directly into owned contiguous
storage. The benchmark times only production sampling, not those additional
showcase conversions.

With C=8, T=600, H=2, F=32, 256x256 RGBA uint8 images and one Torch thread,
the B=32 Source median latency changed from 556.343 to 318.719 ms for one
camera and from 1113.004 to 641.989 ms for two cameras. These are 1.75x and
1.73x component speedups on this shared host, not training acceleration.
Each case has two warmups and seven latency samples; all B=1/32 and
state/single/dual cases, including smaller gains, appear in the report.
The separate 1 ms RSS observation is approximate and is not an allocator peak.

- [Three-table benchmark report](report.md).
- [Raw measurements, hardware, versions and production source hashes](raw.json).
- [RGB normalization/packing equivalence and output ownership checks](payload-validation.json).
- [Validation summary](validation.json).

All 36 implementation/shape/API cases had identical output bytes and RNG
state. Context checks include real padding, and retained outputs remain
unchanged after shared-cache replacement and another sample. Four additional
payload checks cover state, single/dual cameras and a full B=32 dual-camera
batch; normalized and packed tensors match the baseline bit for bit and
survive input overwrite. These checks supplement production regressions for
nested noncontiguous images, repeated indices and transformed padding.
The fused RGB operation also matches the original float32 result for all
256 channel values under float32, float64, float16 and bfloat16 global defaults.

## Reproduce the component comparison

From a checkout containing the optimization, using the py311 environment:

```bash
git worktree add --detach /tmp/ods-copy-baseline 6d86b81c3
python scripts/benchmark/data_pipeline/benchmark_ods_context_copy.py \
  --implementation baseline=/tmp/ods-copy-baseline \
  --implementation optimized=. \
  --output-dir outputs/benchmarks/ods-copy
```

The script uses separate checkout subprocesses, identical seeds and
camera-wise alternating comparison order. Cache construction, correctness
checks, output release and the separate RSS pass are excluded from service
latency. There is no simultaneous producer, GPU transfer or learner.
Candidate production hashes identify the measured working tree independently
of its then-current HEAD. The source and masks are unchanged by the benchmark.

The primary training-efficiency comparison is the separate
[equal-work end-to-end protocol](../../../../../docs/architecture/ods-data-stream-showcase-design.md#6-follow-up-equal-work-end-to-end-training-time).
It measures generation and training together; faster CPU sampling is only a
diagnostic improvement toward that goal.
