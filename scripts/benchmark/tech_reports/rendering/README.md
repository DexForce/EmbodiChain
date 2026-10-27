# Rendering comparisons (R-series)

The directory scaffold for
[issue #679](https://github.com/DexForce/EmbodiChain/issues/679) reserves
`experiments/` for R-series definitions, camera/environment/resolution matrices,
and comparison orchestration, and `platforms/` for platform-specific execution
adapters. Small input descriptions and asset references belong in
`../assets/rendering/`; downloaded binaries use the external asset cache.

The first planned cross-platform slice is R-04, batched camera rendering on
EmbodiChain and Isaac Lab, followed by R-03/R-05/R-06/R-09. Matched ManiSkill
reference points can be added under the issue's frozen comparison conditions.
Record render, observation-delivery, and environment-step boundaries separately;
camera-frame counting, modality/temporal evidence, quality, GPU completion,
and memory scope must be explicit. Each platform runs in an isolated process
and Python environment. New adapters and R-04 are not implemented by this scaffold.

The [existing offscreen denoising workflow](../../rendering/README.md) stays in
`scripts/benchmark/rendering/` with its current entry point and outputs. It is
not by itself the proposed cross-platform R-04 experiment.
See the [technical report layout](../README.md#technical-report-comparison-layout)
for common ownership and artifact conventions.
