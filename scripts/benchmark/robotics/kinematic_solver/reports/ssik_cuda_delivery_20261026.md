# SSIK CUDA delivery benchmark

Measured on 2026-10-25/26, RTX 5090 / Ryzen 9 9950X, NVIDIA driver 595.58.03, Torch 2.7.1+cu128, Warp 1.17.0, one CPU thread.

All four presets retain Pytorch defaults. Explicit SSIK CPU/CUDA modes are delivered with the gates below. This report supersedes earlier default-switch and Torch-only scratch conclusions; historical reports remain unchanged.

[Compact metrics, source hashes, provenance and counterexamples](ssik_cuda_delivery_20261026.json). Large NPZ fixtures and raw timing samples remain in ignored outputs/benchmarks/ssik_cuda_delivery_20261026. The JSON retains the separate measurement recipes as validation evidence, not production entry points; the unified benchmark remains the maintained runner.

## Time & Memory

| Model | Backend | Scope | N | Calls | p50_ms | p95_ms | p99_ms | max_ms | Scratch_MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marvin_bare_left | warp/cpu | streaming_control | 1 | 10000 | 0.340209 | 0.363764 | 0.466759 | 1.68368 | — |
| marvin_bare_right | warp/cpu | streaming_control | 1 | 10000 | 0.341262 | 0.356502 | 0.444755 | 1.05292 | — |
| marvin | warp/cpu | paired_streaming_control | 1 | 10000 | 0.686571 | 0.773389 | 0.950331 | 4.94067 | — |
| marvin_bare_left | warp/cuda | streaming_control | 1 | 10000 | 7.68421 | 8.32977 | 8.73974 | 10.8793 | — |
| marvin_bare_right | warp/cuda | streaming_control | 1 | 10000 | 7.61486 | 8.26891 | 8.68058 | 12.9722 | — |
| marvin | warp/cuda | paired_streaming_control | 1 | 10000 | 14.9741 | 15.8471 | 16.2753 | 20.81 | — |
| marvin_bare_left | warp/cuda | resident | 1000 | 30 | 35.4193 | 36.2471 | 36.486 | 36.5655 | — |
| marvin_bare_left | warp/cuda | resident | 10000 | 30 | 347.201 | 348.536 | 348.996 | 349.144 | — |
| marvin_bare_right | warp/cuda | resident | 1000 | 30 | 35.0262 | 36.1164 | 36.3651 | 36.464 | — |
| marvin_bare_right | warp/cuda | resident | 10000 | 30 | 347.518 | 348.286 | 348.688 | 348.805 | — |
| aloha_left | warp/cuda | fresh-process driver growth: uniform | 10000 | — | — | — | — | — | 334 |
| marvin_bare_left | warp/cuda | fresh-process driver growth: uniform | 10000 | — | — | — | — | — | 334 |
| marvin_bare_left | warp/cuda | fresh-process driver growth: seed_local | 10000 | — | — | — | — | — | 334 |
| arx5 | near_parallel/cpu | streaming_control | 1 | 10000 | 0.225563 | 0.286358 | 0.31429 | 1.18628 | — |
| arx5 | near_parallel/cuda | streaming_control | 1 | 10000 | 0.668448 | 1.59146 | 1.7109 | 2.06367 | — |
| arx5 | near_parallel/cuda | resident | 1000 | 30 | 5.45425 | 5.84263 | 5.97502 | 6.01437 | 4.96786 |
| arx5 | near_parallel/cuda | resident | 10000 | 30 | 52.8553 | 53.6244 | 54.7489 | 55.2067 | 6.48473 |
| arx5 | near_parallel/cuda | conservative tensors + fresh driver increment | 10000 | — | — | — | — | — | 8.48473 |
| arx5 | pytorch_original/cuda | resident | 1000 | 30 | 457.802 | 460.515 | 460.622 | 460.658 | — |
| arx5 | pytorch_original/cuda | resident | 10000 | 30 | 5577.79 | 5827.29 | 6197 | 6344.24 | — |
| rizon4s | cuda_rr | streaming_first_take | 1 | 10000 | 2.88627 | 3.15837 | 3.31949 | 4.48707 | — |
| rizon4s | cuda_rr | streaming_quiet_retake | 1 | 10000 | 2.89163 | 5.41233 | 6.68172 | 7.48833 | — |
| rizon4s | cuda_rr | resident | 1000 | 30 | 110.689 | 111.344 | 111.401 | 111.409 | — |
| rizon4s | cuda_rr | resident | 10000 | 30 | 1076.96 | 1078.49 | 1078.92 | 1079.01 | — |
| rizon4s | pytorch_original | resident | 1000 | 30 | 596.982 | 614.936 | 618.34 | 619.481 | — |
| rizon4s | pytorch_original | resident | 10000 | 30 | 5217.21 | 5500.15 | 5594.29 | 5597.46 | — |
| rizon4s | pytorch_quality_matched | resident | 1000 | 30 | 441.554 | 443.816 | 444.007 | 444.013 | — |
| rizon4s | pytorch_quality_matched | resident | 10000 | 30 | 5228.34 | 5319.04 | 5393.68 | 5414.69 | — |
| rizon4s | cuda_rr | fresh-process driver growth minus output | 10000 | — | — | — | — | — | 53.7234 |

Resident calls use device inputs and benchmark completion synchronization. Streaming calls use changing goals, the preceding returned seed, resets only at path boundaries, and a hard 0.05 rad per-joint bound. Single-arm and paired control have 10,000 completed samples; bulk rows have 30. p99 is a measured tail, not a real-time guarantee. Initialization/JIT, transport and Robot end-to-end timing are separate; this update does not certify transfer-inclusive or live-hardware latency.

Fresh-process driver/NVML increments conservatively include cold module loading and driver stacks; Torch-only workspace statistics cannot certify total CUDA scratch. Common Aloha/Marvin growth exceeds 128 MiB. ARX5 and RR use separate bounded providers. Raw allocation phase snapshots are preserved in JSON.

## Success & Other Metrics

| Model | Device | Wide_10000 | Boundary_1000 | Singular_200 | Unreachable_false_accepts | Whole_path_20000 | False_successes | Max_position_m | Max_rotation_rad | Max_step_rad | Step_violations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| marvin_bare_left | cpu | 1 | 1 | 1 | 0 | 0.98345 | 0 | 9.02737e-06 | 1.45419e-06 | 0.0128771 | 0 |
| marvin_bare_right | cpu | 1 | 1 | 1 | 0 | 0.97185 | 0 | 8.04953e-06 | 1.35085e-06 | 0.0350286 | 0 |
| marvin_gripper_left | cpu | 1 | 1 | 1 | 0 | 0.98345 | 0 | 9.02542e-06 | 1.29425e-06 | 0.012731 | 0 |
| marvin_gripper_right | cpu | 1 | 1 | 1 | 0 | 0.97185 | 0 | 8.04772e-06 | 1.37078e-06 | 0.0350729 | 0 |
| marvin_bare_left | cuda | 1 | 1 | 1 | 0 | 0.98345 | 0 | 9.02748e-06 | 1.45419e-06 | 0.0128772 | 0 |
| marvin_bare_right | cuda | 1 | 1 | 1 | 0 | 0.97185 | 0 | 8.04953e-06 | 1.35085e-06 | 0.0350286 | 0 |
| marvin_gripper_left | cuda | 1 | 1 | 1 | 0 | 0.98345 | 0 | 9.0255e-06 | 1.29425e-06 | 0.0127311 | 0 |
| marvin_gripper_right | cuda | 1 | 1 | 1 | 0 | 0.97185 | 0 | 8.04786e-06 | 1.37078e-06 | 0.0350729 | 0 |
| arx5 | cpu | 0.9999 | 1 | 1 | 0 | 0.95215 | 0 | 9.66069e-06 | 9.96048e-06 | 0.0081058 | 0 |
| arx5 | cuda | 0.9999 | 1 | 1 | 0 | 0.95215 | 0 | 9.66069e-06 | 9.96048e-06 | 0.0081058 | 0 |
| rizon4s | cuda | 0.9999 | 1 | 1 | 0 | 0.2 | 0 | 3.68733e-06 | 9.85952e-06 | 0.0427969 | 0 |

Rates are fractions. Every successful float32 output was independently checked against the original URDF/PK chain, actual joint limits and preset TCP at 1e-5 m/rad. Unreachable rows contain 500 targets per model/device. Finite search failure never establishes global unreachability.

Marvin retains all 128 global samples, a defined seed swivel and one optimized eight-branch sample (1,040 slots). All seven constructed smooth-joint paths complete in every variant. Remaining left/right failures are Cartesian reach-shell violations exceeding the accepted tolerance. CPU/CUDA whole-path masks match exactly. ARX5 correction is explicitly numerical: eight geometric initializations and one always-computed seed initialization, each with at most twelve bounded monotone Newton steps. It is not a pure closed-form general 6R solver. The ARX5 Cartesian continuation failures are not classified as globally unreachable; pointwise unbounded-step IK succeeds. Rizon4S finite-grid misses include a retained reachable witness with no valid conditional branch in an independent CPU QZ sweep.

## Leaderboard

| Model | Mode | Precision | Continuity | Latency | Resource | Default |
| --- | --- | --- | --- | --- | --- | --- |
| Aloha Mini | strict 6R | historical full suite + current regressions | historical; not recertified here | not recertified here | HOLD: +334 MiB | Pytorch |
| Marvin (4 variants) | strict SRS + seed_local | PASS: 100% static | PASS: matches quality-matched baseline | CPU PASS; CUDA HOLD | CUDA HOLD: +334 MiB | Pytorch |
| ARX5 X5A | near_parallel hybrid, 9 slots | PASS: 99.99% wide | HOLD: continuation qualification incomplete | CPU/CUDA PASS | PASS: fresh driver probe | Pytorch |
| Rizon4S | CUDA RR, 16 locks / 384 slots | PASS: 99.99% wide | HOLD: 20% on general paths | HOLD: quiet p99 6.682 ms | PASS: 53.72 MiB | Pytorch |

Rows describe model-specific gates, not unconditional algorithm rankings. Defaults change only after all gates pass. No backend invokes an implicit CPU or numerical-solver fallback. Rizon4S performs prepared polynomial evaluation, cooperative complex QR and back-substitution entirely on CUDA; symbolic derivation is offline.

Rizon4S bulk speedups versus original Pytorch are 5.39x / 4.84x at 1,000 / 10,000 targets. The quality-matched Pytorch comparison is separately recorded (3.99x / 4.85x); original default convergence settings do not meet the strict external acceptance gate on this bulk cohort. Both 10,000-call Rizon timing takes are retained: the first p99 is 3.319 ms, the quiet retake 6.682 ms, so the latency gate is held.

ARX5 is 83.93x / 105.53x faster than its original Pytorch defaults on the identical bulk input cohort (30 completed calls per batch size). This is an original-default timing comparison; the ARX5 baseline was not separately qualified at the strict external pose gate. Target-aware continuation improves the ARX5 whole-path result from 89.205% to 95.215%, with all 7,000 constructed smooth-joint frames passing. The remaining continuation gap holds its default switch.

The Rizon zero-lock DH chart is rederived from the source common-normal geometry (0.88468 mm correction). Every one of the 16 charts has an independent POE/DH FK certificate. Corrected CPU QZ and CUDA have 542 matching conditional branches with no missing or extra branches in the comparison cohort. The unmodified upstream zero-lock two-parallel path is not used as an oracle for the corrected chart.

## Reproduction

Run python scripts/benchmark/robotics/kinematic_solver/run_benchmark.py --solvers ssik, adding the following explicit settings:

- Marvin: --ssik-models marvin_bare_left marvin_bare_right marvin_gripper_left marvin_gripper_right --ssik-srs-search seed_local
- ARX5: --ssik-models arx5 --ssik-geometry-mode near_parallel
- Rizon4S: --ssik-models flexiv_rizon4s --ssik-cuda-rr

The unified correctness and trajectory fixtures use seed 2531; the separately recorded Rizon bulk/control cohort uses seed 20261025. Within each comparison, model, targets, seeds, limits and external acceptance are identical. Keep the full workload, CPU thread count 1, batch sizes 1/8/100/1000/10000 and 1e-5 acceptance. CPU reference installation is explicit via requirements/ssik-reference.txt. Rebuild the RR bundle only with prepare_ssik_rr_rizon4.py against pinned SSIK 4109147f37a7ca3e2516c83853aca583612783f3. The manifest, source/data digests and minimal misses are in the JSON.
