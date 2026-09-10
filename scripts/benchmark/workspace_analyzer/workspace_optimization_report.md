# Workspace optimization measurements

Baseline: `797fe56ed51c0d74dc9337f35f71d358a1ff0e23`; implementation: `perf/workspace-batching`.
Environment: AMD EPYC 9K65, NVIDIA RTX PRO 5000 72GB Blackwell; Python 3.11, Torch 2.7.0+cu128, Warp 1.15.0. CPU runs use four Torch threads. Each row is the median of three warm repeats.

The benchmark uses real UR5/CobotMagic analytic solvers, packaged asset chains and Robot batch methods, with fixed root-pose snapshots. It excludes renderer/simulation creation, persistent caching, compilation and visualization. Asset FK is eager; these timings are not full simulation deployment timings. Fixed-domain runs include sampling, FK/IK, frame conversion, filtering and basic metrics. Bounds rows measure the dynamic-bounds helper alone.

Both revisions use explicit random sampling, seed 42 and identical fixed bounds. This isolates implementation changes from the new Sobol default. K denotes IK seeds per point, B denotes points per batch. Measurements on shared hardware have timing variability; small differences around 1x should be treated as unchanged.

## CUDA timing

| Robot/path | Before (ms) | After (ms) | Before / after |
| --- | ---: | ---: | ---: |
| ur5/joint_space/K1/B1024/N8192 | 15.009 | 14.302 | 1.05x |
| ur5/cartesian_space/K1/B1024/N8192 | 30.321 | 30.683 | 0.99x |
| ur5/cartesian_space/K4/B1024/N8192 | 38.410 | 28.280 | 1.36x |
| ur5/plane_sampling/K1/B1024/N8192 | 29.936 | 29.639 | 1.01x |
| ur5/plane_sampling/K4/B1024/N8192 | 37.896 | 28.291 | 1.34x |
| ur5/dynamic_bounds/N1000 | 879.574 | 1.470 | 598.35x |
| cobotmagic/joint_space/K1/B1024/N8192 | 9.898 | 9.581 | 1.03x |
| cobotmagic/cartesian_space/K1/B1024/N8192 | 21.215 | 20.737 | 1.02x |
| cobotmagic/cartesian_space/K4/B1024/N8192 | 22.145 | 19.960 | 1.11x |
| cobotmagic/plane_sampling/K1/B1024/N8192 | 25.148 | 19.519 | 1.29x |
| cobotmagic/plane_sampling/K4/B1024/N8192 | 20.747 | 21.156 | 0.98x |
| cobotmagic/dynamic_bounds/N1000 | 143.475 | 0.833 | 172.24x |

## CPU timing

| Robot/path | Before (ms) | After (ms) | Before / after |
| --- | ---: | ---: | ---: |
| ur5/joint_space/K1/B1024/N8192 | 18.229 | 12.636 | 1.44x |
| ur5/cartesian_space/K1/B1024/N8192 | 95.348 | 86.985 | 1.10x |
| ur5/cartesian_space/K4/B1024/N8192 | 433.853 | 383.950 | 1.13x |
| ur5/plane_sampling/K1/B1024/N8192 | 90.777 | 84.029 | 1.08x |
| ur5/plane_sampling/K4/B1024/N8192 | 556.121 | 385.301 | 1.44x |
| ur5/dynamic_bounds/N1000 | 209.597 | 1.603 | 130.75x |
| cobotmagic/joint_space/K1/B1024/N8192 | 19.188 | 14.113 | 1.36x |
| cobotmagic/cartesian_space/K1/B1024/N8192 | 17.676 | 15.849 | 1.12x |
| cobotmagic/cartesian_space/K4/B1024/N8192 | 46.012 | 42.893 | 1.07x |
| cobotmagic/plane_sampling/K1/B1024/N8192 | 12.510 | 11.069 | 1.13x |
| cobotmagic/plane_sampling/K4/B1024/N8192 | 28.233 | 25.973 | 1.09x |
| cobotmagic/dynamic_bounds/N1000 | 224.824 | 1.650 | 136.26x |

## Accuracy comparison

| Device | Cases compared | Identical success rates | Identical reported FK residuals |
| --- | ---: | ---: | ---: |
| cuda | 22 | 22/22 | 22/22 |
| cpu | 22 | 22/22 | 22/22 |

UR keeps all 512 candidates (eight branches expanded across periodic joint shifts), its validity thresholds and nearest-seed selection. OPW keeps eight candidates. Existing default asset-FK versus analytic-geometry discrepancies are unchanged; exact candidate equality tests and separate analytic round trips validate allocation changes without silently weakening solver tolerances.

The largest reliable gain is replacing 1,000 individual FK calls in bounds estimation with a batch. Fixed-domain FK/IK gains vary: some workloads improve, while several remain near parity. OPW packs live limits into one host transfer instead of twelve scalar conversions. Internal candidate arrays reuse storage; public returned arrays retain independent ownership.

Memory columns in the raw reports are CPU RSS deltas and PyTorch allocator GPU deltas/peaks. Baseline Warp allocations are outside that GPU counter; these numbers must not be used to claim total VRAM reductions. Candidate capacity stays at the largest requested batch until solver destruction. Compact results reduce retained/cache arrays, not peak sampling allocation.

## Reproduction

From each checkout, run the same script with that checkout as `PYTHONPATH` and working directory:

```bash
PYTHONPATH=. python /path/to/benchmark_robot_workspace.py --device cpu --output outputs/benchmarks/workspace_cpu.md
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python /path/to/benchmark_robot_workspace.py --device cuda:0 --output outputs/benchmarks/workspace_cuda.md
```

The renderer-free benchmark supports CUDA remapping. Live DexSim tests must keep CUDA and Vulkan physical device selection aligned; the validation here ran live tests without `CUDA_VISIBLE_DEVICES` remapping.

## New controls

- `--sampler sobol` is now the CLI/API default; explicit `random` and `uniform` remain supported.
- `--sample-within-constraints` opts into bounded refill within box/ground/exclusion constraints (and configured sphere/plane domains through the analyzer API). Reachability then describes the permitted domain, not the original bounding box.
- `--compact-results` retains reachable points, qpos and aligned scores; dropped rejected-point diagnostics are unavailable for later analysis.
- `BaseSolver.prepare_buffers(max_batch)` optionally reserves scratch capacity; `get_fk_batch`/`get_ik_batch` preserve leading batch axes.

Raw per-run reports are written under `outputs/benchmarks/workspace_{before,after}_{cpu,cuda}.md` in the implementation worktree. Each includes time/memory, success/pose metrics and a complete per-case ranking. Rankings across different analysis domains are descriptive, not an algorithm-quality comparison.

## Validation

Affected regression tests pass in two processes: 169 workspace/solver/config/motion-generator/randomization cases, plus 32 Robot CPU/CUDA cases. This includes real simulation tests, analytic candidate equality, scratch growth and CUDA stream ownership, and compact-cache visualization. The 23 project-context tests also pass. Black 26.3.1, `git diff --check`, context validation and API documentation coverage (1,853/1,853 exports) pass.

The initial combined process passed 199 cases and failed two Robot CUDA FK cases after reaching Torch Dynamo's eight-recompilation limit for `forward_kinematics_tensor`. Running the Robot module separately passed all 32 cases; the other group passed all 169. No production compiler settings were changed to bypass that limit.

```bash
PYTHONPATH=. python -m pytest -q tests/sim/motion/workspace tests/sim/motion/solvers/test_analytic_batching.py tests/sim/motion/solvers/test_base_solver.py tests/sim/objects/test_robot_cfg.py tests/sim/motion/test_motion_generator_batched.py tests/gym/envs/managers/test_workspace_randomization.py --run-gpu
PYTHONPATH=. python -m pytest -q tests/sim/objects/test_robot.py --run-gpu
```
