# Fluid integration implementation plan

## Accepted scope

Start from PR #771 (d5322aa). Use DexSim's existing SPHSolverCfg directly;
add only the existing Newton solver mapping, not a new EC solver config.
Initial fill, fixed particle counts, Newton CUDA DFSPH, one-way coupling.

## Delivery gates

1. **Sim public interfaces (complete)**: FluidObjectCfg / spawn and render
   configuration; declared FluidObject / particle state; manager and Spawn
   lifecycle; rigid boundary state and SDF adaptation; production tests.
2. **Match the existing Sim demo (complete)**: migrate #771 using identical
   particles, trajectory, numerical parameters, camera and material. Run a
   same-environment baseline and migrated short test, then full pouring.
   Compare received/residual/spilled volume, settling, images, runtime and VRAM.
   A short test alone does not pass this gate. Record environment and evidence.
3. **Gym integration (complete after gate 2)**: YAML scene
   ownership, vectorized observations, partial reset, optional particle records.
   DexSim world-filtered SPH neighborhoods/boundaries and per-world solver reset
   are prerequisites; spacing worlds apart is not isolation.

## Sim acceptance

- Native solver object and mapping-valued sph configuration both work.
- Object declaration -> prepare -> GPU particle state -> write/reset -> cleanup.
- Deterministic points and volume sampling; local/world pose conventions.
- Rendering never changes physical particle state; no native solver, builder or
  render-sync access from the migrated tutorial.
- Start with one environment and reject unsupported batch use explicitly.
- Production tests, Black 26.3.1, API coverage and updated project context.

## Evidence

Implementation and validation results will be appended here. No Gym changes
will be represented as complete before full demo parity is established.


## Implemented Sim interfaces

- `FluidObjectCfg`, `FluidSpawnCfg`, `FluidRenderCfg`; points/NPZ, box, sphere,
  cylinder and closed fill-mesh sources. Shared numerical sampling is in compute.
- `add_fluid_object`, `get_fluid_object`, existing `get_asset`, and detached
  world-frame particle data; write/reset/reset_to_pose through Scene batches.
- Direct DexSim SPHSolverCfg and mapping-valued `solver_type: sph` support.
- Rigid kinematic pose/twist writes and native PRE_SUBSTEP trajectory controls.
- Independent rigid visual/collision geometry and pre-baked SDF policy.
- Native render publication, render-only clipping, and detached Viser points.
- This first Sim milestone initially rejected multiple environments; the
  completed Gym phase below lifts that gate only for qualified native artifacts.

## Current qualification limits

- Fixed physical topology after prepare; no particle/rigid topology rebuild.
- `FluidRenderCfg(mode="none")` requires the NoRender backend.
- Native runtime used here is a local DexSim build; repository dependency pin
  is unchanged. This does not qualify every published wheel.
- GPU memory was sampled on matching 2-second runs: baseline 4981 MiB,
  public API 4985 MiB. This does not measure the full-pour peak.

## Review and validation

Architecture/API and engine reviewers checked the implementation independently.
Integer initial velocities, independent-visual-mesh UV selection, optional optical
material forwarding and render-frame cache scope were corrected and regression
covered. The full run must include a pixel/video check: an earlier iteration had
correct particle snapshots but stale offscreen images, fixed by publishing via
`sim.sync_render_state()` before camera acquisition.


## Full-demo parity result

- Runtime: Python 3.11.14, local DexSim 0.5.1+21bfad162518216fdd9cf313c70b15af94e6fa84,
  Torch 2.10.0, Warp 1.15.0, Newton 1.4.0, RTX 5090.
- Same-environment original script versus public-API script: 30.975 seconds,
  76,538 particles, 31 snapshots with **bit-identical positions and velocities**.
  All recorded statistics match exactly, including 139.16475 mL received and
  final receiver speed RMS 0.0048424867 m/s.
- Two videos each decode all 744 frames. Mean pixel absolute difference on the
  0-255 scale: 0.477 (wide), 0.586 (close); moving/pouring and settling frames
  were also visually inspected. Original native linear water color is converted
  explicitly to the public material's sRGB convention.
- Full-loop wall time: original 341.69 s, public API 349.04 s (+2.15%).
  This is a single-run comparison, not a benchmark distribution.
- Evidence: `outputs/fluid-parity.json`, `outputs/fluid-baseline-full/`,
  `outputs/fluid-public-final/`. Generated videos/NPZ remain local ignored outputs.
- Focused CPU/config/lifecycle/visualization/material/context/API-check tests:
  533 passed, 1 skipped in the final combined run. GPU lifecycle/substep/reset
  test: 1 passed. The tutorial also passed a two-substep smoke run.
- GPU lifecycle test verifies declaration, state, native substep control,
  selected reset, repeated rollout, endpoint hold and cleanup.

## Delivered milestones

Sim parity and two-stage Task Program -> fluid Gym integration are complete in
isolated worktrees. Batch support remains gated by loaded native capabilities.
Published native wheel/AOT qualification is a separate release task; no shared
runtime or project dependency pin was changed.

GPU memory evidence: `outputs/fluid-memory-comparison.json` (process samples
at approximately 250 ms cadence; 4981 -> 4985 MiB).


## Completed phase: batch SPH, then Task Program -> fluid Gym

User confirmed the execution route: generate and validate a new trajectory with
Task Program in a MuJoCo Gym environment; replay its recorded robot and cup
states in a separate fluid Gym environment. This remains one-way coupling;
there is no online rigid-fluid joint solver or synthetic grasp attachment.
Generation acceptance and fluid physical metrics are recorded independently.

Dependency worktree: `/home/dex/.codex/worktrees/fluid-sph-worlds/dexsim`,
branch `codex/fluid-sph-worlds`, base `21bfad162518216fdd9cf313c70b15af94e6fa84`.
Native implementation commit: `d2d36fbcb`.
EC worktree remains this checkout, base Sim commit `bbd6928c1`.
Native modifications belong to the existing SPH solver/kernels and backend reset
interface. Compare: per-world solver instances (resource overhead), independent
partition scheduling (larger redesign), or world-filtered batched kernels
(selected, preserves ownership and batched stepping). No threading changes.

First oracle: coincident replicated worlds must have independent particle
neighbors, boundaries, gravity and convergence. Establish baseline failures
before fixing kernels. Then selected-world reset must preserve all other rows.
On green: enable capability-gated EC batch declaration and integrate Gym YAML,
fluid observations/reset, and Task Program trajectory generation + fluid replay.
On red: diagnose the failing kernel/contract in the dependency, without removing
EC capability guards. Runtime tests must print/verify source package identity;
the shared installed DexSim build is not modified.


## Gym and native qualification results

- Native source: world-filtered neighbors and SDF/volume-map boundaries,
  per-world gravity and DFSPH convergence; canonical `E+1` selected reset.
  Pure global legacy single-world particles remain supported. Mixed global/local
  particles and global particles with multiple worlds fail explicitly.
- Before fixes, overlapping worlds shared 16 instead of 8 neighbors and gravity;
  convergence differed by 0.62 m/s and a foreign boundary displaced particles
  by 0.08 m. New native regressions plus existing boundary checks: **14 passed**.
- EC YAML declares fluids through physical environment ownership; packaged
  relative source assets, fluid observations, container refill and capability
  gated batching are implemented. `FluidReplayEnv` reuses normal Gym lifecycle,
  with per-row clocks and partial reset. Robot roots/joint order are validated.
  Action managers and policy dataset/trajectory recording are rejected.
- Task Program source freshly generated in MuJoCo and accepted: lift
  **0.1197915 m**, maximum tilt **66.4883 degrees**, return error **0.00223394 m**.
  Source program hash:
  `6cc3d7aa5a64c4cc86f8d4f60bf80dbad23fca368785e165164cd39f50b288cb`.
- Full Gym replay: **2 x 76,538 particles**, **31 s**, **744 video frames** fully
  decoded and inspected at lift/pour/return. Final received volume:
  **141.98288 / 140.88261 mL**; all sampled particle states finite. Wall time
  **905.33 s**. Source motions are measured states with one-way fluid coupling;
  these results do not claim online robot-fluid force feedback.
- Focused EC config, objects, packaging, context and docs checker tests:
  **259 passed**. Later batch-count regression and refill reuse: **11 passed**.
  Fresh-process live Gym independent completion/reset and Sim selected reset:
  **2 passed**. Final coarse two-row CLI partial-reset and single-row override
  smoke checks also passed.
- Independent reviews corrected wheel asset omission, root pose adoption,
  misleading action recording, dataset mapping compatibility and canonical
  global reset masks. Real Gym tests additionally caught pre-resolution sim
  versus env counts; both replay and CLI now use `EnvCfg.num_envs`.
- Sphinx dummy build succeeded. The full run reported 811 warnings versus
  the earlier 800: one new duplicate fluid field entry was corrected, and
  ten additional warnings concern optional Pinocchio/type imports in the
  isolated environment. The post-fix incremental build also succeeded, with
  1123 warnings from repeated existing API indexing/type resolution and no
  fluid-specific warning entries. This is not a warning-free documentation build.
- API coverage **2503/2503**; source Task Program inspector passed all three
  calls. A built EC wheel contains all six pour-cup assets. Native validation
  uses staged JIT source, not a rebuilt release/AOT wheel.
- Registered `sph_speed_dfsph_4096` benchmark, same RTX 5090 / Newton 1.6 /
  Warp 1.17 configuration, 20 warmup + 100 measured steps: baseline mean
  **3.05956 ms**, changed **3.17811 ms** (+3.87%); p95 **3.24373 / 3.38617 ms**.
  Both runs finite with matching physical summary metrics. Single-run evidence,
  not a statistical performance guarantee; no claim of multi-world speedup.
- Runtime: isolated Python 3.11, Torch 2.10, Newton 1.6.0, Warp 1.17.0,
  MuJoCo/MuJoCo-Warp 3.12.0; DexSim native binaries from local base build and
  independently staged changed JIT source. Shared build and environments remain
  unchanged. NoRender robot smoke teardown still logs native entity balance
  `1024 != 1023`; tests complete, but repeated native resource-leak qualification
  is not established by these short runs.
- Context reviewed and revised: fluid lifecycle/capability, Gym replay and
  native Newton world/reset ownership. Existing Task Program language/semantics
  guidance remains accurate because those contracts are reused unchanged.

Evidence (ignored local outputs): `outputs/fluid-task-program/`,
`outputs/fluid-gym-full/fluid_gym.mp4`, `outputs/fluid-gym-full/fluid_gym_result.json`,
`outputs/fluid-gym-full/contact-sheet.jpg`, `outputs/fluid-gym-final-smoke/`,
`outputs/fluid-benchmark/`, `outputs/fluid-wheel/asset-validation.json`.

Reproduction in this checkout uses the isolated native JIT runtime and virtual
Python at `/home/dex/.codex/worktrees/fluid-sph-worlds/venv/bin/python`, with
`DEXSIM_NEWTON_ARTIFACT_MODE=jit`; `python -m
embodichain.lab.scripts.run_fluid_task_program --num-envs 2` runs both phases.
Old pre-recorded tutorial trajectories lack qualification metadata and are
intentionally rejected by the Gym adapter.
