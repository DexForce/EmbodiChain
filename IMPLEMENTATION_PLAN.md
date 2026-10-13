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
3. **Gym integration (gated, do not start before gate 2 passes)**: YAML scene
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
- No Gym modifications. Multi-environment fluid declarations fail explicitly.

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

## Next milestone

Sim and demo parity are the current delivered milestone. Gym remains unimplemented.
Its next dependency is native world-filtered neighborhoods/boundaries, per-world
DFSPH convergence and selected-world cache reset, followed by EC YAML/manager
integration. Do not remove the single-environment gate before that qualification.

GPU memory evidence: `outputs/fluid-memory-comparison.json` (process samples
at approximately 250 ms cadence; 4981 -> 4985 MiB).
