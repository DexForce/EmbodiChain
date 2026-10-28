# Fluid Gym integration

`EmbodiedEnvCfg.fluid_object` owns fixed-count fluid declarations. Physical
components own these alongside rigid entities; `config_to_cfg()` resolves
fluid point files and explicit relative rigid/visual mesh paths against the
owning component. Spawn rigid references are resolved before fluid binding.
The default pour deployment generates its fill from a local revolved profile;
no particle asset file is required. Pouring meshes use `PourCupAssets/...`
references resolved through the [data asset cache](../data-assets/data-assets.md),
so installed task packages carry configuration without mesh payloads. The runner's spacing override resamples the
same volume recipe and rejects fixed point imports. It scales explicit render
radius, mesh cell size and reconstruction kernel radius by the spacing ratio,
keeping the surface reconstruction consistent with the new particle density. Fill definitions and
in-memory template ownership are documented under
[simulation fluids](../simulation-system/fluids.md).

`managers/observations.py` provides particle state, summary and volume-in-region
observations. Particle arrays are world-frame; rigid `get_local_pose()` is
arena-local. Region measurement and `events.reset_fluid_in_container()` add
arena offsets before comparing or placing particles. Region volume is particle
count times nominal spacing cubed, not a reconstructed-surface measurement.

## Online Task Program workflow

`lab/scripts/run_fluid_task_program.py` runs the configured pour-cup deployment
through ordinary `EmbodiedEnv` and `execute_demo_episode()`. The environment
component selects DexUni with DFSPH settings nested under `sph_cfg`. Robot
drive targets, dynamic grasp contacts and fluid state
advance in one World, with no recorded trajectory input. SPH reads live rigid
boundaries; fluid reaction forces are not fed back into the rigid solver.
The fluid program measures cup lift and alignment, transports the held cup
to a receiver-relative pour pose through the existing MoveHeldObject lowerer,
then pours and returns it. The standalone Sim tutorial uses the same physical
component and relative pour target with freshly planned Atomic Action targets.

Program completion and physical qualification are separate: the runner measures
cup lift, maximum tilt, final return error and received volume. Projected effect
assurance alone does not prove grasp or pouring success. Optional trajectory
recording stores measured online state for inspection. `--seconds` qualifies
only a bounded fragment, not completion of the task.

`FluidReplayEnv` remains an explicit diagnostic utility for accepted historical
recordings; it is not the default deployment or task runner. It requires a
separate kinematic configuration, one Newton substep and fixed-root source
states, and rejects action managers and policy dataset/trajectory recording.

## Rendering and recording

The pouring environment and runner select `fast-rt`. Hybrid is rejected for
native fluid rendering: OptiX-only liquid geometry is absent from its Vulkan
primary surface. Particle-state and video-decode checks alone do not qualify
liquid visibility; use the image-based fluid regression linked below. The
default record view focuses on the cup and receiver; the mesh kernel radius
is 1.5 times the particle spacing to retain a connected surface after settling.
Its interval `record_camera` event uses the existing `record_camera_data`
functor with `fps=None`, recording independent environment views at the
control frequency. The runner only redirects the output directory and sets
the recording limit to the resolved environment count. It explicitly commits
before close, which otherwise discards pending frames. Incomplete or rejected
runs use the recorder's diagnostic purpose rather than qualifying the output.
Selected-row recording and camera consumption are owned by the
[recording functor](../manager-functor/manager-functor.md#camera-recording).
NoRender qualification removes the recorder before constructing the scene.

Production configuration/measurement checks are in
`tests/gym/envs/test_fluid_integration.py`; live isolation checks are in
`tests/sim/objects/test_fluid_batch_runtime.py` and
`tests/sim/objects/test_fluid_online_runtime.py`; actual primary-surface visibility
is checked by `tests/sim/objects/test_fluid_visibility.py`. Generated trajectories/videos
are local qualification evidence, not packaged task assets.
