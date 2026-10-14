# Fluid Gym integration

`EmbodiedEnvCfg.fluid_object` owns fixed-count fluid declarations. Physical
components own these alongside rigid entities; `config_to_cfg()` resolves
fluid point files and explicit relative rigid/visual mesh paths against the
owning component. Spawn rigid references are resolved before fluid binding.
The default pour deployment generates its fill from a local revolved profile;
no particle asset file is required. The runner's spacing override resamples the
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

## Two-stage Task Program workflow

`lab/scripts/run_fluid_task_program.py` generates the existing pour-cup Task
Program in MuJoCo, requires program acceptance and measured cup lift/tilt/return,
and stores provenance in trajectory metadata. A separate process/world runs
`gym/envs/fluid_replay.py::FluidReplayEnv` with live SPH. The physical deployments
remain separate files (`env.newton.yaml` versus `env.fluid.yaml`); the latter
reuses DexSim's SPH solver configuration directly.

Replay broadcasts one accepted source row across environments. It adopts the
recorded fixed robot root, checks joint compatibility, interpolates measured
joints/rigid poses on the physics grid, and publishes kinematic boundary twist.
Each environment has its own clock and selected reset/refill. Newton substeps
must be one; control intervals may contain multiple physics steps. Completion
is truncation; reset is required once every row is exhausted.

This is one-way state replay: policy actions do not drive the robot and fluid
forces do not affect the recorded motion. Action managers, trajectory recording
and active policy dataset recorders are rejected. Task Program acceptance is
source provenance; fluid capture metrics remain separate measured outcomes.

## Rendering and recording

The pouring environment and runner select `fast-rt`. Hybrid is rejected for
native fluid rendering: OptiX-only liquid geometry is absent from its Vulkan
primary surface. Particle-state and video-decode checks alone do not qualify
liquid visibility; use the image-based fluid regression linked below. The
default record view focuses on the cup and receiver; the mesh kernel radius
is 1.5 times the particle spacing to retain a connected surface after settling.
Its interval `record_camera` event uses the existing `record_camera_data`
functor with `per_env=True` and `fps=None`, recording every environment view at
control frequency. The runner only redirects the output directory and sets
the recording limit to the resolved environment count. It explicitly commits
at normal completion before close, which otherwise discards pending frames.
Selected-row recording and camera consumption are owned by the
[recording functor](../manager-functor/manager-functor.md#camera-recording).
NoRender qualification removes the recorder before constructing the scene.

Production configuration/measurement checks are in
`tests/gym/envs/test_fluid_integration.py`; live isolation checks are in
`tests/sim/objects/test_fluid_batch_runtime.py`; actual primary-surface visibility
is checked by `tests/sim/objects/test_fluid_visibility.py`. Generated trajectories/videos
are local qualification evidence, not packaged task assets.
