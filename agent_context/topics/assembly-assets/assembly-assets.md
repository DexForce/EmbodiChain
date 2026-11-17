# Precision assembly assets

Precision assembly is a reusable capability outside GenSim. GenSim pipelines,
Gym tasks and examples consume the same neutral asset API; the toolkit does not
import GenSim, simulation runtime, task packages or benchmark helpers at its root.
Native preparation is an explicit opt-in through `embodichain.toolkits.assembly.native`.

| Responsibility | Owner |
|---|---|
| Pinned mjfit provider, manifests, sampling and USD format export | `embodichain/toolkits/assembly/` |
| Native compound configs and actual hinge articulations | `embodichain/toolkits/assembly/native.py` |
| Robot-held mating geometry recipes | `extensions/embodichain_tasks/embodichain_tasks/manipulation/assembly/assets.py` |
| Reference execution, analytic probes and measurement | `extensions/embodichain_benchmarks/assembly/` |

`build_assembly` produces source body-local geometry, independent visual parts,
mass/COM/inertia, world attachments and source contact metadata. Loading a
checksummed manifest does not import either generator or MuJoCo.
Install the pinned Git revision of mjfit separately for generation. SDK
profiles using `newton[sim]` obtain MuJoCo through Newton's dependency bounds;
there is no assembly-specific extra or duplicate MuJoCo version constraint.
Other profiles need MuJoCo explicitly only when compiling MJCF.

`many.py:build_many_pegs` prepares the public sampler/build/grid composition as
one scene with separately owned plate/peg bodies. Its manifest records each
cell's configuration and source-prefixed names; cells are not replicated
arenas. Explicit outline-accuracy overrides retain both sampled and actual
configurations rather than silently changing the official geometry profile.

`prepare_native_configs(asset, directory)` prepares mixed rigid/hinge assets and
optionally file-backed rigid appearance through one public entry point. Benchmark
execution consumes those configurations without importing private visual writers.
`embodichain.toolkits.assembly.native.to_rigid_config(body)` builds a native compound for free/fixed bodies.
Hinges are rejected there and exported as actual native articulations through
`to_articulation_configs`; USD export includes the same collider ownership and
fixed-anchor hinge trees. Never combine all source convex pieces into one hull
or recompute authored inertia from overlapping pieces. Compiled MuJoCo geom
poses already account for mesh recentering; convert frame and quaternion order
once at the adapter boundary.
Native rigid appearance uses the owner's `visuals` list with body-local poses;
an explicit empty list hides it without removing colliders. Raw source contact
parameters belong to `attrs.material_props`, while contact envelopes, dimension
and priority belong to `attrs.collision_props`. Hinge export validates generated
names and destinations before writing into the output directory.

`embodichain/toolkits/assembly/_visual.py` owns body-local OBJ visuals for native rigid objects and URDF
moving links. The URDF visual origin stays identity because geom transforms
are already baked into that file; collision entries retain their own source
transforms. `usd.py` keeps mesh normals geometry-local under the authored geom
transform. Both preserve source mesh hard edges with explicit face normals.
Native visual cache paths include the asset hash and `_VISUAL_FORMAT_VERSION`;
bump the visual version when changing the rendered file format so the SDK
cannot reuse a stale imported mesh.
Rigid preview preparation validates every body name and OBJ/MTL destination
before writing the batch. Cache symlinks cannot redirect output outside its
preparation root; bodies with no visible surfaces retain empty appearance.

The application-owned `assets.py:build_task_assembly` composes robot-held task geometry using public generators,
including perforated gear cores and bosses. It records the changed topology
separately from the passive generator output. Peg tasks request a local support
floor through public `bottom_thickness`; passive/raw generator defaults remain
unchanged. Existing manifests pin their geometry, including an open bottom,
and do not acquire later task defaults. Task registration, manager
semantics and training deployment belong to the environment/RL owners. Assembly
deployments declare `rl_task` and register the common `EmbodiedEnv` through the
configuration loader, with distinct IDs for distinct deployments. Physical
components explicitly select scene preparation, resource export, action, startup,
interval, reset, observation and reward functors. `preparation.py` expands the
configured manifest into native entities and contact sensors before World creation;
it does not replace actions or inject events/conditions. Native preparation uses
`prepare_native_configs` with configured file-backed appearance, keeping collision
pieces and authored mass/contact data independent of a small number of visual
material groups. MuJoCo-Warp startup impedance writes use the physics-owned
batched gain view once, with the configured arm names and unchanged gain values. `_config.py` validates
`extensions.assembly`, including entity/control-part bindings, sampling counts
and measurement thresholds. Standard profiles explicitly size mating and grasp
sampling buffers for the solver's 4096-contact budget; smaller development
profiles retain their measured overrides. Buffer overflow fails immediately
instead of accepting truncated grasp evidence. The configured interval event advances shared state
once after each control transition, independently of selected termination terms.
Observation, reward and condition functors read that snapshot; reset updates only
selected rows. Task-owned `functors.py` owns observations, reset, rewards and conditions;
`_state.py` owns the shared measurement snapshot. Encoder dimensions follow the
selected arm and orientation representation. Task parameter and geometry
defaults live in `configs/tasks/manipulation/assembly/defaults.yaml` inside the
official task package; deployment declarations override those data-owned values.
The task's `resources.py` serializes assembly manifests/fidelity and delegates
prepared robot-tree copying to `embodichain.data.resources.snapshot_urdf_resources`.
Pure polygon distances belong to `embodichain.compute.geometry`; URDF gravity
feedforward belongs to `embodichain.compute.kinematics`. Robot joint-order and
root-frame adaptation remains task-owned. Preparation/resource-export entry paths
remain stable for frozen current-schema runs. Reset events leave selected
measurements pending until the host finishes `episode_prepare`, so first
observations and labels reflect the final prepared scene. Robot-base Euler
rotations follow the simulator's intrinsic `XYZ` convention. Nut advancement uses
scaled pitch for thresholds and scaled pitch times thread starts for lead.
No import-owned task registration aliases or environment-method replacements are used.

`embodichain_benchmarks.assembly.robot_reference` prepares the historical public Rizon/Robotiq approximation
from pinned source assets, with explicit gripper approximation and fidelity
metadata. Preparation is optional and runs before config loading; replay uses
frozen resources. Prepared models and their manifest/licenses live under the
data root's `generated/assembly/` tree, outside source-cache `extract/` repair.
Source validation never renames a prepared model seen by concurrent readers. Default arm/tool assembly belongs to task embodiment YAML,
not this converter or a combined robot preset.

`embodichain_benchmarks.assembly.precision` checks geometry/USD ownership and runs passive source/native
comparisons. MuJoCo Warp references declare their constants protocol through
`PrecisionRunCfg.warp_reference_initialization`: `cpu_compiled` preserves the
imported reference baseline; `warp_recomputed` explicitly calls public
`set_const_0`. The latter can replace nearly-zero translational inverse weights
with rotational weights, changing contact regularization without changing asset
geometry or mass. Newton's Warp model-property updates already recompute these
constants. Compare declared initialization protocols and effective solver options;
equal requested tolerance or MultiCCD settings alone do not establish physics
alignment. `PrecisionRunCfg.newton_solver` selects the Newton family
(`mujoco_warp` or `dexuni`); omission preserves MuJoCo Warp. The demo CLI uses
`--newton-solver`, and the matrix accepts `--newton-solvers` to compare families
without duplicating independent reference cells. A non-Newton backend rejects
an explicit family. `PrecisionRunCfg.solver` explicitly selects a MuJoCo algorithm comparison;
omission preserves the source setting. Source PGS gears require a distinct CG
comparison on current MuJoCo Warp. `use_mujoco_cpu` selects a separate,
single-world Newton/MuJoCo CPU profile, never an implicit Warp fallback.
Source and active MuJoCo algorithms remain separate from the Newton family.
Shared profile routing forwards the same contact capacity, algorithm/integrator,
iteration limit and explicit calibration knobs to direct MuJoCo Warp or DexUni's
nested `mujoco_options`. Native constrained probes accept `newton_solver` too;
their matrix rejects passive-only algorithm/CPU/calibration overrides.
Results record prepared `newton_solver` and SDK `solver_diagnostics`; DexUni
assembly profiles require measured `pure_mujoco` dispatch. Missing diagnostics,
a mismatched family, or VBD/coupled dispatch cannot silently stand in for this
contact/algorithm profile. Report grouping uses requested family/CPU identity,
including unsuccessful cells; actual dispatch is shown independently. Every
matrix cell owns its preparation directory.

`embodichain_benchmarks.assembly.probes` owns independent constrained circle/square clearance oracles.
Many-pegs endpoint seating and gear ratios do not replace those independent
measurements. Constrained `integration_substeps` defaults to four, resolving
stiff float32 contacts with a finer interval on every reference/native route.
Every settled physical step is scored; duration, constant efforts and the
accuracy budget are preserved. Requested and integration intervals, physical
step counts and sample counts remain distinct in results. The matrix rejects
mismatched worker refinement settings. Refinement increases time/storage work;
one substep remains an explicit reproducibility profile. The harness cannot certify an unavailable health measurement: backend
warnings, unsupported shapes and missing overflow accounting remain explicit.
Single-peg and grid scoring share a bounded seating predicate: centered
endpoints below the valid seat are failures, including floor passage.
`examples/sim/assembly/` documents SDK capabilities and qualified boundaries;
the source benchmark runner is `scripts/benchmark/assembly/precision.py`.
The matrix host initializes installed SDK schemas before authoring USD and
records geometry/ownership failures separately from independent source/direct
cells. Each invocation owns a fresh run directory. Worker exit status, asset
identity and configuration must match before its result is accepted.
Nominal robot preparation is shipped by the robot deployment
layer together with its source asset registration, frame regression and API.

User-facing installation, functional entry points, native RGB examples and the
upstream research citation live in the [Precision Assembly guide](../../../docs/source/features/toolkits/precision_assembly.md).

Focused coverage lives under `tests/toolkits/assembly/`, `tests/sim/assembly/`,
`tests/benchmark/assembly/` and `tests/gym/envs/tasks/test_blind_assembly.py`.
Portable-format tests explicitly block simulator, task, benchmark and generator
imports. Existing manifest schemas and content identities are independent of
Python module locations.
