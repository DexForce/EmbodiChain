# Precision assembly assets

Precision assembly is a reusable capability outside GenSim. GenSim pipelines,
Gym tasks and examples consume the same neutral asset API; the toolkit does not
import GenSim, simulation runtime, task packages or benchmark helpers.

| Responsibility | Owner |
|---|---|
| Pinned mjfit provider, manifests, sampling and USD format export | `embodichain/toolkits/assembly/` |
| Native compound configs and actual hinge articulations | `embodichain/lab/sim/assembly/` |
| Robot-held mating geometry recipes | `extensions/embodichain_tasks/embodichain_tasks/manipulation/assembly/assets.py` |
| Reference execution, analytic probes and measurement | `extensions/embodichain_benchmarks/assembly/` |

`build_assembly` produces source body-local geometry, independent visual parts,
mass/COM/inertia, world attachments and source contact metadata. Loading a
checksummed manifest does not import either generator or MuJoCo.
Install the pinned Git revision of mjfit separately for generation; the
`assembly` extra installs index-distributed MuJoCo only.

`many.py:build_many_pegs` prepares the public sampler/build/grid composition as
one scene with separately owned plate/peg bodies. Its manifest records each
cell's configuration and source-prefixed names; cells are not replicated
arenas. Explicit outline-accuracy overrides retain both sampled and actual
configurations rather than silently changing the official geometry profile.

`embodichain.lab.sim.assembly.to_rigid_config(body)` builds a native compound for free/fixed bodies.
Hinges are rejected there and exported as actual native articulations through
`to_articulation_configs`; USD export includes the same collider ownership and
fixed-anchor hinge trees. Never combine all source convex pieces into one hull
or recompute authored inertia from overlapping pieces. Compiled MuJoCo geom
poses already account for mesh recentering; convert frame and quaternion order
once at the adapter boundary.
For native compound visuals, `visual_parts=None` renders the collision parts;
an explicit empty list hides them. Hinge export validates generated names and
destinations before writing into the output directory.

`lab/sim/assembly/_visual.py` owns body-local OBJ visuals for native rigid objects and URDF
moving links. The URDF visual origin stays identity because geom transforms
are already baked into that file; collision entries retain their own source
transforms. `usd.py` keeps mesh normals geometry-local under the authored geom
transform. Both preserve source mesh hard edges with explicit face normals.
Native visual cache paths include the asset hash and `_VISUAL_FORMAT_VERSION`;
bump the visual version when changing the rendered file format so the SDK
cannot reuse a stale imported mesh.

The application-owned `assets.py:build_task_assembly` composes robot-held task geometry using public generators,
including perforated gear cores and bosses. It records the changed topology
separately from the passive generator output. Peg tasks request a local support
floor through public `bottom_thickness`; passive/raw generator defaults remain
unchanged. Existing manifests pin their geometry, including an open bottom,
and do not acquire later task defaults. Task registration, manager
semantics and training deployment belong to the environment/RL owners. Each
official task supplies its builder to `BlindAssemblyEnv`; the base environment
loads prepared manifests directly and never imports the task extension.

`embodichain_benchmarks.assembly.robot_reference` prepares the historical public Rizon/Robotiq approximation
from pinned source assets, with explicit gripper approximation and fidelity
metadata. Preparation is optional and runs before config loading; replay uses
frozen resources. Default arm/tool assembly belongs to task embodiment YAML,
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
alignment. `PrecisionRunCfg.solver` explicitly selects a solver comparison;
omission preserves the source setting. Source PGS gears require a distinct CG
comparison on current MuJoCo Warp. `use_mujoco_cpu` selects a separate,
single-world Newton/MuJoCo CPU profile, never an implicit Warp fallback.
Source and active solver identities remain separate in results.

`embodichain_benchmarks.assembly.probes` owns independent constrained circle/square clearance oracles.
Many-pegs endpoint seating and gear ratios do not replace those independent
measurements. The harness cannot certify an unavailable health measurement: backend
warnings, unsupported shapes and missing overflow accounting remain explicit.
`examples/sim/assembly/` documents dependency patches and qualified boundaries;
the source benchmark runner is `scripts/benchmark/assembly/precision.py`.

Focused coverage lives under `tests/toolkits/assembly/`, `tests/sim/assembly/`,
`tests/benchmark/assembly/` and `tests/gym/envs/tasks/test_blind_assembly.py`.
Portable-format tests explicitly block simulator, task, benchmark and generator
imports. Existing manifest schemas and content identities are independent of
Python module locations.
