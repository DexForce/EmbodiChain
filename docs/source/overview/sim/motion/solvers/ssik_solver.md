# SSIKSolver

`SSIKSolver` derives analytical IK geometry from the configured robot's actual
serial chain. Its Warp backend runs the same double precision geometric
calculation on CPU and CUDA, then filters candidates by joint limits and
forward-kinematics error. Its optional reference backend uses the pinned
[SSIK implementation](https://github.com/personalrobotics/ssik/tree/4109147f37a7ca3e2516c83853aca583612783f3)
on CPU for cross-checking and offline-prepared algebraic models.

## Supported geometry

| Backend | Supported chains | Execution |
|---|---|---|
| `"warp"` | 6R with parallel joint axes 1, 2 and 3 (zero based); strict 7R with concurrent shoulder and wrist axes | CPU or CUDA; geometric candidates only |
| `"warp"`, `geometry_mode="near_parallel"` | Certified small deviations from three-parallel 6R geometry, including ARX5 X5A | CPU or CUDA; geometric initialization plus bounded original-model numerical polishing |
| `"cuda_rr"` | Verified prepared Rizon4 joint prefix, including Rizon4S with its actual terminal frame | CUDA only; algebraic matrix-pencil roots on the device |
| `"reference"` | SSIK-dispatched 6R/7R models, with an offline artifact required for generic algebraic paths | CPU only; optional SSIK dependency |

The Warp paths cover the Aloha Mini arms and Tianji Marvin arms.
ARX5 and Flexiv Rizon4S require the explicit modes above or prepared CPU
reference artifacts; they do not meet the strict Warp geometry conditions. UR5 and strict
SRS W1 revisions provide additional geometry checks. Classification uses joint
axes and transforms rather than robot names. Approximate SRS, Franka-specific
geometry, and unprepared RR/HP algebraic models remain unsupported.
Unsupported geometry raises `ValueError` during initialization.

Aloha Mini and Tianji Marvin retain Pytorch as their default arm solvers;
Aloha's torso keeps its position-only Pytorch solver. Select SSIK explicitly
and use `srs_search="seed_local"` for Marvin continuous control. All finite
searches can miss solutions.

Default switching is held behind the full acceptance gates. Fresh-process
measurements currently exceed the 128 MiB CUDA scratch budget for the common
Aloha/Marvin candidate kernel because driver allocations are not included in
Torch-only counters. Precision and continuity results do not certify that
resource gate; see the current delivery benchmark report for measured limits.

## Configuration

This example builds a preset's kinematic chain without starting a simulation
or renderer. Preserve its root, end link, joint order and TCP when replacing
the solver configuration.

```python
import torch

from embodichain.lab.sim.motion.solvers import SSIKSolverCfg
from embodichain_robots import AlohaMiniCfg

device = torch.device("cpu")  # Use torch.device("cuda") for CUDA execution.
robot_cfg = AlohaMiniCfg.from_dict({"uid": "aloha_mini"})
part = "left_arm"
old_cfg = robot_cfg.solver_cfg[part]
chain = robot_cfg.build_pk_serial_chain(device=device)[part]

cfg = SSIKSolverCfg(
    urdf_path=robot_cfg.fpath,
    root_link_name=old_cfg.root_link_name,
    end_link_name=old_cfg.end_link_name,
    joint_names=robot_cfg.control_parts[part],
    tcp=old_cfg.tcp,
    backend="warp",
)
solver = cfg.init_solver(device=device, pk_serial_chain=chain)

seed = solver.get_default_qpos_seed().unsqueeze(0)
target = solver.get_fk(seed)
success, qpos = solver.get_ik(target, qpos_seed=seed)
valid, candidates = solver.get_ik(
    target, qpos_seed=seed, return_all_solutions=True
)
```

For Marvin, replace the robot preset with
`TianjiMarvinCfg.from_dict({"uid": "marvin", "with_gripper": False})` and
preserve that preset's TCP in the same way. Both arms and both gripper variants
use their selected source end link. Aloha Mini's torso uses a prismatic joint
and is outside this solver's supported chain types.

| Setting | Default | Meaning |
|---|---|---|
| `backend` | `"warp"` | Explicit execution backend; failures never switch backends |
| `geometry_mode` | `"strict"` | `"near_parallel"` explicitly permits bounded 6R axis deviations and numerical polishing |
| `axis_alignment_tolerance` | `1e-5` | Maximum normalized-axis deviation in near-parallel mode; does not relax FK acceptance |
| `srs_search` | `"uniform"` | `"seed_local"` adds scalar geometric redundancy optimization near the current seed |
| `num_samples` | `128` | Global SRS swivel samples, in addition to a defined seed swivel |
| `position_tolerance` | `1e-5` | Maximum accepted position error, metres |
| `rotation_tolerance` | `1e-5` | Maximum accepted orientation error, radians |
| `max_joint_step` | `None` | Optional hard per-joint distance from the seed or previous path frame, radians |
| `chunk_size` | `512` | Maximum target count per candidate-generation chunk |
| `reference_artifact` | `None` | Path to an offline-generated reference Python artifact |

```{tip}
For streamed control, pass the previous successful joint solution as the next
seed. Set `max_joint_step` to the largest acceptable per-frame joint change.
When no candidate satisfies that constraint, inspect the returned failure mask
before commanding the robot.
```

## Results and solution selection

Direct `get_ik` inputs are root-relative TCP transforms shaped `(N, 4, 4)`.
Seeds may have shape `(N, dof)` or `(1, dof)`. Omitting the seed uses the
effective joint-limit midpoint. The ordinary result chooses the valid branch
nearest to the limit-projected seed using the configured `ik_nearest_weight`.

| Call | Validity shape | Joint shape |
|---|---|---|
| `get_ik(..., return_all_solutions=False)` | `(N,)` | `(N, dof)` |
| `get_ik(..., return_all_solutions=True)` | `(N, K)` | `(N, K, dof)` |
| `get_ik_batch(...)` | `(...)` | `(..., dof)` |

The strict Warp candidate axis has eight slots for 6R, or
`8 * (num_samples + 1)` slots for uniform SRS 7R, or
`8 * (num_samples + 2)` with seed-local search. Prepared RR returns all 384
slots from its 16 locked-joint samples and 24 pencil-root slots per sample.
These are conditional candidates, not a complete continuous 7R manifold.
Near-parallel mode returns nine slots: eight geometric initializations and one
always-computed seed initialization for bounded original-model correction.
This enumerates that hybrid algorithm's candidates; it does not promise all
algebraic roots of a general, nonparallel 6R model.
Slots remain masked if their branch
does not exist, violates limits, exceeds the step constraint, or fails the FK
gate. The batch axis is retained when `N == 1`; an empty batch is supported.
Outputs contain float32 joint positions, and the FK gate checks the returned
precision. Internal geometric calculations use float64.

Failed or invalid slots contain a finite, limit-projected seed and have a
false mask. Treat the mask as authoritative. A failed Warp call does not
invoke numerical IK, SSIK reference, or any other fallback. Finite 7R sampling
does not enumerate its full continuous solution manifold: failure cannot prove
global unreachability. A defined seed swivel is considered alongside the global
grid; it is skipped if the seed swivel is geometrically undefined.

Each geometric branch selects the joint-period representative nearest to the
seed within the effective limits. Multiple winding numbers are not enumerated.
Candidates within `1e-6` radians of a limit may be projected onto that limit to
absorb input and model representation roundoff. The returned joint positions
must still lie strictly within the limits and pass the unchanged FK gates.
For ill-conditioned 6R branches, analytical reconstruction can recover a
representative at a nearby wrist or shoulder limit. It recomputes the coupled
joint angles geometrically and applies the same final limits and FK checks;
it does not enable an iterative IK fallback. Tangent subproblems account for
coordinate roundoff when a caller supplies float32 target matrices.
Singular wrists produce finite seed-dependent representatives of their
continuous family rather than an infinite list.

Nearest mode bounds candidate scratch by `chunk_size`. All-candidate mode
additionally allocates the complete `(N, K, dof)` output, so its output memory
still grows with the total batch and SRS sample count. Returned tensors retain
their values across later solver calls. `prepare_buffers(max_batch)` can
reserve bounded internal scratch before repeated calls.

## Continuous paths and shared methods

`supports_continuous_batch_ik` is true. Through
`robot.compute_batch_ik(..., continuous=True)`, candidates are selected
sequentially using the previous frame's selected joints and the same weights.
The optional `max_joint_step` is enforced against that previous frame. A failed
frame returns failure and retains the preceding pose; later frames continue
from that retained pose. For single-target streaming, maintain the seed in the
caller; the solver has no implicit cross-call joint state.

For SRS `seed_local`, the target-aware path hook regenerates the same candidate
pool at each frame from the preceding result. Its scalar search optimizes
analytical branches over swivel (or the free roll at a collapsed elbow circle),
including Euler pole anchors. It does not solve a Cartesian Jacobian system.
The global samples and unchanged final FK/step gates still apply.
CPU evaluates a complete path in a fused kernel. CUDA queues frame-dependent
calls while parallelizing paths, redundancy samples and branches within each
frame; the previous result stays on the device. This avoids serial FP64 work
in one GPU lane without introducing host result reads or synchronization.

`get_fk`, `get_fk_batch`, `get_qpos_limits`, `set_qpos_limits`, and
`update_with_robot_limit` retain the shared `BaseSolver` behavior. `set_tcp`
updates the end-link-to-TCP transform without rebuilding the chain geometry;
`set_ik_nearest_weight` updates branch-selection weights. Robot-level IK
converts arena-frame poses into the solver root frame. See
{ref}`shared FK/IK conventions <motion-solver-conventions>`.

## Selecting the numerical solver

Explicit solver configurations remain authoritative. To select Aloha Mini's
numerical arm setup while retaining its source frames and TCP:

```python
from embodichain.lab.sim.motion.solvers import PytorchSolverCfg

for part in ("left_arm", "right_arm"):
    current = robot_cfg.solver_cfg[part]
    robot_cfg.solver_cfg[part] = PytorchSolverCfg(
        root_link_name=current.root_link_name,
        end_link_name=current.end_link_name,
        tcp=current.tcp,
    )
```

In serialized configuration, specify `class_type: PytorchSolver` together with
the root and end links (and a nonidentity TCP when present). An override with a
`class_type` replaces that part's solver configuration; an override without it
updates attributes of the current default solver.

## Optional CPU reference and offline artifacts

Install the optional reference separately from the repository root:

```bash
pip install -e .
pip install -r requirements/ssik-reference.txt
```

The reference requirements are fixed to SSIK commit
`4109147f37a7ca3e2516c83853aca583612783f3`. Importing the ordinary solver package
and running the Warp backend do not require SSIK. The reference backend checks
the installed source revision and rejects CUDA devices. The VCS pin is kept
out of published wheel dependency metadata, as required by the release policy.

For the geometry in the example above, construct the reference using the same
CPU chain:

```python
reference_cfg = cfg.copy()
reference_cfg.backend = "reference"
reference_solver = reference_cfg.init_solver(
    device="cpu",
    pk_serial_chain=robot_cfg.build_pk_serial_chain(device=torch.device("cpu"))[part],
)
reference_success, reference_qpos = reference_solver.get_ik(
    target.cpu(), qpos_seed=seed.cpu()
)
```

Generic algebraic models must be built explicitly before initialization. This
command writes a Python artifact and an adjacent JSON manifest:

```bash
python -m embodichain.lab.sim.motion.solvers._ssik_reference \
    --urdf /absolute/path/robot.urdf \
    --root base_link --end flange \
    --output /absolute/path/my_arm_ik.py
```

Set `reference_artifact="/absolute/path/my_arm_ik.py"` on a reference config
using the identical chain. Loading checks the SSIK revision, kinematic
fingerprint, ordered joint names, root/end links and artifact source hash
before import. Do not edit a generated artifact independently of its manifest.
Symbolic generation never runs inside a runtime IK call.

Reference evaluation disables optional rescue and optional refinement. Some
SSIK families still contain internal root polishing or near-geometry numerical
correction; reference success for those families does not establish a pure
closed-form Warp implementation. The reference backend's branch counts are
determined by upstream SSIK rather than the Warp fixed-slot layout.

## Prepared ARX5 and Rizon4S models

For device execution, preserve the preset's original root, end link and TCP as
in the first example, then set `geometry_mode="near_parallel"` for ARX5, or
`backend="cuda_rr"` with a CUDA device for Rizon4S. Neither requires installing
SSIK. Both presets retain their Pytorch defaults pending complete performance
acceptance. ARX5 keeps its authored 85 mm TCP and tool rotation.

ARX5's device path generates eight surrogate geometric branches and one seed
initialization, then performs at most twelve monotone Newton corrections on
the actual 6R model, with bounded joint updates. This is an explicitly selected numerical correction, not a pure
closed-form result. Failure is returned without trying another solver.

The Rizon4S device path evaluates the prepared polynomial tensors, solves their
24-by-24 complex matrix pencils using cooperative Warp QR, and back-substitutes
roots entirely on CUDA. Workspaces are chunked to at most 32 targets; all 384
root slots pass through the original-model FK gate after float32 conversion.
It neither invokes CPU LAPACK nor generates coefficients during a query.
The fixed grid can miss a feasible lock value, especially under a small step
bound; increase of this grid requires another offline derivation and validation.
ARX5's target-aware batch path hook regenerates its nine candidates and their
periodic representatives from each preceding result on CPU and CUDA. For
single-target streaming, pass the preceding result explicitly. Rizon4S's
finite-grid batch selector uses candidates generated from the initial path
seed. Neither mode's bounded continuous coverage is certified to match Pytorch.
Successful static IK and low single-call latency do not establish that gate.

The bundled coefficient manifest records the source revision, hashes and an
independent DH/POE consistency certificate for every locked-joint chart. The
zero-lock chart corrects an upstream DH link distance using the source axis
geometry before offline rederivation. Reproduce the bundle with
`scripts/benchmark/robotics/kinematic_solver/prepare_ssik_rr_rizon4.py` and the
pinned SSIK source. CPU reference artifacts below preserve upstream behavior;
their zero-lock output is not an oracle for this corrected chart.

The bundled robot package contains explicit offline reference artifacts for
`ARX5` (the registered X5A source) and `FlexivRizon4S`. Both presets retain
Pytorch defaults. Install the pinned reference requirements above, then select the
prepared artifact explicitly:

```python
from importlib.resources import files

import torch

from embodichain.lab.sim.motion.solvers import SSIKSolverCfg
from embodichain_robots import ARX5Cfg, FlexivRizon4SCfg

robot_cfg = ARX5Cfg.from_dict({})
artifact_name = "arx5.py"
# For the Rizon 4S source model instead:
# robot_cfg = FlexivRizon4SCfg.from_dict({})
# artifact_name = "flexiv_rizon4s.py"

previous = robot_cfg.solver_cfg["arm"]
artifact = files("embodichain_robots").joinpath("_ssik_artifacts", artifact_name)
robot_cfg.solver_cfg["arm"] = SSIKSolverCfg(
    backend="reference",
    reference_artifact=str(artifact),
    urdf_path=robot_cfg.fpath,
    root_link_name=previous.root_link_name,
    end_link_name=previous.end_link_name,
    tcp=previous.tcp,
)
solver = robot_cfg.solver_cfg["arm"].init_solver(
    device="cpu",
    pk_serial_chain=robot_cfg.build_pk_serial_chain(device=torch.device("cpu"))["arm"],
)
```

The same configuration can be passed to a CPU-device `Robot` through the
normal `solver_cfg` override. CUDA-device robots require a CUDA-capable solver;
reference mode does not move targets to CPU implicitly. The artifacts are package resources, not an automatic
fallback. A modified source chain requires its own verified artifact.

ARX5's rounded half-turn origins produce a `7.24e-6` parallel-axis residual.
Its reference artifact contains general 6R Raghavan–Roth coefficients built
from the actual source; it does not snap the axes to the three-parallel Warp
family. Rebuild it with the ordinary offline command above, using
`--root base_link --end ee_link` and the selected `X5A.urdf`.

The Rizon 4S has non-spherical shoulder/wrist offsets. Its artifact reuses the
pinned SSIK Rizon 4 joint-lock/6R algebraic implementation after an explicit
joint-prefix comparison within `5e-7` representation tolerance. Its 43 mm
terminal-offset difference is handled by a baked rigid end-frame transform.
Source joint names/order, normalized axes and pre-terminal link offsets must
match before generation. The prepared joint-lock reference uses the fixed
16-sample grid with baked RR coefficients. A seed reorders that grid without
adding an unprepared lock value. It does not launch the upstream HP secondary
sweep after an empty result, so failure does not prove global unreachability.
`num_samples` controls SRS sampling and does not change this artifact grid. Final acceptance still uses the complete original
Rizon 4S model and configured TCP, at the unchanged pose tolerances.

Recreate that adapter with an explicit prebuilt selection:

```bash
python -m embodichain.lab.sim.motion.solvers._ssik_reference \
    --urdf /absolute/path/rizon4s.urdf \
    --root base_link --end flange \
    --prebuilt-model flexiv.rizon4_ik \
    --output /absolute/path/flexiv_rizon4s.py
```

Loading verifies the installed pinned prebuilt source and its coefficient
sidecars against the manifest before executing the adapter. A different
shoulder, wrist, axis or joint order is rejected; a shared robot name is not
sufficient evidence of compatibility. The ARX5 and Rizon4S RR paths evaluate numerical eigenvalues/roots and retain
upstream mandatory polishing of already accepted roots, even with optional
near-miss refinement and rescue disabled. These CPU reference paths do not
establish pure geometric Warp or hard real-time performance.

The unified benchmark accepts `--ssik-models arx5 flexiv_rizon4s` with explicit
`--ssik-reference-artifacts MODEL=/absolute/artifact.py` entries. These models
are opt-in; the existing default six-model benchmark remains unchanged.
Unsupported Warp geometry and missing reference preparation are reported as
unmeasured with a reason, while an incompatible supplied artifact is rejected.
Device experiments select `--ssik-geometry-mode near_parallel`,
`--ssik-srs-search seed_local` or `--ssik-cuda-rr`. These flags do not change
the default six-model benchmark or robot presets. Allocation counters and
fresh-process driver memory observations must both be considered when
certifying CUDA scratch; Torch allocator statistics alone omit driver stacks.

## How it works

1. Normalize the actual PK chain into home-frame joint axes, segment offsets
   and fixed end rotation; validate supported geometric structure once.
2. Remove the configured TCP and home rotation from each target.
3. Enumerate coupled shoulder/wrist and elbow branches for three-parallel 6R,
   or elbow and general three-axis rotation branches at sampled SRS swivels.
4. Select periodic representatives within limits and validate candidate FK
   against the target, using the configured position and orientation gates.
5. Return all candidate masks, or reduce to the seed-nearest valid result.

## References

- [Pinned SSIK source](https://github.com/personalrobotics/ssik/tree/4109147f37a7ca3e2516c83853aca583612783f3)
- [SSIK three-parallel decomposition](https://github.com/personalrobotics/ssik/blob/4109147f37a7ca3e2516c83853aca583612783f3/src/ssik/solvers/ikgeo/three_parallel.py)
- [IK-Geo geometric subproblems](https://github.com/rpiRobotics/ik-geo)
- [NVIDIA Warp](https://github.com/NVIDIA/warp)
