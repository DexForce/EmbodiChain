# SSIKSolver

`SSIKSolver` derives geometric IK or selects a verified prepared algebraic
model from the configured robot's actual serial chain. Its Warp backend runs
double precision computation on CPU and CUDA, then filters candidates by joint limits and
forward-kinematics error. Its optional reference backend uses the pinned
[SSIK implementation](https://github.com/personalrobotics/ssik/tree/4109147f37a7ca3e2516c83853aca583612783f3)
on CPU for cross-checking and offline-prepared algebraic models.

## Supported geometry

| Backend | Supported chains | Execution |
|---|---|---|
| `"warp"` | 6R with parallel joint axes 1, 2 and 3 (zero based); strict 7R with concurrent shoulder and wrist axes | CPU or CUDA; geometric candidates only |
| `"warp"`, `geometry_mode="near_parallel"` | Certified small deviations from three-parallel 6R geometry, including ARX5 X5A | CPU or CUDA; geometric initialization plus bounded original-model numerical polishing |
| `"warp"`, `geometry_mode="rr"` | Verified prepared Rizon4 joint prefix, including Rizon4S with its actual terminal frame | CPU or CUDA; Warp algebraic matrix-pencil roots |
| `"cuda_rr"` | Compatibility alias for the prepared RR path | CUDA only; prefer `backend="warp", geometry_mode="rr"` for new configurations |
| `"reference"` | SSIK-dispatched 6R/7R models, with an offline artifact required for generic algebraic paths | CPU only; optional SSIK dependency |

The Warp paths cover Aloha Mini, Tianji Marvin, ARX5 and Flexiv Rizon4S on
CPU and CUDA. ARX5 and Rizon4S use the explicit near-parallel and RR modes
above; they do not meet the strict geometric conditions. UR5 and strict
SRS W1 revisions provide additional geometry checks. Classification uses joint
axes and transforms rather than robot names. Approximate SRS, Franka-specific
geometry, and unprepared RR/HP algebraic models remain unsupported.
Unsupported geometry raises `ValueError` during initialization.

Aloha Mini's two arm defaults use strict Warp SSIK. Both Tianji Marvin
variants use Warp SSIK with `srs_search="seed_local"`. CPU and CUDA use the
same configured geometry and acceptance thresholds, without the optional SSIK
reference dependency. ARX5 defaults to `geometry_mode="near_parallel"`;
Rizon4S defaults to `geometry_mode="rr", rr_search="seed_local"`.
Aloha's torso retains its position-only Pytorch solver.
Explicit `solver_cfg` replacements remain authoritative. All finite searches
can miss solutions; inspect the validity mask before executing a result.

The benchmark reports record the measured models, control paths and hardware
windows separately. CUDA resource measurements include driver allocations
alongside Torch/Warp counters. A preset's default does not establish a latency
guarantee for arbitrary workloads or concurrent rendering.

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
| `geometry_mode` | `"strict"` | `"near_parallel"` permits bounded 6R deviations and polishing; `"rr"` explicitly selects a verified prepared algebraic Warp model |
| `axis_alignment_tolerance` | `1e-5` | Maximum normalized-axis deviation in near-parallel mode; does not relax FK acceptance |
| `srs_search` | `"uniform"` | `"seed_local"` adds scalar geometric redundancy optimization near the current seed |
| `rr_search` | `"fixed"` | For Warp `geometry_mode="rr"` or legacy `cuda_rr`, `"seed_local"` adds local and analytically predicted lock samples using the parameterized RR bundle |
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
`8 * (num_samples + 2)` with seed-local search. Fixed prepared RR returns all
384 slots from its 16 locked-joint samples and 24 pencil-root slots per sample.
RR `seed_local` returns 600 slots from the same 16 global and nine local samples.
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
CPU SRS evaluates a complete path in a fused kernel. Near-parallel and
seed-local RR regenerate their pools from the previous result on either device.
CUDA queues frame-dependent
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

Explicit solver configurations remain authoritative. Restore the original
numerical arm settings while preserving source frames, TCP, limits and weights
with the following replacement. This applies to Aloha Mini, either Marvin
variant, ARX5 and Rizon4S:

```python
from copy import deepcopy

from embodichain.lab.sim.motion.solvers import PytorchSolverCfg

parts = ("arm",) if "arm" in robot_cfg.solver_cfg else ("left_arm", "right_arm")
for part in parts:
    current = robot_cfg.solver_cfg[part]
    robot_cfg.solver_cfg[part] = PytorchSolverCfg(
        urdf_path=robot_cfg.fpath,
        joint_names=robot_cfg.control_parts[part],
        root_link_name=current.root_link_name,
        end_link_name=current.end_link_name,
        tcp=deepcopy(current.tcp),
        user_qpos_limits=deepcopy(current.user_qpos_limits),
        ik_nearest_weight=deepcopy(current.ik_nearest_weight),
        num_samples=30,
        max_iterations=500,
        pos_eps=5e-4,
        rot_eps=5e-4,
    )
```

For a serialized replacement, use the new `PytorchSolverCfg.to_dict()` values
above, including `class_type: PytorchSolver`, the root/end links and the TCP.
A dictionary containing `class_type` replaces the part independently; it does
not inherit those integration fields. A class-type-only override therefore
loses the preset frames and can reset Marvin's TCP. Construct a fresh numerical
config rather than changing the class label on an SSIK dictionary: copying its
`num_samples=128` would retain that value instead of restoring the original
numerical setting of 30. A partial dictionary without `class_type` only changes
attributes of the selected default solver.

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
`backend="warp", geometry_mode="rr"` on CPU or CUDA for Rizon4S. Neither requires installing
SSIK. Both presets retain their Pytorch defaults pending complete performance
acceptance. ARX5 keeps its authored 85 mm TCP and tool rotation.

ARX5's device path generates eight surrogate geometric branches and one seed
initialization, then performs at most twelve monotone Newton corrections on
the actual 6R model, with bounded joint updates. This is an explicitly selected numerical correction, not a pure
closed-form result. Failure is returned without trying another solver.

The Rizon4S device path evaluates the prepared polynomial tensors, solves their
24-by-24 shifted matrix pencils with Warp QR, and back-substitutes roots on
the selected device. CPU uses scalar Warp scheduling and CUDA uses cooperative
warps; they share the coefficient system, samples and projective recovery. Workspaces are chunked to at most 32 targets; every
root slot passes through the original-model FK gate after float32 conversion.
Production queries use neither SciPy/LAPACK nor the official SSIK package,
and never generate symbolic coefficients. The private QZ oracle and optional
SSIK reference are used only for explicit numerical validation.
The default `rr_search="fixed"` preserves the 16-lock coefficient bundle.
For continuous control, select `rr_search="seed_local"` explicitly. Its
parameterized bundle evaluates arbitrary lock angles without runtime symbolic
derivation. Nine extra samples include the preceding lock, symmetric local
offsets and two original-model Jacobian predictions. The predictions select
only a lock angle: all returned joint solutions still come from algebraic RR
roots. They are not iterative Cartesian IK or a numerical fallback. The same
global pool remains present, and hard limits, step bounds and FK acceptance
mask every slot.
The parameterized path checks that the input is a finite rigid pose, then
applies two fixed polar-conditioning steps to the internal rotation used for
coefficient evaluation. This prevents float32 frame-conversion roundoff from
violating the exact SO(3) polynomial identities. It does not repair malformed
targets, change the target used for final original-model FK validation, or
relax either acceptance tolerance.
Existing real roots receive three bounded determinant-Newton corrections on
the original quadratic matrix pencil. CPU and CUDA call the same Warp kernel;
the reciprocal chart keeps large roots numerically bounded. This corrects
shift-invert roundoff without changing the sampled locks, root masks or
candidate count. It is algebraic root refinement, and does not introduce a
joint-space or Cartesian IK fallback. Unsafe corrections retain the last safe
root; the original-model FK gate still decides whether a candidate is valid.
ARX5's target-aware batch path hook regenerates its nine candidates and their
periodic representatives from each preceding result on CPU and CUDA. For
single-target streaming, pass the preceding result explicitly. Rizon4S's fixed
grid batch selector uses the initial path seed. Its `seed_local` path hook
regenerates the full pool using each preceding selected posture. Finite lock
sampling can still miss a feasible interval. Successful static IK and low
single-call latency alone do not establish bounded continuous coverage.

The bundled coefficient manifest records the source revision, hashes and an
independent DH/POE consistency certificate for every locked-joint chart. The
zero-lock chart corrects an upstream DH link distance using the source axis
geometry before offline rederivation. Reproduce the bundle with
`scripts/benchmark/robotics/kinematic_solver/prepare_ssik_rr_rizon4.py` and the
pinned SSIK source. CPU reference artifacts below preserve upstream behavior;
their zero-lock output is not an oracle for this corrected chart.

The separate parameterized bundle uses POE joint-frame factors and affine
sin/cos lock coefficients. Its manifest includes exact reconstruction identities,
coefficient hashes and an independent original-PK pose certificate. Rebuild it
with the same maintained builder, adding `--parameterized --urdf PATH` and
`--ssik-source PINNED_SOURCE`. Both bundles are packaged independently; loading
validates the selected bundle and actual model before device execution.

The bundled robot package contains explicit offline reference artifacts for
`ARX5` (the registered X5A source) and `FlexivRizon4S`. These artifacts are
optional validation tools. Install the pinned reference requirements above, then select the
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

For explicit validation, the same configuration can be passed to a CPU-device `Robot` through the
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

The unified benchmark selects all eight arm/model cases by default and inherits
mode and search settings from each preset. Explicit `--ssik-geometry-mode`,
`--ssik-srs-search` and `--ssik-rr-search` override the corresponding algorithm.
Use `--ssik-models` to select a smaller workload. New RR experiments use
`--ssik-geometry-mode rr --ssik-rr-search seed_local --ssik-devices cpu cuda`;
`--ssik-cuda-rr` retains the historical CUDA-only comparison.

Official SSIK is never imported by a normal Warp benchmark. Enable numerical
alignment explicitly with `--ssik-reference`, and provide
`--ssik-reference-artifacts MODEL=/absolute/artifact.py` for prepared algebraic
models. Reference rows report correctness only; they do not contribute timing
or memory gates. Unsupported geometry and missing preparation remain explicitly
unmeasured, while incompatible artifacts fail initialization. Allocation
counters and fresh-process driver observations must both be considered for
CUDA scratch; Torch statistics alone omit driver allocations.

Add `--ssik-robot-e2e` to run an isolated, engine-backed `Robot` benchmark with
the native `no-render` renderer. This preserves native physics and state
retrieval; headless mode alone can still initialize Hybrid rendering. The
result certifies this control context, not simultaneous GUI/rendering load.
It measures 10,000 changing-target calls with previous-result seeds through
the real root-state query, arena/root conversion and `Robot.compute_ik`.
Setup, physical simulation stepping and external-device transport are excluded.
Outputs undergo independent original-PK FK, effective-limit, step and failure-hold
checks, and validated coverage is compared with the same standalone streaming
targets. An unavailable engine or missing coverage comparison is unmeasured;
neither can pass the gate.
When both arms of a preset are selected, the same entry point also measures
10,000 paired calls through one actual Robot and checks the 8 ms paired gate.
Formal timing also requires a quiet measurement window. Record other GPU
compute processes during the run and retain contended measurements separately;
a successful short diagnostic does not replace the 10,000-call tail-latency
certificate.

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
