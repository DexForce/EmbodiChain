# Codex grasp-to-assembly harness

Move an **assemble object onto a base object** using an existing successful
`assemble/result.json`. Codex proposes a grasp, initial object arrangement, and
approach/insertion parameters from the assembly descriptions and measured meshes.
The host validates each proposal, builds object and TCP waypoints, and plans a
continuous joint trajectory with analytical IK and TOPPRA. Neither mesh nor the
accepted base-relative assembly pose is regenerated.

This tool lives entirely in `grasp_assemble`. The supported robot is currently
**UR5 + DH PGI 140/80**, constructed directly from production simulation APIs.
It does not invoke atomic skills or start a browser viewer.

## Minimal configuration and commands

Only the source assembly result is required:

```json
{
  "assembly_result": "/absolute/path/to/successful/assembly/result.json"
}
```

The harness inherits `base_description`, `assemble_description`,
`action_description`, frame conventions, and detailed design from that result.
Do not copy them into the grasp configuration. An optional `instruction` adds a
short request, such as "Start with the phone lying flat." Runs default to
`outputs/grasp_assemble/<config-file-stem>/` under the repository root;
`output_dir` optionally overrides this. All supplied relative paths resolve from
the configuration directory.

The examples are deliberately small:

- `configs/tree_mug.json`: put a mug on its tree-shaped rack, with an explicit
  20-degree final orientation tolerance for the wide opening.
- `configs/phone_stand.json`: place a phone on its stand.

Both examples explicitly enable the temporary grasp constraint. A minimal
configuration uses finger contacts alone. Source `latest.json` must contain a
successful assembly; otherwise select a successful cycle's `result.json`
explicitly.

Run from the repository root:

```bash
conda activate embodichain2
python scripts/tools/grasp_assemble/generate.py \
  --config scripts/tools/grasp_assemble/configs/tree_mug.json
python scripts/tools/grasp_assemble/visualize.py \
  --result outputs/grasp_assemble/tree_mug/latest.json
```

For another assembly, use its example configuration:

```bash
python scripts/tools/grasp_assemble/generate.py \
  --config scripts/tools/grasp_assemble/configs/phone_stand.json
python scripts/tools/grasp_assemble/visualize.py \
  --result outputs/grasp_assemble/phone_stand/latest.json --headless
```

`--assembly-result PATH` overrides the configured source. To start from an
existing grasp for the **same source assembly JSON hash**, add
`--grasp-result PATH` to generation. Codex can reuse its object-local grasp while
adjusting the task parameters and initial arrangement.

Generation saves a unique run and updates `latest.json`. Visualization rechecks
the frozen source and replans before execution. In the native window, press Enter
in the terminal to execute, then Enter again to close the final scene. `--headless`
executes without a window and records `pick_place.mp4`. Generation and visualization
are separate commands, so a Codex call does not block an active render loop.

## What the agent generates

Each `evaluate` action proposes three related parts:

| Part | Meaning |
| --- | --- |
| `T_assemble_tcp` | A 4 × 4 TCP pose in the original exported assemble-object frame |
| `layout` | Initial base/assemble XY coordinates and yaw offsets relative to the robot root |
| `task_plan` | Initial assemble orientation, clearance, approach/insertion/retraction parameters, and an explanation |

The host constructs the prompt from the source design, mesh bounds, calibrated
gripper geometry, fixed assembly transform, effective configuration, and previous
geometry/planning observations. No mug-specific grasp or compulsory inversion is
embedded in this process: the target orientation comes from the accepted assembly
matrix. A phone may start flat and be tilted into its stand; a mug may start
upright and be inverted onto a support.

The agent's task plan contains:

| Field | Meaning |
| --- | --- |
| `assemble_initial_rpy_degrees` | Initial object-to-world extrinsic XYZ rotation before the layout yaw offset |
| `clearance` | Extra height used to clear the base during transport |
| `pre_grasp_distance` | Open-finger approach distance along TCP +Z |
| `insertion_direction_base`, `insertion_distance` | Direction toward the target and distance from pre-insertion to target |
| `retract_direction_base`, `retract_distance` | Gripper withdrawal after release |
| `reason` | Explanation of the proposed approach |

Directions are expressed in the **base-object frame**, and the host normalizes
them. Distances are meters; task-plan angles and layout yaw offsets are degrees.
The insertion direction points **toward** the final target:

```text
direction_world = R_world_base @ insertion_direction_base
p_preinsert = p_target - insertion_distance * direction_world
```

The gripper retreats along its independently chosen direction after release.
For example, an object can approach along an inclined slot instead of always
moving vertically downward. Invalid transforms, non-finite parameters, zero
direction vectors, and out-of-range distances are rejected before planning.
Clearance and insertion distance must be within 0.02–0.50 m; pre-grasp and
retraction distances must be within 0.02–0.30 m. These are proposal bounds;
the full sampled corridor must still pass geometry and robot checks.

When an initial pose is omitted, the host uses transformed mesh vertices to
ground the object; the mesh origin need not be its lowest point. The base remains
Z-up, matching the source assembly's gravity convention. The initial assemble
orientation comes from the agent and may be arbitrary. The host also projects the
uniform-density mesh center of mass onto the bottom support polygon and requires
at least a 5-degree estimated tipping margin. This rejects narrow-edge balancing
before planning and gives the agent feedback to choose a broader resting face.
The support estimate is a geometric heuristic, not a dynamics guarantee.

New jobs with an omitted pose default to bounded layout optimization. The agent
chooses initial positions and yaw while the host checks object separation, robot
distance, reachability, joint excursion, and collisions. The supported robot root
is at the world origin with identity rotation, which the host verifies. Moving the
base moves the target through the same fixed `T_base_assemble`.

Advanced overrides remain available:

- Explicit `T_world_base` and `T_world_assemble_initial` matrices preserve their
  supplied heights/orientations; automatic grounding and initial roll/pitch/yaw
  do not replace them. If both are supplied, layout defaults to `fixed`.
  `layout.mode: "optimize"` enables XY/yaw search relative to those seeds.
- Explicit motion parameters take priority over the corresponding agent proposals.
  Sampling, timing limits, joint weights, and execution tolerances remain host
  settings; the agent cannot relax them to pass a failed candidate.
- `layout` bounds constrain the search. A failed bounded search reports diagnostics
  rather than claiming no feasible grasp exists.

Optimization requires the configured minimum number of distinct layouts, then
selects the accepted candidate with the lowest measured score:

```text
score = sum_i(weight_i * travel_i_degrees)
        + 2 * max_i(weight_i * range_i_degrees)
```

`motion.joint_cost_weights` defaults to `[1, 1, 1, 1, 1, 0.2]`, so the sixth joint
costs less when it can replace larger arm movements. The same weights guide IK
branch selection. Full turns count as actual motion; hard joint limits, maximum
steps, collision checks and derivative limits remain unweighted.

## Frames, waypoints, and continuous timing

All transforms use column vectors:

```text
T_world_assemble_target = T_world_base @ T_base_assemble
T_world_tcp_pick = T_world_assemble_initial @ T_assemble_tcp
T_world_tcp_place = T_world_assemble_target @ T_assemble_tcp
```

The same `T_assemble_tcp` is used throughout transport, with no regrasp. TCP +Z
points from the palm toward the fingertips; TCP X is the closing direction. The
TCP is 0.160 m along the gripper root's +Z. The host reads the actual asymmetric
finger collision meshes: their usable open gap is about 0.074 m. A broad object
must expose a thinner reachable feature to this gripper.

The host constructs this route from the resolved task plan:

| Waypoint | Motion |
| --- | --- |
| `grasp` | Approach along TCP +Z with open fingers, then close |
| `lift` | Raise the object to a mesh-dependent clearance height |
| `rotate` | Reach the target assembly orientation at clearance |
| `transit` | Move above the pre-insertion point while staying clear of the base |
| `hover` | Reach the start of the chosen insertion corridor |
| `place` | Follow the insertion direction to the accepted assembly pose |

After placement, the fingers open and the gripper withdraws along the selected
retraction direction. The name `hover` is retained for the pre-insertion waypoint;
it need not lie vertically above the target.

The robot initializes at the open pre-grasp state; travel from an arbitrary home
configuration is excluded. The host samples the Cartesian route and enumerates
analytical IK solutions, including valid full-turn equivalents. A weighted
shortest-path search selects a continuous branch sequence. It tries finer
sampling and alternative rotation senses when needed, then checks the fitted
trajectory against collision geometry.

Only gripper state changes separate the arm timing:

| TOPPRA block | Complete arm path |
| --- | --- |
| `approach` | Open pre-grasp → grasp |
| `carry` | Grasp → lift → rotate → transit → pre-insert → place |
| `retract` | Released pose → withdrawal |

Each moving block uses **one TOPPRA solve**. The entire carry also uses one
simulator playback call. Internal waypoint annotations do not insert rest
commands or restart playback. Finger closure/opening occur between these blocks
while the arm stays stationary.

TOPPRA supplies a velocity/acceleration-constrained speed profile for the joint
spline. The host fits nonnegative cubic Hermite path speed, integrates it into
monotone progress, and evaluates analytical joint velocity, acceleration and jerk.
It selects a verified speed fit, dilates time as needed, adds a 2% margin, and
rounds to the control grid. Joint velocity and acceleration are continuous and
zero at block endpoints. Jerk may change at knots. Internal turns can slow down
to meet the limits, but no zero speed is imposed at a waypoint. This timing is
numerically checked, not globally time-optimal or a continuous-time certificate.

Optional host timing overrides include:

```json
{
  "motion": {
    "velocity_limit": 0.6,
    "acceleration_limit": 1.2,
    "jerk_limit": 6.0,
    "joint_cost_weights": [1, 1, 1, 1, 1, 0.2],
    "duration_scale": 1.0
  }
}
```

Units are rad/s, rad/s², and rad/s³. `duration_scale` can further slow a trajectory.
The physics timestep is 0.01 s. The installed `toppra` package in `embodichain2`
is used without downloading another trajectory library.

## Validation and physical execution

The host revalidates the source assembly on its original solids and checks mesh
and source JSON hashes. Grasp preflight uses original-mesh FCL and solid
intersection queries, preserving cavities. The assembly generator uses VISACD
for coarse collision checks, and native replay uses VISACD convex decomposition
for the moving rigid body. For a candidate, the host checks open-finger approach,
bilateral finger proximity, carried-object
clearance, release and withdrawal, robot-link collisions against the base/ground,
TCP endpoint accuracy, joint continuity, and sampled motion derivatives. It also
checks the final smoothed trajectory through FK. Retiming does not repair an
obstructed geometric path.

The first contact of each finger is measured by binary search on its actual
prismatic joint. Replay closes 3 mm beyond those measured contact positions,
capped at the 0.04 m joint limit. This check does not prove force closure or
frictional retention. Robot self collision and continuous swept-volume clearance
are not currently certified.

`execution.grasp_mode` is `contact` unless explicitly set to `fixed_constraint`.
The latter creates a native fixed joint only after verifying the measured closed
grasp. Its frames preserve the measured object-to-gripper transform without
snapping the object to the target. The joint is removed before opening, including
on exception cleanup. This mode tests transport and placement with assisted
retention; it does not demonstrate a friction-only grasp.

```bash
python scripts/tools/grasp_assemble/visualize.py \
  --result outputs/grasp_assemble/phone_stand/latest.json \
  --grasp-mode fixed_constraint --headless
```

The CLI grasp-mode override affects only that replay. Default holds are 100
physics steps before release and 240 after withdrawal. The base is a static
triangle mesh; the assemble object is a dynamic body with convex decomposition,
using the current fixed 0.15 kg simulation mass. Shape generalization does not
automatically infer material properties or mass.

Generation's `success: true` means **geometry and planning accepted**;
`physical_success: null` means execution has not yet been measured. In
`execution.json`, `trajectory_completed` and `physical_success` are separate.
Final pose success requires position error within 15 mm and orientation error
within `execution.rotation_tolerance_degrees`. New jobs with an omitted pose
default to 10 degrees; `tree_mug.json` explicitly allows 20 degrees. Rotation error
uses full SO(3), so rotation about an object's axis still counts. These tolerances
do not alter the fixed assembly target, grasp-attachment check, or FK accuracy gate.

Replay observes `grasp_closed`, `before_release`, and `retracted` without pausing
at internal carry waypoints. A completed trajectory can still fail after release.
The command exits with code 0 only when the physical result passes its tolerances.

The supported scope is a single rigid object placed in a gravity-supported
assembly with one retained grasp. Force fits, screw motion, deformable objects,
regrasping, and arbitrary robots require additional planning/execution support.
An agent-generated insertion direction must still pass the numerical checks;
the harness does not guarantee that every natural-language assembly is feasible.

## How `codex exec` is called

Each decision is a fresh ephemeral call. The complete task and current feedback
are included in its prompt; a previous desktop conversation is not required.
The strict actions are `evaluate`, `finish`, and `fail`. `finish` must name an
accepted candidate, and the host independently rechecks it before saving success.
The agent proposes numerical parameters; the host owns geometry, robot commands,
acceptance and persistence.

The argument list is equivalent to:

```bash
codex exec --ignore-user-config --ephemeral --skip-git-repo-check \
  --sandbox read-only --disable shell_tool --disable apps --disable multi_agent \
  -c project_doc_max_bytes=0 -c 'web_search="disabled"' \
  --json --color never --output-schema /absolute/run/action.schema.json \
  --output-last-message /absolute/run/turn_01.json \
  --cd /absolute/run - < /absolute/run/turn_01.prompt.txt
```

`codex.model` optionally adds `--model MODEL_ID`. The existing Codex login is
used. `--output-schema` constrains the final action JSON; `--json` emits a separate
event stream. The host reads the final response file, saves both log streams,
and records the exact arguments in `turn_NN.command.json`. No model tools, Blender
generation, or shell commands are run by these grasp decisions.

## Results and replay compatibility

Each run retains the source assembly and hash, original and resolved config,
agent task plan, selected layout, `T_assemble_tcp`, candidate checks, and complete
action/observation trace. Motion artifacts include:

- `trajectory.npz`: positions, velocities, accelerations, jerks, and arrival
  intervals `dt`.
- `plan.json`: phase ranges, FK/collision/derivative checks, planning time,
  joint costs, and three timing blocks. `waypoint_passages` reports block-relative
  passage times, global sample indices, and arm-speed norms.
- `replay_plan/plan.json`: the plan recomputed from saved inputs before execution.
- `execution.json` and `pick_place.mp4`: measured physical outcome and recording.

Carry phase ranges retain `lift`, `rotate`, `transfer`, `approach_target`, and
`place` as annotations. The command line reports elapsed generation/planning
seconds, trajectory duration, weighted joint cost, travel, and peak derivatives.
`validation.toppra_calls` records one solve per moving arm block.

Existing saved TOPPRA jobs with explicit world poses remain replayable. Their
stored parameters take precedence; absent execution rotation tolerances retain
the older 20-degree replay default. Unsupported timing strategies and obsolete
`sample_count` settings are rejected. Historical execution measurements are not
rewritten when a result is replanned.

## Measured generic example

The phone/stand run `outputs/grasp_assemble/phone_stand/grasp_e572cgce/` exercised
the full Codex parameter-generation loop and native physical replay. Codex chose
a broad-face initial resting pose, a width pinch near the phone's upper edge,
and vertical insertion and withdrawal. It corrected rejected grasps from collision
feedback without manually supplied initial matrices or path parameters.

Generation took 217.00 s. Replay planning took 1.61 s and produced a 29.21 s
commanded trajectory: approach 1.62 s, carry 24.84 s, retract 1.51 s, plus finger
commands and the grasp hold. Each arm block called TOPPRA once. Separate release
and settling holds are excluded from that duration.

Native replay passed with **0.67 mm position error and 0.08-degree rotation error**
after settling. The temporary grasp constraint was released before opening and
was inactive at the end. The run's `execution.json` and `pick_place.mp4` retain
the measurements and recording. This qualifies the tested phone and stand;
other shapes still require their own planning and physical checks.

The existing mug result was also replayed with the generalized code, preserving
its saved poses and legacy route. It passed at 2.5 mm / 15.18 degrees under its
explicit 20-degree tolerance; this compatibility recording is stored separately
in `outputs/grasp_assemble/tree_mug/generic_compatibility/`.
