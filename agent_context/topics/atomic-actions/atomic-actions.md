# Atomic Skills (`atomic_actions`)

## Scope

Atomic Skill is the capability term; `AtomicAction` is its implementation
base type. The typed planning and execution framework lives in
`embodichain/lab/sim/atomic_actions/`. It owns action goals, resource binding,
planning, transport-neutral commands, execution sessions, recovery, tracking,
and effect-verification requests.

There are two supported caller paths:

1. Atomic Skills: direct callers construct `ActionInvocation` values.
2. Task Program: task-level semantic programs are validated, grounded, and
   lowered internally to Atomic Skills.

`embodichain.lab.task_program.semantics` is declarative only. It owns calls, scene
and robot-profile contracts, effects, and evidence declarations. It is not an
execution facade.

## Ownership graph

```text
Direct Python caller                    Task Program
ActionInvocation                       semantic program
          |                                  |
          |                       validate + ground + lower
          +------------------+---------------+
                             |
                             v
                    AtomicActionEngine
                  resolve -> plan -> start
                             |
                             v
                    ExecutionSession
          context revisions + row-local recovery
                             |
                             v
                    ExecutionRunner
        observe -> schedule -> dispatch -> verify -> stop
                             |
                             v
        EndpointCommandRouter + transports + clock
```

Planning never steps simulation and never treats command completion as proof of
a physical effect. Runtime hosts may collect versioned physical evidence after
each fresh observation and pass it to an evidence-aware effect verifier, phase
gate, or held-object guard. Evidence acquisition stays outside planning and
does not mutate symbolic task state; verification results are the only path
that commits or invalidates observed effects.

`MoveHeldObject` accepts ordered object-pose waypoints without rebasing the
verified grasp. Its goal optionally frees only the final world yaw; default
goals remain exact. Heading alternatives are checked as complete motion paths,
with accepted rows retained independently and selected offsets in diagnostics.

`SlideGoal.joint_target` optionally supplies a public `SlideJointTarget` with an
absolute prismatic coordinate and a calibrated axis sign. Slide resolves the
remaining distance from same-timestamp joint/robot observations per planning row;
already-satisfied rows hold, invalid observations fail, and mixed active directions
are rejected. Omitting the target preserves fixed-distance `SlideOptions` and the
original Slide tutorial. Slide remains open-loop; this extension does not certify
contact or physical goal satisfaction.

## Package map

| Concern | Source of truth |
|---|---|
| Core action and goal contracts | `atomic_actions/core.py`, `goals.py`, `affordance.py` |
| Affordance sampling values and deterministic branch selection | `atomic_actions/affordance_sampling.py` |
| Articulation affordance geometry adapter | `atomic_actions/articulation_geometry.py` |
| Invocation, binding, and policies | `invocation.py`, `bindings.py`, `policies.py`, `control.py` |
| Robot/task/scene state | `state.py`, `scene.py` |
| Runtime physical evidence | `evidence.py`, `scene.py`, `runner.py` |
| Plans and runtime commands | `plans.py`, `runtime_commands.py` |
| Session state machine | `execution.py` |
| Verification request/result values | `verification.py` |
| Runner and transports | `runner.py`, `transports.py`, `tracking.py` |
| Engine and built-in registration | `engine.py`, `primitives/`, `__init__.py` |
| Simulation ports | `sim_adapter.py` |
| Task Program semantic declarations | `embodichain/lab/task_program/semantics/` |
| Task Program lowering/execution | `embodichain/lab/task_program/compiler/`, `runtime/` |
| Gym lifecycle bridge | `embodichain/lab/gym/envs/task_program/bridge.py` |

## Read on demand

- [Planning and execution](execution.md): invocation resolution, control grid, row-local recovery and verification.
- [Affordance sampling](affordance-sampling.md): legal geometric variation, reproducible branch identity, action feasibility, and the sim-only host boundary.
- [Articulation geometry](articulation-geometry.md): topology/mesh ownership and directional affordance contracts.

The `scripts/tutorials/atomic_action/place.py` tutorial configures a
task-scoped Newton contact candidate on the cube and gripper collision links
before `SimulationManager.prepare()`. Shared tutorial helpers also author
`condim=4` on gripper/contact links and grasped objects before preparation.
Newton manipulation overlays use `ke=4e4`, `kd=4e2`,
`torsional_friction=0.1`, and `rolling_friction=0.01` on the gripper and
directly manipulated object contact surfaces. `condim=4` activates the
torsional term in MuJoCo-Warp; the rolling coefficient is retained for
solvers/`condim=6` that support it.
The profile was selected from Newton's native-contact examples and a fixed-
seed Default comparison; it improves contact force-closure in the replay but
does not promise identical open-loop Place behavior, whose attachment,
mimic-joint, and release timing semantics remain backend-sensitive.
The values are intentionally scoped and are not world defaults. Other surfaces
keep their source contact dimension. Replay does not rewrite MuJoCo model
arrays after CUDA Graph capture; contact storage is allocated for the authored
dimensions. The packaged PGI/Franka embodiment and
repeated-pick/open-drawer environment components carry the same Newton-only
overlays, while their declared Default backend continues to consume only the
portable fields.

## Semantic integration boundary

`semantics` contains:

- `calls.py`: Pick, Place, HandOver, registered declarative calls, and catalog;
- `scene.py`: canonical refs, registry, affordances, collision roles;
- `profiles.py`: robot resources, endpoints, presets, `EffectAssurance`;
- `effects.py` / `evidence.py`: effect and measured-evidence contracts; and
- `integration.py`: provider-free manifests, binding, diagnostics.

It does not contain compiler, executor, parallel scheduler, or application
facade modules. Task Program owns the internal semantic lowering lifecycle.

Every `SkillPolicyPreset` selects explicit effect authority:

- `EffectAssurance.VERIFIED`: curated Pick/Place/HandOver calls require exact
  monitor mappings and measured evidence.
- `EffectAssurance.PROJECTED`: monitor mappings are forbidden; expected
  symbolic effects are projected after command completion.

Projected assurance is not physical task success.

Use `/add-atomic-action` to extend goals, planners, registration and focused tests.
Keep planning side-effect-free; actions do not own task sequencing or simulator stepping.

The direct-simulation sample host is `embodichain run-atomic-action --skill`:
`lab/scripts/run_atomic_action.py` owns selection, batching and cleanup;
`lab/scripts/_atomic_action/` owns the ten example scenes and internal sample defaults.
The required skill selects the scene; physics, robot and planner are direct
selectors. Robot compatibility is checked before scene import or simulator
construction: Pickup, Place, MoveJoint and Handover accept the three sample
presets, while the other scenes require UR5. Layouts, skill options and
expansion settings remain sample-owned rather than externally configurable.
The dual-arm sample factory in `lab/scripts/_atomic_action/scenario.py` retains
the arm preset unless Python callers select `solver="pytorch"`. The Handover
scene currently makes that selection for Franka; UR presets keep analytical UR
IK. Cartesian preparation uses the shared
[batched motion facade](../motion-planning/planner-details.md#normalization-and-capabilities).
Mounted-gripper TCP calibration belongs to `lab/scripts/_atomic_action/tcp_presets.py`.
The UR5/PGI, Franka/PGI and UR10/Robotiq mounting poses are independent; named
tool points retain the existing general, grasp and legacy PGI dual calibrations.
Handover explicitly mounts the Robotiq 2F-140 asset on every supported arm.
UR5 and Franka own separate Robotiq Handover TCP matrices (0.21 m local Z
and a 90-degree local-Z rotation); the generic dual factory retains PGI unless
that gripper is selected. The scene's antipodal generator uses its matching
140 mm opening geometry, and closure commands follow the six Robotiq joints'
mimic directions with a 0.4 rad master angle.
Scene callers select those points, and single/dual robot factories forward full
poses to their solvers. Explicit Python PGI `tcp_z` overrides affect only local
Z while preserving rotation and lateral offsets. TCP calibration changes the
kinematic tool frame, not physical geometry.
The matching tutorial files are compatibility entry points to those same scenes.
Affordance branch overrides stay independent of physical slots, then the host
adapts each compiled reference to `motion.expansion` for spatial variants.
Contact phases and hand columns stay exact; protected phase samples are copied
from the source after adaptation, including near-identical zero-time joins.
Original action joins and intervals are restored before replay so callbacks
retain their source indices. MoveJoint compiles each required joint target as
a separate invocation, preserving its ordered waypoints as action boundaries
through control-grid retiming and spatial expansion. Pour's single
rotation/return segment is protected from spatial perturbation because its
required interior target has no exported
waypoint boundary; its preceding PickUp free phases remain editable. Slide
replans its push from each row's robot/handle observations after pull. Every
Slide reference remains row-local through expansion and replay, including its
start, locked contact path, timing and sampling provenance; shared affordance
branch identity does not imply equivalent measured states. Output status records
planning/playback completion, with no measured physical-success authority or
Gym dataset collection. Usage lives in the host's `README.md`.

## Recommended change sites

| Change | Owning location |
|---|---|
| Goal/options or action behavior | matching file under `atomic_actions/primitives/` plus `goals.py` when shared |
| Articulation-link sampling or affordance geometry metadata | `atomic_actions/articulation_geometry.py`; keep `objects/articulation.py` limited to meshes, FK, and topology |
| Engine catalog or lifecycle | `engine.py` |
| Session transitions or recovery | `execution.py` |
| Verification value contracts | `verification.py` |
| Scheduling, dispatch, and safe stop | `runner.py` |
| Runtime payload/transport | `runtime_commands.py`, `transports.py` |
| Scene observation | `scene.py`, `sim_adapter.py` |
| Robot resources/presets | `semantics/profiles.py` |
| Semantic call/effect declarations | `semantics/calls.py`, `effects.py`, `evidence.py` |
| Task-level semantic sequencing | `embodichain/lab/task_program/` |

## Focused validation

```bash
pytest -q tests/sim/atomic_actions
pytest -q tests/lab/task_program/semantics
pytest -q tests/lab/task_program
python docs/scripts/check_api_docs.py
```

For public API changes also run the docs checker tests and Sphinx dummy build.
For simulator adapters add an environment-level test that exercises normal
`env.step()` consumption and safe cancellation.
