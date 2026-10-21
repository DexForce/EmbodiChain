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
Held-object transport and Pour keep the binding's configured grasp command; a
contact-blocked measured aperture does not retain closing effort.

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

Sample Newton contact overlays are local to the selected scene and gripper;
they are not world defaults or a guarantee of open-loop physical success.
Author contact dimensions and material settings before `prepare()`, never by
rewriting captured solver arrays. The current values live in
`lab/scripts/_atomic_action/support.py` and the selected physical components.

For UR5 with `dexuni`, the paper-cup `move_held_object.py` tutorial opts into
VBD contact history with `latest` collision matching, sliding friction 4 on the
gripper alone, and a 0.0115 m candidate close target. Transport orientation
is not yet repeatably aligned. The native contact alpha and the cup
and ground materials are unchanged. `create_tutorial_simulation` accepts
`newton_solver_options`, and `add_tutorial_robot` accepts
`newton_gripper_friction` and optional `arm_damping`; these are authored before
model construction. This
calibration also applies to the UR5-only `pour.py` tutorial with its 0.013 m
cube close target. The UR5 plastic-tray `coordinated_pickment.py` tutorial uses
the same contact settings with a 0.023 m close target; the dual-arm helper
`add_dual_tutorial_robot` also accepts `newton_gripper_friction`.
The current Handover fixture uses Robotiq 2F-140 and retains its fixture-owned
angular close targets and full TCP poses in `lab/scripts/_atomic_action/handover.py`.
The UR5 DexUni `place.py` profile uses a 0.013 m close target, gripper friction 4,
contact history with `latest` matching, contact alpha 0.5, and arm damping 1000.
The lower arm damping reduces tracking lag when the fingers open; finger
drives and authored release waypoints remain unchanged. Complete cached
replays preserve the grasp to release, but landing orientation still varies;
this is a scoped improvement, not a claim of full solver equivalence.
These explicit scene choices do not change world defaults
or establish a generic object-width-to-gripper-target mapping.

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

## Direct-simulation sample host

`embodichain run-atomic-action --skill` selects a sample scene.
`lab/scripts/run_atomic_action.py` owns selection, batching and cleanup;
`lab/scripts/_atomic_action/` owns scenes, robot compatibility, solver choices,
and local defaults. Tutorial files forwarding there are compatibility entries.
Read the [host README](../../../embodichain/lab/scripts/_atomic_action/README.md)
for supported combinations, fixture calibration, expansion and output artifacts.

Mounted-gripper TCPs belong to `lab/scripts/_atomic_action/tcp_presets.py`.
They are independent full poses per fixture, forwarded to single/dual-arm
solvers; changing a TCP changes the kinematic tool frame, not physical geometry.

Affordance branch identity stays separate from physical slots. The host adapts
compiled references to `motion.expansion` while preserving contact/hand samples,
action boundaries and callback timing. Measured state remains row-local,
including Slide's replanned push after pull. Completion reports planning and
playback only; it does not establish measured success or Gym dataset collection.
See [motion expansion](../motion-planning/motion-planning.md) for the shared
operators and host ownership boundary.

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
