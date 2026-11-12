# Atomic Skills (`atomic_skills`)

## Scope

Atomic Skill is the canonical capability term and `AtomicSkill` is its
implementation base type. Atomic denotes the registration/invocation unit; a skill
may contain multiple phases, replans and retries. The typed planning and execution framework lives in
`embodichain/lab/sim/atomic_skills/`. It owns skill goals, resource binding,
planning, transport-neutral commands, execution sessions, recovery, tracking,
and effect-verification requests.

There are two supported caller paths:

1. Atomic Skills: direct callers construct `SkillInvocation` values.
2. Task Program: task-level semantic programs are validated, grounded, and
   lowered internally to Atomic Skills.

`embodichain.lab.task_program.semantics` is declarative only. It owns calls, scene
and robot-profile contracts, effects, and evidence declarations. It is not an
execution facade.

## Ownership graph

```text
Direct Python caller                    Task Program
SkillInvocation                       semantic program
          |                                  |
          |                       validate + ground + lower
          +------------------+---------------+
                             |
                             v
                    AtomicSkillEngine
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

`RecoveryPolicy.planning_budget` selects an outer complete-request search ledger
and optional cooperative deadline. Nested calls retain an existing parent's
ledger and deadline; None adds no ledger or wall budget. `PlanningRecovery`
values contain only immutable finite data and name their skill/revision.
`SkillDescriptor.planning_recovery_type` accepts one exact type or an explicit
tuple; unsupported declarations fail before skill preparation. These contracts
add no alternate built-in skill implementation or source-specific decoder.

Press keeps its default open-loop path. Explicit `PressPlanningRecovery` permits
at most two complete B1 plans, including nominal, and consumes the declared
contact-roll alternative only after failure. Failed phases use Motion's existing
continuous IK on the same targets and time grid: `lab/sim/motion/ik.py` adapts the
robot while `compute/kinematics/ik_path.py` owns bounded seed search. Press owns
only the failed-phase trigger and contact-roll alternative. Robot's
`compute_batch_ik(continuous=True)` selects solver candidate branches; it does
not implement this recovery's velocity window, perturbation order or RNG scope
and is not an interchangeable recovery backend. Final emitted full-robot velocity
qualification remains rejection-only; no transformed plan triggers another search.
The configured decoder accepts only the fixed `press_bounded` revision-one
declaration. `validate_press_extra_distance` validates declaration-owned extra
travel without changing skill defaults or deciding measured Press success.

`SkillDescriptor.supports_segment_motion_recovery` is read-only and derived
from the actual Atomic validation override. A copied class marker does not
advertise a handler. Semantic preflight rejects unsupported segment policy
before acquisition; the default false field is omitted from canonical descriptor
metadata, while true capabilities participate in the effective fingerprint.

`MotionPolicy` separates whole-action sample retries, segment recovery and a
final full-robot speed gate. The complete builder retains exact request/engine
identity, applies scoped transforms and performs rejection-only qualification.
`plan_preview` accepts only a same-observation projected context with explicit
preview role and a different occurrence; it uses the same engine. Preview
callbacks never become global engine configuration. Unsupported segment recovery
rejects before provider preparation, and external nominal motion is not replaced
by native sampling retries. No new Source skill implementation is registered.

`ExecutionPlanLifecycle` is an optional host-owned adoption port. It receives an
owned validated attempt before dispatch. Installation rejection latches the
session: later ticks/revisions cannot resume commands. `ExecutionRunner` keeps
its normal cancel-then-hold safe stop before the once-only abort notification.

Planning never steps simulation and never treats command completion as proof of
a physical effect. A gripper-close hold inserted into a moving arm path does not
preserve solved timing: retain required stop boundaries and validate the compiled
control-grid path, not only native planner samples. Runtime hosts may collect
versioned physical evidence after
each fresh observation and pass it to an evidence-aware effect verifier, phase
gate, or held-object guard. Evidence acquisition stays outside planning and
does not mutate symbolic task state; verification results are the only path
that commits or invalidates observed effects.

`MoveHeldObject` accepts ordered object-pose waypoints without rebasing the
verified grasp. Its goal optionally frees only the final world yaw; default
goals remain exact. An explicit `HeldObjectPoseGoal.object_id` requires that exact
input object before native or nominal planning and reports retained attachment
through the existing terminal effect-verification boundary. Omission keeps the
original effectless transport behavior and numerical path. Heading alternatives
are checked as complete motion paths,
with accepted rows retained independently and selected offsets in diagnostics.
Held-object transport and Pour keep the binding's configured grasp command; a
contact-blocked measured aperture does not retain closing effort.

Optional `hand_geometry_cfgs` enter through `AtomicSkillEngine` and its existing
`SkillPlanningServices`. `hand_geometry_cfg(target_id)` returns an independent
Lab configuration and fails for undeclared targets; it never infers a model or
captures geometry. Consumers retain declared pad-link order and replace closing
commands with the current resolved endpoint commands, including invocation
overrides. [Robot geometry ownership](../robot-system/robot-system.md#sampled-hand-geometry)
defines the capture boundary. Task Program declarations do not enter Atomic APIs.

`SkillPlanningServices.prepare_grasp_query()` resolves the declared hand
calibration by target ID and binds Scene capture to GraspKit's typed geometric
policy and the existing isolated Newton backend. The opt-in context manager
closes all query resources on normal exit, failure or cancellation. Opening
grounding and optional rigid-scene queries share the explicit observation scope;
undeclared calibration fails before capture. It selects no experiment revisions,
installs no alternate skill registry and leaves ordinary PickUp defaults unchanged.
Historical revision mapping belongs only to the frozen regression fixture.

`PickUpOptions.approach_direction` accepts one world vector or one vector per
planning row. PickUp and AxisAlign normalize each row independently and use the
same direction for candidate eligibility and pre-grasp translation. Calibrated
fixed grasps can explicitly select `approach_along_grasp_axis` so native
pre-grasp translation follows each realized TCP +Z after object rotation.
The default retains the world-vector contract. A benchmark
that freezes per-row grasp realisations must forward their directions to the
actual action; replacing them with one shared vector changes the request.
`approach_waypoints` adds evenly spaced poses on the straight pre-grasp line
to the same motion-policy call, so a free-space backend cannot sweep the target
on a diagonal approach. Zero keeps the three-pose plan.

`AxisAlignOptions.alignment_twist_rad` selects a scalar or per-row rotation
about the aligned target axis. Zero keeps shortest-arc behavior; an independent
benchmark generator can freeze another full-pose realisation without changing
the directed axis goal. `alignment_offset_m` similarly selects a scalar-vector
or per-row world translation from the lifted object centre; zero retains the
fixed-centre target. Neither option establishes path or physical feasibility.

`PourGoal.target` optionally supplies a `PourTarget` with a live receiving
opening and source-mouth geometry. The action derives the tilt direction,
source-body clearance and mouth-compensated approach/tilt/return path from the
held transform. Omission retains the original local-axis rotation mode.
Geometry belongs to the goal/integration; motion timing, IK and collision
checks remain with the selected planning policy. Validate executed mouth
alignment and height throughout tilted motion with host observations, rather
than treating an accepted final pose as evidence of pouring.

ExecuteDemoSkill executes task-specific demonstration assets with measured runtime
contracts through the same nominal-source and runner interfaces. Its internal
events do not create independent recovery boundaries.

PickUp, MoveHeldObject and Place can also assemble a pre-planning nominal motion
source. The source does not own semantic effects or environment lifecycle; see
[Real2Sim motion](../data-pipeline/real2sim.md) for source and replay limits.

`SlideGoal.joint_target` optionally supplies a public `SlideJointTarget` with an
absolute prismatic coordinate and a calibrated axis sign. Slide resolves the
remaining distance from same-timestamp joint/robot observations per planning row;
already-satisfied rows hold, invalid observations fail, and mixed active directions
are rejected. Omitting the target preserves fixed-distance `SlideOptions` and the
original Slide tutorial. Slide remains open-loop; this extension does not certify
contact or physical goal satisfaction.

`source_native_contact` composes preserved explicit source policies with the
shared Newton collision pipeline over the same GraspKit closing/opening/rest
algorithms. Contact decisions retain the query owner's conservative interval
semantics; threshold ambiguity remains unavailable. Scene keeps
native geometry capture, including the caller-selected sampling grid. The source
adapter owns only policy mapping, provenance and query cleanup; ordinary
`SkillPlanningServices.prepare_grasp_query` retains Newton and shared defaults.
`grasp_geometry.GraspOpeningTarget` is the single value definition, re-exported
with the same identity from the existing Atomic/runtime imports. No concrete
skill or alternate registry is installed by these query builders.

## Package map

| Concern | Source of truth |
|---|---|
| Core action and goal contracts | `atomic_skills/core.py`, `goals.py`, `affordance.py` |
| Affordance sampling values and deterministic branch selection | `atomic_skills/affordance_sampling.py` |
| Articulation affordance geometry adapter | `atomic_skills/articulation_geometry.py` |
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
`examples/sim/atomic_skill/support.py` and the selected physical components.

Scene-specific DexUni contact options are authored through
`create_tutorial_simulation(newton_solver_options=...)` before model construction.
`add_tutorial_robot` owns optional `newton_gripper_friction` and `arm_damping`
overrides; `scenario.add_dual_tutorial_robot` owns dual-arm friction and optional
mimic stabilization.
Scene modules retain calibrated close targets, fixture TCPs and grippers, including
Handover's Robotiq 2F-140 fixture. Read those example owners for current values;
calibration does not establish generic physical success or solver equivalence.
The coordinated-placement host opts into mimic stabilization and two-way DexUni
feedback, checks measured
lift and release placement, and retains commanded jaw preload when reconciling
measured arm/held-object state. A contact-blocked jaw measurement is not its
holding command. Its mesh fixtures use
the authored Z-up frame; shared pan geometry selects the handle and basin along
local Y. The three original-GLB benchmark suites instead declare
`support_mesh_frame: native_glb`, selecting along local Z in the native Y-up
mesh. Propagate this frame through grasp, target and footprint qualification;
retain each suite's matching asset scale and root pose. Keep these asset-specific
conventions in the scene/geometry owners.
The host caches textured OBJ geometry so native renderers and NoRender load the
same baked asset frame.
Release checks reuse complete projected mesh-footprint coverage and sampled
support-surface gaps. The example rejects negative sampled gaps and requires
an interior-floor witness within 8 mm to reject overhang, penetration, hovering
and rim-spanning outcomes while permitting curved undersides; the target-input
audit retains its separate collision-free-proposal semantics. This is finite
geometric evidence, not a proof of continuous 3D collision freedom or contact load.
Keep the base-to-table height invariant when raising the fixture.
Read held poses without resetting them at the replan boundary.
The launcher preserves an omitted device through simulation configuration so the
selected backend owns its default; explicit device selections retain priority.

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

Use `/add-atomic-skill` to extend goals, planners, registration and focused tests.
Keep planning side-effect-free; actions do not own task sequencing or simulator stepping.

## Direct-simulation example

`embodichain run-example sim/atomic_skill --skill` selects a fixed sample scene;
`run-atomic-skill` remains a compatibility alias. The generic launcher in
`embodichain/cli/run_example.py` resolves scripts directly from the ordinary
`examples/` tree and forwards arguments through `_example_runner.py`.

The example lives in `examples/sim/atomic_skill/` and is shipped as ordinary
distribution data with the rest of that tree. Its `__main__.py` owns
typed configuration and validated selection; `host.py` owns compilation,
expansion and output. Scene modules own physical setup and local defaults.
Runnable scene demonstrations and physical helpers live there. Specialized
examples such as coordinated manipulation and observed recovery run by file path through
`embodichain run-example`. The tutorial directory contains `single_arm.py`
(PickUp→Place), `dual_arm.py` (HandOver), and `embodiment_switching.py`
(robot/tool configuration reuse). They reuse example setup while explaining
typed application composition. The switching walkthrough owns its CLI matrix;
`examples/sim/atomic_skill/embodiments.py` owns its fixtures and measured playback
checks and recording cameras. Its task recipes remain hardware-independent;
swapping embodiments requires compatible resources and a physically feasible scene. The
[switching tutorial README](../../../scripts/tutorials/atomic_skill/README.md)
owns its eight-combination matrix and links the checked-in MP4 previews under
`docs/source/_static/atomic_skills/embodiment_switching/`. Read the
[example README](../../../examples/sim/atomic_skill/README.md)
for supported combinations, fixture calibration, playback and output artifacts.

`--variants N` requests N executions including the reference and creates N
parallel environments; the default is one. Supported affordance and trajectory
factors apply without a Cartesian recipe bank or collection lifecycle.
Unsupported explicit factors fail before simulation. Mounted-gripper TCPs are
example-owned full poses forwarded to single/dual-arm solvers.

The engine's optional `plan_transform` runs before concatenation and projected
state propagation. The example and Task Program share
`atomic_skills/_plan_expansion.py`, using `SkillPlanTemplateAdapter` and
`motion.expansion` to rebuild commands, velocities and phase metadata together.
Core simulation and Task Program code do not depend on example modules.
Built-in phase permissions may only narrow; editable joints come from explicit
motion bindings. Contact/hand samples, endpoints, required targets, stationary
participants, zero-time joins and callback indices remain exact.

Logical variant identity stays separate from physical slots. Measured state
remains row-local, including Slide's replanned push after pull. Playback and
exports share a control clock with explicit settling and final holds.
Completion reports planning and playback; it does not establish measured
success or Gym dataset collection. Scenes remain fixed; spatial generalization
belongs to benchmark hosts. See
[motion expansion](../motion-planning/motion-planning.md) for the shared
operators and host ownership boundary.

## Recommended change sites

| Change | Owning location |
|---|---|
| Goal/options or action behavior | matching file under `atomic_skills/primitives/` plus `goals.py` when shared |
| Articulation-link sampling or affordance geometry metadata | `atomic_skills/articulation_geometry.py`; keep `objects/articulation.py` limited to meshes, FK, and topology |
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
pytest -q tests/sim/atomic_skills
pytest -q tests/lab/task_program/semantics
pytest -q tests/lab/task_program
python docs/scripts/check_api_docs.py
```

For public API changes also run the docs checker tests and Sphinx dummy build.
For simulator adapters add an environment-level test that exercises normal
`env.step()` consumption and safe cancellation.
