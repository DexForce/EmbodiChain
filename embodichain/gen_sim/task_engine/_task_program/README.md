# GenSim Task Program Integration

This package owns GenSim declarations and provider assembly, not another
Task Program executor. The shared compiler, semantic runtime, execution
sessions, Gym bridge, and demo executor remain authoritative.

The execution/compiler implementation follows the `2620929c` baseline. The
limited public additions are optional `SlideGoal.joint_target` and strict decoding
of the existing empty `MoveEndEffectorOptions`. E3, E5 and a
single-object E2 have passed fresh physical regressions against it. The full
can task and final multi-seed qualification remain separate gates. See
`BASELINE_LIMITS.md` for the independent recovery gap and the registered-call
phase-protection gap. The latter blocks complete qualification even when a
normal physical rollout succeeds; terminal acceptance is not in-flight safety.

## Local Boundary

- `assembly.py` composes the existing simulation integration with task-owned
  named stability presets. Its factory only selects provider instances and
  returns the exact public `TaskProgramEnvironmentAdapter`.
- `grasp_filter.py` wraps the existing parallel-jaw sampling service. Immutable
  geometry rules filter unrestricted single-arm proposals by object-local
  region and release clearance. Stock PickUp still chooses symmetric variants,
  solves candidate IK, builds trajectories, and publishes measured effects.
- `stability.py` implements `SegmentPostPolicyPort`. It observes object poses
  and object-to-endpoint drift while yielding ordinary target-qpos holds.
  Only Gym consumes those commands and advances simulation.
- `align_held.py` binds a declared object axis using the latest scene
  observation. It emits a normal `MoveHeldObject` goal with a minimal rotation;
  it neither solves IK nor overwrites the runtime's measured grasp state.
  The explicit `current_object_pose` selector changes orientation without
  moving the object to a nominal scene position. E1 upright placement reuses
  E2's constrained acquisition and staging recipe. E4 upright transfer aligns
  on the source arm before HandOver and corrects on the receiver afterward.
- `stack_place.py` registers a separate stack-placement call and option preset.
  It reuses the original Place goal/effect binding but does not inherit the low
  table-placement TCP cap, which could otherwise eliminate the release retreat.
- `release_clearance.py` binds generated staging heights and a fingerprinted
  withdrawal distance to the existing `MoveEndEffector` skill. The free hand
  first clears vertically, then moves toward its own arm root before Park.
  It refuses a live held attachment and never solves IK or applies commands.
- `motion.py` supplies a planning policy for multi-waypoint EEF approaches.
  Cartesian samples are solved by the original motion generator; command
  timing, execution, cancellation and recovery remain owned by the core.
- `articulation_binding.py` inspects and fingerprints a single-prismatic USD and
  generates explicit Slide/withdraw/Park calls. `articulation_slide.py` binds
  these to existing public skills and measures joint retention via post-policies.
  It does not add a private executor or claim contact-qualified motion.
  Initial table penetration over 2 mm or a buried handle fails before publication
  and at runtime binding. Native joint limits must match the scaled declaration;
  stale public limits are refreshed during assembly using the already-active
  native values, before the first scene observation.
  Scene geometry is measured in the base-link frame that runtime reset preserves.
  Proxy-envelope fitting is an explicit scene-revision operation, never a hidden
  loading fallback or a change to the source USD.
- `constraints.json` is a closed, versioned, inert bundle artifact. The
  integration fingerprint includes its complete contents and the program.
  Unknown fields and old bundles without this artifact fail before execution.
- A policy records measurements and time-window results. It never modifies
  `TaskState`, row eligibility, the program counter, recovery queues, or final
  task success. The core bridge combines its result with the skill result.

`wait_stable` retains its declared meaning: an entity must satisfy a named
stability preset. Upright and stacking presets require a contiguous stable
window. Held-object presets additionally fail on observed grasp slip or target
loss instead of restarting the observation window after a drop.

Placement acceptance is also repeated at the final cleanup segment. A can
passing an early stable check must remain upright after hand clearance and
Park; a stack must remain supported for another complete stable window after
withdrawal. These are ordinary shared segment post-policies and validators,
not a separate task-result update path.
Stack stability measures translation and rotation of both objects against one
shared window. Movement of either object restarts the complete observation
window; a stationary upper object cannot conceal motion of its support.

Coordinated placement additionally checks the scene-generated destination;
being stationary at a wrong location is not sufficient. Coordinated hold
checks that destination and both attachments. Upright HandOver hold checks
the requested axis and the receiving attachment without inventing a position
goal. Ordinary position validators, including Pour return targets, are also
rechecked after cleanup. Upright acquisition and staging use the canonical
`PreparedScene.table_top_z`, not a second required copy in AABB metadata.

The planner's predicted holder map only selects declarations (for example,
omitting a duplicate Pick before a continuation HandOver). It is not runtime
ownership: the shared compiler and runtime still require measured attachment
evidence and block calls after failure.

Baseline registered lowerers may not return action options. GenSim therefore
binds each constrained Pick's declared approach direction into a task-scoped
`gen_sim.pick.<task-group>` policy alias before compilation. Directions come
from scene declarations and preceding planned alignment, not runtime option
mutation. The factory and lowerer identity classes specialize only this inert
call ID because the baseline requires class-level identities; they introduce
no new execution behavior or Atomic Skill implementation.

Each invocation uses the baseline's single shared world direction. Geometry
filtering and stock IK remain row-local. A rejected or empty proposal row uses
the sampler's existing infinite-cost failure convention, never successful
padding or a private eligibility update. End-specific, best-grasp and paired
grasp protocols retain their baseline behavior; constrained Pick requires
unrestricted sampling without a fixed grasp or post-sampling rotation.

The coordinated Robotiq recipe admits 5 mm contact pairs and uses the measured
65 mm pad extent around the configured TCP. The generic 130 mm conservative
envelope is retained for inclined single-arm grasps because the current
three-box collision approximation couples palm position to finger length.
Both service declarations are fingerprinted. Physical joint limits, hand
commands, and contact/holding acceptance thresholds are unchanged.

Stacking from a low handover grasp includes explicit staging, free-arm park,
upper regrasp and final placement calls. The shared Task Program observes every
transition. This is a compiled recipe, not a private retry/state machine.

## Current Migration Status

`configured.py` owns task service decoding and delegates public option decoding
directly to the baseline. It does not inject extra fields into public Options.
`services.py` owns immutable Pick, MoveHeldObject, relative Place and coordinated
transport routes and lowerers. Baseline component composition, common scalar
decoders, Park and Pour are reused without adding fields to the shared YAML
schema. The shared configured service modules and official pour-water format
have been restored to the baseline.

The public language/compiler/validator stack has also been restored: GenSim
upright and stability requirements now use the existing named post-policy port,
not new public validator types. Earlier broad shared Atomic Skill and compiler
changes were withdrawn; the optional Slide joint target is a separate extension.
MoveHeldObject now binds one exact target; alternative
declarations are rejected instead of being silently discarded. Public
HandOver owns its original release sequence without an extra settle phase.
Simulation-factory internals used during assembly are isolated compatibility
dependencies; no Session or Bridge method is replaced.

The fingerprint manifest uses `semantic_integration_fingerprint/v2` with the
explicit `gen_sim.task_program/2620929c/v3` adapter contract. Old bundles are
rejected before component loading and must be regenerated.
Version 3 adds the support object's stable window and a final failed-attempt
camera fetch before reset. The v2 physical runs are retained as historical
evidence, not qualification of this revised acceptance behavior.

E1-E6 are the Task Engine execution scope. E7-E9 routing is rejected before
semantic graph generation and bundle asset writing. Initial E6 support requires
one environment and a fixed-base, single-prismatic, self-contained metre-authored
USD with uniform scale, a zero closed endpoint, and one identifiable collision
handle. Opening and closing select absolute native joint endpoints, not a fixed
movement independent of current state. A one-second window must stay within
1 mm joint spread and the target tolerance (opening at most 3 cm, closing at most
5 mm, both capped at one quarter of the joint span), after Slide, withdrawal, and
Park. Withdrawal also requires measured hand-open posture. This proves neither
contact-supported sliding nor geometric hand separation or in-flight safety.
Arbitrary scenes and robots are not physically qualified by these checks.

Physical acceptance requires the user's complete can-stacking and original
tray-holding CLI runs, with measured stable end states and recorded failures.
Provider-level CLI probes are necessary but do not substitute for these runs.
