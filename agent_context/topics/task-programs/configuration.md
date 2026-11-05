# Task Program language and semantic integration

Read this when the request needs these details. [Topic overview](task-programs.md).

## Language contract

`TaskProgramCfg` owns:

- one `program_id`;
- exact `TaskProgramIntegrationCfg` IDs;
- optional overall language ``instruction`` and named targets; and
- one bounded program tree.

Supported nodes are `SequenceCfg`, `RepeatCfg`, `SegmentCfg`, `InvokeCfg`, and
`ParallelCfg` with an owned `BarrierCfg`. Built-in call configs are `PickCfg`,
`PlaceCfg`, and `HandOverCfg`; `RegisteredSemanticCallCfg` is the allowlisted
extension form.

The program's root `instruction` owns its overall task language; each explicit
segment may declare a finer `instruction`. The compiler freezes both levels,
including repeated/parallel segments. The Gym bridge and common demo executor
bind them to the actual episode-selected program; omitted segment labels are
marked as overall-task fallbacks. See `resolve_demo_instruction` in `demo.py`
for task/direct-call/legacy precedence, shared with handwritten demonstrations. See
the [persistence contract](../data-pipeline/data-pipeline.md#persistence-contract)
for the independent task/subtask language mapping.

Choose segment boundaries around independently describable subgoals under their
required incoming task state. A segment may own one or several Semantic Calls;
sequential splitting retains task state and downstream planning look-ahead.
Keep release settling and final-state validators on the subgoal that establishes
those outcomes. Atomic trajectory phase names or arbitrary waypoint counts do
not establish semantic boundaries for an unannotated recorded demonstration.
Language instructions express intent; physical acceptance still depends on the
selected effect authority and validators, not on the presence of a label.

Measured segment validators can check lift from the segment's incoming pose,
world-axis position relative to another live object, and object uprightness.
`object_near_relative_target` compares an object's measured position against a
fresh reference position plus a declared world-axis displacement; it does not
reuse the reference pose captured for planning.
They belong to the program's acceptance contract; a projected call completing
does not replace these measurements. The Gym tutorial at
`scripts/tutorials/gym/simple_task_program.py` shows Cartesian end-effector
motion with a tutorial-local registered call, the existing `MoveEndEffector`
Atomic Skill, and the common Gym environment and demonstration executor.

Unknown fields, duplicate keys, non-finite values, invalid exact types,
excessive depth/nodes/repeats, cyclic or executable registered payloads, and
unresolved references fail before live providers are touched.

Task Program poses follow the EmbodiChain quaternion contract. `PoseCfg` and
serialized targets require `quaternion_xyzw`; `SemanticPose` stores and reports
the same order. The configured hand-over service uses
`final_quaternion_xyzw`. Identity is `[0, 0, 0, 1]`; legacy `*_wxyz` keys are
unknown fields and are rejected rather than silently reinterpreted.

`TaskProgramCompiler` resolves canonical scene references, expands bounded
repeats and cyclic targets, assigns stable segment/call indices, preserves
parallel branches, and returns an immutable `CompiledTaskProgram`.

## Semantic integration

`task_program.semantics` contains:

- `calls.py`: Semantic Calls and the call catalog;
- `scene.py`: canonical references, registry, affordances, and collision roles;
- `profiles.py`: robot resources, endpoints, Atomic Skill bindings, policy
  presets, and `EffectAssurance`;
- `effects.py` / `evidence.py`: typed effects and measured evidence; and
- `integration.py`: provider-free manifests, static binding, and diagnostics.

`SimulationTaskProgramRegistration` is the standard composition root. Its
fingerprint covers scene/profile declarations, the Semantic Call catalog,
settling presets, grounders, endpoint transports, evidence and safety
factories, and registered-call lowerers. Adapter creation calls
`assert_unchanged()` and revalidates all live bindings.
`fingerprint_payload()` returns an independent, revalidated JSON declaration.
Reconstructing a registration with `dataclasses.replace` does not duplicate its
scene-derived standard grounders; duplicate custom grounders remain errors.

`runtime_services.settle_presets` is the closed serialized route to this same
registration-owned table. Omission retains the existing dynamic presets; an
explicit table owns its complete named selection. Components may contribute
disjoint names, but duplicate ownership is rejected. Ordinary monitor fields
retain `DynamicSettleMonitorCfg`; explicit `measurement` selects
`SimulationSettleMonitorCfg`. A declared condition `kind` selects
`SimulationStabilityConditionCfg`, with normalized owned geometry and timing.
The explicit `joint_position_and_stability` kind selects
`SimulationJointStabilityConditionCfg` and requires a canonical articulation,
exact native joint, target and tolerance. These are policy parameters, not
static articulation geometry or names inferred by an engine. Its optional invalid
row policy and bounded raw retention belong to execution, and participate in the
registration fingerprint; ``to_condition_dict()`` keeps only normative values.
Support conditions include an explicit sensor UID/capacity declaration. The
scene binding supplies native pair UIDs, and the full definition is fingerprinted.

Configured environments use runnable `task.<embodiment>.yaml` deployments with
three typed selections: `environment.component`,
`task_program.{program,integration,execution_policy}`, and
`embodiment.component`. The reusable `env.yaml` owns only
embodiment-independent Gym values and physical simulation entities. All
ordinary deployment component paths resolve from `task.<embodiment>.yaml`.
Official `extensions/embodichain_tasks/configs/` references for program, integration and
execution policy use the imported SDK's packaged configs. The shared
[Gym component path contract](../env-framework/configuration.md#reusable-gym-deployment-components)
defines the owner-directory and CWD-shadow behavior.
`config_to_cfg()` checks that semantic binding targets exist in the physical
scene, validates the task's required scene/embodiment contracts, composes the
immutable integration catalog, injects its trusted profile/scene/preset
selection into the unbound program, and registers the common `EmbodiedEnv`.
The CLI may override only the program. Components have closed fields and
intentionally omit compatibility `version` values.

External callers compose the same semantic deployment through
`integrations.load_configured_task_program_deployment`, implemented in
`integrations/composition.py`. It consumes the resolved embodiment skill profile
and the deployment's Task Program selectors, returning a fresh
`ConfiguredTaskProgramDeployment` with trusted selection, registration and a
lazy `TaskProgramAdapterFactory`. It does not register a Gym ID or create a live
simulator or adapter. Physical component checks and ordinary program loading
remain host responsibilities; configured nominal sources retain their static
program preflight. Configured occurrence bindings likewise preflight the exact
program before composition returns. Private loader aliases remain available for existing internal
callers, with the same result types and composition rules.

Ownership is explicit rather than a generic deep merge:

- `program.yaml` owns task flow, targets, post-policies, and validators; it
  contains neither robot IDs nor an `integration` selection;
- task-local `integration.yaml` owns required contracts, its nested semantic
  `scene_binding`, semantic defaults, action options, effect monitors, and
  task-specific runtime services;
- task-local `env.yaml` owns only physical simulation entities and ordinary
  Gym environment values, so it can also be reused by handwritten trajectories;
- `configs/components/embodiments/*.yaml` owns simulation robot construction,
  the sensor suite, and an optional `skill_profile` containing logical
  resources/endpoints, command presets, and embodiment-specific services;
- `configs/components/execution_policies/*.yaml` owns planner
  selection/configuration, motion, tracking, recovery, runner, and
  effect-assurance policy. Planner collision objects are attached from the
  selected live scene and are not serialized in the policy.

`task_program.execution_policy` accepts a string path or a closed
`{component, overrides}` selector. `_resolve_execution_policy()` in
`integrations/_configured_composition.py` applies allowlisted motion/recovery
scalar overrides to an owned copy, then validates the resolved values with the
normal policy decoders. Contracts, planner selection, tracking and effect
assurance remain component-owned. Paths in either form resolve from the runnable
deployment. Reuse the base policy and declare tuning in the selector instead of
copying a policy for each robot or sampling rate; the supported fields and
example are in the [task package usage](../../../extensions/embodichain_tasks/README.md).

Named execution-policy `presets` patch policy groups over the unchanged base
preset. Task `profile.preset_options` owns action options and effect monitors for
those names. `runtime_services.invocation_policy_bindings` binds the complete
compiled-program fingerprint, a sequential execution occurrence, and one declared
preset. Repeated calls retain separate occurrence IDs. The catalog validates the
selected preset during linking, and the bridge consumes the same immutable
bindings; configured and explicit binding tuples cannot be combined.

`simulation/planning_services.py` owns the closed geometry-service declaration.
Exactly one component owns it; composition snapshots it and stores it in the
integration fingerprint without native work. The standard simulation factory
creates the explicitly selected shared Motion collision and Scene receiver
providers only after the host exists. Configured native receiver geometry needs
a complete ordered collision roster, observed root and actual native revision;
it retains Default/B=1 qualification. Gym checks declared backend/batch eligibility
before creating Sim. Planning observations compare the full native source stamp
before and after capture, and closed factories cannot authorize new captures.
Unavailable service fields are rejected at decoding rather than selecting a
fallback. The decoder is the authority for currently supported fields.

Configured `motion` policies accept optional finite, strictly positive
`velocity_limit` and `acceleration_limit` values. Configured Pour action options
accept a finite, non-negative `dwell_duration`; unknown timing keys still fail
the closed-field decoder. Omitting `velocity_limit` preserves the Robot's
ordered per-joint speed caps; an explicit scalar request is intersected with
those caps rather than replacing them.

Configured motion also accepts explicit `interpolation_mode` and `ik_seed_mode`.
`fixed_grid` requires `acceleration_limit: null` and no scalar speed override;
omission retains the existing adaptive policy and acceleration default.
`sequential` selects preceding-valid-posture IK seeds. These non-default modes
require `strategy: ik_interp`; see the [motion boundary](../motion-planning/motion-planning.md).

The configured `pour` lowerer's optional `target` binds a receiving object,
source/receiver mouth frames and usable opening radii. It lowers to a
`PourTarget` with a late-bound receiving pose; the Atomic Skill computes the
preparation and pouring path from the held transform. Keep these geometry
declarations in the integration and timing in the execution policy, rather
than placing a precomputed bottle world pose in the program. Omitting `target`
retains the original internal-axis pouring behavior.

The configured lowerer allowlist also exposes profile-owned Park, AxisAlign,
relative Place and coordinated transport. Their declarations in
`integrations/_configured_services.py` select exact Atomic Skill descriptors;
the closed decoder in `configured.py` owns accepted options and route fields.
Relative Place resolves a live reference plus world-axis displacement and
requires the verified held-object transform. Coordinated transport must select
the real primitive's release mode. Do not install frontend skill replacements.
`move_held_object` additionally accepts an explicit `routes` declaration instead
of its legacy single-target fields. Each route binds a canonical object, target
ID and an inert absolute or scene-relative pose. These declarations live in
`simulation/targets.py` and `simulation/transport.py`; lowering only constructs
typed goals, retained-attachment effects and lookahead targets. It does not read
live poses, select motion policy or replace the shared Atomic Skill. The original
single-target decoder/factory and effect behavior remain unchanged. Mixed forms
and duplicate target IDs are rejected. The optional
`RegisteredSemanticLowererSceneValidator` contract lets an immutable factory
preflight its references against `SceneManifest` before runtime assembly; routed
transport uses it for canonical object/reference checks.
The routed-2 factory uses named transport goals and verified retention-only
phases: grounding checks the exact object and every eligible input-hold row,
then guards `transport` and requests terminal attachment verification. Projected
presets reject this contract before motion planning or dispatch. The legacy
single-target factory keeps its existing effectless behavior and identity.

The same closed route also exposes retained alignment,
released clearance and paired idle clear/restore declarations. Their factories
bind explicit operands into the ordinary shared Atomic skills, with target math
in Compute. The catalog checks idle pairing, required discrete queries, exact
Cartesian policy modes and B=1 eligibility before simulation. Planning and
physical effect verification remain separate; bounded staging/released-session
policies reject until their owning providers are available.

Container affordances may declare a local-z release clearance while retaining
their original semantic target for final acceptance.

`motion.plan_opts.kind: trapezoidal` selects typed backend options for
`stop_at_waypoints`, `blend_tolerance`, and `collinearity_tolerance`. It requires
`strategy: motion_gen` and selects the Trapezoidal planner contract; incompatible
planner requirements fail during decoding. Policy scalar velocity/acceleration
limits are carried into these explicit options, while omitted limits retain
Robot speed caps and the planner's acceleration default. Sampling count remains
owned by `motion.sample_count`.

Grasp generators select toolkit-owned geometry with a string `model` ID;
an inline mapping is a custom calibration and its `model_id` does not trigger
catalog lookup. The catalog is owned by
`embodichain/toolkits/graspkit/pose_generator.py`, with source provenance in
`embodichain/toolkits/graspkit/README.md`. Embodiments may omit algorithm and
collision fields to use GraspKit defaults; task-specific tuning belongs in
the allowlisted `grasp_pose_generator_overrides` mapping.

`runtime_services.hand_geometries` declares explicit hand calibration by grasp
endpoint target ID. Its `sampled_parallel_jaw` entries reference an existing
endpoint command preset and declare motion/hand parts, the hand root, ordered
pad links and sampling bounds. `_configured_hand_geometry.py` strictly decodes
these values and creates owned `RobotHandGeometryCfg` data without reading a
robot or capturing geometry. An embodiment and task cannot both own the same
target, and this calibration has no partial override map. Omission preserves
the original integration fingerprint. Live assembly passes the data through
the existing Atomic planning services; current invocation commands remain
authoritative when a skill samples closing motion. See the
[Lab capture contract](../robot-system/robot-system.md#sampled-hand-geometry)
for supported topology, geometry and cache lifetime.

The reference embodiment `skill_profile.contract_id` and `profile_id` values
are unversioned. Versioned Gym, task-integration, or scene-registry IDs are
separate identity domains and do not imply a skill-profile version.
For `joint_position_constraint` evidence, `object_ids` is an optional explicit
narrowing. When omitted, runtime assembly scopes the embodiment service to the
selected scene's graspable rigid objects, so reusable embodiments do not name
task-local objects.

Deployment embodiment overrides affect only the robot simulation fields and
are restricted to `uid`, `init_pos`, `init_rot`, and `init_qpos`. Sensor lists
are selected atomically with the embodiment. Task grasp-generator overrides
are similarly allowlisted. This gives one embodiment to many tasks and one
task to many compatible embodiments without copying either declaration or
admitting arbitrary merge semantics. `repeated_pick_place` and `open_drawer`
each provide UR5 and Franka deployments as reference compositions.

Physical environment, embodiment, and standalone scene expansion is owned by
`gym/utils/_component_composition.py` and also runs for ordinary handwritten
Gym tasks. Task Program integration and nested scene-binding composition is
owned by `task_program/integrations/_configured_composition.py`. An embodiment
component may omit `skill_profile`, while a scene component never owns Task
Program metadata.

Configured articulation-link Slide services sample initial articulation
geometry only when `translation_axis` is omitted. Supplying the compatibility
axis preserves the mesh-only legacy path and bypasses initial point-cloud
sampling. This keeps the explicit axis usable when non-unit `body_scale` is
rejected by the point-cloud adapter; it does not make automatic sampling
scale-aware.

Selecting a physical component does not by itself parameterize hard-coded
Python control-part names or trajectory dimensions. A configured Task Program
must select an embodiment `skill_profile`; its integration must declare a
`scene_binding` satisfying the scene and execution-policy contracts. The
removed embodiment-level and scene-level `task_program` metadata keys are not
accepted.

Within `integration.yaml.scene_binding`, every affordance is nested in an
`affordances` list under its owning `rigid_objects`,
`rigidized_articulations`, `articulations`, or `links` entry. The child keeps a
globally unique `entity_id` and a closed `kind` discriminator
(`antipodal_grasp`, `support_surface`, or `container`); the YAML does not repeat
ownership with `object_id` or `parent_id`. `rigidized_articulations` owns a
required `locked_qpos` mapping and binds a physical `simulation.articulation`
UID as a `SceneObjectRef`. Its `antipodal_grasp` child owns a required
`grasp_link`; ordinary rigid-object antipodal fields (`mesh_env_id` and
`internal_axis`) remain unchanged and are not accepted on that link-backed
child. Configured rigidized articulations accept only `collision_role: none`:
their link-local grasp mesh does not provide planner collision geometry.
Joint-position and cross-arena transform tolerances may only tighten the safe
maxima of `1e-3` and `1e-5`; native joint-limit coincidence is proven with a
fixed internal epsilon rather than an author-controlled tolerance. The
configured decoder derives those relations and normalizes the authoring
hierarchy into the flat `SimulationSceneBinding` / `SceneRegistry` index.
Scene-level affordance collections are not accepted.

Placement affordances default to composing `object_target_pose` in their parent's
frame. Select `object_target_frame: world_axes` when a target should follow the
parent's measured position while its offset and orientation stay in world axes.
This distinction matters for left/right sorting after reference-object rotation.
World-axis targets fingerprint both their frame and declared pose; static
manifests and live registries retain the same identity.

`object_relative_position` validators default to `reference_frame: world_axes`,
so reference rotation does not rotate their offset or per-axis tolerances.
Select `reference_frame: parent` to measure both in the live reference frame,
matching a target whose relative pose composes the complete parent transform.


## Asset role requirements

An integration may declare optional `asset_requirements`; decoding validates its
closed role/relation contract without contacting providers. Offline asset
materialization owns selection and emits a new bound deployment before ordinary
compilation. See [asset generalization](../asset-generalization/asset-generalization.md)
for supported adaptations, provenance and integrity boundaries.

Embodiment grasp services may explicitly select `approach_deviation_angle`,
`max_candidates` and `center_mode` (`bounds` or `centroid`). Omitted fields
retain toolkit defaults and the prior integration fingerprint. These numerical
search declarations do not change contact/effect evidence requirements.
