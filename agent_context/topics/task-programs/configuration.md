# Task Program language and semantic integration

Read this when the request needs these details. [Topic overview](task-programs.md).

## Language contract

`TaskProgramCfg` owns:

- one `program_id`;
- exact `TaskProgramIntegrationCfg` IDs;
- optional named targets; and
- one bounded program tree.

Supported nodes are `SequenceCfg`, `RepeatCfg`, `SegmentCfg`, `InvokeCfg`, and
`ParallelCfg` with an owned `BarrierCfg`. Built-in call configs are `PickCfg`,
`PlaceCfg`, and `HandOverCfg`; `RegisteredSemanticCallCfg` is the allowlisted
extension form.

Unknown fields, duplicate keys, non-finite values, invalid exact types,
excessive depth/nodes/repeats, cyclic or executable registered payloads, and
unresolved references fail before live providers are touched.

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

Configured environments use runnable `task.<embodiment>.yaml` deployments with
three typed selections: `environment.component`,
`task_program.{program,integration,execution_policy}`, and
`embodiment.component`. The reusable `env.yaml` owns only
embodiment-independent Gym values and physical simulation entities. All
deployment component paths resolve from `task.<embodiment>.yaml`.
`config_to_cfg()` checks that semantic binding targets exist in the physical
scene, validates the task's required scene/embodiment contracts, composes the
immutable integration catalog, injects its trusted profile/scene/preset
selection into the unbound program, and registers the common `EmbodiedEnv`.
The CLI may override only the program. Components have closed fields and
intentionally omit compatibility `version` values.

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
- `configs/components/execution_policies/*.yaml` owns motion, tracking,
  recovery, runner, and effect-assurance policy.

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
`affordances` list under its owning `rigid_objects`, `articulations`, or `links`
entry. The child keeps a globally unique `entity_id` and a closed `kind`
discriminator (`antipodal_grasp`, `support_surface`, or `container`); the YAML
does not repeat ownership with `object_id` or `parent_id`. The configured
decoder derives that relation and normalizes the authoring hierarchy into the
flat `SimulationSceneBinding` / `SceneRegistry` index. Scene-level affordance
collections are not accepted.
