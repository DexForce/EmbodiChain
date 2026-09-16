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
- the selected execution-policy file owns motion, tracking, recovery, runner,
  and effect-assurance policy. Reusable policies live under
  `configs/components/execution_policies/`; a task-specific policy can live
  under the task's `task_program/` directory without moving these fields into
  its integration.

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

## Configured articulation and Pour services

`integrations/configured.py` decodes the closed service declarations in
`integration.yaml.runtime_services.registered_semantic_lowerers`;
`integrations/_configured_services.py` owns their factories and typed goal
lowering. These declarations select allowlisted implementations rather than
serialized Python callables. The articulation services bind to ordinary
`scene_binding.articulations` and `links` entries; they do not introduce new
nested affordance-kind discriminators.

| Service `kind` | Registered call ID | Program `arguments` |
|---|---|---|
| `articulation_link_press` | `simulation.articulation_link_press` | `target`: configured link ID |
| `articulation_link_twist` | `simulation.articulation_link_twist` | `target`: configured link ID |
| `articulation_link_open_door` | `simulation.articulation_link_open_door` | `handle`: configured link ID; `open_fraction`; optional `open_fraction_range` |
| `pour` | `simulation.pour` | `object`: configured object ID |

The Press, Twist, and OpenDoor services require `articulation_id`,
`articulation_simulation_uid`, and `link_entity_id`. Their optional
`target_pose_mode` defaults to `live`, which produces a `SceneEntityPose`;
`snapshot` clones the selected link's pose from the current planning context.
Factories check the semantic link's parent, native link name, and exact engine
robot before creating fresh lowerers. They do not step the simulator or change
object state.

- Press samples initial target-local geometry to derive contact and the
  prismatic pressing direction. Its options use `kind: press`.
- Twist additionally requires link-local `grasp_position: [x, y, z]` and
  optionally declares nominal `grasp_roll` (default zero) about the grasp's
  forward axis and `grasp_roll_range: [min, max]` in radians. The range contains
  the nominal roll and supplies absolute rotations, applied once. It samples
  target-local geometry, requires one unambiguous revolute ancestor, and retains
  that joint's name and limits. Its options use `kind: twist`; optional
  `twist_angle_range` expresses accepted signed motion, separately from contact
  symmetry.
- OpenDoor uses `OpenDoorAffordance.from_articulation()` to resolve the handle
  mesh and hinge geometry. Its optional `hinge_joint_name` disambiguates the
  ancestor, and `opening_direction` is exactly `-1` or `1` (default `1`). The
  program owns the absolute goal `open_fraction` in `[0, 1]`; an optional
  `open_fraction_range` must contain that nominal fraction and stay within the
  same interval. Its options use `kind: open_door`.

Press/Twist distances, angles, interpolation counts and approach settings live
in the typed `profile.action_options` templates, keyed by registered call ID.
OpenDoor keeps trajectory/interpolation settings there while the call carries
its task goal. Unknown program arguments are rejected. The selected policy
continues to own overall motion strategy and sample count. These services do
not add physical-effect monitors or advertise held-object transport look-ahead.

Configured articulation-link Slide services sample initial articulation
geometry only when `translation_axis` is omitted. Supplying the compatibility
axis preserves the mesh-only legacy path and bypasses initial point-cloud
sampling. This keeps the explicit axis usable when non-unit `body_scale` is
rejected by the point-cloud adapter; it does not make automatic sampling
scale-aware. The new Press and Twist services use the same initial-geometry
adapter and do not expose this Slide compatibility override.

The Pour service requires only `object_id`. That object's nested
`antipodal_grasp` declares a nonzero local `internal_axis`, which upgrades its
payload to `AxisAlignAffordance`. Pick stores the selected object-to-EEF
transform; Pour requires the object to remain exclusively held, rotates it
about that local axis, then returns to its starting orientation while retaining
the grasp. `profile.action_options.simulation.pour` uses `kind: pour` and owns
`rotate_angle`. This is an object-motion example, not a liquid simulation.

The UR5 deployment entry points are
`embodichain_tasks/configs/tasks/manipulation/{press,twist,open_door,pour}/task.ur5.yaml`.
Each directory owns `env.yaml` and `task_program/{program,integration}.yaml`.
Press and Pour select the shared `trajectory_open_loop.yaml`; Twist and
OpenDoor select their task-local `task_program/execution_policy.yaml` to own
their larger trajectory budgets. All four use projected effect assurance;
successful command execution alone is not measured physical task acceptance.
Twist selects the reusable `ur5_dh_pgi_140_80_tcp170.yaml` embodiment, whose
170 mm tool-center offset and `0.036` grasp command place the finger tips around
the knob contact center. Its profile ID is `ur5_dh_pgi_140_80_tcp170`; the arm
drive gains and sensor suite match the standard embodiment. The other three
deployments retain `ur5_dh_pgi_140_80.yaml` with its 150 mm offset and `0.040`
grasp command.

OpenDoor places the microwave at `[-0.90, 0.20, 0.4]` in its environment
component so the nominal handle approach stays inside the UR5 workspace and
avoids an extended-elbow IK branch switch. Its integration overrides the hand
generator with `sample_count: 10000`, `opening_margin: 0.03`, and
`force_refresh: true`. These settings provide denser handle contact candidates
and regenerate older, coarser disk annotations when a mesh backend is first
created; that backend is then reused by the generator. They leave the shared
embodiment's grasp sampling defaults unchanged.

The read-only deployment inspector checks decoding, composition, physical UID
bindings, catalog preflight and lowering before a separate live environment run.
