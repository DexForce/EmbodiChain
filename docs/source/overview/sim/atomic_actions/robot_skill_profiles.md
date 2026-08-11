(robot-skill-profiles)=

# Robot skill profiles

```{currentmodule} embodichain.lab.sim.skills
```

A {class}`RobotSkillProfile` describes how robot-independent atomic-skill
requirements map onto one robot embodiment. Configure the robot's resources,
semantic commands, default choices, and policy presets once; task code can then
select skill-local participants instead of constructing an `ActionBinding` from
robot-specific control-part names.

The model is deliberately generic. It does not define global `arm` and `tool`
fields. Each atomic skill publishes its own participant slots and endpoint
requirements, while a robot resource may expose any endpoints appropriate to
that embodiment: manipulation motion and grasping, a mobile base, a torso, or a
whole-body controller.

## Contracts on the two sides

An atomic action owns a
{class}`~embodichain.lab.sim.atomic_actions.SkillBindingContract`:

- a {class}`~embodichain.lab.sim.atomic_actions.SkillResourceSlot` names each
  skill-local participant, such as `primary`, `source`, or `destination`;
- a {class}`~embodichain.lab.sim.atomic_actions.SkillEndpointRequirement`
  declares the all-of capabilities and typed semantic commands needed from that
  participant;
- {class}`~embodichain.lab.sim.atomic_actions.DisjointSlotEndpoints` declares
  endpoint views that must not share physical channels within one participant;
  coupled whole-body views may overlap when the skill does not declare this
  constraint; and
- {class}`~embodichain.lab.sim.atomic_actions.DisjointResourceSlots` requires
  multi-participant skills to select physically disjoint resources.

The robot side supplies {class}`RobotResource` values. A resource exposes named
{class}`ResourceEndpoint` values and may contain other resources through
`members`. Members form a directed acyclic graph and describe the physical
claim; endpoint capabilities are always explicit and are never inherited or
inferred from names. {class}`ControlPartEndpoint` is the built-in joint-backed
endpoint type, not the resource schema itself.

```text
skill contract                         robot profile

slot primary                          resource left_participant
+-- endpoint motion  <--------------> +-- endpoint motion -> left_arm
`-- endpoint grasp   <--------------> `-- endpoint grasp  -> left_hand
      capabilities + commands                 + members/physical claim
```

Binding the profile to an engine resolves each endpoint through a registered
{class}`ResourceEndpointAdapter` and validates physical claims, known
solver-backed kinematics capabilities, command types and dimensions, complete
defaults, policy presets, and installed skill contracts. The resulting
{class}`BoundRobotSkillProfile` exposes only installed, agent-visible skills
with at least one valid resource assignment.

Endpoint and resource declarations are snapshotted when owned by a resource,
profile, or resolved binding. Custom endpoint types whose payloads cannot be
deep-copied must override {meth}`ResourceEndpoint.snapshot` and return a new
value of the same exact type.

## Configure a manipulation participant

The following profile groups two physical leaves into one participant. The
`motion` and `grasp` endpoint names come from the built-in manipulation
contracts; they are local protocol names, not global robot-resource categories.

```python
import torch

from embodichain.lab.sim.atomic_actions import (
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_CAPABILITY,
    ControlPartCommandProfile,
    HandOverOptions,
    MotionPolicy,
    PickUpOptions,
    PlaceOptions,
)
from embodichain.lab.sim.skills import (
    COMPOSITE_EFFECT_MONITOR_ID,
    COMPOSITE_EFFECT_MONITOR_REVISION,
    ControlPartEndpoint,
    EffectMonitorRef,
    ResourceBinding,
    RobotResource,
    RobotSkillProfile,
    SkillPolicyPreset,
)

left_motion_capabilities = frozenset(
    {
        CARTESIAN_POSE_CAPABILITY,
        FORWARD_KINEMATICS_CAPABILITY,
        BATCH_INVERSE_KINEMATICS_CAPABILITY,
    }
)

profile = RobotSkillProfile(
    profile_id="example_robot",
    resources={
        # Physical leaves own disjoint robot joints.
        "left_arm_leaf": RobotResource(
            resource_id="left_arm_leaf",
            endpoints={"control": ControlPartEndpoint("left_arm")},
        ),
        "left_hand_leaf": RobotResource(
            resource_id="left_hand_leaf",
            endpoints={"control": ControlPartEndpoint("left_hand")},
        ),
        # A skill selects this participant as one indivisible resource.
        "left_participant": RobotResource(
            resource_id="left_participant",
            endpoints={
                "motion": ControlPartEndpoint(
                    "left_arm",
                    capabilities=left_motion_capabilities,
                ),
                "grasp": ControlPartEndpoint(
                    "left_hand",
                    capabilities=frozenset({GRASP_CAPABILITY}),
                ),
            },
            members=("left_arm_leaf", "left_hand_leaf"),
        ),
    },
    command_profiles={
        "left_hand": ControlPartCommandProfile.joint_positions(
            open=torch.tensor([0.04, 0.04]),
            grasp=torch.tensor([0.0, 0.0]),
        ),
    },
    defaults={
        "pick_up": ResourceBinding(
            resources={"primary": "left_participant"},
        ),
    },
    presets={
        "default": SkillPolicyPreset(
            preset_id="default",
            action_option_templates={
                "pick": PickUpOptions(),
                "place": PlaceOptions(),
                "hand_over": HandOverOptions(),
            },
            motion_policy=MotionPolicy(strategy="ik_interp"),
            effect_monitors={
                semantic_id: EffectMonitorRef(
                    COMPOSITE_EFFECT_MONITOR_ID,
                    COMPOSITE_EFFECT_MONITOR_REVISION,
                    {
                        "attached_translation_threshold": 0.02,
                        "detached_translation_threshold": 0.05,
                        "consecutive_samples": 2,
                    },
                )
                for semantic_id in ("pick", "place", "hand_over")
            },
        ),
    },
    default_preset="default",
)
```

During binding, each resolved endpoint also receives a logical
`task_state_key` and immutable, channel-keyed `effect_sources`. By default the
logical key is the selected resource ID, so the `motion` and `grasp` endpoints
of `left_participant` share one symbolic held-object state even though they use
different control parts. An effect source contains an `EffectEvidenceAddress`;
it is intentionally separate from the endpoint's command-only
`RuntimeEndpointTarget`.

Every `ControlPartEndpoint.control_part` must be a key in
`robot.control_parts`. A composite endpoint may reuse a member's control part,
but all joints controlled directly by the composite must already be covered by
its members. Two physical leaf resources may not claim the same joint; model a
shared physical part once and reference that leaf from multiple composites.

`command_profiles` are generic IDs selected by endpoint adapters; the built-in
control-part adapter defaults the ID to its `control_part`, and the engine
installs those profiles into the current action core automatically.
One-dimensional joint-position commands are broadcast across environments.
Their last dimension must equal the resolved endpoint's degree of freedom. Use
invocation-level command overrides for object- or environment-specific values.

## Safe preset and dynamic collision worlds

When the authoritative scene registry declares dynamic collision entities and
`safe` is reachable through the integration-wide, per-skill, or
profile-default preset selection, semantic integration validates that path
conservatively during binding. The `safe` preset must use `motion_gen`, and the
active motion generator must explicitly support dynamic collision worlds;
otherwise binding fails before provider observation, planning, or command
emission.

A linked call receives an effective immutable preset snapshot with
`DynamicCollisionMode.REQUIRED`; the source profile preset is not mutated.
Other presets, and scenes without dynamic collision entities, retain their
configured collision mode.

## Configure semantic action behavior with the preset

`SkillPolicyPreset.action_option_templates` is the required, typed action-
behavior table for semantic calls that can select the preset. Each key is the
exact semantic call ID (`pick`, `place`, `hand_over`, or
`operate_articulation`), and each value must be the target action's exact frozen
`ActionOptions` dataclass. Static linking rejects a missing entry, an unknown
call ID, or an options value of the wrong exact type before simulation starts.

The preset owns independent snapshots of each template. Pick and HandOver
grounding only replace their compiler-owned dynamic target fields; distances,
directions, waypoint counts, and other reusable behavior remain configuration.
A registered semantic lowerer may build a goal but cannot return replacement
options. This keeps task extensions from silently moving action parameters back
into Python code.

Pick's `downstream_object_target_poses` and HandOver's
`middle_object_pose`/`final_object_pose` are reserved for the semantic compiler
and must remain empty in a template. Planner choice, sample count, tracking,
recovery, runner timing, and effect monitors stay in their dedicated preset
fields rather than `ActionOptions`.

## Select semantic effect monitors with the preset

A {class}`SkillPolicyPreset` owns one coherent runtime choice: planning and
recovery policy, runner cadence, and the exact semantic-effect monitors used to
confirm physical postconditions. `effect_monitors` maps a semantic call ID to a
versioned {class}`EffectMonitorRef`. Its parameters are bounded declarative
values; executable objects, tensors, cyclic containers, and non-finite numbers
are rejected.

When `effect_monitors` is omitted, the preset selects the built-in
pose-relation hysteresis monitor for `pick`, `place`, and `hand_over`. Passing an
explicit empty mapping disables that default; static analysis then reports
`missing_effect_monitor` if a curated effectful call selects that preset. A
manifest also rejects monitor entries whose semantic ID is absent from its call
catalog, and the compiler requires the exact monitor ID/revision and validates
its parameters before grounding.

The semantic compiler creates a fresh monitor for every grounded call. Pick
expects one attached destination relation, place one detached source relation,
and handover both source-detached and destination-attached relations in the
same observation. The monitor compares fresh backend evidence with owned
object-to-endpoint baselines; it never treats the planned `StateDelta` or
current `TaskState` as proof that the physical effect occurred. Invalid or
missing per-environment evidence remains unresolved. Consecutive-sample state
survives request-mask shrinkage within one attempt and resets when recovery
installs a new attempt.

```{note}
The monitor contract is backend-neutral. Simulation, hardware perception, or
controller feedback supplies typed pose-relation evidence. The semantic
runtime adapter that connects that evidence to `ExecutionRunner` is separate
from the profile and monitor configuration.
```

## Bind, discover, and resolve

Pass the profile to
{class}`~embodichain.lab.sim.atomic_actions.AtomicActionEngine`. The engine
installs its command profiles and binds it after loading built-in actions:

```python
from embodichain.lab.sim.atomic_actions import AtomicActionEngine

engine = AtomicActionEngine(motion_generator, skill_profile=profile)
bound = engine.skill_profile
assert bound is not None

# This is the embodiment-filtered semantic catalog, not every installed action.
assert "pick_up" in bound.skills

resolved = bound.resolve("pick_up")
assert resolved.resource_ids == {"primary": "left_participant"}
binding = resolved.action_binding
preset = bound.preset(skill_id="pick_up")
```

{meth}`BoundRobotSkillProfile.resolve` returns a {class}`ResolvedSkillBinding`
containing the selected logical resources, their adapter-resolved endpoints,
their combined {class}`ResourceClaim`, and an engine-owned generic
{class}`~embodichain.lab.sim.atomic_actions.ActionBinding`. A
semantic compiler uses that binding and the selected preset when constructing
an invocation; profile resolution does not plan or execute the action itself.

If exactly one assignment is valid, resolution selects it. If several remain,
the caller must provide enough skill-local selections or the profile must define
a complete per-skill default:

```python
left = bound.resolve("pick_up", selections={"primary": "left_participant"})
candidates = bound.candidates("pick_up")
```

Incomplete defaults are rejected when the profile is bound. Without an
unambiguous choice, resolution raises {class}`AmbiguousSkillBindingError` rather
than selecting a resource by declaration order. An unsupported selection raises
{class}`UnsupportedSkillError` with endpoint, capability, command, or claim
rejection details.

`engine.actions` remains the direct-core implementation registry.
`engine.skills` is the installed semantic catalog before embodiment filtering,
and `bound.skills` is the profile-supported catalog. Registering or replacing an
action invalidates the bound profile; bind it again before discovery or
resolution.

## Extend the graph beyond manipulation

Resource and capability identifiers are open strings. A joint-driven mobile
robot can model a base and a whole-body controller without changing the profile
schema:

```python
base = RobotResource(
    resource_id="base",
    endpoints={
        "motion": ControlPartEndpoint(
            "base",
            capabilities=frozenset({"motion.planar_pose"}),
        )
    },
)
torso = RobotResource(
    resource_id="torso",
    endpoints={"motion": ControlPartEndpoint("torso")},
)
whole_body = RobotResource(
    resource_id="whole_body",
    endpoints={
        "motion": ControlPartEndpoint(
            "full_body",
            capabilities=frozenset({"motion.whole_body"}),
        )
    },
    members=("base", "torso", "left_arm_leaf", "right_arm_leaf"),
)
```

Here `base`, `torso`, and `full_body` must be real, non-empty robot control
parts, and the `full_body` joint set must be covered by the listed members. A
future locomotion or whole-body skill can require the corresponding endpoint
and capability in its own binding contract. Existing built-in actions do not
consume these example capabilities.

Non-joint controllers add one endpoint declaration type and one adapter. The
adapter returns {class}`EndpointResolution` with a typed immutable
{class}`~embodichain.lab.sim.atomic_actions.RuntimeEndpointTarget`, an optional
command-profile key, joint IDs when applicable, and adapter-defined claim
tokens. The generic graph, matching, command, default, and conflict code does
not change. For example, a twist controller can return a target addressed to a
`base_velocity` transport and `claim_tokens={"controller:base"}` with no joint
IDs. Exclusive endpoints must provide joint IDs or claim tokens; a read-only or
otherwise shareable virtual endpoint must opt into `exclusive=False`
explicitly.

Adapters are registered by exact endpoint type. The built-in
{class}`ControlPartEndpointAdapter` cannot be overridden; define a distinct
endpoint subtype and adapter when controller semantics differ. An adapter may
set `requires_command_profile=True` when a missing generic command-profile ID
must make profile binding fail immediately.

The standard Expert Program simulation declaration accepts these endpoints
directly for robots that expose the normal full-state/qpos action base; a task
does not need a custom runtime factory solely to register the endpoint and Gym
transport:

```python
profile = SimulationRobotSkillProfileBinding(
    profile_id="mobile_v1",
    resources=(
        RobotResourceBinding(
            resource_id="mobile_base",
            endpoints={
                "motion": MobileVelocityEndpoint(
                    controller_id="base_controller",
                    capabilities=frozenset({"motion.base.velocity"}),
                )
            },
        ),
    ),
)

adapter = create_simulation_expert_program_adapter(
    env,
    registration=SimulationExpertProgramRegistration(
        scene_binding=scene_binding,
        robot_profile_binding=profile,
        endpoint_adapters=(MobileVelocityEndpointAdapter(),),
        runtime_transports=(MobileVelocityGymEncoder(),),
    ),
)
```

The adapter and encoder publish exact class-level declarations before a live
robot is created. The adapter declares its endpoint type, runtime target types,
transport IDs, and versioned tracking/evidence routes. The encoder declares its
transport ID plus exact target and payload types; each target and payload type
declares the same `TRANSPORT_ID`. Registration rejects missing, unused,
duplicate, or conflicting declarations, and runtime profile binding verifies
that `adapter.resolve()` returns only those declared routes. A stateful adapter,
transport, grounding provider, or safety factory must be a frozen dataclass whose
configuration is recursively immutable; mutable leaves such as lists, mappings,
sets, byte arrays, and tensors are rejected before registration.

The standard factory currently accepts its built-in tracking feedback,
projector, evaluator, and effect-evidence routes only for
`ControlPartEndpoint`. In C1, every custom endpoint adapter must declare empty
route sets and therefore supports timed/open-loop execution only. Custom
closed-loop mobile or whole-body tracking/evidence needs a registration-owned
provider factory in C2; task code must not supply a live provider side channel.

`RobotResourceBinding` snapshots arbitrary typed `ResourceEndpoint` values.
`ControlPartResourceBinding` remains the stricter joint-backed convenience and
continues to validate native control parts, joint IDs, and command-preset
widths.

Endpoint registration is not a navigation or whole-body planner. Existing
built-in semantic skills do not consume the example base/whole-body
capabilities. A reusable capability must also install its semantic descriptor
and lowerer, atomic planner, command payload, safe-state transport behavior, and
effect integration as applicable. The current standard Gym encoder composes
custom transports over a full-qpos hold and the standard simulation factory
owns a `MotionGenerator`; a truly jointless or natively structured controller
therefore needs a reusable base-action composition/provider integration. This
does not require base- or whole-body-specific fields in the generic profile or
runtime core.

Task vertical slices may declare a typed profile binding locally while the API
stabilizes. Repeated use should move that binding into an embodiment-owned
profile catalog so new tasks select it instead of redefining robot data.

A resolved action binding is keyed only by the skill-local
`(slot_id, endpoint_id)` pair. A reusable non-joint capability supplies a
matching {class}`~embodichain.lab.sim.atomic_actions.RuntimeCommandPayload`, a
shared atomic skill that emits
{class}`~embodichain.lab.sim.atomic_actions.RuntimeCommandFrame` values, and an
{class}`~embodichain.lab.sim.atomic_actions.EndpointCommandTransport` registered
with {class}`~embodichain.lab.sim.atomic_actions.EndpointCommandRouter`. The
core binding, session, runner, and router do not need controller-specific
changes. Once that shared capability exists, new tasks and robot variants reuse
it through profile and task configuration rather than task-specific motion
code.

```{important}
`ResourceClaim` combines leaf IDs, concrete joint IDs, and adapter claim tokens.
It and explicit disjoint constraints detect physical overlap for binding. A
claim alone does not enable or prove safe parallel action execution. The
separate explicit `ParallelSkillRuntime` can coordinate disjoint branch lanes,
but it merges command frames only through an authoritative
`ParallelCommandSafetyValidator`. Joint-backed plans may retain a full-robot
trajectory for feedback and offline compilation, while runtime dispatch remains
scoped to the endpoints in each command frame.
```

See {doc}`index` for the direct atomic-action core and
{doc}`../scene_registry` for canonical scene identity and snapshots.
