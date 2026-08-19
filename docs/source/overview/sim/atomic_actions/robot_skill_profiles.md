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
    MotionPolicy,
)
from embodichain.lab.sim.skills import (
    ControlPartEndpoint,
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
            motion_policy=MotionPolicy(strategy="ik_interp"),
        ),
    },
    default_preset="default",
)
```

Set `SkillPolicyPreset.required_planner` only when a preset depends on one
planner backend, typically because it carries backend-specific typed planning
options. Profile binding checks that requirement against the engine's configured
backend and fails early on a mismatch. Leave it as `None` for portable presets.

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
It and explicit disjoint constraints detect physical overlap for binding and
future scheduling work. They do not enable parallel action execution. The
runtime does not merge concurrent endpoint-command streams. Joint-backed plans
may retain a full-robot trajectory for feedback and offline compilation, but
runtime dispatch is scoped to the endpoints in each command frame.
```

See {doc}`index` for the direct atomic-action core and
{doc}`../scene_registry` for canonical scene identity and snapshots.
