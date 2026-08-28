(semantic-skills)=

# Semantic skills

```{currentmodule} embodichain.lab.semantic_skills
```

Semantic skills are EmbodiChain's **declarative vocabulary** for describing
robot-independent manipulation intent and the integration data needed to make
that intent meaningful. The package contains no public compiler, executor, or
action-generation facade.

There are exactly two supported action-generation entry points:

| Entry point | Use it for |
|---|---|
| {doc}`Atomic Actions <atomic_actions/index>` | Python code that already owns typed goals, bindings, policies, and execution ports. |
| {doc}`Expert Program <atomic_actions/expert_programs>` | Task-level JSON/YAML or programmatic sequences of semantic calls, including repeats, segments, validation, and parallel barriers. |

`semantic_skills` supplies contracts to Expert Program; it is not a third
execution path. In particular, semantic calls do not expose `run()`, `start()`,
or `step()` methods.

```text
embodichain.lab.semantic_skills
  calls + catalog
  scene + affordances
  robot resources + policy presets
  effects + evidence declarations
             |
             v
embodichain.lab.expert_program
  schema -> decode -> validate -> compile
             |
             v
Gym/simulation adapter -> AtomicActionEngine -> controller transports
```

## Package responsibilities

| Contract family | Main types | Responsibility |
|---|---|---|
| Semantic calls | {class}`Pick`, {class}`Place`, {class}`HandOver`, {class}`RegisteredSemanticCall`, {class}`SemanticCallCatalog` | Immutable object-centric intent and discoverable schemas. |
| Scene semantics | {class}`SceneRegistry`, {class}`SceneObjectRef`, {class}`SceneAffordanceRef`, {class}`SceneManifest` | Canonical identity, topology, affordances, collision roles, and provider-free snapshots. |
| Robot semantics | {class}`RobotSkillProfile`, {class}`RobotResource`, {class}`ResourceEndpoint`, {class}`SkillPolicyPreset` | Embodiment resources, default bindings, command templates, and policy selection. |
| Effect contracts | {class}`SemanticEffectSpec`, {class}`EffectMonitorRef`, {class}`EffectEvidenceCollector` | Typed postconditions and measured evidence routing. |
| Integration | {class}`SemanticIntegrationManifest`, {class}`BoundSemanticIntegration` | Static compatibility checks between the catalog, scene, profile, and atomic-action catalog. |

Executable lowerers, schedulers, and semantic-call execution state are internal
to Expert Program. Applications should not import underscore-prefixed modules
from `embodichain.lab.expert_program`.

## Built-in calls

The catalog returned by {func}`builtin_semantic_call_catalog` contains three
curated calls:

| Call | Intent |
|---|---|
| {class}`Pick` | Acquire a registered object, optionally through an explicit grasp affordance. |
| {class}`Place` | Release a held object at an absolute pose, on a support, or inside a container. Exactly one target form is allowed. |
| {class}`HandOver` | Transfer a held object between two robot resources, with an optional explicit final target. |

{class}`SemanticPose` is an immutable absolute pose. Typed references such as
{class}`SceneObjectRef` and {class}`SceneAffordanceRef` keep aliases and native
simulator names out of task declarations.

Extensions use {class}`RegisteredSemanticCall`. Its payload must contain only
declarative values. Live simulator objects, tensors, callables, classes, and
modules are rejected. The surrounding Expert Program registration must provide
the matching allowlisted lowerer; a payload cannot choose executable code.

## Scene and integration snapshots

Register canonical scene entities once:

```python
from embodichain.lab.semantic_skills import (
    SceneEntityRegistration,
    SceneObjectRef,
    SceneRegistry,
)

scene_registry = SceneRegistry(
    (
        SceneEntityRegistration(
            ref=SceneObjectRef("cube"),
            state_provider=cube_state_provider,
            semantic_type="cube",
        ),
    )
)
```

The registry owns identity and live providers. A {class}`SceneManifest` is its
provider-free snapshot: creating one does not read simulation state. An
integration manifest combines that snapshot with one robot profile and call
catalog:

```python
from embodichain.lab.semantic_skills import (
    SceneManifest,
    SemanticIntegrationManifest,
    builtin_semantic_call_catalog,
)

manifest = SemanticIntegrationManifest(
    scene=SceneManifest.from_registry(scene_registry),
    robot_profile=robot_profile,
    call_catalog=builtin_semantic_call_catalog(),
)
```

Expert Program uses this data for provider-free validation and then binds the
same declarations to a live atomic-action engine. Changed scene metadata,
robot resources, or action-catalog revisions invalidate the snapshot instead
of being accepted silently.

See {doc}`scene_registry` for registration and collision-world rules, and
{doc}`atomic_actions/robot_skill_profiles` for resource binding.

## Explicit effect assurance

Every {class}`SkillPolicyPreset` must select one {class}`EffectAssurance`:

| Value | Meaning |
|---|---|
| {attr}`EffectAssurance.VERIFIED` | Semantic state advances from an installed physical-effect monitor and measured evidence. Curated Pick, Place, and HandOver calls must each have a monitor. |
| {attr}`EffectAssurance.PROJECTED` | After command completion, the expected symbolic effect from the action plan is projected. Effect monitors are forbidden. |

There is no implicit default and no implicit built-in monitor installation.
This makes an integration's authority visible in configuration and review.

```python
from embodichain.lab.semantic_skills import (
    EffectAssurance,
    EffectMonitorRef,
    SkillPolicyPreset,
)

verified = SkillPolicyPreset(
    preset_id="safe",
    action_option_templates=action_options,
    effect_assurance=EffectAssurance.VERIFIED,
    effect_monitors={
        "pick": EffectMonitorRef("composite", "1"),
        "place": EffectMonitorRef("composite", "1"),
        "hand_over": EffectMonitorRef("composite", "1"),
    },
)

projected = SkillPolicyPreset(
    preset_id="trajectory_demo",
    action_option_templates=action_options,
    effect_assurance=EffectAssurance.PROJECTED,
)
```

Projected assurance is useful for trajectory demonstrations, but command
completion is not proof of physical task success. Dataset acceptance should
use verified assurance or explicit Expert Program segment validators when the
physical result matters.

## Effects and evidence

Effect monitors evaluate typed clauses over evidence batches. Evidence comes
from registered providers such as control-part state, articulation joint state,
pose relations, contacts, constraints, or force channels. A command
acknowledgement is never physical evidence.

Held-object guards and phase-effect gates may stop an atomic plan at named
boundaries when measured state is missing or contradicted. They can remove an
action-authorized symbolic relation, but they never create simulator
constraints, freeze bodies, or overwrite object poses.

## Where to make changes

| Change | Owning location |
|---|---|
| Call values and public catalog | `embodichain/lab/semantic_skills/calls.py` |
| Scene references, registry, and affordances | `embodichain/lab/semantic_skills/scene.py` |
| Robot resources and policy presets | `embodichain/lab/semantic_skills/profiles.py` |
| Semantic effect declarations | `embodichain/lab/semantic_skills/effects.py` |
| Evidence contracts and collection | `embodichain/lab/semantic_skills/evidence.py` |
| Integration snapshots and diagnostics | `embodichain/lab/semantic_skills/integration.py` |
| Program schema, decoder, and compiler | `embodichain/lab/expert_program/` |
| Live Gym/simulation assembly | `embodichain/lab/gym/envs/expert_program/` |

## Further reading

- {doc}`atomic_actions/expert_programs` — the semantic task execution entry point
- {doc}`atomic_actions/index` — direct typed atomic-action execution
- {doc}`scene_registry` — canonical scene identity and affordances
- {doc}`atomic_actions/robot_skill_profiles` — embodiment resources and presets
- {doc}`/tutorial/semantic_skills` — configuring semantic contracts through Expert Program
