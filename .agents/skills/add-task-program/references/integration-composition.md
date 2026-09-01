# Integration Composition

Generate or repair the task-local trusted binding between a provider-independent
program and reusable physical/profile/policy components.

## File and exact ownership

```text
<task config>/task_program/integration.yaml
```

Required top-level fields:

- `integration_id`
- `program_id`
- `requires`
- `scene_binding`
- `profile`

`runtime_services` is optional.

The `program_id` must match `program.yaml`. `requires` owns
`scene_contract` and `embodiment_contract`.

## Deterministic generation workflow

### 1. Read the selected physical environment

Resolve `environment.component` or inline scene fields. Build exact native UID
sets:

- rigid roots from physical `background` and `rigid_object`;
- articulation roots from physical `articulation`.

Every semantic root's `simulation_uid` must be present in the corresponding
set.

### 2. Read the selected embodiment profile

From `skill_profile`, copy no task-local data. Record:

- `contract_id` and `profile_id`;
- logical resources;
- endpoints and capabilities;
- command presets;
- optional embodiment-owned runtime services.

Use those logical resource IDs in task defaults. Never infer resources from
robot joint names when a profile is present.

### 3. Read the execution policy

Confirm its required embodiment contract equals the integration requirement.
The policy owns its preset and effect-assurance authority. Do not duplicate its
motion/tracking/recovery/runner values in the task integration.

### 4. Build `scene_binding`

Declare a stable scene `contract_id` and `registry_id`. Add only entities the
program or runtime services need. Nest affordances under their owning rigid
object, articulation, or link:

```yaml
scene_binding:
  contract_id: my_scene_contract
  registry_id: my_task_scene
  rigid_objects:
    - entity_id: cube
      simulation_uid: cube
      dynamics: dynamic
      semantic_type: cube
      affordances:
        - entity_id: cube_grasp
          kind: antipodal_grasp
```

Affordance `entity_id` values are globally unique. Supported configured kinds
must come from the current decoder; do not invent a discriminator.

### 5. Build task profile additions

`profile` requires:

- `defaults`: Semantic Call resource slots to profile resource IDs;
- `action_options`: configured Atomic Skill options for each call family;
- `effect_monitors`: mappings required by verified execution, or empty for
  projected execution.

Optional `skill_presets` and `grounding_providers` must use registered
provider-free declarations.

### 6. Add only necessary runtime services

Task-local services may include current allowlisted grasp generators, hand-over
pose providers, registered Semantic Call lowerers, control-part evidence, and
grasp-generator overrides. The composition boundary is explicit, not a generic
deep merge:

- task and embodiment service keys must not collide unless the schema exposes
  an explicit override field;
- an override target must already exist;
- registered call IDs must be unique and have exactly one matching lowerer;
- no dotted imports or arbitrary callables are accepted.

For reusable joint-position constraint evidence, omit task-object
`object_ids` unless explicit narrowing is required; runtime assembly scopes
the service to graspable objects in the selected scene.

## Effect assurance

`verified` execution advances semantic state only from measured evidence and
requires the relevant monitor mappings. `projected` execution forbids those
monitor mappings and projects the expected symbolic effect after command
completion.

Keep separate:

1. effect monitor: one call's physical postcondition and recovery;
2. segment post-policy: environment advancement/settling;
3. segment validator: task or dataset acceptance.

## Failure handling

If generation discovers a missing:

- physical entity: fix the environment or the requested logical mapping;
- compatible resource/capability: route to `$add-embodiment-component`;
- Atomic Skill: route to `$add-atomic-action`;
- Semantic Call/allowlisted provider: route to `$add-semantic-call`.

Do not encode a guessed value to make static validation pass.
