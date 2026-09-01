---
name: add-task-program
description: Create, compose, attach, validate, or repair an EmbodiChain Task Program. Use for program.yaml, integration.yaml, TaskProgramCfg, scene_binding, configured Task Program deployments, Semantic Call workflows, or Task Program expert demonstrations.
---

# Add Task Program

Build a provider-independent task flow and bind it to trusted physical
components without leaking robot or simulator details into the program.

## Route from the prompt

Select one or more modes and read only their references:

| Prompt asks for | Mode | Reference |
|---|---|---|
| Task flow, nodes, targets, calls, segments, repeats, parallel blocks, Python `TaskProgramCfg` | Author program | `references/program-authoring.md` |
| Generate/fill `integration.yaml`, map scene UIDs, choose profiles/services, fix contract mismatch | Compose integration | `references/integration-composition.md` |
| Add `task.<embodiment>.yaml`, reuse env/embodiment/policy components, add another embodiment | Attach deployment | `references/deployment.md` |
| Validate, inspect, diagnose, migrate, or repair an existing Task Program deployment | Validate/repair | `references/validation.md` |

For a complete new Task Program expert task, use the modes in this order:

1. author program;
2. compose integration;
3. attach deployment;
4. validate.

Use `$add-task-env` for a missing physical task/environment baseline.
Use `$add-embodiment-component` for a missing reusable robot/sensor/skill
profile. Use `$add-semantic-call` only when the program needs a Semantic Call
that the selected catalog cannot discover.

## Load current context

Read `agent_context/MAP.yaml`, resolve `task-programs`, and verify the
matched source-of-truth files before editing. Also load `env-framework` when
changing a Gym deployment.

Inspect the selected:

- physical environment component;
- embodiment component and optional `skill_profile`;
- execution policy;
- nearest official Task Program example; and
- tests covering the changed configuration boundary.

Do not read the human-facing Sphinx docs unless the user asks for them.

## Preserve ownership boundaries

```text
program.yaml
  task flow, targets, post-policies, validators

integration.yaml
  integration identity and required contracts
  semantic scene_binding
  task semantic defaults/options/effect monitors
  task-specific allowlisted runtime services

env.yaml
  physical scene and ordinary Gym environment values

components/embodiments/*.yaml
  simulation robot, sensors, optional reusable skill_profile

components/execution_policies/*.yaml
  motion, tracking, recovery, runner, effect assurance

task.<embodiment>.yaml
  runnable ID and component selections
```

Never put robot IDs or an integration selection into serialized
`program.yaml`. Never put Task Program metadata in `env.yaml` or a physical
scene component. Do not create `scene.yaml`; semantic scene data is nested in
`integration.yaml.scene_binding`.

Components use closed fields and do not carry compatibility `version`
values. Do not serialize dotted imports, arbitrary callables, or executable
payloads.

## Integration generation principles

When generating an integration, derive values from selected components:

- map logical scene entities to native UIDs that actually exist in the
  physical environment;
- take the embodiment contract, resources, endpoints, capabilities, command
  presets, and reusable services from `skill_profile`;
- take the execution-policy contract and runtime preset from the selected
  policy;
- declare only task-specific semantic defaults, action options, effect
  monitors, scene affordances, and services in the integration.

Do not invent a resource ID, endpoint, capability, runtime service kind, or
Semantic Call. If the current catalog/decoder cannot represent the request,
stop config generation at that boundary and route to the owning extension
skill.

One task program may be deployed against multiple embodiments only when the
integration's required contract is satisfied by each selected profile and
policy. Deployment overrides do not repair a semantic contract mismatch.

## Validation ladder

Use the smallest level that proves the requested work:

1. strict file decode and closed-field validation;
2. component resolution and physical UID validation;
3. scene/embodiment/policy contract composition;
4. catalog preflight and Semantic Call lowering;
5. focused unit/integration tests;
6. live environment execution with effect assurance and validators.

Run the bundled read-only inspector for complete configured deployments:

```bash
python .agents/skills/add-task-program/scripts/inspect_deployment.py \
  embodichain_tasks/configs/tasks/<category_path>/<task_name>/task.<embodiment>.yaml
```

This command performs levels 1-4 without constructing a simulator. It does not
prove physical success.

## Completion checklist

- [ ] Prompt was routed to the required mode reference(s)
- [ ] Program remains provider- and embodiment-independent
- [ ] Integration values were derived from current physical/profile/policy data
- [ ] Every `scene_binding` root maps to a declared physical UID
- [ ] Contract IDs agree across integration, embodiment, and policy
- [ ] Runtime services and registered calls are allowlisted and fully covered
- [ ] Deployment paths resolve relative to the runnable task config
- [ ] Static inspector passes
- [ ] Focused tests pass
- [ ] Physical qualification was run or explicitly reported as not run
