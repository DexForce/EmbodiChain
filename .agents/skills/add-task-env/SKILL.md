---
name: add-task-env
description: "Route and scaffold a new EmbodiChain task by solution type: handwritten expert trajectory, Task Program expert trajectory, simulator RL, lightweight RL, or an environment-only baseline. Use when creating a task, environment, expert demo, or RL task."
---

# Add Task Environment

Own the task identity, task-first layout, physical environment, registration
boundary, and solution routing. Do not assume every new task needs a Python
environment subclass.

## Route from the user's prompt

Classify the requested solution before creating files. Read only the matching
reference files.

| Prompt signal | Route | Required reference |
|---|---|---|
| `handwritten`, `scripted`, `demo segments`, `create_demo_segments`, custom Python planning | Handwritten expert trajectory | `references/handwritten-expert.md` |
| `Task Program`, `program.yaml`, `integration.yaml`, declarative task, Semantic Call, MLLM-generated program | Task Program expert trajectory | `references/task-program-expert.md` |
| Generic `expert trajectory` or `expert demo` with no implementation signal | Resolve handwritten versus Task Program from required behavior; ask only if still ambiguous | One selected expert reference |
| `RL`, `PPO`, `GRPO`, `APG`, policy, reward learning, training config | Reinforcement learning | `references/rl.md` |
| Task/environment/scene only, with no requested solution | Environment-only baseline | This file only |

Routing rules:

- If the prompt names multiple solutions, read every matching reference and
  keep all artifacts under one task-first directory.
- If the prompt merely says `expert trajectory` and provides no signal for
  handwritten versus Task Program, determine whether the requested behavior
  is declarative and supported by existing Semantic Calls. Ask only when that
  choice materially changes the requested deliverables.
- Treat environment-only as the shared baseline, not as a fourth solution
  implementation.
- Task Program authoring and integration composition are owned by
  `$add-task-program`. New reusable robot/skill-profile declarations are owned
  by `$add-embodiment-component`.

## Load current project context

Read `agent_context/MAP.yaml`, then load `env-framework`. Also load:

- `task-programs` for the Task Program route;
- `rl-learning` for the RL route; and
- any matched robot, sensor, manager, or randomization topic needed by the
  requested environment.

Verify paths and configuration fields against the current source of truth.
Do not infer schemas from older task examples alone.

## Establish task identity

Resolve from the prompt or nearby conventions:

- `<category_path>`: a task family plus optional subdomain, such as
  `manipulation/tableware`;
- `<task_name>`: snake case;
- one unique runnable environment ID per deployment;
- intended embodiment variants; and
- selected solution routes.

Organize by task identity, never by solution method. Use:

```text
embodichain_tasks/embodichain_tasks/<category_path>/<task_name>.py
embodichain_tasks/configs/tasks/<category_path>/<task_name>/
```

Keep a single task Python entry point flat at `<task_name>.py`. Do not create a
same-named Python package, `scenario` package, `mdp` package, or a task-local
`task_program` Python package when configuration and manager functors express
the behavior.

## Build the shared environment baseline

Choose the narrowest representation that supports the requested routes.

### Reusable physical environment

For new task-first compositions, prefer:

```text
<task config>/env.yaml
```

This is a component, not a runnable deployment. It owns:

- `environment_id`;
- ordinary environment values such as episode limits and environment count;
- physical simulation entities under `simulation`; and
- manager configuration under `env`.

It must not contain a runnable `id`, Task Program fields, semantic scene
bindings, a robot, or sensors. Task Program semantic roots and affordances
belong to `integration.yaml.scene_binding`.

### Runnable deployment

Prefer one thin deployment per embodiment or execution variant:

```text
<task config>/task.<variant>.yaml
```

It owns the runnable `id` and selects reusable components. A typical physical
deployment selects:

```yaml
id: MyTask-v1

environment:
  component: env.yaml

embodiment:
  component: ../../../components/embodiments/<embodiment>.yaml
```

The original inline Gym format remains supported. When extending an existing
inline `env.json` or `env.yaml`, preserve that representation unless the user
asked for component extraction. Never select a component and repeat its owned
inline fields in the same deployment.

### Optional Python entry point

Create
`embodichain_tasks/embodichain_tasks/<category_path>/<task_name>.py` when the
route requires import-owned behavior or registration:

- an environment-only registered Gym task;
- handwritten expert behavior;
- simulator RL registration, even when managers/config own its behavior; or
- a lightweight learning environment.

A supported configuration-defined Task Program is the exception: it normally
omits the task module and dynamically registers the common `EmbodiedEnv` while
loading its runnable deployment.

Keep `@register_env` or `@register_learning_env` in the task-named module. Task
discovery recursively imports these modules, so category `__init__.py` files do
not need per-task re-exports.

## Implement the selected route

After the shared baseline exists, follow the selected specialized reference:

- `references/handwritten-expert.md`
- `references/task-program-expert.md`
- `references/rl.md`

Reuse existing manager functors and registered components. Invoke
`$add-functor` only when a missing observation, reward, event, action, dataset,
or randomization term is required. Use `$add-test` for test structure.

## Validate proportionally

At minimum:

1. parse every new or changed runnable config;
2. prove registration/discovery for Python-owned tasks;
3. run the route-specific focused tests;
4. run `embodichain list-task` when task-discovery metadata changed; and
5. format changed Python files with the pinned project Black version.

Do not claim an expert trajectory is qualified from schema validation alone.
Physical expert behavior requires an environment run with its validators and
persisted completion result. Do not claim an RL task works from config parsing
alone; at least construct/reset the selected environment and run a minimal
trainer-routing smoke test when dependencies permit.

## Completion checklist

- [ ] Prompt was routed to the correct specialized reference(s)
- [ ] Task family, optional subdomain, task name, and runnable IDs are stable
- [ ] Python and config paths share the same task-first hierarchy
- [ ] Environment component and runnable deployment ownership are not mixed
- [ ] A Python task module exists only when the selected route needs it
- [ ] No solution-method directory or same-named task package was introduced
- [ ] Existing components and manager functors were reused where possible
- [ ] Route-specific tests and focused validation passed
