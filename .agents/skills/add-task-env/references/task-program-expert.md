# Task Program Expert Trajectory

Use this route for declarative expert behavior composed from Semantic Calls,
including programs authored by a human, agent, or MLLM frontend.

Invoke `$add-task-program` and let it own program authoring, integration
composition, deployment attachment, and static validation. This reference owns
only the surrounding task/environment decisions.

## Canonical layout

```text
embodichain_tasks/configs/tasks/<category_path>/<task_name>/
├── env.yaml
├── task.<embodiment>.yaml
└── task_program/
    ├── program.yaml
    └── integration.yaml
```

- `env.yaml` is a reusable physical-environment component.
- `program.yaml` is provider- and embodiment-independent task flow.
- `integration.yaml` owns contracts, semantic `scene_binding`, profile
  defaults, action options, effect monitors, and task-specific runtime
  services.
- `task.<embodiment>.yaml` is the runnable deployment.
- Execution policy and embodiment declarations are selected from reusable
  components.

Do not create `scene.yaml`; semantic scene data is nested under
`integration.yaml.scene_binding`. Component files do not carry compatibility
`version` fields.

## Runnable selector

```yaml
id: MyTask-v1

environment:
  component: env.yaml

task_program:
  program: task_program/program.yaml
  integration: task_program/integration.yaml
  execution_policy: ../../../components/execution_policies/<policy>.yaml

embodiment:
  component: ../../../components/embodiments/<embodiment>.yaml
```

Paths resolve relative to the runnable deployment. Deployment embodiment
overrides are limited to `uid`, `init_pos`, `init_rot`, and `init_qpos`.

## Python boundary

A supported configuration-defined Task Program does not need
`<task_name>.py`. Loading the deployment composes the typed components,
checks physical scene targets and contracts, preflights the program, and
dynamically registers the common `EmbodiedEnv`.

Create a task Python class only when the prompt also requests independent
Python-owned behavior that cannot be represented by this path. Do not add a
class merely to hold the Task Program.

## Environment obligations

The physical environment must declare every native UID selected by
`scene_binding`. The embodiment component must provide a compatible
`skill_profile`. A reusable embodiment must not name task-local objects;
for joint-position constraint evidence, omit `object_ids` to scope evidence
to the selected scene's graspable rigid objects unless explicit narrowing is
required.

## Validation

Run the static deployment inspector supplied by `$add-task-program`, then the
focused configured-integration tests. Static success proves schema,
composition, contract, scene-target, and lowering consistency. It does not
prove physical success. Qualify the expert path with a real environment run,
the configured effect assurance, segment validators, and persisted completion
metadata.
