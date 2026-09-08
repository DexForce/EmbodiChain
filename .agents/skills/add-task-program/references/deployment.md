# Task Program Deployment

Attach a program and integration to a physical environment, embodiment, and
execution policy through a thin runnable Gym config.

## Canonical selector

```yaml
id: MyTask-v1

environment:
  component: env.yaml

task_program:
  program: task_program/program.yaml
  integration: task_program/integration.yaml
  execution_policy: ../../../components/execution_policies/trajectory_open_loop.yaml

embodiment:
  component: ../../../components/embodiments/ur5_dh_pgi_140_80.yaml
```

All component paths resolve from the runnable deployment file. Use YAML
components. Do not add `task.component`; that compatibility boundary has been
removed.

## Composition rules

- Select either `environment.component` or corresponding inline environment
  and physical scene fields, never both.
- Select either `embodiment.component` or inline `robot`/`sensor`, never
  both.
- A configured Task Program must select an embodiment with `skill_profile`.
- An environment component is Task Program independent and may also serve a
  handwritten or RL deployment.
- A physical scene component contains no Task Program metadata.
- Embodiment overrides are limited to `uid`, `init_pos`, `init_rot`, and
  `init_qpos`; the sensor suite is selected atomically.

## Multiple embodiments

Reuse one program across variants. Reuse one integration only when all
variants satisfy its required embodiment contract and task defaults address
logical resources shared by those profiles. Otherwise create an intentionally
different integration/policy pairing; do not fork the provider-independent
program just to rename robot parts.

Give every runnable deployment a unique Gym ID. Loading a configuration-owned
ID is process-local and occurs when the config passes through
`config_to_cfg()`; task-package import alone does not register it.

## Python boundary

Do not create a task subclass for a fully configuration-defined deployment.
The loader registers the common `EmbodiedEnv`. If the task also provides a
separate handwritten or simulator-RL solution, its flat task module and
deployment can coexist under the same task identity.
