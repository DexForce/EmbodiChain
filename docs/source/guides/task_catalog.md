# Task Catalog

Use the task catalog to find a logical task, select a runnable deployment, and
inspect the evidence available for it.

## Discover and inspect tasks

```bash
embodichain list-task --category manipulation
embodichain show-task embodichain_tasks:repeated_pick_place
```

`show-task` reports the task description, named deployments, robot embodiment,
physics backend, supported uses, configuration paths and supplied validation
records. Use the qualified `package:task_key` when different packages share a
task key. Each displayed launch command selects a concrete configuration.

The catalog distinguishes three artifacts:

| Artifact | Responsibility | Example |
|---|---|---|
| Task catalog | Human-facing identity and named deployment references | `catalog.yaml` |
| Runnable deployment | Gym ID and selections of environment, embodiment and optional program | `task.ur5.yaml` |

Catalog metadata does not register a Gym ID. Existing tasks without metadata
remain discoverable. Capabilities describe available execution routes; they are
not evidence that a deployment passed a physical evaluation. In particular, RL
support is associated with the deployment referenced by a trainer configuration,
not every deployment in a directory containing an `agents` folder.

## Export a local task gallery

From a source checkout, export a static HTML gallery without importing simulator
registries:

```bash
embodichain list-task \
  --config-root embodichain_tasks=embodichain_tasks/configs/tasks \
  --category manipulation \
  --export-html task-gallery.html
```

Open `task-gallery.html` in a browser. `--config-root` points to a `configs/tasks`
directory; repeat it for additional packages. This static mode does not inspect
runtime registrations, so additional registered capabilities may be available.
Omit `--config-root` to use installed task discovery.

The generated gallery links local task resources where available. Keep those
resources in place when using the exported file; it is not a portable archive of
all configurations. Missing previews and validation results are shown as
unavailable.

The repeated-pick-place catalog is authored alongside its deployments:

```{literalinclude} ../../../embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/catalog.yaml
:language: yaml
```

`default_deployment` names a declared deployment. Each `config` references a
task-relative JSON, YAML or YML runnable configuration; optional `validation`
references a task-relative JSON report. Select a different deployment to change
physics backends: launcher `--physics` does not override the backend owned by
the physical environment component.

For launch and recording options, see {doc}`run_env`.
