# Task Catalog and Physical Objectives

Use the task catalog to find a logical task, select a runnable deployment, and
inspect the evidence available for it. An optional physical objective measures
what happened in the simulated world independently of the program that generated
the actions.

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
| Runnable deployment | Gym ID and selections of environment, embodiment and optional program/objective | `task.ur5.objective.yaml` |
| Physical objective | Measured goal regions, order, duration and stability limits | `objective.yaml` |

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

## Measure ordered physical outcomes

The optional `ur5_objective` deployment attaches `objective.yaml` through
`objective: {component: objective.yaml}`. It observes the cube reaching A, then
B, then A again:

```{literalinclude} ../../../embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/objective.yaml
:language: yaml
```

Bounds are inclusive and expressed in environment-local metres. Each region
requires consecutive stable samples spanning 0.1 seconds, with linear speed at
most 0.05 m/s and angular speed at most 0.2 rad/s. The bounds describe the settled
cube center, not the gripper release pose.

Historical milestones persist, but leaving the final region or exceeding its
speed limits revokes current physical success. Re-entry requires a fresh stable
hold. Progress advances once per control step; snapshot queries do not advance
time, and partial reset clears only the selected environment rows.

```{note}
Stable position and velocity do not prove gripper detachment. This first
predicate measures ordered stable-region occupancy. It does not change
authoritative Task Program completion, segment validation, environment
termination or dataset persistence.
```

## Compare expert execution and dynamic replay

Run one expert episode and replay its recorded controller actions against the
same physical objective:

```bash
python -m embodichain.lab.scripts.evaluate_task_objective \
  --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.objective.yaml \
  --output-dir objective-baseline --seed 0 --num_envs 1 --headless --device cuda
```

For a controlled variation, use a new output directory and add
`--initial-position-jitter 0.005`. This samples per-episode X/Y offsets within
±0.005 m through reset events and records the actual initial pose. The runner
accepts jitter up to 0.02 m. Reproducibility assumes the same configuration,
runtime, seed and reset schedule; it is not a cross-backend determinism guarantee.

The runner writes `report.json` and, after expert execution, `expert.pt`.
Its default 20-step settling tail is included in the diagnostic trajectory
without extending accepted demonstration segments. Dataset saving is disabled.
Dynamic replay validates the expert action schema, joint order and cadence;
kinematic playback is not physical qualification. This first sample supports
one environment and replays only the recorded action horizon.

| Report field | Meaning |
|---|---|
| `execution_outcome` | Full program completion or completed replay horizon |
| `physical_outcome` | Independently measured objective progress and final truth |
| `demo_acceptance` | Existing expert acceptance result; not applicable to replay |
| `persistence_status` | Dataset and diagnostic trajectory write status |

Reports also retain resolved configuration, authored component contents/hashes,
code revision/dirty status, effective seed and actual sampled initial pose.

### Current validation status

The checked-in {download}`startup record <../../../embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/validation/default-startup.json>`
documents a native run blocked during initialization: the installed DexSim
descriptor rejected `com_quaternion`, which the existing spawn adapter requires.
No physical rollout was measured. The gallery therefore shows a startup error
and unavailable physical outcome; it does not claim success or measured task
failure. A compatible DexSim build is needed to complete expert/replay physical
qualification.

For launch and recording options, see {doc}`run_env`. For objective types and
decoders, see the objective sections in {doc}`/api_reference/public_api`.
