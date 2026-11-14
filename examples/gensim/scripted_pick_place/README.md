# Scripted GenSim pick/place reproduction

This example prepares fixed inputs, runs the shared GenSim workflow, and measures
wall-clock time through native execution and cleanup. It demonstrates the
previous dual-UR5 table/cube case without a live model or reconstruction service.

The recorded outcome is **pickup succeeds, placement release verification fails**.
It is a reproducible failure investigation, not a successful expert dataset
collection or qualification of real photograph-to-scene generation.

## Requirements and inputs

Install the project with its GenSim extra in a supported DexSim environment:

```bash
pip install -e ".[gensim]"
embodichain run-example gensim/scripted_pick_place --help
```

The worker uses four Torch CPU threads, seed 0, Default physics, no native renderer
and no camera acquisition. The robot's assets may be downloaded on the first run;
downloaded assets, assembled URDFs and grasp caches are retained between runs.
No model credentials are required.

| Input | Value |
| --- | --- |
| Photograph | Generated 100 × 80 white placeholder |
| Instruction | `Put the cube on the table` |
| Model calls | Reviewed responses in `responses.json`, injected by service role |
| Scene | Procedural table and 4 cm cube, with fixed poses |
| Embodiment | Official dual UR5 with DH-PGI-140-80 grippers |
| Installation | Position `[1.45, 0, 0.25]`, rotation `[0, 0, -90]` degrees |
| Resource binding | Logical `source` → `cube_native`; default manipulator → `right` |
| Execution policy | Official `motion_gen_verified.yaml` |

Robot installation and physical timing belong to [runtime.yaml](runtime.yaml).
The white image has no visual evidence for the scripted object declarations.
The image planning report's visual coverage validates those declarations against
the supplied contract; it does not prove that a model recognized objects in this
placeholder.

## Run and repeat

Use a new output directory for each attempt. Both commands execute a fresh worker
process; they rebuild the procedural inputs and runtime deployment rather than
replaying a compiled program.

```bash
embodichain run-example gensim/scripted_pick_place --output outputs/gensim-scripted-a
embodichain run-example gensim/scripted_pick_place --output outputs/gensim-scripted-b --compare outputs/gensim-scripted-a
```

Run the commands separately when your shell stops on nonzero exits. An observed
placement failure returns **exit code 1**. Inspect `worker.log` and the execution
report: a dependency, asset or startup error can also return a nonzero status.
The example does not turn an expected failure into workflow success.

The application path is:

```text
fresh Python worker
  → prepare placeholder image, scripted responses and procedural GLB scene
  → load the versioned run/runtime configuration
  → Task Engine interpretation, image grounding and semantic graph publication
  → reuse the prepared scene through the workflow's existing scene input
  → bind logical entities/resources and select a measured landing pose
  → create Z-up native assets and a configured Task Program deployment
  → strict component composition, program decoding and catalog preflight
  → initialize/reset Gym and execute Pick → Place through Atomic Skills
  → verify measured effects, persist reports and close owned runtime resources
  → worker shutdown
```

The example supplies `GenSimServices` through the workflow's existing Python
injection boundary and supplies a prepared scene. The business workflow is shared
with `gensim run`; this example has its own `run-example` launcher. It does not
import tests, patch production functions, or invoke live model/asset generation.
Use the [live-service configuration example](../README.md) for real inputs.

## Intermediate outputs and frontend

The output directory contains:

| Path | Meaning |
| --- | --- |
| `input.png`, `responses.json`, `runtime.yaml`, `run.yaml` | Frozen example inputs |
| `scene/scene_export/` | Newly prepared procedural source scene and GLBs |
| `run/task/semantic_task_graph.html` | Standalone interactive planning frontend |
| `run/task/semantic_task_graph.json`, `run/task/task_spec.json` | Canonical calls and normative specification |
| `run/binding_report.json`, `run/preflight_report.json` | Native object/resource bindings and static checks |
| `run/deployment/` | Bound graph, program, integration, native assets and runnable `task.yaml` |
| `run/execution_report.json`, `run/run_manifest.json` | Native segment outcomes and workflow status |
| `run/service_calls.json` | Role-bound service audit, identifying injected adapters |
| `timing.json`, `process_timing.json` | Workflow and complete worker wall-clock measurements |
| `comparison.json` | Optional semantic-artifact and segment-outcome comparison |
| `worker.log` | Complete worker stdout/stderr |

Open the HTML directly, or serve only the example output directory:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory outputs/gensim-scripted-a
```

Then visit `http://127.0.0.1:8000/run/task/semantic_task_graph.html` and select
**Calls / 调用** to see Pick → Place. The planning frontend remains an **unbound,
not-run** view. It does not ingest the native execution report automatically.
Its displayed wall time covers instruction/image semantic planning only; it is
not the end-to-end duration. Read `run/execution_report.json` for execution and
`process_timing.json` for full worker time.

## Timing and reproducibility

`worker_process_wall_seconds` starts immediately before spawning the worker and
ends after it exits. It includes Python startup/imports, preparing the image and
scene, planning, binding/asset conversion/preflight, environment startup/reset,
native episode execution, report publication, cleanup and Python shutdown.
It excludes the parent launcher's short startup and optional post-run comparison.

`workflow_wall_seconds` measures configuration loading through the shared
workflow's return and service closure. The procedural scene is prepared before
this interval; its preparation time is reported separately. Simulation time and
video duration are separate measures and must not be substituted for either
wall-clock interval.

Validation of the packaged launcher against the workflow at `9438f69e4` recorded:

| Measurement | Attempt 1 | Attempt 2 |
| --- | ---: | ---: |
| Full measured worker process | 33.76 s | 29.98 s |
| Shared workflow interval | 26.71 s | 22.39 s |
| Episode control steps | 283 | 283 |
| Pickup segment | steps 0–180: succeeded | steps 0–180: succeeded |
| Placement segment | steps 180–283: runtime_failed | steps 180–283: runtime_failed |

Those figures used Python 3.12.14/DexSim 0.6.0rc1 with four Torch CPU threads,
no rendering and populated robot/grasp caches. They are observations, not timing
targets for other machines or live model generation.

The earlier instrumented driver measured 29.48 s and 30.12 s end-to-end, with
17.60 s and 18.16 s inside its preloaded CLI interval. Its import/preparation
boundaries differed from this launcher's; compare the complete process scope
rather than treating these inner intervals as equivalent.

The original 283 steps represent 11.32 simulation seconds at a 0.04 s control
period. A recorded execution video lasted 11.3 s; that video was an execution-only
rerun and does not measure reconstruction or deployment construction.

`--compare` checks the input files, resolved robot/policy snapshots, unbound
semantic graph, source scene, logical/native bindings, program semantics,
segment outcomes and step spans. Program IDs derived
from output paths are ignored. It does not compare floating-point trajectories
or certify physical success. In the earlier per-step comparison against the
recorded execution, object position differed by up to 0.46 mm and orientation by
1.68°, despite matching outcomes and step spans.

A changed robot, SDK, solver, cache or hardware may change timing and outcomes.
A failure at `source_released` is a measured placement failure, not completion of
a successful expert trajectory. The separate normative TaskSpec verdict remains
`not_evaluated`.
