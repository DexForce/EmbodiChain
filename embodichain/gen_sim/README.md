# GenSim commands

Use the unified namespace; every command also works with `python -m embodichain`.

```bash
embodichain gensim --help
embodichain gensim asset ingest --input raw_asset --category cup --output outputs/assets
embodichain gensim scene generate --image table.png --output outputs/scene
embodichain gensim scene edit --scene outputs/scene --prompt "Move the cup to the left"
embodichain gensim scene expand --scene outputs/scene --config layouts.yaml
embodichain gensim scene preview --scene outputs/scene --viser
embodichain gensim task plan --image table.png --instruction "Put the cup on the tray" --output outputs/task
embodichain gensim services validate --config model_services.json --env-file services.env
embodichain gensim services doctor --config model_services.json --env-file services.env
embodichain gensim ui --host 127.0.0.1 --port 7860
```

Domain operations share their application implementation with the existing
`asset-engine`, `scene-engine`, `scene-randomization`, `preview-scene`, and
`task-engine` commands. Existing argument spellings remain accepted. New commands
use `--input`, `--output`, `--scene`, `--prompt`, and hyphenated option names.
The new scene operation is named `expand`; the legacy top-level command remains
`scene-randomization`.

Scene editing modifies the selected workspace in place. Task planning requires a
fresh output directory and ends with an unbound semantic graph and offline HTML
viewer. Its output alone does not construct a robot or execute a task.

The UI launcher uses the active Python interpreter, preserves the selected SDK
import root, and retains the existing UI's process cleanup and authentication.
Install the `embodichain[gensim]` extra for its optional dependencies.
Model implementations and serving frameworks remain operator-selected; see the
[deployment and BYOS guide](services/DEPLOYMENT.md) and the versioned
[HTTP protocol](services/spec/v1/README.md).

## Generate, bind and execute

For a self-contained input fixture, see the
[scripted pick/place reproduction](../../examples/gensim/scripted_pick_place/README.md):

```bash
embodichain run-example gensim/scripted_pick_place --output outputs/gensim-scripted-a
```

It uses scripted model responses and a procedural scene, then real native
binding, Task Program preflight and verified execution. Each attempt saves the
graph frontend, reports and wall-clock timing. The observed placement failure
retains a nonzero exit status. Its timing does not qualify live image/asset
generation. For real inputs, configure the model services and use `gensim run`:

```bash
embodichain gensim run --config examples/gensim/run.yaml \
  --image /absolute/path/to/table.png --output outputs/gensim-run-001
```

The [example configuration](../../examples/gensim/README.md) selects a trusted
robot embodiment, a file-owned physics backend, and a verified execution policy.
Configuration paths resolve relative to their owning file; explicit CLI path
overrides resolve relative to the caller's working directory. Model services
retain the discovery and credential rules in [services/README.md](services/README.md).
The workflow borrows one service run across task planning, scene generation and
runtime entity grounding. It uses the existing `task_image_grounding` role to
match logical references to the generated inventory and declared robot resources.

A run performs image task planning, scene generation (or read-only reuse through
`--scene`), native entity/resource binding, strict configured deployment
composition and catalog preflight, then one batched demonstration episode through
the ordinary Gym/Task Program execution owner. It does not use the removed
action-agent pipeline. Native physical execution and its result are required;
a graph-only run cannot produce workflow success.

Automatic geometry binding covers observed rigid objects, antipodal pickup and
placement on a measured near-level support or planar container floor. Placement
reuses the shared landing-region selector, including footprint and occupied-space
checks. A run-owned Z-up OBJ copy bakes the exported Y-up scale and coordinate
conversion directly into the vertices, with isolated material/texture sidecars.
This avoids renderer-dependent GLB up-axis conversion while preserving the
original scene export.

Registered calls require compatible configured lowerers and runtime services.
The optional `binding` mapping supplies task-owned `profile`, `runtime_services`,
`scene_binding` additions and `segments` post-policy/validator declarations
through the existing closed decoders. Missing native parts, hidden or ambiguous
references, unsupported geometry and incompatible capabilities reject rather
than selecting a substitute or claiming execution readiness. Model grounding
never creates capabilities, rewrites control-part names or chooses physics.

## Outputs and verdicts

A new run directory contains:

- `source_image.*`: the input frozen for the run, with a SHA-256 in the manifest;
- `task/`: source semantic graph, TaskSpec, image grounding and offline viewer;
- `scene/`: generated workspace, unless an existing workspace was selected;
- `deployment/`: robot/policy snapshots, native mesh copies, integration,
  bound graph, program and runnable `task.yaml`;
- `binding_report.json`, `preflight_report.json`, `execution_report.json`;
- `run_manifest.json` and `service_calls.json`.

Existing run directories are never overwritten. Failed runs retain completed
artifacts and stage/error information. The execution report retains the native
demo segment outcomes and measured-effect verdict; truncation and failed
execution produce a nonzero CLI exit status. Exceptions and interrupts retain
their failure status and release owned runtime resources.

The source task graph stays unbound. The bound graph and explicit deployment are
separate artifacts. The workflow records `task_spec_verdict: not_evaluated`:
a measured Task Program execution result is not a separate normative TaskSpec
certificate.

The standalone planning frontend does not automatically display native execution
results. Its elapsed-time field covers semantic planning only. The scripted
example's `process_timing.json` covers the complete measured worker lifetime;
its `comparison.json` separates matching semantic artifacts/step spans from
floating-point trajectory reproducibility.
