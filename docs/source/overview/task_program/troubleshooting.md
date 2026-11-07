(task-program-troubleshooting)=

# Troubleshooting

Start with the first failure in the output. Keep the deployment, seed, local
program and full log together. For the Quickstart, reproduce with seed 7,
`--num_envs 1 --max_episodes 1`, Default physics, and data saving disabled.
If the failure occurs during execution, add `--debug-mode` to the complete
Quickstart command to print the failed demo result and per-segment diagnostics.

The two errors explicitly marked **reproduced** below were observed on the
{doc}`validation` baseline. Other entries describe diagnostic branches; a symptom
alone is not proof of its listed possible causes.

## The program or a component file cannot be found

**First check:** the path named by the exception and the current working directory.
`--task-program program.yaml` is relative to your shell directory. Deployment
component paths resolve relative to the deployment file. A physical environment
component by itself is not a runnable deployment.

**Diagnose:** rerun `embodichain show-task embodichain_tasks:repeated_pick_place`
and compare its deployment with the Quickstart's complete package reference.
Check that the downloaded file is YAML rather than an HTML error page. For a
local deployment, inspect its `environment`, `embodiment`, and `task_program`
references in that order.

**Repair:** use an existing readable program path and a complete deployment.
Do not copy only `task.franka.yaml` into an empty directory while retaining
relative component paths. Python `load_task_program()` takes a filesystem path;
it does not implement the launcher's package-resource lookup.

## A scene object is unknown

**Reproduced:** changing the first Pick to `object: missing_cube` raises:

```text
TaskProgramValidationError: $.program.body.items[0].steps.call.object: program.body.items[0].steps.call.object: Unknown scene entity 'missing_cube'. [reference_validation_failed]
```

**First check:** that exact program path. The bracket index is zero-based.
Programs use `entity_id` from `integration.yaml.scene_binding`, which maps to a
physical `simulation_uid`. They happen to both be `cube` here; that is not a
universal naming rule. An affordance such as `cube_grasp` is another semantic
entity, not a substitute for the object itself.

**Repair:** restore `object: cube`. To add a genuinely new object, update its
physical environment and semantic binding through the integration workflow in
{doc}`scene_registry`; renaming the program reference alone cannot add it.
This reproduced error occurred before environment construction and stepping.

## A field is rejected

**Reproduced:** adding `speed: 0.5` under the repeat node raises:

```text
TaskProgramDecodeError: $.program.speed: Unknown field 'speed'; allowed fields are ['body', 'count', 'kind']. [unknown_field]
```

**First check:** indentation and the node named in the error. A field valid in
one configuration owner is not automatically valid in another.

**Repair:** remove the field. Use the full downloadable programs as structural
references, and consult [motion timing](recipes.md#change-motion-timing) for
execution-policy changes. This reproduced error also occurred before stepping.

## A call is unknown or has no lowerer

**First check:** the call's `kind`, and for a registered call its ID/revision,
against the selected integration's catalog. A spelling mistake, unsupported
call family, or missing registered lowerer are distinct possibilities.

**Diagnose:** restore the original supported program. If that works, compare the
new call with the integration's available calls. Integration authors can run
the [read-only deployment inspector](validation.md#maintainer-checks) to perform
catalog preflight and static lowering without constructing a simulator.

**Repair:** use a supported call or add it through the trusted integration.
Do not put Python imports or executable functions in YAML. See
[registered calls](architecture.md#registered-calls) and {doc}`robot_profiles`.

## The program, scene, robot or policy contracts do not match

**First check:** `program_id` and the components selected by `--gym_config`.
`--task-program` only changes the program; it does not replace the integration.

**Diagnose:** compare the integration's required scene and embodiment contracts
with its scene binding and selected embodiment/policy. Also check physical UIDs
against the selected environment. Reusing a program from an unrelated task or
mixing component versions can cause this symptom.

**Repair:** restore a compatible complete deployment and its matching program ID.
Do not rename contract IDs just to silence validation. For deliberate component
composition, follow {doc}`/tutorial/task_program`.

## Grasp generation or motion planning fails

**First check:** the first failing segment/call in the execution log. A static
preflight pass establishes structural compatibility, not physical feasibility.

**Diagnose:** run the original program at seed 7 in one environment; compare the
object pose, target pose, selected robot and grasp affordance. Possible causes
include an unreachable target, incorrect orientation/frame, inadequate clearance,
or an unavailable grasp/IK solution. Preserve the failure trace from `--debug-mode`.

**Repair:** restore a known working target, then change one value at a time. If
the original deployment fails, inspect installation/assets and robot setup before
changing the program. See {doc}`robot_profiles` and
{doc}`/overview/sim/atomic_actions/index` for integration-level planning diagnostics.

## Motion finishes but position or stability checks fail

**First check:** the failed Place segment's `metadata.validation` and
`metadata.post_policies`, in the debug trace or saved episode sidecar.
A completed progress bar or `projected` runtime result does not override a failed
measurement.

**Diagnose:** inspect `position_error`, `object_position`, `target_position`,
`target_value_index`, and the settling status/thresholds. Check that the resting
target list has the same XY/order as the release list and uses the correct center
height. Possible causes include a missed grasp, collision, residual motion,
or a validator that mistakenly checks the elevated release pose.

**Repair:** correct the target semantics or the physical execution problem, then
rerun unchanged acceptance criteria. Do not enlarge the tolerance solely to
obtain a passing dataset. `wait_stable` checks settling; it does not force a cube
into the requested position. See [acceptance boundaries](architecture.md#runtime-lifecycle).

## The task ran but no dataset was saved

**First check:** remove `--filter_dataset_saving` and `--preview`, select a
deployment with a dataset recorder, and use `--dataset-dir` as in the Quickstart.
`--output-dir` is not the ordinary recorder output override.

**Diagnose:** look for the recorder's `Saved dataset to:` path and finalization
message. Check invalid-attempt warnings: a rejected episode may be discarded.
At this baseline the collection summary can say “episode saved” even when saving
is disabled, so it is insufficient evidence. Do not confuse a visualization video
with a state/action dataset.

**Repair:** run the checked program with saving enabled, then use
`preview_lerobot_data ... --latest --expect-segments 4`. A readable dataset root
contains `meta/info.json`. If interrupted during persistence, follow
{doc}`/guides/offline_expert_datasets` for validation/recovery; do not manually
invent a completion record.

## One environment works but randomized or batch runs fail

**First check:** identify the exact selected deployment and whether it enables
expansion, randomized resets, or different collection defaults.

**Diagnose:** retain the seed, environment-row identity, recipe/candidate selection
and per-row failure data. Check object reachability, target frames and acceptance
for the failing row. One successful seed does not establish valid coverage for
other layouts. Reproduce the smallest failing configuration before scaling up.

**Repair:** fix the scene/configuration or narrow its supported variation based
on measured results. Requalify the intended seeds and batch settings. See
{doc}`/guides/run_task` and {doc}`/tutorial/data_expansion`; expansion is outside
the Quickstart's qualification scope.

## DexSim fails during import

**First check:** run the installation guide's import verification in the same
Python environment as the CLI. During this work, a development DexSim tree failed
with `cannot import name 'reconstruct_cable_rings_kernel'`; a mismatched Warp
installation then produced `DexSim AOT runtime mismatch`.

**Repair:** install the SDK-matched DexSim wheel and its matching dependencies
using the package indexes in {doc}`/quick_start/install`. The validated setup uses
DexSim `0.5.1rc1` with Warp `1.17.0`. Fix environment compatibility before physical
validation; a static config check cannot substitute for a working simulator.
