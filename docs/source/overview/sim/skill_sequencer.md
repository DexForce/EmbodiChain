# Browser skill-sequence authoring

```{currentmodule} embodichain.lab.visualization.authoring
```

The skill sequencer turns the Viser browser scene into a small authoring tool.
Instead of editing a Python script to try a different manipulation order, you
click a target in the 3-D view, stack a few Atomic Skill *cards* in a side
panel, compile them into one trajectory, watch a translucent preview, and only
then run the sequence on the real simulated robot.

It is a thin, opt-in layer on top of {doc}`viser_visualization`: nothing is
registered unless your application creates the objects below, and a browser
session without the panel behaves exactly as before.

## Quick start

From the repository root:

```bash
python scripts/tutorials/visualization/skill_sequencer.py --viser
```

The terminal prints the server endpoint, normally `http://127.0.0.1:8080`.
Open it while the simulation is running and work through the **Skill sequence**
panel:

1. **Pick a target.** Click the cube in the 3-D view. The panel's
   *Picked entity* line shows its UID.
2. **Add a card.** Choose a skill in the *Skill* dropdown and press
   **Add card**.
3. **Configure it.** Select the card in *Selected card*, then press
   **Bind selected entity** for `pick_up`, or enter a target position and press
   **Apply target position** for `move_end_effector` and `place`.
4. **Compile.** Press **Compile sequence**.
5. **Preview.** Use *Play / Pause*, *Step forward/backward*, and the *Frame*
   slider. The real robot does not move.
6. **Execute.** Press **Execute sequence** and watch the cards advance.

The same tutorial has a browser-free self-check that drives the whole pipeline
programmatically and asserts its invariants. It is the fastest way to confirm
the feature works in a new environment:

```bash
python scripts/tutorials/visualization/skill_sequencer.py --headless_smoke
```

## Panel controls

| Control | Meaning |
|---|---|
| Summary line | Card count, whether the sequence is compiled, total waypoints, and the newest status or error message. |
| *Cards* list | The sequence in execution order. Each row shows a state marker, the skill, the bound entity, and the card's compiled waypoint range. |
| *Selected card* | The card that the edit buttons act on. |
| *Skill* | Skill used by the next **Add card**. |
| **Add card** / **Remove card** | Append or delete one card. |
| **Move up** / **Move down** | Reorder the selected card. |
| *Picked entity* | UID of the newest click-pick in the 3-D view. |
| **Bind selected entity** | Attach the picked entity to the selected card. |
| *Target x/y/z (m)* + **Apply target position** | Set the selected card's target position. |
| **Compile sequence** | Plan every card into one trajectory. |
| **Execute sequence** | Replay the compiled trajectory on the real robot. |
| *Preview* controls | **Play / Pause**, **Step backward**, **Step forward**, and the *Frame* slider drive the translucent preview. |

Every edit invalidates a previous compilation: the waypoint ranges disappear and
the cards return to their configured state. Compile again before previewing or
executing.

## Card states

A card carries exactly one of five states, shown as a colored marker.

| Marker | State | Meaning |
|---|---|---|
| 🟡 | `unconfigured` | A required input is missing: a target entity for `pick_up`, or a target position for the other skills. The sequence cannot be compiled. |
| ⚪ | `ready` | Fully configured, and eligible for compilation or execution. |
| 🔵 | `running` | Execution is inside this card's waypoint range. |
| 🟢 | `succeeded` | Execution left this card's waypoint range without an error. |
| 🔴 | `failed` | Compilation or execution failed. The row's second line carries the planner's diagnostic. |

A failed compilation stops at the first unplannable card: that card turns
`failed`, and the cards after it stay `ready` without a waypoint range.

## Preview versus execution

The two run paths are deliberately different.

**Preview** is read-only. {class}`SequencePreview` evaluates the robot's
analytic forward kinematics on the compiled joint positions and publishes the
resulting link poses as translucent *preview nodes*, anchored on the real
robot's current link poses. It never writes joint targets, never steps physics,
and never moves an object. Pausing, seeking, or looping the preview therefore
leaves the simulation exactly where it was, and a preview that coincides with
the current configuration overlaps the rendered robot exactly. An orange
polyline traces the compiled end-effector path.

**Execution** is the real thing: the robot is commanded with the compiled
joint positions and physics advances between waypoints, so objects are grasped,
moved, and released. It cannot be undone; rebuild the scene to start over.

## Supported skills and parameters

The compiler currently accepts three Atomic Skills.

| Skill | Required input | Optional parameters |
|---|---|---|
| `move_end_effector` | `position` | `rotation` (3×3, defaults to a top-down TCP orientation), `sample_count` |
| `pick_up` | a bound target entity | `approach_direction` (defaults to `(0, 0, -1)`), `pre_grasp_distance`, `lift_height`, `hand_interp_steps`, `sample_count` |
| `place` | `position` | `rotation`, `lift_height`, `hand_interp_steps`, `sample_count` |

The panel exposes the required inputs only. Optional parameters are available
through {class}`UpdateCard`, which an application can emit from its own
controls.

`pick_up` and `place` require a `"grasp"` entry in the session's control-part
mapping, and the engine must provide a grasp-pose generator and a control
profile for that part.

## Application wiring

Five objects connect a simulation to the browser panel. The shortest working
path, assuming a robot, an atomic-action engine, and a Viser-enabled simulation:

```python
from embodichain.lab.visualization.authoring import (
    AuthoringBridge,
    AuthoringSession,
    PreviewPlaybackCfg,
    SequencePreview,
    SkillSequencePanel,
)

session = AuthoringSession(robot, engine, sim, {"motion": "arm", "grasp": "hand"})
runtime = sim.visualization_runtime  # requires visualization.allow_commands
preview = SequencePreview(session, runtime.exporter, PreviewPlaybackCfg(autoplay=True))
preview.register()
runtime.refresh_scene()  # preview nodes need a new manifest
bridge = AuthoringBridge(
    session,
    runtime,
    SkillSequencePanel(),
    preview,
    stepwise_execution=True,
)
bridge.register()

while running:
    if not bridge.execution_active:
        sim.update(step=1)
    preview.advance()
    bridge.update()
    preview_updates, overlays = preview.capture_inputs()
    runtime.capture(
        sim_step=step,
        sim_time=sim_time,
        overlays=overlays,
        preview_updates=preview_updates,
    )
```

| Object | Thread | Responsibility |
|---|---|---|
| {class}`AuthoringSession` | simulation | Owns the cards, compiles them through the atomic-action engine, and executes the result. Never touches Viser. |
| {class}`SequencePreview` | simulation | Playback cursor over the compiled trajectory; produces preview-node poses and the path overlay. |
| {class}`SkillSequencePanel` | visualization worker | Renders an immutable {class}`PanelViewState` into browser controls and emits immutable command values. |
| {class}`AuthoringBridge` | simulation | Drains click-picks and panel commands, applies them, and publishes the new view state. Turns every failure into a status string instead of raising. |
| {class}`StepwiseExecution` | simulation | Replays a compiled sequence under host-loop control so the browser keeps updating while the robot moves. |

## Keeping the browser responsive during execution

{meth}`AuthoringSession.execute` replays the whole trajectory inside one call.
That is fine for a script, but a host loop cannot publish anything while it
blocks: the browser freezes and every card jumps straight from `ready` to
`succeeded` once the run is over.

Passing `stepwise_execution=True` to {class}`AuthoringBridge` makes an
**Execute sequence** click start a {class}`StepwiseExecution` instead. Each
{meth}`AuthoringBridge.update` call then advances the run by
`execution_steps_per_update` waypoints and returns, so the panel shows the
`running` → `succeeded` transitions as they happen. Two rules apply:

- The host loop must not step the simulation itself while
  {attr}`AuthoringBridge.execution_active` is true, because each tick already
  steps physics.
- Sequence edits are refused while a run is active, so a browser click cannot
  invalidate the trajectory being replayed. {meth}`AuthoringBridge.cancel_execution`
  abandons a run.

The blocking path remains the default and is unchanged, so existing
integrations keep their behavior.

## Known limitations

- Only `move_end_effector`, `pick_up`, and `place` can be compiled. Other
  Atomic Skills are rejected by {class}`AddCard`.
- The panel edits a card's target entity and target position only. Orientation
  and the per-skill tuning parameters need an application-supplied control that
  emits {class}`UpdateCard`.
- A card's target position is a single point; multi-waypoint `place` goals are
  reachable only through the atomic-action engine directly.
- One session drives one robot and one environment. The preview renders the
  environment selected by {attr}`PreviewPlaybackCfg.env_id`.
- Compilation is synchronous on the simulation thread. A long grasp-sampling
  pass makes the browser wait, unlike execution.
- Executing a sequence is not undoable, and nothing resets the scene between
  runs. Freeze a grasped object's dynamics from the execution callback if a
  replayed grasp throws it, as the tutorial does.
- Click-picking accepts rigid objects only. Picking any other asset kind leaves
  a status message and no binding.
- Browser commands require `visualization.allow_commands`, which `--viser`
  enables. Without it the panel is built read-only.

## Related pages

- {doc}`viser_visualization`
- {doc}`atomic_actions/index`
- {doc}`sim_manager`
