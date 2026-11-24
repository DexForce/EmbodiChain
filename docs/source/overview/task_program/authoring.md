# Configuration authoring

Task Programs accept shorthand and explicit forms for the `program` tree. The
loader expands shorthand into a canonical abstract syntax tree (AST) before
strict decoding. Both forms use the same validation for fields, repeat bounds,
references and resource requirements.

## Shorthand and explicit forms

Both excerpts pick up `book` and place it at `presentation`, a named target
declared under `targets`.

### Shorthand YAML

A list gives the order of operations, and each action name is a key.

```yaml
program:
  - pick:
      object: book
  - place:
      object: book
      at: presentation
```

### Explicit YAML

Each node declares its `kind`. An `invoke` node contains a `call` with its own
`kind`, and the placement target is a `target_ref` object.

```yaml
program:
  kind: sequence
  items:
    - kind: invoke
      call:
        kind: pick
        object: book
    - kind: invoke
      call:
        kind: place
        object: book
        at:
          kind: target_ref
          target: presentation
```

## Choosing a form

- Prefer **shorthand YAML** for new task files and examples so readers can
  scan action names and segment labels directly. Follow the file's existing
  form for local edits, including agent-assisted edits. Individual nodes in
  one YAML program can use either form.
- Use **explicit JSON** when an agent sends a model response to
  `decode_mllm_task_program()`. A schema can select a node type using `kind`,
  and tools can address calls through paths such as `program.items[0].call`.
  The MLLM frontend looks up `call` at its canonical path in the original
  response, so emit explicit `invoke` nodes with `call` objects. The host
  supplies `integration`, and the frontend accepts sequential programs with
  `pick`, `place` or `hand_over` calls.
- Work with the **normalized AST** when inspecting or transforming a program
  in code. Use `normalize_task_program()` before traversing the tree so the
  tool can handle both forms through the same node structure.

## Implement a task with repository skills

In a source checkout, name the skill in your coding agent's prompt and provide
the runnable deployment path plus the outcome you want to observe. The agent
instructions live under `.agents/skills/`. Start with `$add-task-program` when
the deployment already supplies the scene, robot, and actions you need.

> Use `$add-task-program` with the repeated Pick/Place Franka deployment.
> Change the repeat count to two and check the cube's resting position after
> each Place. Show which four segment occurrences the inspector resolves.

| Implementation task | Skill and inputs |
|---|---|
| Write or change the program | Give `$add-task-program` the subgoals, scene entity IDs, and acceptance checks. Its authoring, integration, and deployment workflows cover the program, semantic bindings, and runnable YAML. |
| Create the physical task environment | Give `$add-task-env` the scene requirements and select the Task Program expert trajectory route. Then use `$add-task-program` to bind the flow. |
| Deploy the flow on another robot | Give `$add-embodiment-component` the robot preset, sensors, and required resource endpoints. Use the resulting skill profile in the deployment. |
| Expose an existing Atomic Skill to program authors | Give `$add-semantic-call` the intended call arguments and the Atomic Skill they should lower to. |
| Implement a new motion primitive | Give `$add-atomic-skill` the typed goal, planning behavior, and expected effect. Use `$add-semantic-call` when the program also needs a new call for it. |

For an existing failure, use the [debug workflow](troubleshooting.md#debug-with-a-repository-skill).

## Example with a measured hold

```yaml
program_id: present_book
instruction: Pick up the book and present it.
targets:
  presentation:
    kind: pose
    frame: world
    position: [0.6, 0.0, 1.1]
    quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]
program:
  - segment: acquire
    steps:
      - pick:
          object: book
  - segment: present
    steps:
      - move_held_object:
          object: book
          at: presentation
    post:
      - hold:
          object: book
          at: presentation
          duration_s: 2.0
          position_tolerance: 0.02
          orientation_tolerance: 0.10
```

This illustrative program requires an integration binding `book`, compatible
resources, and a verified execution policy with Pick and transport monitors.
The target describes the **object** pose, not the end effector. Positions and
position tolerances are in metres; orientation tolerances are in radians.

## Structure and identity

| Explicit `kind` | Purpose | Shorthand |
|---|---|---|
| `sequence` | Run children in order. | A list or `sequence: {items: [...]}` |
| `repeat` | Run an entire body a fixed number of times. | `repeat: {count: 3, body: ...}` |
| `segment` | Label a subgoal and attach checks to its boundary. | `segment: name` with `steps` and optional `instruction`, `contract`, `post` and `validators` |
| `invoke` | Run one Semantic Call. | One action key, such as `pick: {object: book}` |
| `parallel` | Run compatible branches and join at a barrier. | `parallel: {branches: [...], barrier: ...}` |

Lists with multiple items expand to a `sequence`. A single-item list expands
directly to that item. Use `sequence: {items: [...]}` to keep a single-item
sequence node. Each shorthand action mapping has exactly one action name and
includes its required fields. The barrier still declares `kind: barrier`, and
target definitions and segment contract entries also require `kind`. Check the
[parallel execution requirements](architecture.md#parallel-programs) before
binding a parallel block to a deployment.

The normalizer retains canonical node locations for compilation and maps errors
back to authored locations. Equivalent shorthand and explicit ASTs have the
same compiled identity. Historical explicit inputs retain their fingerprints.

## Choose a Semantic Call

| Action key | What to declare |
|---|---|
| `pick` | Name the object. Set `grasp` to select a particular grasp affordance. |
| `place` | Name the held object and choose one destination using `at`, `on`, or `inside`. `at` selects a named pose target. `on` and `inside` select support and container destinations in the semantic scene. |
| `hand_over` | Name the object to transfer. Optionally set `final_target` to deliver it to a named target after transfer. |
| `move_held_object` | Name the held object and its target with `at`. The verified transport call preserves the grasp. |
| `registered` | Specify the integration's `call_id` and its declared `arguments`. |

Use the scene's `entity_id` values and the logical resources declared by the
selected robot profile. Built-in calls accept a `resources` mapping to select
configured resource bindings explicitly. If a slot is omitted, the selected
integration can supply its default. The [deployment inspection](deployment.md#inspect-the-resolved-deployment)
shows the available call contracts and effective bindings.

## Define segments and their outcomes

Use a segment for a subgoal you can describe and check given its incoming
state, such as lifting a block clear of the table. Its `steps` can contain one
call or a sequence of calls. The root `instruction` describes the episode,
and each segment's `instruction` describes its subgoal for recorded data.

- Put actions that run after motion, such as waiting for stability, in `post`.
- Put acceptance checks at the end of the segment in `validators`.
- Use `contract.prerequisites` for conditions needed at entry and
  `contract.outcomes` to identify the results that establish the subgoal.
  A `validator` outcome refers to a zero-based index in that segment's
  `validators` list.

The Stack Two Blocks example refers to the named `lifted` condition from its
selected evaluation component. Its contract makes the validator result an
outcome of `lift_block`.

```{literalinclude} ../../../../embodichain/extensions/tasks/configs/tasks/manipulation/tableware/stack_blocks_two/task_program/program.measured.yaml
:language: yaml
:start-at: - segment: lift_block
:end-before: - segment: stack_block
```

Select reusable named conditions with `evaluation.component` in the runnable
deployment. For episode-level checks and the distinction between call effects
and segment acceptance, see [verify the run](deployment.md#verify-the-run).

## Shared targets and transport

`pose` is a single world pose. `cyclic_pose` selects from its pose list using the
iteration of the nearest enclosing `repeat`, cycling back to the first pose
after the last. String `at` and `final_target` values expand to `target_ref`
objects.
A `scene_relative` target declares `entity`, `frame`, `position` and
`quaternion_xyzw`. `frame: parent` composes the full reference transform;
`world_axes` follows its position while keeping offset and orientation in world
axes. Compilation validates references without observing live state. Each call
resolves and freezes a relative target from its initial planning observation.
A scene-relative `object_near_target` validator consumes the matching call's
frozen target in the same sequential segment; unmatched or parallel acceptance
is rejected during preflight. Use `object_near_relative_target` when acceptance
is deliberately defined against a fresh reference observation.

Typed `move_held_object` selects an exact held object, preserves its verified
grasp and uses the existing MoveHeldObject Atomic Skill. Configure a default
resource under `profile.defaults.move_held_object`, options under
`profile.skill_options.move_held_object`, and a standard composite monitor under
`profile.effect_monitors.move_held_object`. A projected policy cannot execute
this typed retention contract.

The existing `simulation.move_held_object` registered identifier remains valid.
Routed calls accept `target` plus optional `object` and `reference`, validated
against their route. Legacy single-target calls retain their target-only and
effectless behavior. Changing a legacy route into a typed call is an explicit
assurance change and needs qualification.

## Named Demo Skills

Register task-owned assets in `integration.yaml`:

```yaml
runtime_services:
  demo_skills:
    grasp_book:
      call_id: task.grasp_book
      asset_manifest: ../skills/grasp_book/demo_skill.json
      asset_sha256: REPLACE_WITH_VERIFIED_CONTENT_SHA256
      contact_sensor_uid: grasp_contacts
      tcp_contact_links: [left_pad, right_pad]
```

Replace the digest with the asset's actual `content_sha256`, and use the actual
sensor UID and calibrated link names from the selected embodiment. A Demo Skill
that publishes a terminal attachment requires measured contact as well as the
object-to-TCP relation. Paths resolve from
the integration component, or from the deployment for inline integrations;
existing `{asset, path}` references also work. Assets are never identified by
file location alone.

A program invokes the alias with explicit asset-defined roles:

```yaml
program:
  - demo_skill:
      skill: grasp_book
      roles:
        object: book
```

Roles, required initial hold, terminal hold/release, phases, events and permitted
adaptation come from the verified asset. Defaults may reuse the integration's
single-arm Pick binding; otherwise declare `profile.defaults.execute_demo_skill`
or select the resource explicitly on each call.
The selected resource must explicitly advertise the required joint-position
capability. Required contact constraints also need `contact_sensor_uid` and,
for TCP contact, explicit `tcp_contact_links` in the registration. Metric
providers remain trusted host services, never executable YAML.

The deployment generates registered lowerers, required options/monitors and a
source for every expanded occurrence. Repeats receive distinct call indices;
new-style bindings are sealed to the resulting program fingerprint. Explicit
old source declarations remain supported and are never silently re-signed.
Automatic and explicit ownership of the same occurrence is an error.

Current configured Demo Skill execution requires one environment and a
sequential program. Asset adaptation bounds still apply. Internal phases are
monitoring and edit boundaries within one planning/recovery boundary. An asset
with a verified terminal attachment can continue into native transport/place;
a native Pick can supply an asset's declared initial attachment.

## Continuous measured holds

`hold` requires verified effect assurance, a registered rigid object, the normal
execution clock and a preceding verified attachment. It preserves configured
joint drive targets through the ordinary Gym action path. Every acknowledged
observation must satisfy the standard attachment evidence and target pose.
A matching preceding transport target is reused; otherwise the target is
resolved at hold entry. The window lasts `ceil(duration_s / step_dt)` steps.

Failure or missing evidence invalidates its row for the whole window. Cancellation
and interrupted clocks cannot produce success. Results record requested and
actual duration, frozen target, observation counts, maximum measured position
and orientation errors, accepted rows and failure reasons. `wait_stable`
continues to wait for its existing stability condition; it is not a duration hold.

## Inspect before execution

Use the [deployment inspector](deployment.md#inspect-the-resolved-deployment)
to check the authored program against its integration. Compare
`normalized_program` with the intended node structure and use the authored
paths in `calls` to locate expanded occurrences in the source YAML.

The {download}`validation record <../../../architecture/task-program-authoring-validation.json>`
separates unchanged program identities, execution outcomes, measured acceptance,
and remaining native SDK or source-entry qualification gaps. It does not mark
physical qualification complete when a required evidence channel is unavailable.
