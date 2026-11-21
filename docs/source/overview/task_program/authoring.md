# Configuration authoring

Task Programs accept both the explicit `kind` AST and concise action mappings.
The loader expands shorthand before strict decoding; unknown fields, bounded
repeats, references and resource requirements retain their normal checks.
An action item contains exactly one action name. Objects are always explicit.

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

Lists express sequences. A single item expands directly to that item. An
explicit `sequence: {items: [...]}` retains an intentional sequence wrapper.
Use `segment: name` with `steps`, optional `instruction`, `post` and `validators`.
`repeat: {count: 3, body: ...}` keeps bounded lexical repeat semantics.
`parallel: {branches: [...], barrier: ...}` keeps the existing explicit safety
and barrier requirements. Registered extensions use `registered: {call_id: ...,
arguments: ...}`. String `at` and `final_target` values expand to `target_ref`.

The normalizer retains canonical node locations for compilation and maps errors
back to authored locations. Equivalent shorthand and explicit ASTs have the
same compiled identity. Historical explicit inputs retain their fingerprints.

## Shared targets and transport

`pose` is a single world pose; `cyclic_pose` retains repeat-local selection.
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

```bash
python .agents/skills/add-task-program/scripts/inspect_deployment.py \
  path/to/task.yaml --json --explain
```

The report includes normalized input, authored locations, effective contracts,
resolved default resource IDs, targets, generated sources and execution limits.
Static checks establish configuration consistency; measured effects and task
validators establish physical success during normal execution.

The {download}`validation record <../../../architecture/task-program-authoring-validation.json>`
separates unchanged program identities, execution outcomes, measured acceptance,
and remaining native SDK or source-entry qualification gaps. It does not mark
physical qualification complete when a required evidence channel is unavailable.
