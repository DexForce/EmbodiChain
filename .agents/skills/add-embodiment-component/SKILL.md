---
name: add-embodiment-component
description: Add or update a reusable EmbodiChain embodiment YAML component that owns one simulation robot, its sensor suite, and an optional Task Program skill_profile. Use for configs/components/embodiments, new Task Program-compatible robots, or multi-embodiment task deployment.
---

# Add Embodiment Component

Create the reusable physical and semantic declaration that lets one robot
configuration serve many tasks.

## Establish scope

An embodiment component belongs at:

```text
embodichain_tasks/configs/components/embodiments/<embodiment_id>.yaml
```

It owns exactly:

- `embodiment_id`;
- one `simulation` robot mapping;
- the complete `sensor` list;
- optional `skill_profile`.

It does not own a task, physical scene, Task Program, execution policy,
task-local object IDs, or runnable Gym ID. Component files do not use a
compatibility `version` field.

## Load current context

Read `agent_context/MAP.yaml`, then resolve:

- `robot-system`;
- `sensor-system`; and
- `task-programs` when a `skill_profile` is requested.

Inspect the closest current component under
`embodichain_tasks/configs/components/embodiments/` and the intended runnable
task deployment.

## Reuse or add the robot

If an existing registered robot config can express the request, select it in
`simulation` and add only necessary config overrides. If a new
`RobotCfg`/URDF/control-part implementation is required, invoke `$add-robot`
first. This skill does not duplicate robot registration logic in YAML.

Keep the sensor suite atomic with the embodiment. Sensor parent link names must
exist on the selected robot.

## Add `skill_profile` only when needed

Handwritten and RL deployments may use an embodiment without semantic
metadata. A configured Task Program requires `skill_profile` with:

- `contract_id`: the compatibility contract tasks/policies require;
- `profile_id`: stable unversioned profile identity;
- `resources`: logical task-facing resources;
- `command_presets`: named control commands;
- optional `runtime_services`.

Each resource declares logical endpoints. Endpoint declarations map to actual
`control_part` names and list only capabilities the selected robot/runtime
can provide. Grasp endpoints that require open/grasp commands select a matching
command preset.

Use logical resource IDs such as `primary_manipulator`, `left`, or `right`;
Task Program integrations bind Semantic Call slots to these IDs. Do not expose
task-local names as reusable resources.

## Runtime service ownership

Put embodiment-calibrated services in `skill_profile.runtime_services`, such
as gripper-specific antipodal grasp generation or control-part evidence. Put
task-specific targets, poses, and overrides in the task integration.

For reusable joint-position constraint evidence, omit `object_ids` unless the
embodiment truly supports only an explicit global set. Omission lets runtime
assembly scope evidence to graspable objects in the selected task scene.

Only use service kinds accepted by the current configured decoder. Never
serialize a dotted import or callable.

## Attach without copying

A runnable deployment selects:

```yaml
embodiment:
  component: ../../../components/embodiments/<embodiment_id>.yaml
```

Optional deployment overrides are limited to `uid`, `init_pos`, `init_rot`,
and `init_qpos`. Do not repeat inline `robot` or `sensor` fields.

One task can provide multiple thin deployments selecting compatible
embodiments. Keep the program embodiment-independent and validate the
integration and policy contracts against every variant.

## Validation

At minimum:

1. decode the YAML as a closed embodiment component;
2. build the selected robot config and verify control parts;
3. verify sensor parent links and shapes;
4. verify every profile endpoint/control part/capability and command preset;
5. compose at least one intended task deployment;
6. for Task Program use, run
   `$add-task-program`'s deployment inspector;
7. run a minimal live robot/sensor smoke test when available.

## Checklist

- [ ] Component owns exactly one robot and one atomic sensor suite
- [ ] Existing robot config was reused, or `$add-robot` was completed first
- [ ] Profile is omitted when no semantic consumer needs it
- [ ] Contract/profile/resource/preset IDs are stable and unversioned
- [ ] Endpoint control parts exist on the robot
- [ ] Declared capabilities and commands are actually supported
- [ ] Reusable services contain no task-local object IDs or poses
- [ ] At least one intended deployment composes successfully
