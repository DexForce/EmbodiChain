# Repeated Pick and Place

The Task Program repeats cube transfers between two authored target poses. The
physical environment is selected separately from the robot embodiment and program.

| Deployment | Robot component | Physics | Configuration |
|---|---|---|---|
| franka (default) | franka_panda | default | `task.franka.yaml` |
| ur5 | ur5_dh_pgi_140_80 | default | `task.ur5.yaml` |
| franka_newton | franka_panda | newton | `task.franka.newton.yaml` |
| ur5_newton | ur5_dh_pgi_140_80 | newton | `task.ur5.newton.yaml` |
| franka_rlinf | franka_panda_vla | default | `task.franka.rlinf.yaml` |
| franka_rlinf_joint | franka_panda_vla | default | `task.franka.rlinf_joint.yaml` |
| franka_rlinf_expert | franka_panda_vla | default | `task.franka.rlinf_expert.yaml` |

Inspect the available deployments:

```bash
embodichain show-task embodichain_tasks:repeated_pick_place
embodichain run-env --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml
```

Each deployment owns its physics backend through its environment component. Choose
a deployment for another backend; `--physics` does not switch an existing one.
Task Program execution is the supported expert demonstration route. These original
examples do not independently qualify measured physical placement. No preview or
physical qualification result is claimed by this catalog.

The task-local `catalog.yaml` names deployments; it does not register a Gym ID.
Generate a simulator-free local gallery from the source checkout with:

```bash
embodichain list-task --config-root embodichain_tasks=embodichain_tasks/configs/tasks --category manipulation --export-html task-gallery.html
```

## Dataset action contract

The LeRobot recorder keeps joint position as the default primary action. A
recorded row is aligned as `(observation_t, action_t)`: the observation is read
before the controller applies the action, and the next row observes the
resulting simulation state.

| Mode | Primary `action` | `observation.state` | Auxiliary EEF field | Controller boundary |
|---|---|---|---|---|
| `joint` (default) | Active joints in the configured joint order | Measured active-joint qpos | Optional measured `observation.eef_pose` | `controller.qpos` |
| `eef` | `[x, y, z, roll, pitch, yaw, gripper]` | Measured active-joint qpos | Optional measured `observation.eef_pose` | IK-generated `controller.qpos` |

EEF actions are absolute arena-frame targets. Rotation is XYZ Euler/RPY in
radians, and gripper values are normalized to `[-1, 1]`. The requested EEF
target is captured from the raw policy command; it is never reconstructed from
the measured pose. The executed joint command and action contract are recorded
in the episode metadata. Select EEF recording explicitly:

```yaml
params:
  action_mode: eef
  record_eef_observation: true
```

Datasets generated before this contract was introduced should be treated as a
separate schema and regenerated before SFT or policy evaluation.
