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

## Configured trajectory-generation showcase

Run the same deployment with the separate source-neutral Generation Profile:

```bash
python examples/sim/motion/repeated_pick_place_generation_showcase.py \
  --task-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml \
  --generation-profile embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/generation.demo.yaml \
  --output-dir /tmp/repeated-pick-place-generation \
  --seed 7 --device cuda --headless
```

The showcase executes candidate ordinals 0, 1, and 2 as separate Task Program
episodes and writes JSON provenance plus joint-trajectory plots. Add
`--save-video` for one MP4 per candidate. The profile varies only the Place
retract phase on the original control grid; Pick, release/contact samples, and
phase endpoints remain unchanged.

The explicit environment seed rewinds reset and grasp-sampling streams before
each candidate so all three ordinals start from the same reference episode.

This is a projected-assurance visualization. Completion means the configured
command sequence completed; it is not measured task success, receipt-confirmed
coverage, or a qualified dataset commit.

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

| Contract representation | Primary `action` | `observation.state` | Auxiliary EEF field | Controller boundary |
|---|---|---|---|---|
| omitted (main-compatible default) | Active joints in the configured joint order | Measured active-joint qpos | None | Not recorded separately |
| `joint_position` | Active joints in the configured joint order | Measured active-joint qpos | Optional measured `observation.eef_pose` | Not recorded separately |
| `joint_position_velocity` | Active-joint `[qpos, qvel]` | Measured active-joint qpos | Optional measured `observation.eef_pose` | Not recorded separately |
| `eef_pose_parallel_gripper` | `[x, y, z, roll, pitch, yaw, gripper]` from `arm_action`, then `gripper_action` | Measured active-joint qpos | Optional measured `observation.eef_pose` | Not recorded separately |
| `joint_position_parallel_gripper` | Seven arm joints from `arm_action`, then one `gripper_action` scalar | Measured active-joint qpos | Optional measured `observation.eef_pose` | Not recorded separately |

EEF actions are absolute arena-frame targets. Rotation is XYZ Euler/RPY in
radians, and gripper values are normalized to `[-1, 1]`. The requested EEF
target is captured from the Action Manager's validated flat policy command; it
is never reconstructed from the measured pose. Policy feature width, names, and
slices come from the manager's ordered descriptors, which are stored as
`embodichain.action_terms` in the LeRobot action feature metadata and episode
sidecar. Configurations that omit `action_contract` keep the expert joint schema
unchanged. Select EEF policy recording explicitly:

```yaml
params:
  action_contract:
    version: 1
    representation: eef_pose_parallel_gripper
    record_eef_observation: true
```

Contract datasets use LeRobot's per-frame `task` / `task_index` mapping for
segment instructions. Main-compatible datasets retain the legacy
`subtask_index` sidecar behavior. Datasets with different declared action
representations are distinct feature schemas and cannot be merged directly.
