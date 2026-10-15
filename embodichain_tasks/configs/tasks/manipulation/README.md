# Manipulation Atomic Skill trajectory expansion

Activate the `embodichain` conda environment and run from the repository root:

```bash
conda activate embodichain
embodichain run-env --gym-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml
```

These deployments use the scenes and gripper calibrations from
`embodichain run-atomic-action --physics default --robot ur5 --planner trapezoidal`.
Single-arm samples use UR5 with DH PGI-140-80; Handover uses two UR5 arms with
Robotiq 2F-140 grippers. The microwave, drawer, cubes, support surface and robot
mounts retain the corresponding Atomic sample layouts. Each episode starts from
its configured scene and robot state. Repeated Pick/Place performs three cycles
within each episode. Handover runs one complete pickup/transfer/release per
episode; OpenDrawer opens and closes once per episode. Both collect three
sweeps of their 16 recipes (48 episodes), resetting all objects and robot joints
between rounds. Other samples also repeat through independently reset episodes.

| Atomic Skill | Gym deployment relative to this directory |
|---|---|
| pickup | `repeated_pickup/task.ur5.yaml` |
| pickup + place | `repeated_pick_place/task.ur5.yaml` |
| pour | `repeated_pour/task.ur5.yaml` |
| press | `repeated_press/task.ur5.yaml` |
| slide | `repeated_slide/task.ur5.yaml` |
| twist | `repeated_twist/task.ur5.yaml` |
| handover | `repeated_handover/task.dual_ur5_dh_pgi_140_80.yaml` |
| move_joint | `repeated_move_joint/task.ur5.yaml` |
| axis_align | `repeated_axis_align/task.ur5.yaml` |
| open_door | `repeated_open_door/task.ur5.yaml` |
| move_held_object | `repeated_move_held_object/task.ur5.yaml` |
| push_object | `repeated_push_object/task.ur5.yaml` |

The existing tableware and Rubik's-cube tasks retain their separate physical
scenes. `repeated_move_held_object` and `repeated_push_object` expose their
implemented Atomic Skills in the same UR5 reference setup.

## Expansion and collection

The default fixed scene has four Affordance branches and four trajectory
variants, giving 16 episode recipes. MoveJoint and PushObject have four
trajectory recipes. Variant zero is the reference; the remaining variants use
`joint_residual` or `via_points`, with normalized amplitude `0.005` and
deduplication tolerance `0.002`. Sampling uses the same shared phase permissions
as the Atomic launcher. Contact paths, phase endpoints, required joint targets,
arrival intervals and gripper columns remain fixed. Pour's interior tilt target
is protected; the preceding Pick provides its trajectory variants.
Handover only varies the empty-hand pickup approach, ending at the exact
pre-grasp waypoint. Its final reach, lift, receiving approach, handover and
placement retain every reference sample, including internal waypoints and the
gripper schedule. The stationary arm stays fixed during the pickup approach.
Both hosts use a Robotiq closing target of 0.5 rad to retain the can throughout
the transfer; the previous 0.4 rad target allowed nominal trajectories to slip.

The common policy is `components/expansion_policies/atomic_fixed_scene.yaml`.
Task-local `expansion.yaml` files declare the episode call schedule and output
location; Repeated Pick/Place and OpenDrawer retain their existing `expansion/`
resource paths. Object-position and visual randomization are disabled by default.
RGB recording uses the selected embodiment's sensors.

The default parallel width is two environments. Episode limits range from
2,000 to 8,000 control steps according to trajectory duration, avoiding the
former eager allocation of 16 10,000-frame RGBA buffers while retaining longer
Trapezoidal paths. Recipe count is independent of this parallel width.
The shared collection selects recipes cyclically, so `--max_episodes` can
request more than one complete sweep. Every collected round has its own episode
identity and recorder boundary; reset jumps are not stored as continuous motion.

Use explicit recipe indices to inspect selected variants or reduce a run:

```bash
embodichain run-env \
  --gym-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
  --headless --num-envs 2 --max_episodes 4 \
  --expansion-recipe-indices 0 1 4 5 \
  --output-dir /tmp/pick-place/expansion \
  --dataset-dir /tmp/pick-place/datasets
```

`--num-envs` also sets expansion slot capacity. Explicit recipe selections must
contain exactly `--max_episodes` indices. Add `--disable-sensor` for joint/state
collection without RGB. Re-running the command creates the recorder's next
numbered dataset; give `--output-dir` a fresh path to retain each run's manifest.
Each manifest records requested/selected variants, row identities and committed
episodes. Each new batch restores the configured initial state before planning.

For example, add `--max_episodes 96` to the Handover or OpenDrawer command to
collect six complete sweeps. Omitting it collects the default three sweeps.

These samples use projected effects and report command completion. Measured
physical objectives are separate. Handover preserves its landing-position
validator in `repeated_handover/task.ur5.objective.yaml`; Repeated Pick/Place retains
`repeated_pick_place/task.ur5.objective.yaml`. The existing Default/Newton selectors on Repeated
Pick/Place and OpenDrawer continue to select their separately owned environments.
