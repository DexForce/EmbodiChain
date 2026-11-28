# Physically validated grasp labels

The example pairs the registered CobotMagic V100 gripper
with the registered `CoffeeCup/cup_translated.ply` mesh. The YAML sets the cup's
mass, friction, scale, collision mesh and support axis. The registered hand
supplies its TCP and open/close commands. The example overrides hand stiffness
(200) and maximum effort (30). Sampling, validation and pose NMS use their
defaults. Run from the repository root with the PhysX/CUDA environment active.

```bash
python -m embodichain affordance label-grasps \
  --config examples/toolkits/grasp_labeling/cobotmagic_cup.yaml \
  --output outputs/grasp-labeling/cobotmagic_cup.pt
```

## Sample a smaller set

For a quick end-to-end check, run a smaller budget in a separate output.

```bash
python -m embodichain affordance label-grasps \
  --config examples/toolkits/grasp_labeling/cobotmagic_cup.yaml \
  --stable-poses 3 --max-drops 12 --directions 4 --candidates-per-direction 4 --sample-count 2000 \
  --num-envs 4 --output outputs/grasp-labeling/cobotmagic_cup_quick.pt
```

The full example releases the cup 5 cm above the table. With
`sampling.support_up_axis: [0, 0, 1]` the cup's local +Z points toward its
rim. Initial rotations target upright, side-lying and inverted supports with
random tilt and yaw. The **settled** inclination determines the mode (within
45° of vertical for upright or inverted). For other objects, omit the axis to
sample uniform random rotations. Settled poses are then grouped by the table
normal in object coordinates and the centroid of the measured object/table contact
points, transformed into that same frame. The selection distributes the
budget across observed modes and regions, then favors distinct rotations within
them. A mode must settle in physics to enter the dataset.

The default budget keeps up to five placements at least 20° apart from 100
drops. From 64 upper-hemisphere viewpoints the hand travels toward the object.
Each placement and direction receives up to 100 candidates from 20,000 sampled
rays. The grasp contact-frame approach must be within 5° of its requested direction.

Drops use up to 100 arenas here. Validation packs candidates across directions
and placements into 2,400 arenas. Each row restores its own frozen object pose
and starts with open fingers. Only the final partial batch needs padding.
On a smaller GPU, add just `--num-envs 128` to the full command to run
at most 128 validations at once. The five-placement, 64-direction,
100-candidate-per-direction budget stays the same.

## Configure another mesh or hand

Pass `--mesh path/to/object.obj` and `--gripper-urdf path/to/hand.urdf` to
override the example assets. With a catalog model and no URDF override, the CLI
uses the model’s standalone hand, including its mesh paths and mount calibration. CLI paths resolve from the current directory.
YAML paths resolve relative to the YAML file or via the EmbodiChain data registry.
The `rigid` section uses `RigidObjectCfg` units and supports `body_scale`,
`attrs.mass_props`, `attrs.material_props` and `shape.collision`.

For another catalog hand, set `gripper.model`, supply its matching standalone
URDF, and update or omit the per-hand overrides. For an unregistered
parallel-jaw URDF, replace `model` with a `description` mapping containing
`model_id`, `root_link`, `collision_links`, `pad_links` (left and right),
`root_from_contact_frame`, `min_opening_width` and `max_opening_width`.
Set `root_from_tcp`, `open_qpos` and `close_qpos` alongside `description`.
Both commands must cover every moving hand joint, including mimic followers.

The command checkpoints frozen placements, candidate poses and trial outcomes
alongside the output. Add `--resume` to the same command after an interruption
or incomplete contact observations. The final file contains CPU tensors.
Successful candidates are sorted by generator cost and deduplicated across
placements in the object frame (3 mm, 5 degrees by default).

```python
import torch

labels = torch.load("outputs/grasp-labeling/cobotmagic_cup.pt", weights_only=True)
object_from_tcp = labels["grasp_pose"]              # [N, 4, 4], translation in metres
opening_m = labels["open_length"]                   # [N], antipodal contact-point distance
placement = labels["object_pose_index"]             # [N]
world_from_object = labels["object_pose"][placement]
world_from_tcp = world_from_object @ object_from_tcp
directions = labels["approach_direction_world"]     # [N, 3], inward unit vectors
```

The width is a geometric contact distance. During physics the hand starts at
`open_qpos` and closes towards `close_qpos`. Each retained label passed in
its indexed placement. A completed run with zero successes saves an empty
`[0, 4, 4]` pose tensor and an empty width vector.

## Validate a supplied TCP

`GraspValidator.validate` takes one object-from-TCP matrix per arena. All
translations are in metres. For a replay, use the saved object-relative TCP
and the arena-from-object transform where it was tested. A different hand
calibration, object scale, friction or mass changes the physical experiment.
This standalone example checks two top-down poses on a small block.

```python
import numpy as np
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.affordance.grasp.grasp_validation import GraspValidator

block = RigidObjectCfg.from_dict({
    "uid": "block",
    "shape": {"shape_type": "Cube", "size": [0.065, 0.065, 0.12]},
    "init_pos": [0, 0, 0.061],
    "attrs": {
        "mass_props": {"mass": 0.015},
        "material_props": {"static_friction": 0.99, "dynamic_friction": 0.97},
    },
})
validator = GraspValidator(block, "FrankaPandaHand", num_envs=2, physics="default")
try:
    tcp = np.tile(np.diag([1., -1., -1., 1.]), (2, 1, 1))
    tcp[:, 2, 3] = 0.035
    tcp[1, 0, 3] = 0.15  # Deliberate miss.
    for result in validator.validate(tcp):
        print(result["status"], result.get("failure_stage"), result.get("hold"))
finally:
    validator.close()
```

Set `headless=False` to watch arena 0. Use `restore_placement(T)` before
`validate` to replay one recorded table pose across all arenas, or pass a
tensor of shape `[num_envs, 4, 4]` for mixed placements. To acquire resting
poses yourself, initialize with `settle_on_init=False`, call `drop_objects`
with object-to-arena rotations, then restore its supported, settled results.
Passing `return_support_centers=True` also returns the object-frame centroids
of the contacts measured on the table at the end of each drop.

A trial approaches, closes, lifts 10 cm, optionally travels 3 cm out and back
along arena +X and +Y, and holds for three seconds. The target must clear the
table by at least 2 cm, maintain pad contact without table contact throughout
each scored window, and have both pads touching at the end of that window.
Set `physics="default"` for GPU PhysX or `physics="newton"` for two-way DexUni.
The result identifies the first failed phase. Incomplete sensor observations
are retried on resume. A pass records a particular simulated trial. Retain the
measurements when comparing borderline grasps or different backends.

## Inspect validation evidence

The file records `physics`, `engine_version`, the mounted `root_from_tcp`,
`sampling`, and each placement's `support_mode`, `support_region`,
`support_normal_object` and measured `support_center_object`. Every completed
candidate has a row in `trials`, including failures and poses suppressed by NMS.
Use the trial's own TCP and placement when replaying a suppressed pose.

```python
t = labels["trials"]
k = 0  # First retained label in a nonempty result.
members = (t["label_index"] == k).nonzero().flatten()
rep = labels["representative_trial_index"][k]
tested_world_tcp = labels["object_pose"][t["object_pose_index"][members]] @ t["poses"][members]
print(labels["quality_fields"], t["hold"][members], t["disturbance"][members])
print(t["widths"][rep], t["direction_index"][rep], t["candidate_index"][rep])
```

The quality columns are maximum gripper-relative COM slip (mm), maximum
gripper-relative object rotation (degrees), and fraction of samples with
bilateral pad contact. Values are relative to the start of each window.
An unreached or disabled window has NaN measurements. `failure_stages` decodes
the integer stage. Failed trials have `label_index == -1`. Each passing trial
references its first greedy cost-ordered NMS representative. An NMS group can
contain trials from different placements. `arena_index` identifies the
physical row of the measurement. `run_stats.real_trials / run_stats.arena_slots`
measures validation-slot utilization for the current invocation, including retries
in that invocation. The CLI prints wall time through the final file write. The per-mode
`run_stats["support_coverage"]` counts placements, tested candidates, passes
and retained representatives.

## Precompute the atomic-skill planning cache

Run the following from the checkout before the first planning call. It uses the
deployment's mounted hand, open/grasp commands, backend, scaled affordance mesh
and material parameters. It prints the cache file in
`~/.cache/embodichain/grasp_annotator_cache/validated/`. Repeat `--pair` for
additional hand/object/affordance triples from the same deployment.

```bash
python -m embodichain affordance precompute-grasp-cache \
  --gym-config embodichain/extensions/tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml \
  --pair eef cube cube_grasp
```

The CLI prepares the configured endpoint commands and the final TCP adjustments
from the deployment's skill presets. For
invocations that change the TCP or override the commands, use the mounted
generator's provider with the same effective parameters as the skill.

```python
cache_path = generator.validated_labels.precompute(
    affordance.mesh_vertices,
    affordance.mesh_triangles,
    affordance.validation_attrs,
    hand_commands=(open_qpos, grasp_qpos),  # One vector per command in control-part order.
    pose_adjustment=adjustment,  # Same adjustment dictionary as the planning invocation.
)
```

Both preparation paths use the same cache identity as planning. A cache miss
during planning reports the missing key and the preparation command. The
ordinary `affordance label-grasps --output` command writes a standalone dataset. Warming
a deployment through `precompute-grasp-cache` resolves its own physical identity.

## CobotMagic cup reference run

One fresh seed-0 run of the full example on an RTX 5090 D v2, GPU PhysX,
DexSim 0.6.0, and four CPU math threads gave the following coverage. Each NMS
representative is counted under the placement of its tested TCP.

| Settled support | Placements | Tested TCPs | Passed physics | NMS representatives |
| --- | ---: | ---: | ---: | ---: |
| Upright | 2 | 8,909 | 8,632 | 6,890 |
| Side-lying | 2 | 7,048 | 6,926 | 5,695 |
| Inverted | 1 | 4,741 | 4,708 | 3,904 |
| Total | 5 | 20,698 | 20,266 | 16,489 |

All 100 drops settled. Nine batches of 2,400 validation arenas assigned
20,698 of 21,600 slots to real candidates (95.82%). The CLI took 146.38 s
from process start to exit (112.65 retained labels/s). Scene setup through
writing the final tensor took 127.35 s (129.48 retained labels/s).

A separate three-arena replay used the first retained placement of each cup
mode and its first passing, first failing and last passing TCP in trial order.
All nine restored transforms matched their recorded placements to within
4.8e-7 maximum absolute matrix-element error. Mixing placements while keeping
each TCP in its original arena reproduced all nine pass/fail outcomes and
failure stages. Repeating the grouped run also matched all nine. A second
mixed layout that reassigned arenas matched eight pass/fail outcomes and seven
failure stages. One previously passing TCP failed on lift, and one failed TCP
stopped during disturbance instead of lift. These numbers describe individual
physics observations for this backend and run.
