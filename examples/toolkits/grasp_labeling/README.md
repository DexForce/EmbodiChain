# Physically validated grasp labels

The example pairs the registered CobotMagic V100 gripper
with the registered `CoffeeCup/cup_translated.ply` mesh. The YAML sets the cup's
mass, friction, scale and convex decomposition, the hand's TCP and drives, and
the sampling budget. Run from the repository root with the PhysX/CUDA environment active.

```bash
python -m embodichain label-grasps \
  --config examples/toolkits/grasp_labeling/cobotmagic_cup.yaml \
  --output outputs/grasp-labeling/cobotmagic_cup.pt
```

For a quick end-to-end check, run a smaller budget in a separate output.

```bash
python -m embodichain label-grasps \
  --config examples/toolkits/grasp_labeling/cobotmagic_cup.yaml \
  --stable-poses 2 --directions 4 --candidates-per-direction 4 --sample-count 2000 \
  --num-envs 4 --output outputs/grasp-labeling/cobotmagic_cup_quick.pt
```

The full example drops randomly rotated cups from 5 cm above the support and
accepts up to five settled orientations at least 20 degrees apart within 100 drops.
It samples 64 uniform upper-hemisphere viewpoints and approaches towards the
object. The actual contact-frame approach axis must lie within 5 degrees of the
requested direction. Each placement/direction gets up to 100 candidates from
the 20,000-ray antipodal surface sample. Every candidate starts from its recorded
resting pose with open fingers, then goes through close, lift, X/Y disturbance and hold.
Fewer available placements or candidates reduce the number of trials.

The default batch has 2,400 arenas. Candidates from different directions share
a physics batch when they use the same resting placement. Set `--num-envs` to override the YAML for another GPU or a
smaller sampling budget.

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
