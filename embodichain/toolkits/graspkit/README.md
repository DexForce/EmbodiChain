# GraspKit parallel-jaw geometry presets

`get_parallel_jaw_gripper_model(model_id)` returns a fresh
`ParallelJawGripperModelCfg`. The static catalog contains planning dimensions;
robot assembly, asset loading, TCP transforms and generator policy have their
own owners. Inline configurations remain available for calibrated hardware.

## Default use

The reusable Franka Panda embodiment selects `franka_panda_hand` for `hand`.
The CobotMagic embodiment selects `cobotmagic_v100_gripper` for its declared
`right_eef` grasp endpoint. The Tianji Marvin embodiment selects
`tianji_marvin_gripper` for both `left_hand` and `right_hand`, reusing the
registered `TianjiMarvinCfg` with `with_gripper: true`. For example:

```yaml
runtime_services:
  grasp_pose_generators:
    right_eef:
      kind: antipodal_parallel_jaw
      model: cobotmagic_v100_gripper
```

These declarations use the library defaults in
`AntipodalGraspPoseGeneratorCfg`, `ParallelJawGraspCollisionCfg` and
`GraspAnnotationCfg`. Task integrations can tune the supported fields through
`grasp_pose_generator_overrides`; the shared embodiment retains its defaults.
An inline `model: {model_id: ...}` is a custom configuration and does not look
up the named preset.

Direct callers use the same defaults:

```python
from embodichain.toolkits.graspkit import get_parallel_jaw_gripper_model
from embodichain.toolkits.graspkit.pg_grasp import AntipodalGraspPoseGenerator

generator = AntipodalGraspPoseGenerator(
    get_parallel_jaw_gripper_model("franka_panda_hand")
)
```

## Geometry semantics

All dimensions are in metres. Grasp-frame X is the opening axis, Z is the
approach axis, and Y is the finger-width axis. Opening width is the full
distance between the opposing contact surfaces, not one finger's travel.
The three presets below use a 1 mm minimum usable opening, matching the generic
model's planning cutoff. Franka and CobotMagic close at joint position zero;
Marvin opens at zero and closes by decreasing joint position to -0.05 m.

Finger dimensions and palm depth are axis-aligned collision-mesh extents,
rounded upward to the next millimetre. They are scalar size approximations:
the current model does not retain mesh offsets, curved surfaces or independent
palm width/thickness. Full robot collision checking and TCP alignment remain
the host integration's responsibility.

| Model ID | Min opening | Max opening | Finger length | Finger width | Finger thickness | Palm depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `franka_panda_hand` | 0.001 | 0.080 | 0.054 | 0.022 | 0.027 | 0.092 |
| `cobotmagic_v100_gripper` | 0.001 | 0.100 | 0.077 | 0.056 | 0.025 | 0.074 |
| `tianji_marvin_gripper` | 0.001 | 0.095 | 0.103 | 0.043 | 0.034 | 0.082 |

## Versioned geometry sources

The source archives are the official assets selected by
`embodichain/data/assets/robot_assets.py`, pinned here to dataset revision
`48317bd8733182f36d306cc2a7a60ae5e47ed2c9`. Their MD5 checksums were verified
against the download registry. These presets
describe those simulation assets; the CobotMagic V70 variant needs its own ID.

### Franka Panda hand

Source: [FrankaV4.zip](https://huggingface.co/datasets/DexForceAI/embodichain_data/blob/48317bd8733182f36d306cc2a7a60ae5e47ed2c9/robot_assets/FrankaV4.zip).
Archive MD5: `0c613d884798e8b084604381641cc8e1`.

URDF: `Panda/PandaWithHand.urdf`, SHA-256
`a4a2d3265dadd69c4c7d8ed9238ae44494c217db6f66f586fcc40e8754ce2315`.
The packaged Panda model names its hand links `fr3_*` and selects the white
Franka Hand meshes. Both finger joints declare 0–0.04 m travel, giving the
nominal 0.08 m full opening. The finger joints open along hand-frame Y and
approach along hand-frame Z, so mesh X maps to `finger_width`, Y to
`finger_thickness` and Z to `finger_length`.

Collision meshes under `Panda/meshes/robot_ee/franka_hand_white/collision/`:

| File | Measured extent (m) | SHA-256 |
| --- | --- | --- |
| `finger.stl` | X: 0.021002460, Y: 0.026429254, Z: 0.053766549 | `f36550380d0208f35279b2514db133bfdde5b9242ec35e8ed7970d1ffe985501` |
| `hand.stl` | Z: 0.091886953 | `94493e94f30fe940f2c8ca2f155c3bbe67bbff406d3edf5e261670d2f0f6e2ed` |

### CobotMagic V100 gripper

Source: [CobotMagicArmV4.zip](https://huggingface.co/datasets/DexForceAI/embodichain_data/blob/48317bd8733182f36d306cc2a7a60ae5e47ed2c9/robot_assets/CobotMagicArmV4.zip).
Archive MD5: `8cc54c240c2f26e84e22250c8364b0ec`.

URDF: `CobotMagicWithGripperV100.urdf`, SHA-256
`4340728998d271c3fbc677016baeac1bea014888e1c08fe40e6969b27b61dee6`.
Both finger joints declare 0–0.05 m travel, giving the 0.10 m full opening.
Their joint origins rotate the finger meshes by a quarter turn about X;
mesh Z becomes the opening axis and mesh Y becomes the approach axis.
Consequently mesh X maps to `finger_width`, Y to `finger_length` and Z to
`finger_thickness`. The gripper-base mesh's Z extent supplies `palm_depth`.

Collision meshes under `Collision/`:

| File | Measured extent (m) | SHA-256 |
| --- | --- | --- |
| `link7_acd.obj` | X: 0.055600829, Y: 0.076499882, Z: 0.024499951 | `333fc9cbbcfffe3703d446d452c31f1da391169ac1e3ec0b635c3853affa8525` |
| `link8_acd.obj` | Same rounded extents as `link7_acd.obj` | `7f4aa3068769b89119bbcb6668eb546909f83abeae17e64e3e427956fd1547bf` |
| `gripper_base_v100_acd.obj` | Z: 0.073242376 | `f048466991e252651527e54ac55df064dd8bad79cc523f27c4dd54d156a08218` |

### Tianji Marvin gripper

Source: [TianjiMarvin.zip](https://huggingface.co/datasets/DexForceAI/embodichain_data/blob/48317bd8733182f36d306cc2a7a60ae5e47ed2c9/robot_assets/TianjiMarvin.zip).
Archive MD5: `af91dbcc352fae3469a3190d0c6e1756`.

URDF: `robot_with_ee_acd.urdf`, SHA-256
`691a5f5a08d3335a1e1a50487b0f886f17eb30da4ea7703bf320a0bf9eac874f`.
The four finger joints span -0.05–0 m; on each hand finger 2 mimics finger 1.
Position zero opens the hand and -0.05 m closes it, so the embodiment's command
presets use `[0.0, 0.0]` and `[-0.05, -0.05]` respectively. The bare
`robot_acd.urdf` variant has no gripper preset.

Transform each collision mesh by its URDF joint-origin rotation and translation
into the corresponding `*_hand_base_link` frame. Base-frame X is the opening
axis, Z is the approach axis, and Y is the finger-width axis. The maximum
extent across all four fingers gives length 0.102398208 m, width 0.042925822 m
and thickness 0.033736937 m. The larger palm depth is 0.081209033 m. These
dimensions round upward to the shared preset's millimetre values.

The 0.10 m total joint travel is not the open gap: at zero position, the
opposing finger meshes' inner X bounds are 0.095514838 m apart on the left
hand and 0.095523180 m on the right. The preset takes the smaller gap and
rounds downward to 0.095 m rather than using the nominal travel. Source mesh
offsets and joint-axis tilts still belong to the full robot collision model;
the scalar planning preset retains their measured extents only.

Collision meshes under `Collision/`:

| File | SHA-256 |
| --- | --- |
| `left_hand_base_link_auto_convex.obj` | `7da437d15d0ff48bbc2de3bc6130dbf883cef9662495174dbade3b1344aaf75d` |
| `right_hand_base_link_auto_convex.obj` | `24d69d0ac945323ad4114eccf24f4c99db7e69dba9b59aeaa5ef2bbc35c43920` |
| `left_hand_finger_1_link_auto_convex.obj` | `1baa6e94c0598ca3f9c210928c44b9275767ad792f4078f7b26efef045da14fe` |
| `left_hand_finger_2_link_auto_convex.obj` | `e52c9365e6c67a0a99857364b11e64bbe65b2ed23861bb1e2fea956ccc12d25d` |
| `right_hand_finger_1_link_auto_convex.obj` | `c28eb3fc1ce437c73413999a4d492787e2d98ace879e21672fa3ae052f90bcde` |
| `right_hand_finger_2_link_auto_convex.obj` | `7925becf7dd7de15aab37f0d611eb81c89b2465ee8d6ee8dcdf202f16e788fb2` |
