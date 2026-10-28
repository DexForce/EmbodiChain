# GraspKit parallel-jaw geometry

`AntipodalGraspPoseGenerator` requires an explicit `collision_geometry`.
`ParallelJawGripperModelCfg` supplies the model identity and minimum/maximum
contact opening; it does not approximate fingers or the palm with boxes.
`get_parallel_jaw_collision_geometry` resolves the registered hand's actual
URDF collision meshes, open joint state and calibrated TCP. Custom hardware
must provide its own geometry; missing geometry is an error.

## Construct a generator

```python
from embodichain.toolkits.graspkit import get_parallel_jaw_gripper_model
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGenerator,
    get_parallel_jaw_collision_geometry,
)

generator = AntipodalGraspPoseGenerator(
    get_parallel_jaw_gripper_model("franka_panda_hand"),
    collision_geometry=get_parallel_jaw_collision_geometry("franka_panda_hand"),
)
```

Task Program, benchmark and example integrations use the same service and
geometry contract. The algorithm, collision policy and annotation/cache policy
retain their separate configuration objects. Marvin requires an explicit
`side="left"` or `side="right"` because its two hands have different meshes.
A runtime TCP calibration can be supplied explicitly with `root_from_tcp`;
this changes the returned TCP poses without moving the physical geometry.

For a custom hand, construct `FixedGripperCollisionGeometry.from_urdf` with
its root/TCP, collision links, actual open joint positions, canonical-to-TCP
transform and point spacing. Use `link_from_tcp` for a virtual TCP and explicit
mesh-path mappings when a registered archive requires them. The loader records
URDF/mesh hashes, resolved joint transforms and the sampled geometry. It does
not construct a robot or simulator.

## Frames and collision scope

Sampling, candidate ranking and collision checks use a canonical contact frame:
X opens the jaws, Z approaches the object, and the origin is the antipodal-pair
midpoint. Opening width is the full distance between contact surfaces. Public
pose outputs are converted once to the geometry's declared runtime TCP; callers
must not add another hand-specific rotation. Explicit canonical poses can use
`geometry.canonical_to_tcp_poses` at the same boundary.

For Panda, native `fr3_hand_tcp` closes along Y. The conversion is `Rz(-pi/2)`
with no translation; the contact pads lie near TCP Z=0. The benchmark's policy
and runtime still use the same TCP. Other hand calibrations come from their
actual asset and embodiment definitions.

Collision geometry uses the complete declared hand/finger mesh surfaces at a
fixed open state. Candidate contact widths do not move the fingers beyond that
physical command. All mesh vertices and deterministic triangle-surface samples
are checked in bounded candidate/point blocks. This finite surface check does
not certify continuous mesh intersection, closing contact, arm/environment
collision, swept paths or physical grasp success.

| Model ID | Minimum contact opening (m) | Maximum contact opening (m) |
| --- | ---: | ---: |
| `dh_pgi_140_80` | 0.005 | 0.100 |
| `franka_panda_hand` | 0.001 | 0.080 |
| `cobotmagic_v100_gripper` | 0.001 | 0.100 |
| `tianji_marvin_gripper` | 0.001 | 0.095 |
| `pika_gripper` | 0.001 | 0.094 |

These are contact-pair eligibility bounds. The actual URDF/open-state geometry
owns collision checking.

## Versioned geometry sources

The source archives are the official assets selected by
the asset registry and `extensions/embodichain_robots/`, pinned here to dataset revision
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
approach along hand-frame Z. Their mesh origins and offsets are retained in
the collision template.

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
The collision template preserves those joint transforms and the full palm
geometry. The archive's `../Collision` references require an explicit mapping
to its packaged `Collision/` directory.

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
and thickness 0.033736937 m. The larger palm depth is 0.081209033 m. These are
descriptive asset measurements; the checker uses the transformed mesh surfaces.

The 0.10 m total joint travel is not the open gap: at zero position, the
opposing finger meshes' inner X bounds are 0.095514838 m apart on the left
hand and 0.095523180 m on the right. The preset takes the smaller gap and
rounds downward to 0.095 m rather than using the nominal travel. The gripper
collision template retains the source mesh offsets and joint-axis tilts for the
selected side.

Collision meshes under `Collision/`:

| File | SHA-256 |
| --- | --- |
| `left_hand_base_link_auto_convex.obj` | `7da437d15d0ff48bbc2de3bc6130dbf883cef9662495174dbade3b1344aaf75d` |
| `right_hand_base_link_auto_convex.obj` | `24d69d0ac945323ad4114eccf24f4c99db7e69dba9b59aeaa5ef2bbc35c43920` |
| `left_hand_finger_1_link_auto_convex.obj` | `1baa6e94c0598ca3f9c210928c44b9275767ad792f4078f7b26efef045da14fe` |
| `left_hand_finger_2_link_auto_convex.obj` | `e52c9365e6c67a0a99857364b11e64bbe65b2ed23861bb1e2fea956ccc12d25d` |
| `right_hand_finger_1_link_auto_convex.obj` | `c28eb3fc1ce437c73413999a4d492787e2d98ace879e21672fa3ae052f90bcde` |
| `right_hand_finger_2_link_auto_convex.obj` | `7925becf7dd7de15aab37f0d611eb81c89b2465ee8d6ee8dcdf202f16e788fb2` |

### PikaGripper

Source: `PikaGripper/PikaGripper.urdf`, registered by `eef_assets.PikaGripper`.
Archive MD5: `8a9d0779e7dbfa7bf5d4713175df014e`; URDF SHA-256:
`cb99efefd74f26011c56d5afcf13839c13a5374160fecb02ac02ba9fa172d952`.

The asset includes its adapter. Native gripper-base X approaches, Y opens,
and Z is the finger-width axis. The Pika TCP aligns these with grasp Z, X,
and Y respectively. Source finger bounds give approach extent 0.11214 m,
width 0.03811 m and thickness 0.05700 m. Palm approach extent is 0.08480 m.
The full open gap measured from opposing inner mesh bounds at q=0.05 m is about
0.09404 m; the preset
rounds down to 0.094 m. Joint zero closes the jaws, and finger 2 mimics finger 1.
These are simulation mesh dimensions, not hardware specifications.

| Collision mesh | SHA-256 |
| --- | --- |
| `gripper_base_link.STL` | `ac4593d07d4d95436bb125e76df22332414c24f287ff3de0b35c3336cdfca5e6` |
| `gripper_left_link.STL` | `16de0af984683c4286bf826557c1ea3c40380c64f890969cde5c880726c7474b` |
| `gripper_right_link.STL` | `0a10906fa2ed940db0de90b0f171db3cc550c8cfd56b83bebd33b9a8955e0b66` |
