# W1 V030 with PikaGripper embodiment

[`dexforce_w1_030_pika.yaml`](dexforce_w1_030_pika.yaml) selects the existing
`DexforceW1` preset with `version: v030` and `with_default_eef: false`, then
appends two standalone Pika grippers. The preset supplies the body components,
source joint groups, FEP solvers and body/arm physics. The embodiment adds only
the tool components, finger groups, TCPs, gripper/head drive overrides, ready
pose and robot-mounted RGB-D cameras. A separate assembly output name keeps
the tool-equipped cache distinct from the bare body. It does not introduce a
W1 hand-brand enum or robot catalog variant.

The component anchors the base and provides the `dual_arm_parallel_gripper`
skill profile. Task control uses `left_arm`, `left_eef`, `right_arm`, and
`right_eef`: 14 arm joints and two independent finger joints. The assembled
robot has 24 joint coordinates, including two mimic followers, and 22
independent joints. Its six torso/head joints hold their initial positions.

## Pika frames and commands

The Pika asset includes its flange adapter and attaches through
`GRIPPER_ROOT_LINK`. Its native gripper-base X axis approaches the object and
Y opens the jaws. The configured grasp TCP has X along jaw opening and Z
along approach. Its flange-relative translation is `[0.00064, 0, 0.160]` m;
the rotation is a -90 degree turn about flange Z. This is a model-derived
reference in the distal pad region, not a factory TCP. Cube integrations place
it 20 mm above the object center, leaving the fingertips clear of the table.

`FINGER1_JOINT` is actuated and `FINGER2_JOINT` mimics it. Each source joint
spans 0–0.05 m: `[0.05, 0.05]` opens and `[0, 0]` closes the jaws. The
`pika_gripper` GraspKit geometry is measured from the packaged collision meshes;
see [geometry provenance](../../../../../embodichain/toolkits/graspkit/README.md).
The embodiment sets gripper drives to 1000 N/m stiffness, 100 N s/m damping,
8 N effort and 0.25 m/s velocity, with finger friction 2.0. These are
simulation settings, not hardware ratings.

The ready pose uses a shallow `[-0.15, 0.30, -0.15]` rad leg-chain bend,
keeping the upper body upright. The head physically pitches down 40° through
NECK2 for tabletop observation, within its -40° to +45° source limits; NECK1
stays at zero. Mirrored arms lower the
elbows about 0.25 m below the shoulders, with about 97° of elbow flexion and
18° of upper-arm abduction. Left J1–J7 are
`[0, -1.25, 0, -1.70, -0.32, 0.30, 0]` rad; the right arm mirrors them as
`[0, 1.25, 0, 1.70, -0.32, -0.30, 0]` rad. The TCPs are approximately
`[0.5215, ±0.2673, 1.0623]` m relative to the robot base; jaw-opening axes
are level and the grippers point forward and about 10° downward. The shared
task embodiment places the base at `[0.25, 0, 0]` m. Task work surfaces are
0.90 m high, about 0.16 m below the ready TCPs. Every W1/Pika deployment
inherits this pose.
Source initial joint order is breadth-first: ANKLE, KNEE, BUTTOCK,
WAIST, NECK1, LEFT_J1, RIGHT_J1, NECK2, then interleaved left/right J2–J7 and
four finger coordinates. It differs from URDF joint-element order.

## Mounted RGB-D cameras

These are designed simulation mounts with ideal metric depth. They do not
model stereo matching, camera noise or hardware calibration. All three belong
to `RobotCfg.sensors`; the embodiment's independent `sensor` list stays empty.
An explicit `sensors` override replaces the suite; `sensors: []` clears it.
Gym's `enable_sensor` setting controls the complete suite. Window and
offscreen rendering inherit the default DLSS Ray Reconstruction pipeline.

| UID | Parent link | Resolution | Near/far (m) | Purpose |
| --- | --- | --- | --- | --- |
| `cam_high` | `head_pitch_j2_link` | 960×540 | 0.05 / 3 | Tabletop, both arms and handover area |
| `cam_left_wrist` | `left_gripper_base` | 640×360 | 0.01 / 2 | Left fingertip approach and local grasp |
| `cam_right_wrist` | `right_gripper_base` | 640×360 | 0.01 / 2 | Right fingertip approach and local grasp |

Nominal fields of view are 87° horizontal and 58° vertical. Head-camera origin
is `[0.105, 0, 0.060]` m in `head_pitch_j2_link`. Its optical axis is aligned
with the head link's +X axis and image up with +Z; the mount adds no pitch.
The world camera pose is the moving head-link pose times this fixed local
extrinsic. With NECK2 at zero it looks horizontally and cannot see the nearby
work area; the 40° ready head pitch supplies the downward tabletop view.
Wrist origins
are `[0.055, 0, 0.135]` m in their gripper-base frames, looking toward
`[0.195, 0, 0.00064]` m with native +Z as image up. Mounts use explicit OpenGL
poses (-Z forward, +Y up), converted from these viewing directions. The
head origin clears the head's forward mesh bound. Raising the wrist origins
keeps the housing out of the central image while retaining the near grasp
below image centre and preserving context ahead of the fingertips. The
mount was checked at recorded approach, grasp and transport states.

AgileX publishes a [wrist fisheye observation screenshot](https://github.com/agilexrobotics/pika_ros/blob/master/img/image_4.png)
and [RGB-D recording thumbnails](https://github.com/agilexrobotics/pika_ros/blob/master/img/image_11.png).
These guide image composition only. Pika's RGB-D and fisheye devices are
separate streams, exposed separately by the [official SDK](https://github.com/agilexrobotics/pika_sdk/blob/master/pika/gripper.py).
The simulation suite retains its 87°×58° RGB-D projection; the public fisheye
picture does not establish matching RGB-D intrinsics or a camera-to-gripper
calibration. The mount coordinates above remain nominal simulation parameters.

## Task deployments

All nine deployments use `physics: default`, CPU simulation state, and the same
embodiment. Native camera rendering and DLSS RR continue to use the GPU.
Each lives under `configs/tasks/manipulation/<task>/task.dexforce_w1_030_pika.yaml`
in `embodichain_tasks`. Existing default deployments remain selected by default.

| Task | Gym ID prefix | Capability |
| --- | --- | --- |
| `repeated_pick_place` | `TaskProgramRepeatedPickPlace` | Verified expert workflow |
| `hand_over` | `HandOver` | Verified expert workflow |
| `tableware/blocks_ranking_rgb` | `BlocksRankingRGB` | Projected expert workflow |
| `tableware/stack_blocks_two` | `StackBlocksTwo` | Projected expert workflow |
| `tableware/pour_water` | `PourWater` | Projected expert workflow with physical segment validators |
| `tableware/blocks_ranking_size` | `BlocksRankingSize` | Environment and success predicate |
| `tableware/match_object_container` | `MatchObjectContainer` | Environment and success predicate |
| `tableware/place_object_drawer` | `PlaceObjectDrawer` | Environment and success predicate |
| `tableware/stack_cups` | `StackCups` | Environment and success predicate |

Every ID appends `-DexforceW1V030Pika-v1` to its prefix. The four environment-only
tasks retain their existing evaluation methods and do not supply expert plans.
Pouring uses mouth-controlled preparation, tilt, return and withdrawal, with
the existing bottle/cup final-pose validators.
Its environment preserves the authored Hybrid renderer and the existing bottle/
cup frames. The handover deployment uses a standing 60×60×240 mm bar. The
source grasps its upper portion and the receiving hand takes the lower
portion after lift, leaving separate contact regions for the Pika palms.
Its transfer flow and measured assurance
remain the same. Deployments reuse the shared execution policies with explicit
overrides bounding joint speed at 0.5 rad/s and retain the original
projected/verified distinction. Handover disables the approximate
planning collision world while retaining physical contacts and measured gates.

The work surfaces share a 0.90 m top height; object spawns and fixed program
targets follow the same support height. All deployments inherit the upright
ready pose and face the workbench. Repeated pick/place uses the left-hand
workspace at `[0.68, 0.22]` and `[0.76, 0.12]` m in world XY. Stacking
uses `[0.70, 0.24]` and `[0.70, 0.08]` m, with bounded reset jitter to keep
both targets in that workspace. Cube grasps approach 30° forward of vertical
so the wrists can reach without raising the elbows above the shoulders.
The upright handover bar is centered at `[0.62, 0]` m, shared by both arms.
Pouring starts from the shared ready pose and derives a 60° tilt direction
from the live cup position. Robot +Y is left, so this layout makes the right
arm pour leftward. The integration declares the bottle mouth at local
Z=0.0697246 m and cup opening at local Z=0.0438780 m, with radii 0.012 m and
0.025 m. The skill computes side clearance and mouth height from the source
body bounds, translating the bottle as it tilts and reverses. There is no
fixed pre-pour bottle-centre target. TOPPRA times the complete keyframe paths
under the policy's joint limits. The pouring table mesh has a +0.05 m local
top; its origin is 0.85 m high to match the 0.90 m working surface.
The mouth-controlled pouring deployment reserves 6000 steps at 100 Hz for
preparation, tilt, return and release, including the recording buffer budget.

The workbench layout passes reset and hold checks for all nine environments.
Expert qualification on seeds 0, 1 and 2 passes all 15 runs, including the
handover receiving and release evidence gates and the existing final task
validators. These seeds qualify the configured simulation scenarios; they do
not establish arbitrary scene or hardware robustness.

```bash
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.dexforce_w1_030_pika.yaml --headless --seed 0
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/tableware/blocks_ranking_size/task.dexforce_w1_030_pika.yaml --preview
```

Use `--disable-sensor` to skip acquisition. Existing recorders in RGB block
ranking, block stacking and pouring include all three RGB-D streams and the
16 independent task joints, and identify the robot as `DexforceW1V030Pika`.

The sensors and rollout buffers retain RGBA uint8 images and metric float32
depth. RGB is recorded as video; full-resolution depth uses the existing
`gray12le` HEVC sidecar with linear quantization over 0.005–3 m and lossless
encoding of the 12-bit codes. The maximum quantization error is approximately
0.37 mm inside that range; the sidecar metadata records the units and decoding
parameters. The reader returns metric float32 arrays. This avoids materializing
millions of depth pixels as numeric LeRobot feature values during persistence.
The depth encoder requires libx265 support; the recorder's numeric fallback
requires substantially more memory. Set `keep_numeric_fallback: true` only
when exact original float32 values are required and sufficient RAM is available.

Episode buffer validity masks identify initialized frames; unused image
capacity is allocated without filling it. Memory still grows with episode
length, so the deployments use one environment by default.
