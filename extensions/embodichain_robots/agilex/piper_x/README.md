# AgileX PiPER-X

`AgilexPiperX` is a fixed-base six-axis bare arm. Its `arm` control group
runs from `base_link` to `flange_link` and uses numerical pose IK. The reusable
`piper_x_agx_gripper` embodiment mounts the separately registered
`AgxGripper` asset and adds one `eef` command for the jaw opening in meters.

```bash
embodichain data download --name AgilexPiperX
embodichain data download --name AgxGripper
embodichain preview-robot --robot AgilexPiperX
```

The catalog's portable URDF removes the source's virtual world joint and adds
the PiPER-X flange link and fixed joint from the official tool Xacro. It
retains the source arm kinematics, inertials, DAE visual meshes and STL
collision meshes. The simulation preset caps J6 at ±165° and uses the
manual's per-axis speed ratings. Drive gains and the initial pose are
simulation settings.

Both assets fetch only the required source files from
[AgileX agx_arm_urdf at 680fa92](https://github.com/agilexrobotics/agx_arm_urdf/tree/680fa925cd00db7e2628563d2edc5540963045de).
Each source file is pinned by SHA-256. The original URDFs, PiPER-X mounting
Xacro and [MIT license](https://github.com/agilexrobotics/agx_arm_urdf/blob/680fa925cd00db7e2628563d2edc5540963045de/LICENSE)
are retained in the cache. The derived URDFs use relative mesh paths and can
be loaded without ROS or Xacro.

The mounted embodiment uses the source Xacro's 4.5 mm flange-Z offset and
90° Z rotation. Its TCP is centered between the fingertips 142.5 mm along the
flange Z axis. Use the bare preset to preview the flange, or select
`piper_x_agx_gripper.yaml` as a Gym deployment's embodiment component.
