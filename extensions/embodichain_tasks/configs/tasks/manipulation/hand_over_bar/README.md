# Hand over a rigid bar

The configured Task Program transfers a fixed 0.18 × 0.05 × 0.05 m bar from the
left gripper to the right gripper, then releases it on the tabletop. The reusable
Dual UR5/DH embodiment and verified motion policy own robot calibration and
execution. The environment owns the Default physics backend and static supports.
No task-specific Python environment or action loop is needed.

Use `examples/expansion/scene_robot_feasibility/search.bar_dual_ur5.yaml` with
`embodichain search-scene-feasibility` to search permitted bar/support-group poses
and the installation around the mounting midpoint. All candidates preserve the
robot configuration, task roles and physical parameters. This workflow adds
independent self/inter-arm/payload/world collision checks and ordered measured
lift, receiving-hand possession and release evidence. The ordinary Task Program
effect/terminal validators have their own narrower evidence scope.

Measured successful episodes qualify their exact layouts and execution attempts;
region sampling remains exploratory until the frozen-distribution validation
provides sufficient statistical evidence. Failed/unverified episodes are retained
as diagnostics and do not enter the successful dataset.
