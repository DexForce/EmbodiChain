# Whiteboard Draw Table

The Franka FR3 starts with an uncapped marker between its fingers. Its expert
draws the frame, header, and two column dividers using inverse kinematics,
feedback from the measured nib pose, and a pressure loop. The marker is a
free articulation with a spring-loaded nib. Its mechanical collars help keep
it in the gripper.

Ink is stamped into a 1 mm raster from measured nib-to-board contacts at every
physics step.
Success requires at least 90% coverage on each of the seven lines, no gap
longer than 10 mm, and at most 1 cm² outside the prescribed line bands. The
environment checks the same measured ink during expert execution and dynamic
replay once the held pen tip is at least 10 mm clear of the board.

The workcell follows the drawing geometry and acceptance thresholds of the
[RoboWorld whiteboard benchmark](https://github.com/gym-anything/robotics/tree/main/benchmarks/roboworld/environments/whiteboard)
as used in Eden. EmbodiChain uses an original [compliant marker asset](https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/2a304556866773bacb1593638d6b771a83104040/scene_assets/whiteboard_marker_v1/WhiteboardMarkerV1.zip).
The native scene renders the robot and marker. The demonstration's ink inset
shows the contact-derived raster.
