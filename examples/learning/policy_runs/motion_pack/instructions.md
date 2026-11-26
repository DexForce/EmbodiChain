# EmbodiChain UR5 motion session

The qualified UR5, cube, platform and camera are already installed by the
operator's `ur5_motion_host`. The attached EmbodiChain MCP server owns motion
jobs, physics, grasp evidence, receipts, and video. Use the supplied Session ID.

Discover capabilities then submit the requested skill and poll its run ID.
Each command includes the explicit ID from the task message. Retain the
verified grasp reference for the next turn. The final Place receipt contains
absolute paths to the trajectory, verifier output and video.

For recovery, the operator supplied `output/recovery.py` to rebuild the
fixture on a fresh World, and the motion launcher executes the same fixture
with the dedicated writer. A replacement World has its own authorization and
grasp references. Report the launcher, receipts and verified outputs in
`output/REPORT.md`. After the receipt confirms the requested phase, report
its result and finish the turn.
