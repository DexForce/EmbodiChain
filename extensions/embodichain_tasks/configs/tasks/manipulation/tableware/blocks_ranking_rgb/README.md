# Blocks Ranking RGB

`task.cobotmagic.yaml` implements `BlocksRankingRGB-v1` as a configured Task
Program using the shared `EmbodiedEnv`. The expert uses known block identities
and measured object poses. Camera observations remain available for downstream
learning.

The program executes four segments in order: the right arm picks the red block
and places it left of green, then the left arm picks the blue block and places
it right of green. Each pick requires a measured lift of at least 50 mm.

Placement targets are grounded from the green block's current position when
the call starts. Their X offsets are -80 mm for red and +80 mm for blue in
world axes; green-block yaw does not rotate those offsets. Each block is
released 80 mm above green's origin, preserving its held orientation, then
must settle. Final XY errors must be below 35 mm and 30 mm respectively.
The final blue-placement gate also rechecks red's position relative to green.

`primary_manipulator` remains the right arm in the shared CobotMagic profile;
`secondary_manipulator` maps to the left arm. Both arms use the same calibrated
object-to-TCP side grasp, with translation `[0.025, 0, -0.025]` m. Its rotation
removes the opposite initial wrist-yaw offsets so either arm can reach the
block and its placement target. This task-level calibration bypasses
antipodal sampling. The shared policy uses 90 motion samples and each pick
has a 45-step grasp dwell. Commands and settling holds
pass through ordinary `env.step()` without task-specific dynamics clearing.
Internal semantic effects are projected; segment acceptance is measured.

Run from the repository root with the project's simulation Python environment:

```bash
python -m embodichain run-task \
  --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/tableware/blocks_ranking_rgb/task.cobotmagic.yaml
```

Static deployment inspection checks schema, component composition, resources,
and Semantic Call lowering. Physical qualification additionally requires live
execution and successful measured segment results.

Local Default-backend checks with sensors disabled passed all four expert segments
and dynamic action replay for seeds 0, 1 and 2. Final checks used measured relative
XY placement and linear/angular stability, independently of program completion.
These observations cover those seeds and execution settings, not every randomized
scene or renderer.

The dual-UR5 and Tianji Marvin deployments reuse this same program,
reset distribution and measured acceptance gates. Their task-local integrations
select the corresponding calibrated grasp and dual-arm policy. See the
[deployment matrix](../../../../../README.md#dual-arm-task-deployments) for
configuration selection and sensor differences.
