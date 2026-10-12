# Atomic Skill CLI scenes

Activate the project environment and run any scene from this directory:

```bash
conda activate embodichain
embodichain run-atomic-action --config embodichain/lab/scripts/_atomic_action/configs/place.yaml --headless
```

The ten configuration names are `pickup`, `place`, `pour`, `press`, `slide`,
`twist`, `handover`, `move_joint`, `axis_align`, and `open_door`. They share the
implementations used by `scripts/tutorials/atomic_action/`, including scene
assets, robot mounts/TCPs, initial poses, lighting, motion options, and replay
callbacks. Place and Pour retain their PickUp precondition. Slide pulls the
drawer, executes it, reads the moved handle, then plans and executes the push.
Handover retains the dual-arm robot and soda-can/support-surface scene.

Configurations describe a direct simulation host, so they use `--config`.
Gym/Task Program deployments continue to use `run-env --gym_config`.
The `physics` field owns the backend; `--physics` can confirm it. To run Newton,
copy a config and declare `physics: newton`; the same scene-specific Newton
contact overlays and solver settings used by the examples are applied.

## Expansion

The default configurations produce four affordance branches and four trajectory
variants per branch (16 recipes). MoveJoint has no affordance and produces four
trajectory variants. Variant zero keeps its reference trajectory. Affordance
branch identities select distinct feasible `AntipodalAffordance` grasp candidates
for Pickup, Place, Pour, AxisAlign and Handover; articulation/contact skills use
their native affordance freedom. Candidate pools can be too small to provide
unique grasps; candidate IDs and reuse flags are preserved in the manifest.

`expansion.trajectory.spatial` uses the same production `joint_residual` and
`via_points` operators as the repeated-pick-place expansion. Scale `0.005` and
deduplication tolerance `0.002` match that deployment. Contact/manipulation
phases on the button, knob, drawer or door, hand transitions, phase endpoints,
and hand joint columns stay exact. Motion phases may change between their
endpoints; no timing factor changes the replay callbacks or control clock.

Recipe index is `affordance_branch * variants_per_family + trajectory_variant`.
`runtime.num_envs` is the parallel capacity, and larger selections run in batches.
Every batch rebuilds the original scene. Seeds and branch identities are stable
across batch sizes. A shortage of accepted trajectory variants fails explicitly.

```bash
# One nominal scene, useful for checking assets/planning.
embodichain run-atomic-action --config embodichain/lab/scripts/_atomic_action/configs/place.yaml --headless --no-expansion --num-envs 1

# Nominal + both trajectory operators + a different grasp, four parallel rows.
embodichain run-atomic-action --config embodichain/lab/scripts/_atomic_action/configs/place.yaml --headless --num-envs 4 --recipe-indices 0 1 2 4

# Use the example's viewer and record replay videos.
embodichain run-atomic-action --config embodichain/lab/scripts/_atomic_action/configs/handover.yaml --num-envs 4 --save-video --keep-alive
```

Omitting `--headless` uses the config's viewer choice. Replay starts automatically;
`--keep-alive` retains the final scene until Ctrl+C. `--viser --headless
--keep-alive` retains browser visualization. `--real-time` adds the original
example's wall-clock playback pacing. `output.save_video` enables the example
camera/MP4 recorder; `--renderer no-render` is available for faster checks without
video. Headless alone does not disable rendering.

## Output and qualification

The output directory contains `expansion_manifest.json` and one subdirectory per
batch. Each replay writes a JSON record and, by default, a `.pt` file with
`positions`, `reference_positions`, `dt`, `env_ids`, and recipe provenance.
Slide writes separate pull and push records. `--output-dir` overrides the
configured directory. Relative configured paths resolve beside the YAML file.

Status `executed` means planning, expansion and simulation playback completed.
As with the original examples, these scenes have no measured physical success
validator; `measured_task_success` remains null. They do not generate LeRobot
datasets. The repeated-pick-place Task Program collection pipeline continues to
own its acceptance, scene/visual expansion, and structured dataset persistence.
