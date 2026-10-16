# Franka online PickPlace showcase

See [validation results](VALIDATION.md) for the measured engineering checks and
the reference learner's unmet task-learning target.

This task physically grasps a 5 cm cube, lifts it, places it at the marked goal,
opens the gripper and withdraws. The expert and student use the same measured
success predicate: at least 5 cm of lift, goal error at most 3 cm, ground-height
agreement, open fingers, a withdrawn TCP and 0.3 seconds of low-speed stability.
The cube is not attached or teleported during execution.

From the repository root, with a working DexSim installation and robot assets:

```bash
python examples/data_pipeline/ods_pick_place_showcase.py \
  --sim-device cuda:0 --train-device cuda:0 \
  --output /tmp/pick-place-online
```

The runner first qualifies the expert on 40 independent seeds and requires at
least 38 successes. It then starts a CPU shared ODS buffer, a recurrent PyTorch
BC learner, an independent evaluation process, Viser at
http://127.0.0.1:8080 and TensorBoard at http://127.0.0.1:6006. Browser commands
cannot control the evaluation robot. Training observations are robot qpos/qvel,
cube pose and goal position; cameras are used only for visualization and videos.

The default is five rounds of 1,000 updates, with 32 episode slots and a manual
refill per subsequent round. Each round waits for fresh publication before its
last training update. The learner never reads expert segment labels or calls a
planner during evaluation. Eight active joint targets control the seven arm
joints and one active finger; the robot handles the mimic finger.

Useful variants:

```bash
# Engineering smoke only: two qualification scenes and ten training updates.
python examples/data_pipeline/ods_pick_place_showcase.py --smoke \
  --no-viewer --no-video --output /tmp/pick-place-smoke

# Full expert qualification without training.
python examples/data_pipeline/ods_pick_place_showcase.py --qualification-only \
  --no-viewer --no-video --output /tmp/pick-place-expert

# Fixed initial data, same model seed and update budget as the online run.
python examples/data_pipeline/ods_pick_place_showcase.py --mode frozen \
  --output /tmp/pick-place-frozen

# Continue optimization into a new run with freshly generated ODS data.
python examples/data_pipeline/ods_pick_place_showcase.py \
  --resume /tmp/pick-place-online/round_1_step_1000.pt \
  --output /tmp/pick-place-resumed
```

Use a fresh output directory for each invocation. Checkpoints are immutable.
The run saves resolved experiment arguments, the data-source identity and joint
order, normalization and RNG/optimizer state, per-seed evaluation JSON,
TensorBoard events and initial/final comparison videos. The first three fixed
evaluation seeds are recorded, including failures. Choose unused browser ports
with `--viser-port` and `--tensorboard-port` when running multiple experiments.

Training, development and final-test seeds are separate. Twenty fixed
development scenes select a checkpoint; 100 untouched final-test scenes assess
that selected checkpoint and the initial model. A final success rate of 80%
is an experimental target, not a guaranteed property. Inspect `summary.json`
for the measured result and consumed ODS generations. A smoke run is never
reported as meeting the learning target. Online data need not outperform the
frozen control on every run.

This is online expert-data BC. It does not implement DAgger, feedback-directed
collection or exact replay resume. Resume restores the learner and starts a new
ODS run; its new source identity is recorded explicitly.
