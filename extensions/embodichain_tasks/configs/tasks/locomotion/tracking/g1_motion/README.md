# G1 motion tracking

`UnitreeG1MotionRL-v1` tracks G1 reference motion in a DexUni environment using its rigid-body MuJoCo-Warp route. The deployment advances four 5 ms substeps per 20 ms control interval. The supplied PPO configuration runs 32 environments for 20 updates to check task and trainer integration.

## Prepare motion data

Place the LAFAN1 G1 retargeting source under `data/LAFAN1_Retargeting`, with CSV files in `g1/` and the matching URDF and meshes in `robot_description/g1/`. From the repository root, run:

```bash
python examples/learning/motion_tracking/prepare_lafan1.py --source data/LAFAN1_Retargeting
```

The command resamples the 30 Hz CSV files to 50 Hz and writes `LAFAN1G1/motion.npz` and `LAFAN1G1/robot/` under the EmbodiChain data root. Set `EMBODICHAIN_DATA_ROOT` before preparation and training to select another data location. The complete input contains 40 clips and 441,140 resampled frames. The archive records the joint order and uses XYZW quaternions.

## Train and evaluate

```bash
python -m embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/locomotion/tracking/g1_motion/agents/ppo.yaml
```

Training writes a timestamped `outputs/g1_motion_ppo_smoke_*` directory containing configuration snapshots, TensorBoard metrics, a checkpoint and `run-manifest.json`. Evaluate the newest run from the start of each reference clip:

```bash
RUN_DIR=$(ls -dt outputs/g1_motion_ppo_smoke_* | head -n 1)
python examples/learning/motion_tracking/evaluate.py "$RUN_DIR"
```

Evaluation disables observation noise and removes the 500-step training limit. Each clip runs until physical failure or its final reference frame. `evaluations/motion-clips.json` records per-clip returns and failure status. Normalized return divides accumulated reward by the full reference length times the maximum per-step tracking reward, including the unexecuted tail after a failure in the denominator.

## Task interfaces

| Interface | Content |
|---|---|
| Policy input | 930 history features followed by 730 future reference features |
| Critic input | 245 state features, 730 goal features, 33 physical parameters and one remaining-time feature |
| Policy action | 29 values: leg setpoints relative to the resting pose; upper-body residuals relative to the next reference frame |
| Executed target | Resting pose plus scaled setpoint action minus encoder bias |
| Reset | Reference position and velocity; selected rows receive repeated initial observations in their history |
| Failure | Root-height or tilt tracking error above the configured threshold |
| Truncation | Reference clip end or the training step limit; hard failure takes precedence |
| Pre-action snapshot | `capture_transition_start()` returns owned state, frame, physical context and previous policy action |
| Transition snapshot | Owned post-step state, frame, history, goal, policy action, normalized setpoint, target position, continuation probability and `usable` flag |
| Divergent state | Hard termination, finite observation replacement and `usable=False`; excluded from WM supervision |

The base physical deployment uses nominal physical parameters, uniform clip/start selection and observation noise. The online model-based configs additionally enable adaptive time-bin sampling, reset perturbations, friction randomization, encoder bias and torso center-of-mass offsets. Each measured transition carries the physical context used by its environment.

The task reuses `MotionLibrary` for reference data and `TreeKinematics` for differentiable body states. Tensor tracking calculations in `locomotion/tracking/_task.py` provide the observation, action mapping, goals, reward and continuation used by the physical environment and the model environment.

## Fit a world model from physical transitions

Run the collection and supervised fitting example after preparing the same data:

```bash
python examples/learning/motion_tracking/fit_world_model.py \
  --config extensions/embodichain_tasks/configs/tasks/locomotion/tracking/g1_motion/env.newton.yaml \
  --output outputs/g1_world_model_seed0
```

The default run uses 160 physical environments and collects 512 vector steps.
The first 128 environments supply 65,536 training transitions. The last 32 supply
16,384 held-out transitions; neither their states nor their normalizer statistics
enter training. Collection uses reference joint-position commands, with Gaussian
action noise of standard deviation 0.1 in 75% of environments. This supervised
experiment measures dynamics fitting before connecting learned-policy collection.

The model has three 2048-wide SiLU layers and 9,166,918 parameters. It predicts
heading-frame increments from 245 state features, 29 executed setpoints and 33
physical parameters. Training uses shared SL, Adam at 2e-4, a gradient norm limit
of one, 500 updates and complete eight-step windows of batch size 128. Network
arithmetic uses BF16 by default; `--fp32` disables it. FK and state integration
remain FP32. `world_model_weights.yaml` records the frozen G1 recipe's joint error
weights so changes to geometry or weighting-pose selection do not change the loss.

The autoregressive objective sums the eight per-step errors and averages over
windows. Each per-step error is the feature mean of squared state error divided
by measured increment variance, with the configured joint weights. No predicted
state is detached within the window. Only training transitions update moments.

`run.json` records the recipe; `metrics.jsonl` records every optimizer update;
`result.json` compares the final model with its initialization and a state-hold
baseline on fixed held-out windows. `world_model.pt` stores final model,
normalizers and optimizer state. Replay remains in memory during the run.
The command fails if final eight-step normalized error does not improve over
both baselines. A run takes about two minutes after simulator initialization on
a single RTX PRO 5000 in the recorded 500-update experiment.

## Online world-model and APG training

The model-based recipe alternates measured collection, shared supervised updates
and imagined APG Actor-Critic updates. Its functional, four-clip development and
40-clip convergence configurations are `agents/model_based.yaml`,
`agents/model_based_development.yaml` and `agents/model_based_full.yaml`.

See the [online training guide](../../../../../../../examples/learning/motion_tracking/README.md)
for the fixed data source, complete commands, network sizes, gradient boundaries,
checkpoint recovery, output schema and physical convergence criteria.

## State synchronization and complete-clip evaluation

The MuJoCo-Warp configuration uses `update_data_interval: 0` so that native state advances continuously across the four substeps. This avoids converting floating-base velocity between world and body coordinates before each substep. After `write_state()` writes selected robot rows, the task calls the public `Scene.reset(arenas=..., reset_clock=False)` API to publish their new starting state and clear their integration history. This covers episode initialization, automatic resets and explicit state injection. Other arenas and physical parameters remain unchanged; shared simulation time continues, and the next step recaptures the CUDA Graph.

Setting `update_data_interval: 1` retains per-substep synchronization for diagnostic comparisons and does not add a Scene reset after task state writes. Evaluation preserves the checkpoint's original configuration unless an explicit evaluation configuration selects another interval; the checkpoint metadata continues to describe the original training run. Continuous native integration requires DexSim's selected-world DexUni reset and child-solver CUDA Graph state-preservation fixes.

For complete-clip evaluation, `fixed_clip_assignment: true` assigns world `i` to motion clip `i` on every automatic reset. It requires one world per clip and disables random starts and adaptive sampling. The evaluation entry point enables this option and records the assigned and final episode start frames.
