# Learn reaching through differentiable FK

Train a small Franka joint predictor using **TCP pose labels**. A fixed
`DifferentiableFKLayer` maps its predicted joints to TCP transforms; position and
rotation errors backpropagate through FK into the network. The learned model is a
local approximate inverse mapping. Robot geometry stays fixed.

```text
current joints + target TCP → MLP → predicted joints → FK → pose loss
                              ↑                             │
                              └──────── backward ───────────┘
```

The example uses `learning.models.MLP` and the common `TensorBatchSource`,
`SupervisedLearner` and `SupervisedTrainer`. It needs no simulator, IK solver,
physics backend or target-joint labels. Models and task losses remain example-local.

## Train and view

From a checkout with EmbodiChain installed:

```bash
pip install -r examples/learning/differentiable_fk/requirements.txt

embodichain run-example learning/differentiable_fk/learn_pose_reaching \
  --device cpu --output outputs/learn_fk_reaching
embodichain run-example learning/differentiable_fk/view_pose_reaching \
  --run outputs/learn_fk_reaching --port 8103
```

Open `http://127.0.0.1:8103`; forward the port when running on a remote host.
The blue Franka shows the zero-initialized network's output, equal to the current
posture. The green Franka shows one frozen prediction from the validation-selected
model. Each robot has its own copy of the same root-frame target. Large axes mark
the goal, small axes mark the achieved TCP, and yellow lines show position error.
The sidebar reports position/rotation errors, independent validation curves, and
before/after metrics over **all** held-out test samples.

Choose a test case or edit its XYZ and roll/pitch/yaw offsets. Reset restores the
saved target; changing the case also clears offsets. Rotation edits compose
`Rz(yaw) @ Ry(pitch) @ Rx(roll)` in the chain root frame. Edited targets are labeled
separately from the saved evaluation, may be unreachable, and may exceed the
learned local domain. Their actual errors remain visible. The viewer uses one CPU
worker, coalesces requests and rejects stale results; it never updates weights.

`python -m examples.learning.differentiable_fk.learn_pose_reaching` and
`python -m examples.learning.differentiable_fk.view_pose_reaching` are equivalent
checkout entry points. Training also accepts `--device cuda`, `--seed` and
`--updates`; the viewer accepts `--host`, `--port`, and `--run`.

## Learning contract

- The robot chain is `base → fr3_hand_tcp` from the official Franka URDF.
  Positions use metres; joints and rotation residuals use radians.
- Sample current joints in the intersection of the official ready posture
  ±0.10 rad and physical limits with a 0.15 rad margin. Generate target joints by
  adding independent ±0.10 rad perturbations, apply FK, and discard those target
  joints. Training and saved datasets contain only current joints and TCP poses.
- Input features have 13 dimensions: seven current-joint offsets from ready,
  three current-to-target position errors and three root-frame rotation errors.
  Their respective scales are 0.10 rad, 0.05 m and 0.20 rad.
- The MLP is `13 → 64 → 64 → 7` with SiLU. Its final layer starts at zero.
  Predictions are `q_current + 0.15 * tanh(raw)`. The admitted current-state
  margin keeps every prediction within physical limits.
- Loss is the mean squared norm of position/0.05 and rotation/0.20 residuals,
  plus `1e-3 * mean(||q_pred - q_current||²)`. There is no unique ground-truth
  joint solution for a redundant arm; the posture term favors nearby solutions.

Defaults are float32, CPU, 1,024 training examples, 128 validation examples and
128 independent test examples. Model seed is 17; corpus seeds are 23/24/25 and
the SL sampling seed is model seed + 12. Adam uses learning rate 0.001, batch 128
and 1,000 updates. Validation every 50 updates selects the checkpoint. Test data
never selects weights, feature scales or hyperparameters. These held-outs measure
local interpolation, not global workspace generalization or multiple-seed robustness.

## Artifacts and checks

The output contains common-SL `run.json`, `metrics.jsonl` and `last.pt`, plus:

- `predictor.pt`: exported validation-selected weights, FK identity and config;
- `datasets.npz`: exact train/validation/test inputs and TCP labels;
- `evaluation.npz`: every test prediction, position/rotation error and success flag;
- `metrics.json`, `report.md`, `learning.png`: full-test summaries and validation curves.

The default complete run requires held-out position and rotation mean errors to
each improve by at least 50%. Reports include means, P95 and the fraction meeting
**both 2 mm and 2°**, retaining every failed sample. Execution checks verify
pose-only parameter gradients, finite bounded outputs, unchanged FK state and
CPU-reloaded prediction agreement. Use a fresh output directory for a new run;
the common trainer protects existing run history.

For a bounded execution check:

```bash
embodichain run-example learning/differentiable_fk/learn_pose_reaching \
  --smoke --output outputs/learn_fk_reaching_smoke
embodichain run-example learning/differentiable_fk/view_pose_reaching \
  --run outputs/learn_fk_reaching_smoke --port 8104 --smoke
```

Training smoke uses 20 updates and 64/16/16 samples. It checks the pipeline without
requiring the learning-improvement gate. Viewer smoke constructs both robot
scenes, edits one target, checks bounded frozen predictions and HTTP, then exits.
Production FK tests own the numerical and general gradient contracts.

The preview shows kinematic configurations. It does not execute a trajectory or
validate collisions, contacts, dynamics or hardware control.
