# Learn grasp orientation through differentiable IK

Train a small network to choose a cylinder grasp's yaw from **current joints and
grasp position**. The position and downward approach remain fixed. During training,
`DifferentiableIKLayer` maps the predicted pose to joints, and joint displacement
plus pose residuals backpropagate into the predictor. After training, frozen
weights handle new tasks with one network evaluation and one fixed-step IK solve.

```text
(q_current, grasp_position) → MLP → yaw → TCP target → differentiable IK
                                  ↑                     ↓
                                  └── task loss ← joint configuration
```

The predictor lives only in this example and uses `learning.models.MLP`. It does
not need an RL policy interface, simulator, learned dynamics, or expert joint
labels. The teacher-supervised comparison uses numerical yaw labels explicitly.
Numerical optimization and grid search are integrated into the same training and
evaluation entry point as reference baselines.

## Run and preview

From the repository checkout in an installed EmbodiChain environment:

```bash
conda activate py311
pip install -r examples/learning/differentiable_ik/requirements.txt

# Train three paired seeds, evaluate held-outs, and benchmark complete inference.
python -m examples.learning.differentiable_ik.learn_grasp_yaw
python -m examples.learning.differentiable_ik.view_learned_yaw \
  --run outputs/learn_grasp_yaw --port 8101

# Small CPU run for execution validation; this is not the reference experiment.
python -m examples.learning.differentiable_ik.learn_grasp_yaw \
  --device cpu --seeds 17 --updates 50 --train-count 128 --test-count 16 \
  --output outputs/grasp_learning_smoke
python -m examples.learning.differentiable_ik.view_learned_yaw \
  --run outputs/grasp_learning_smoke --port 8102 --smoke
```

Open `http://127.0.0.1:8101` (forward the port for a remote host). The three Franka
meshes show current-heading, selected learned predictor, and an online numerical
reference. Choose **Held-out split**, **Case**, or **Model**. **Move cylinder**
automatically runs frozen prediction; **Compute online reference** explicitly runs
the more expensive numerical optimizer and grid. Reference poses are hidden until
computed for the edited target. Changing inputs discards stale work. The viewer
uses one CPU worker and never trains or changes the loaded parameters.

The validation curve shows training progress for the fixed first seed. Saved
cases are the first eight examples from each test split, without selecting by
improvement. All models use validation-selected checkpoints. Edited targets are
interactive demonstrations and are not added to the reported evaluation sets.

## Learning and comparison protocol

The network is a `10 → 64 → 64 → 1` SiLU MLP (4,929 parameters). Its output is a
bounded yaw correction, within ±90° of current TCP heading. The zero-initialized
output starts exactly at the heading baseline. Input scaling uses declared task
scales, not test-set statistics. The two network methods share initialization,
mini-batches, Adam learning rate 0.002, and 200 updates of batch 64. The one-scalar
constant baseline receives the same updates with learning rate 0.04.

- **heading**: retain current TCP X heading projected into root XY.
- **constant**: one absolute yaw fitted across training tasks through IK.
- **learned**: conditional predictor trained directly on downstream IK task loss.
- **supervised**: the same predictor trained with periodic angle loss against
  labels from the numerical optimizer on training tasks only.
- **online**: 50 Adam updates per query, starting at heading and retaining fixed
  yaw as another feasible incumbent. Returns the best feasible inner-budget yaw;
  if none exists, heading fallback is explicitly flagged.
- **grid**: best feasible solution among 129 angles spanning [-π, π].

Training and online angle search use 24 IK steps. Loss is squared joint endpoint
displacement plus `1e4 * position_error² + 100 * rotation_error²`, summed over each
residual vector. Every final method is independently evaluated with 64 IK steps,
the same current joints as seed, hard limits, and the layer's strict success
thresholds (0.5 mm / 0.0005 rad). Higher-budget validation can differ from the
training objective, so final success and residuals are always reported.

## Held-outs and reproducibility

The fixed experiment has 1,024 training tasks, 128 independent validation tasks,
and 128 tasks in each test split. Initialization seeds are 17, 29, and 43.

Current joints vary by ±0.12 rad around the official ready posture, with another
±0.7 rad variation in the last joint, intersected with physical limits. Target X/Z
vary by ±5 cm around ready TCP; training/validation Y varies from -5 to +1 cm.

- **iid**: fresh tasks from that admitted domain.
- **spatial**: target Y is +2.5 to +5.5 cm, disjoint from training Y.
- **state_shift**: shift current joints 1 and 3 by +0.20/+0.15 rad; use training
  target-position ranges. No shifted states are used to choose checkpoints.

Each set uses an independent CPU RNG stream. Identical scalar seeds do not imply
bitwise identical FK across devices. Full runs save the exact task arrays. Only
validation task loss selects model checkpoints; test targets do not generate
training labels. The three optimization seeds share test sets, so their variation
is not a population confidence interval.

All failures remain in success rates and all-target residual/loss summaries.
Relative costs use paired-valid masks with counts recorded in `metrics.json`.
The experiment covers local variations; it does not demonstrate global workspace
coverage or a globally optimal grasp.

## Artifacts and correctness

Full runs save `datasets.npz`, `teacher_labels.npz`, per-seed model state dicts,
`paired_results.npz`, `metrics.json`, `report.md` (three tables), `learning.png`,
`validation.json`, and `replay.npz` under `outputs/learn_grasp_yaw/` by default.
Generated results, checkpoints and preview images are local artifacts and are
not committed. Train once before starting the viewer with the same output path.
`torch.load(..., weights_only=True)` loads the model; its robot fingerprint must
match the configured Franka geometry.

Executable checks cover float64 finite-difference gradients through the complete
24-step policy/IK loss for current state, target position, and a network parameter;
all network parameter gradients are finite and nonzero on the check problem.
Saved predictions are checked after CPU reload, frozen input gradients remain
available, FK independently verifies success/task invariants, inputs stay intact,
and unreachable targets retain failures and reference fallback flags. Existing
production tests own the general compute/layer gradient contracts.

The viewer renders numerical grasp candidates. Joint endpoint displacement is
not execution time, energy, or a motion trajectory; collisions, contacts and
physical dynamics are not checked.

The design borrows the use of robot computations inside learning objectives from
[cuRobo's neural-network example](https://github.com/NVlabs/curobo/blob/v0.7.8/examples/torch_layers_example.py).
That reference trains through FK/collision operations; this example trains task
pose selection through our unrolled IK layer.
