# Native palm-up Wuji20 in-hand rotation

This example trains a fresh Wuji Hand V1 right-hand policy with 20 physical
joints through EmbodiChain `SimulationManager`, Newton/MJWarp and EmbodiChain
APG with a graph-connected terminal value bootstrap. The hand's local palmar
`+X` normal must face world `+Z`; native body poses are checked at construction.
Object weight acts down world Z and the requested rotation is about world Z.

The task uses eight cylinder sizes (0.8–1.5), fixed 50 g mass, authored Wuji PD
gains/torque limits, fixed contact parameters, gravity-compensated hand links,
random cylinder axial phase and observation noise. It starts with verified
opposing-pad contacts. Maintaining the grasp and rotating must be learned;
the initial-state cache does not claim 20-second stationary holding or grasp
acquisition. This is a Wuji adaptation, not unchanged Sharpa domain randomization
or an algorithm comparison. Privileged friction is a unit multiplier, not the
absolute native sliding-friction coefficient (2.0).

Use the native dependencies in the sibling [Sharpa example](../sharpa_inhand/README.md),
a clean `Mingrui-Yu/sharpa_rl_unilab` checkout pinned to
`9816885a3b1fbdb947db71e72c27ed75effcbf9f`, and the Wuji archive's `mjcf/right.xml`
and matching `meshes/right` tree. Keep the archive's license/attribution. A run-owned
learner derivative records each 20-DOF dimension substitution; the upstream
checkout stays read-only and its simulator is never constructed.

```bash
python examples/learning/wuji_inhand/prepare.py \
  --asset /path/to/WujiHand --output outputs/wuji_prepared --count 128 \
  --center-x -.115 --center-height .565
python examples/learning/wuji_inhand/screen.py \
  --assets outputs/wuji_prepared --output outputs/wuji_screen --steps 200
python examples/learning/wuji_inhand/build_initial_cache.py \
  --prepared outputs/wuji_prepared --screen outputs/wuji_screen \
  --output outputs/wuji_initial_assets
```

Preparation uses FK fitting plus a bounded controller preload. Native screening
records measured joint/object poses and exact fingertip-geometry forces. Cache
construction selects at least 250 ms of prior thumb/two-opposing-finger contact,
low measured endpoint velocities and at most 3 mm object penetration. It retains
47 coordinates: measured q20, object position3/quaternionWXYZ4 and holding target20.
The optional screening feedback controller is only a preparation experiment;
no such controller is active in training or policy inference.

```bash
python examples/learning/wuji_inhand/train.py \
  --upstream /path/to/sharpa-reference \
  --reference-config /path/to/reference/training/config.yaml \
  --assets outputs/wuji_initial_assets --output outputs/wuji_train \
  --num-envs 256 --iterations 5000 --seed 4
```

The full profile retains a 2048×3 world model, batch 2048 with ten model updates
per iteration, actor batch 1024, horizon 32, source critic fitting/optimizers,
mixed precision, CPU replay capacity 2,048,000 and activation recomputation.
Each iteration collects five control steps per arena after 131,072 warmup
transitions. `--smoke --num-envs 16 --iterations 4` checks wiring with smaller
models/budgets. Checkpoints pair learner state and CPU replay. Resume checks
physical sources, assets, gains and shared-runtime receipts and retains
transition counters; it reinitializes the simulator rather than claiming an
identical physical trajectory continuation.

Rotation and position rewards require soft pad participation, opposition and
side-height quality, computed through differentiable source-hand FK. The native
success rule checks measured pad forces > .02 N, a thumb/opposing-finger radial
dot < -.15 and pad centers within 80% cylinder half-height. At least two non-thumb
partners must oppose the thumb. Physical episodes terminate after 1 s continuously
invalid grasp or escape from the reset height band. Normal finger exchanges are
permitted. The legacy evaluator's `dropped` field counts either failure reason.

```bash
python examples/learning/wuji_inhand/evaluate.py \
  --checkpoint outputs/wuji_train/native_teacher_05000.pt \
  --upstream /path/to/sharpa-reference --assets outputs/wuji_initial_assets \
  --output outputs/wuji_validation --num-envs 256 --stage validation \
  --seeds 83101 83102 83103 83104 83105 83106
```

Evaluation verifies frozen source/asset/runtime receipts and retains all native
poses, tip positions, forces, rewards and grasp flags. Every size must pass
20-second survival/failure, full-revolution, positive-speed and final-five-second
rotation criteria, plus valid opposing-pad grasp during at least 80% of an
episode. Use development scenes for selection, then validation and an unused
holdout. `--zero-action` is an unqualified control. Qualify a result only with
a passing holdout of at least 1,536 episodes; rotation statistics alone are
insufficient.
