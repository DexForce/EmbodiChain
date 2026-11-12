# Sharpa in-hand model-based APG

This implementation is an approximate Newton/URDF experiment. Equivalent-task
algorithm comparisons require the original MJCF task, complete HORA observation
contract, domain randomization and fixed evaluation protocol first. The
[reference reproduction workflow](../../../../../../../examples/learning/sharpa_inhand/README.md#upstream-baseline-reproduction)
keeps that upstream baseline separate from the existing model-only APG config.

This task rotates a 50 g, 40 mm diameter, 32 mm high cylinder about world Z
using the right Sharpa Wave 01 hand. It implements an initial privileged-state
model-based APG baseline, with 22 incremental joint commands, 240 Hz Newton
MuJoCo-Warp physics and 20 Hz controls. The workflow uses EmbodiChain's
SimulationManager, device-bounded TransitionReplay, task-weighted RWM fitting and
SupervisedLearner, and existing complete-rollout APG trainer.

The online workflow is in
[examples/learning/sharpa_inhand](../../../../../../../examples/learning/sharpa_inhand/README.md).
It alternates physical collection, multi-step world-model fitting and APG
updates, then evaluates the actor in an independent real-physics environment.
Model-only reward is not physical task success.

## Task contract

- Actor inputs: 109 privileged simulator-state features, including cylinder pose
  in fixed-palm axes, velocity, hand joint state, current targets and original
  reset object/hand references. Both physical pose and reward references remain
  observable when grasp-cache rows change.
- Actions: 22 bounded increments, at most 1/24 rad per control step. Replay and
  RWM record the executed absolute joint targets after clamping.
- Canonical RWM state: 62 values, with the cylinder as root and the 22 hand joints.
- Replay context: 47 values containing current targets and original reset
  object/hand anchors. It is used for task reset/reward, not as future model input.
- Fixed-palm model context: current predicted object XYZ and full rotation;
  recomputed during autoregression because the hand stays fixed as the object
  turns. Canonical RWM yaw invariance is not valid for this fixed-palm system.
  Native-cylinder world-frame models may instead retain XYZ and world axis,
  excluding the physically irrelevant own-axis phase.
- Physical episode: height excursion over 20 mm terminates; 400 controls truncate.
- Physical reward: reference rotation, translation, pose, estimated PD torque,
  estimated work and position terms, scaled by the 0.05 s control period.
  Object/hand reward references are fixed source defaults; reward rates and
  rotation metrics use completed control-period pose differences.
- Model rollouts freeze world-model weights and statistics. Autoreset preserves
  terminal reward/metrics; detach does not reset values or consume RNG.
- Real transitions are copied before reset. Collection boundaries explicitly end
  replay episodes so windows cannot cross resets or evaluation phases.

The first version fixes object scale/mass/friction and randomizes only the grasp
cache row. It does not implement the reference's tactile history, HORA student,
domain-randomized task, critic bootstrap or SuperMimic optimizer. Its scores
must not be presented as an exact reproduction of that training setup.

## Train an exported model with the common RL CLI

The online workflow produces `world_model.pt`, containing fitted weights,
measured start states, reset contexts, ordered limits and physical provenance.
Point `trainer.learning_env.cfg.model_checkpoint` in `agents/apg.yaml` to it:

```bash
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/sharpa_inhand/agents/apg.yaml
```

This command trains inside the frozen learned model. Use the online workflow's
physical evaluation for outcome claims; model-only evaluation is disabled.
