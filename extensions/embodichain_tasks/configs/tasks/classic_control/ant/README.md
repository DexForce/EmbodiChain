# Ant neural dynamics workflow

These files configure `embodichain dynamics`; they are not Gym deployments and
have no runnable environment ID. The simulator host constructs the pinned NeRD
reference asset through EmbodiChain and DexSim Spawn. The generated manifest
provides the actual physics, asset and codec identity; training inherits it.

Run from the repository root (replace `smoke` with `full` for the single-GPU profile):

```bash
embodichain dynamics generate --config extensions/embodichain_tasks/configs/tasks/classic_control/ant/dynamics.generate.smoke.yaml
embodichain dynamics train --config extensions/embodichain_tasks/configs/tasks/classic_control/ant/dynamics.train.smoke.yaml
embodichain dynamics export --checkpoint outputs/dynamics/ant/smoke/train/best.pt --output outputs/dynamics/ant/smoke/export
embodichain dynamics eval --config extensions/embodichain_tasks/configs/tasks/classic_control/ant/dynamics.eval.smoke.yaml
embodichain dynamics eval --config extensions/embodichain_tasks/configs/tasks/classic_control/ant/dynamics.rollout.smoke.yaml
```

Paths inside YAML resolve relative to the YAML file. CLI path overrides resolve
relative to the current directory. Output directories are under repository
`outputs/`, which is ignored by Git. Exports require a new destination directory.
Generation resumes with `--resume`; training resumes with
`--resume outputs/dynamics/ant/smoke/train/last.pt`. Exact resume preserves
budget, seed, device, configuration, immutable data fingerprint, optimizer, RNG
and the committed sampler cursor. Use `--fine-tune path/to/export` for a new run.

The smoke profile uses CPU, 64 short trajectories, a small Transformer and
100 optimizer steps. It tests the complete plumbing and does not qualify model
accuracy. Ant smoke uses effort_scale=0.05 and default initial states; this is deliberately a different distribution from the official-range full profile.

The full profile uses one CUDA device, the official six-layer, twelve-head
Transformer with embedding width 384, history 10, batch 512, and
4,995,000 Adam updates. This matches 999 trained epochs × 5000 updates in the
saved official configuration (epoch zero only evaluates). The learning rate is
held for 5000 updates, with offset one and a 1000-interval denominator, matching
the official epoch timing. Gradient clipping is a norm limit of 1.0; it does not
truncate temporal gradients. Validation uses the first 100 complete/partial
batches; final offline evaluation reads the entire test split. The expected
training budget is substantial and is not run automatically by tests.

Default generation samples the full recorded force range: cart force ±1500
with no pole actuation, or Ant scalar joints ±400. Teachers use MuJoCo-Warp with
10 substeps per 1/60-second macro step. CPU fixtures explicitly request separate
worlds. Source MJCF motor objects and solver XML options are not installed into
the teacher: the host applies actual generalized forces and records its solver
configuration. This profile is a current Newton teacher, not Warp 1.5.1 paper
reproduction.

## Official comparison controls

Default training uses deterministic sampling with replacement and streamed
population moments over overlapping training windows. These choices are explicit;
they do not reproduce the official dataloader's permutation or its running
normalizer initialization automatically.

For a paired official comparison, pass the **same** immutable data, trajectory
split, initialized weights, batch order, normalizer artifact and update budget to
both trainers. `--statistics stats.pt` loads a tensor-only artifact containing
`schema_version: 1`, `inputs: {name: {mean, var}}`, and
`output: {mean, var}`. No validation/test samples may enter these statistics.
`--sample-order order.npy` reads a memory-mapped integer array of shape
`[total_steps, batch_size]`; each row indexes the full-window training dataset.
A `.pt` tensor is also accepted for short fixtures. The array hash and committed
cursor are checked on resume; large `.npy` orders are never loaded in full.

Use `--seed 0` through `--seed 4` with distinct `--output-dir` values for the
five paired training seeds. Export each validation-selected checkpoint separately
and use evaluation's `--model` and `--output` overrides to retain every result.
Full rollout configurations request 2048 trajectories at horizons 1, 10, 100,
1000; repeat with `motion_mode: passive` and with the qualified fixed policy.
A common seed alone does not guarantee identical minibatches across trainers.

No committed configuration is evidence that the full run passed. Required reports
include short numerical/Adam parity, both teachers × both trainers, five seed
pairs, confidence bounds, non-finite/failure rates, fixed-policy return and
software/asset/checkpoint hashes. `qualification.paired_noninferiority` implements
the seed-paired one-sided 95% bootstrap gate; fewer than five seeds is pending.
A 100-step smoke cannot replace the full budget or paper reproduction.

## Registered differentiable task

After exporting a matching model, `env.nerd.yaml` creates the registered
`AntDynamics-v1` task. Its checkpoint path resolves from the
working directory, unlike dynamics workflow file paths (config-relative).

```bash
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/classic_control/ant/env.nerd.yaml --preview
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/classic_control/ant/agents/apg.nerd.yaml
```

Use preview for the Gym scene: this task does not implement an expert trajectory
collector. The APG configuration uses the existing learning-environment factory
and trainer, with graph-connected state/reward hooks. Its policy checkpoint is
separate from the NeRD dynamics-model export.
