# Cartpole neural dynamics workflows

These configuration files run the offline `embodichain dynamics` workflow;
they are not runnable Gym task deployments.

```bash
embodichain dynamics generate --config extensions/embodichain_tasks/configs/tasks/classic_control/cartpole/dynamics.generate.smoke.yaml
embodichain dynamics train --config extensions/embodichain_tasks/configs/tasks/classic_control/cartpole/dynamics.train.smoke.yaml
embodichain dynamics export --checkpoint outputs/dynamics/cartpole/smoke/train/best.pt --output outputs/dynamics/cartpole/smoke/export
embodichain dynamics eval --config extensions/embodichain_tasks/configs/tasks/classic_control/cartpole/dynamics.eval.smoke.yaml
```

Generation calls Newton's built-in analytical solver directly. It does not
initialize DexSim or require its custom solver/Scene extension. The `full` and
`warp-reference` presets retain the recorded model/data scales and parameter
profiles; a profile does not promise identical dynamics across Newton versions.

The pretrained evaluation preset reads the published data/model directories
shown in Features → Neural Dynamics Models. Downloaded datasets retain their
original teacher provenance. New direct-Newton datasets use a distinct physical
signature; replacing provenance fields does not align different teachers.

See [Neural Dynamics Models](../../../../../../../docs/source/features/neural_dynamics.md)
for downloads, resume, fine-tuning and portable exports. Live neural simulation
and differentiable task integration are tracked separately in PR #822.
