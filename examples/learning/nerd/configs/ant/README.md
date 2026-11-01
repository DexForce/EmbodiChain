# Ant NeRD configurations

These files belong to the [NeRD showcase](../../README.md), not runnable Gym
task deployments. The default `train` command uses `sl.train.yaml` and the
common supervised trainer:

```bash
embodichain run-example learning/nerd prepare --robot ant --resource-root ./outputs/nerd-resources
embodichain run-example learning/nerd train --robot ant --resource-root ./outputs/nerd-resources --device cuda:0 --output-dir ./outputs/nerd/ant/train
embodichain run-example learning/nerd export --checkpoint ./outputs/nerd/ant/train/last.pt --output ./outputs/nerd/ant/export
embodichain run-example learning/nerd eval --robot ant --resource-root ./outputs/nerd-resources --device cpu --model ./outputs/nerd/ant/export
```

The SL recipe is bounded to 1,000 updates, with an explicit resident-cache
budget. Use `--device cpu` for CPU execution. To use `embodichain train-sl`, copy
`sl.train.yaml`, set `data.path` and `trainer.output_dir`, and resolve relative
paths from the working directory. The factory is installed under
`embodichain_models.dynamics.nerd.sl_recipe`, so no example imports are needed.
The final SL checkpoint does not imply validation-selected accuracy.

Teacher collection defaults to `dynamics.generate.smoke.yaml`. Prepare assets
with `prepare --assets-only`, then run `generate --robot ant` with the same
`--resource-root` and an explicit `--output-dir`. Newton's analytical solver
owns integration; no DexSim custom solver or Scene extension is required.
The `full` and `warp-reference` presets retain their recorded scales and
physical profiles, which do not promise identical dynamics across versions.

An explicitly supplied `dynamics.train.*.yaml` retains the historical trainer
for comparison. Those configuration paths resolve relative to the file;
CLI path overrides resolve from the working directory. Historical `best.pt` is
export-only; `last.pt` restores optimizer/RNG/sampling and the selected best
weights. The `full` historical preset disables resident caching rather than
assuming the entire dataset fits a fixed device budget.

The pretrained evaluation default uses the pinned published dataset/model from
`prepare`. Downloaded data retains its Scene teacher provenance; new direct
Newton data has a distinct physical signature. See the
[NeRD guide](../../../../../docs/source/features/neural_dynamics/nerd.md) and
[research acceptance](../../research/acceptance.md) for format and evidence
boundaries. Preset names do not establish accuracy qualification.
