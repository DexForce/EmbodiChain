# Ant neural dynamics

These files configure the `embodichain dynamics` workflow. They are separate
from runnable Gym deployments. See the
[Neural Dynamics Models feature](../../../../../../docs/source/features/neural_dynamics.md)
for the architecture, published datasets and models, comparison videos and
supported deployment boundaries.

## Configuration profiles

| Profile | Purpose |
| --- | --- |
| `dynamics.*.smoke.yaml` | Small CPU collection/training/evaluation workflow |
| `dynamics.*.full.yaml` | Reference network scale and long training budget for a newly generated native dataset |
| `dynamics.generate.warp-reference.yaml` | Current-Newton reference-asset collection with Featherstone |
| `dynamics.eval.pretrained.yaml` | Offline test evaluation of the published Hugging Face dataset/model pair |
| `dynamics.rollout.pretrained.yaml` | Released NeRD model versus its matching Newton reference teacher |
| `env.nerd.yaml` | Registered Gym scene using an export from the matching native workflow |
| `agents/apg.nerd.yaml` | Policy learning through the dynamics environment |

The published reference models use `warp_reference` physics. They do not match
the default native/MuJoCo-generated dataset or the native Gym preset merely
because the robot has the same name. The pretrained rollout preset reconstructs
the asset, materials and solver configuration expected by the released model.

## Use published data and models

Download `neural_dynamics/nerd/ant/v1` from the dataset repository
[DexForceAI/embodichain_datasets](https://huggingface.co/datasets/DexForceAI/embodichain_datasets)
into `outputs/datasets`, and from the model repository
[DexForceAI/embodichain_model](https://huggingface.co/DexForceAI/embodichain_model)
into `outputs/models`. Keep the path prefix when downloading; the feature page
provides a `snapshot_download` example.

Run from the repository root:

```bash
embodichain dynamics eval --config extensions/embodichain_tasks/configs/tasks/classic_control/ant/dynamics.eval.pretrained.yaml
embodichain dynamics eval --config extensions/embodichain_tasks/configs/tasks/classic_control/ant/dynamics.rollout.pretrained.yaml
```

Offline evaluation needs no simulation. Live rollouts require an installed
DexSim build with named solver extensions and Scene functional/snapshot APIs.
Use `--device cpu` or `--device cuda:0` to select the device, and `--output` with a
new path to retain repeated rollout results. The reference models are research
artifacts; consult their model cards for measured error and supported versions.

## Train and export a new model

```bash
embodichain dynamics generate --config extensions/embodichain_tasks/configs/tasks/classic_control/ant/dynamics.generate.smoke.yaml
embodichain dynamics train --config extensions/embodichain_tasks/configs/tasks/classic_control/ant/dynamics.train.smoke.yaml
embodichain dynamics export --checkpoint outputs/dynamics/ant/smoke/train/best.pt --output outputs/dynamics/ant/smoke/export
embodichain dynamics eval --config extensions/embodichain_tasks/configs/tasks/classic_control/ant/dynamics.eval.smoke.yaml
```

Smoke profiles verify executable flow, not accuracy. For a different dataset,
copy the appropriate training config and set `dataset` and `output_dir` to its
location and a new training directory. For a different physical system, collect
new teacher data and train a matching model.

Configuration paths resolve relative to the YAML file; CLI path overrides
resolve relative to the working directory. Exports require a new destination.
Resume an identical training run with `train --resume PATH/last.pt`, or initialize
a new run from a portable export with `train --fine-tune PATH`. Validation data
selects the checkpoint; the test split remains held out.
