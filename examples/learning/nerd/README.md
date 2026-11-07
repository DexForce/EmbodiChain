# NeRD showcase

This example prepares pinned resources, fits causal robot dynamics through the
common SL trainer, exports portable models and evaluates held-out transitions.
Cartpole and Ant are the supported reference systems. Teacher collection and
historical trainer comparisons remain private support workflows.

Reusable networks, physical codecs, committed trajectory readers and the NeRD
SL recipe live in `embodichain_models.dynamics.nerd`, bundled in the main wheel.
Core `embodichain.learning.sl` owns the optimizer and training lifecycle; shared
MLPs remain in `embodichain.learning.models`. The runtime model package does not
import this example or simulation.

## Default workflow

The launcher works from a source checkout or installed wheel and keeps the
working directory. Use a caller-owned resource directory:

```bash
embodichain run-example learning/nerd prepare --robot cartpole --resource-root ./outputs/nerd-resources
embodichain run-example learning/nerd eval --robot cartpole --resource-root ./outputs/nerd-resources --device cpu
embodichain run-example learning/nerd train --robot cartpole --resource-root ./outputs/nerd-resources --device cuda:0 --output-dir ./outputs/nerd/cartpole/train
embodichain run-example learning/nerd export --checkpoint ./outputs/nerd/cartpole/train/last.pt --output ./outputs/nerd/cartpole/export
embodichain run-example learning/nerd eval --robot cartpole --resource-root ./outputs/nerd-resources --device cpu --model ./outputs/nerd/cartpole/export --output ./outputs/nerd/cartpole/trained-eval.json
```

Use `--robot ant` for Ant or `--device cpu` for CPU training. Commands expose
`--help`. `train` defaults to the robot's `sl.train.yaml`, with 1,000 updates;
this bounded recipe is not a convergence qualification. Exported `last.pt`
weights are the final committed weights, not an implicit validation selection.
Resume with the same recipe using `train --resume PATH/last.pt`; each new run
and portable export requires a fresh destination.

`prepare` pins dataset revision `b9ec58f36da99cc99a93f97adfe74ebc9fd66586`,
model revision `78e581526570069410bf269b38c0c52c800b24a7` and reference-asset
revision `97e202e41fe7fb504656a6b163291305e400780d`. `_resources.py` owns their
repository IDs, paths and manifest/member checksums. Raw reference files and
license/attribution are downloaded from
`DexForceAI/embodichain_data/demo/nerd/reference_assets/v1`; they are not in Git
or the wheel. Use `prepare --assets-only` for collection-only setup and
`--local-files-only` for offline cache checks. Without `--resource-root`, the
example uses `EMBODICHAIN_NERD_RESOURCE_ROOT` or the showcase cache under
`XDG_CACHE_HOME`/`~/.cache`.

## Custom configurations

The [Cartpole](configs/cartpole/README.md) and [Ant](configs/ant/README.md)
configuration directories contain common SL recipes and historical presets.
For the generic entry point, copy `sl.train.yaml`, set `data.path` to the
prepared dataset and `trainer.output_dir` to a fresh output, then run:

```bash
embodichain train-sl --config /path/to/recipe.yaml
```

The factory is the installable
`embodichain_models.dynamics.nerd.sl_recipe:build_supervised_trainer`; it does
not require example bootstrap. Paths in a common SL recipe resolve from the
working directory. `--resource-root` replaces paths only for the showcase's
implicit default presets, not an explicitly supplied configuration.

Explicit `dynamics.train.*.yaml` files select the historical trainer, retaining
its checkpoint format, sample ordering and numerical schedule for comparisons.
Only that configuration accepts historical `--fine-tune` and `--sample-order`.
Its model-only `best.pt` supports export; optimizer recovery uses `last.pt`.
Historical configuration paths resolve relative to the file, and explicit CLI
path overrides resolve from the working directory.

## Teacher and research workflows

```bash
embodichain run-example learning/nerd prepare --robot cartpole --resource-root ./outputs/nerd-resources --assets-only
embodichain run-example learning/nerd generate --robot cartpole --resource-root ./outputs/nerd-resources --device cpu --output-dir ./outputs/nerd/cartpole/generated
```

Collection uses Newton's built-in analytical solver without a custom DexSim
solver or Scene initialization. Its direct-builder physical identity differs
from the published Scene-generated dataset. Preserve recorded asset, solver,
version and normalizer identities when comparing models. Continuing collection
requires a matching collector source; readable legacy manifests do not imply
cross-version collection resume.

[Reference experiments](research/README.md) and the
[frozen acceptance protocol](research/acceptance.md) own research restrictions,
source revisions and quality gates. Relocating an implementation or passing a
new smoke does not rerun earlier frozen evidence. Offline prediction, live
rollout and paper-scale accuracy are separate claims.

See the [NeRD guide](../../../docs/source/features/neural_dynamics/nerd.md) and
[model API](../../../docs/source/api_reference/model_catalogs.rst) for the
resource format and runtime interfaces.
