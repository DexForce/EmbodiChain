# Expansion and generalization examples

This directory groups examples for data-generation expansion and asset
generalization. Algorithms and lifecycle ownership remain in the SDK; examples
provide runnable entry points, local configurations and inspectable outputs.

| Directory | Capability | Entry points |
| --- | --- | --- |
| [asset_catalog](asset_catalog/README.md) | Asset-package annotations, explicit/directory Providers and verified snapshots | Provider usage and catalog format |
| [asset_generalization](asset_generalization/README.md) | Semantic asset matching, deployment materialization, batch collection and measured acceptance | `run.py` for Cube primitives; `pour_water.py` for bottle/cup replacement |
| [trajectory](trajectory/README.md) | Configured combined schedules and physical trajectory variants | `repeated_pick_place_expansion_combined.py`, `repeated_pick_place_expansion_showcase.py`, `task_environment_augmentation_showcase.py` |

Asset-generalization Task Programs and physical environments live in
`asset_generalization/configs/`. These are example-local configurations, not
additional tasks installed into `embodichain_tasks`. They reuse the SDK's robot
embodiments, policies and asset registry.

Run from the repository root with compatible SDK dependencies:

```bash
# Prepare primitive configurations; add --execute for measured collection.
PYTHONPATH="$PWD" python examples/expansion/asset_generalization/run.py \
  --output /tmp/expansion-primitive --batches 2

# Prepare semantically matched bottle/cup variants; add --execute to collect.
PYTHONPATH="$PWD" python examples/expansion/asset_generalization/pour_water.py \
  --output /tmp/expansion-pour --batches 2

# Preflight a combined expansion schedule without constructing a simulator.
python examples/expansion/trajectory/repeated_pick_place_expansion_combined.py \
  --output-dir /tmp/expansion-schedule
```

Each subdirectory documents whether an entry point only prepares configurations,
runs physical execution, measures acceptance, or confirms recorder submission.
The PourWater example measures rigid pouring motion, not liquid transfer.

## Asset preparation and functor lifecycle

The current asset-generalization API operates before environment construction:
Provider snapshots → semantic matching → verified deployment → fresh environment
per asset batch → acceptance → recorder finalization. Selecting a different mesh
also changes collision geometry, bindings and geometry-dependent calibration.

Existing manager functors receive an already constructed environment. Even
`startup` events run after `BaseEnv` prepares the native scene. Reset/interval
events are appropriate for supported pose, material and physical-property
randomization on existing entities. The current asset pipeline rebuilds the
environment between asset batches rather than replacing meshes inside reset.

A configuration-driven pre-build or batch-stage functor adapter is not currently
implemented. Such an adapter would wrap the existing Provider/matcher/materializer
APIs at their construction boundary; it would not belong in an ordinary reset
event or run Provider I/O during Task Program compilation.
