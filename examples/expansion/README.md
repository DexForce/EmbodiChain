# Expansion and generalization examples

This directory groups examples for data-generation expansion and asset
generalization. Algorithms and lifecycle ownership remain in the SDK; examples
provide runnable entry points, local configurations and inspectable outputs.

| Directory | Capability | Entry points |
| --- | --- | --- |
| [asset_catalog](asset_catalog/README.md) | Asset-package annotations, explicit/directory Providers and verified snapshots | Provider usage and catalog format |
| [asset_generalization](asset_generalization/README.md) | Semantic asset matching, deployment materialization, batch collection and measured acceptance | `run.py` for Cube primitives; `pour_water.py` for bottle/cup replacement |
| [trajectory](trajectory/README.md) | Pure-trajectory deformation/stitching, configured schedules and physical variants | `pure_trajectory_augmentation_showcase.py`, `repeated_pick_place_expansion_combined.py`, `repeated_pick_place_expansion_showcase.py`, `task_environment_augmentation_showcase.py` |

Asset-generalization Task Programs and physical environments live in
`asset_generalization/configs/`. These are example-local configurations, not
additional tasks installed into `embodichain_tasks`. They reuse the SDK's robot
embodiments, policies and asset registry.

Run from the repository root with compatible SDK dependencies:

```bash
# Render pure-trajectory deformation and multi-demo stitching on CPU.
python examples/expansion/trajectory/pure_trajectory_augmentation_showcase.py \
  --output-dir /tmp/embodichain-pure-trajectory

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

Asset preparation runs before collection: Provider snapshots → semantic matching
→ verified deployments. Collection constructs one environment, then applies each
prepared combination through the existing `replace_assets_from_group` event.
The event calls `SimulationManager.replace_rigid_objects()` once for the selected
bodies and installs the deployment's adapted Task Program binding. Robot,
unselected bodies, sensors and recorder instances remain in the environment.
Each batch creates a fresh scene-acceptance snapshot and each Task Program bridge
builds fresh live providers and a planner collision world.

The existing `entity_cfg` / `folder_path` mode continues to sample directory
meshes. The new `deployment` mode consumes the artifacts returned by
`prepare_asset_batches`; matching and Provider I/O are not reset-time work.
`collect_asset_batches` dispatches the event at a cleared B=1 batch boundary,
after old recording buffers have been consumed and before new recording state
is seeded. Unsupported changes to the robot, backend, control layout or
unselected objects are rejected before native mutation. Native preparation
failures stop collection because the backend does not provide rollback.

The primitive Cube baseline remains a separate, direct-configuration example.
Its primitive shapes do not require asset catalogs.
