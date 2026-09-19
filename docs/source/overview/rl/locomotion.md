# Locomotion RL tasks

Locomotion environments use the same task registration, configuration loader, action manager and PPO entry point as the other official tasks.

| Task configuration directory | Environment ID | Action / actor / critic dimensions |
|---|---|---|
| `locomotion/velocity/g1_flat` | `UnitreeG1FlatRL-v1` | 29 / 98 / 113 |
| `locomotion/velocity/h1_2_flat` | `UnitreeH1_2FlatRL-v1` | 27 / 92 / 107 |
| `locomotion/velocity/go1_flat` | `UnitreeGo1FlatRL-v1` | 12 / 48 / 72 |
| `locomotion/velocity/go2_flat` | `UnitreeGo2FlatRL-v1` | 12 / 47 / 74 |
| `locomotion/velocity/anymal_c_flat` | `ANYmalCFlatRL-v1` | 12 / 48 / 48 |
| `locomotion/velocity/microduck_flat` | `MicroDuckFlatRL-v1` | 14 / 53 / 68 |
| `classic_control/humanoid` | `HumanoidRun-v1` | 17 / 63 / 63 |

Paths in this table are relative to `embodichain_tasks/configs/tasks/`.
Each directory contains `env.yaml` and `agents/ppo.yaml` for the default backend, plus `env.newton.yaml` and `agents/ppo.newton.yaml` for Newton. Capability registration is `RL`.

## Assets and cache

The download classes resolve seven published ZIP archives in the `robot_assets/` directory of [DexForceAI/embodichain_data](https://huggingface.co/datasets/DexForceAI/embodichain_data). Models are downloaded on first use and cached outside the Python wheel.

```bash
embodichain data list --category robot
embodichain data download --name UnitreeGo2Locomotion
```

```python
from embodichain.data import get_data_path

robot_path = get_data_path("UnitreeGo2Locomotion/go2.usda")
```

`EMBODICHAIN_DATA_ROOT` selects the cache. A matching local file under this root takes precedence; otherwise the registered download class uses the standard download/extract cache. H1_2 requires its own asset archive and must not be substituted with the older H1 asset.

Preserve every file in a robot archive, including USD payload layers, meshes, textures and license notices. MicroDuck 3D models have the upstream Creative Commons BY-SA-NC designation, distinct from the code license. Humanoid has authored tiny positive inertias on auxiliary links; compatible runtimes must preserve them as nonzero.

## Controls and observations

Velocity tasks convert normalized joint actions to offsets around the task's default pose. Task definitions retain joint order, gains, command ranges, reward weights and episode limits. Contacts are collected after each physics substep, including contacts that do not persist until the end of a control interval. Actor and privileged critic observations remain separate.

Humanoid uses clipped effort actions (`[-0.4, 0.4]`) multiplied by per-joint motor gears and limited by joint effort bounds. The benchmark's progress potential divides distance by the physics timestep; it deliberately retains that original reward scaling. The maximum episode length is 900 control steps at 60 Hz.

## PPO configuration

For example, the prepared Go2 training entry is:

```bash
embodichain train-rl --config embodichain_tasks/configs/tasks/locomotion/velocity/go2_flat/agents/ppo.yaml
```

Local smoke checks cover environment construction, stepping, finite tensors, selective reset and policy forward execution, with zero training updates.

See the [task API reference](../../api_reference/locomotion_tasks.rst) for configuration and MDP types.
