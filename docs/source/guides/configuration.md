# Configuration Guide

EmbodiChain uses a declarative configuration system built on Python dataclasses. This guide explains the key patterns: `@configclass`, `FunctorCfg`, and JSON/YAML configuration files.

---

## The `@configclass` Decorator

All configuration objects use the `@configclass` decorator, which is similar to Python's `@dataclass` with additional validation and serialization support.

```python
from embodichain.utils import configclass
from dataclasses import MISSING


@configclass
class MyManagerCfg:
    param_a: float = 1.0
    param_b: str = MISSING  # Required — must be set by caller
    param_c: int = 10
```

- **Optional parameters** have default values.
- **Required parameters** use `MISSING` as the default — callers must provide them.
- All parameters are typed for IDE auto-completion and static analysis.

---

## Configuration Hierarchy

EmbodiChain configs form a nested hierarchy:

```
EmbodiedEnvCfg
├── sim_cfg: SimulationManagerCfg
│   ├── render_cfg: RenderCfg
│   ├── physics_config: PhysicsCfg
│   ├── gpu_memory_config: GPUMemoryCfg
│   └── visualization: VisualizationCfg
├── robot: RobotCfg
│   ├── urdf_cfg: URDFCfg
│   ├── drive_pros: JointDrivePropertiesCfg
│   └── solver_cfg: Dict[str, SolverCfg]
├── sensor: List[SensorCfg]
├── events: EventCfg
├── observations: ObservationCfg
├── rewards: RewardCfg
├── actions: ActionTermCfg
├── dataset: DatasetFunctorCfg
└── extensions: Dict[str, Any]
```

Each sub-config can be set independently, allowing fine-grained control over the environment.

---

## Functor Configuration

Functors are configured through specialized config classes that inherit from `FunctorCfg`. The base class has three fields:

```python
@configclass
class FunctorCfg:
    func: Callable | Functor = MISSING   # The function or class to call
    params: dict[str, Any] = dict()      # Keyword arguments
    extra: dict[str, Any] = dict()       # Optional metadata
```

### Specialized Config Classes

| Config Class | Extra Fields | Used By |
|---|---|---|
| `ObservationCfg` | `mode`, `name` | ObservationManager |
| `EventCfg` | `mode`, `interval_step`, `is_global` | EventManager |
| `RewardCfg` | `weight`, `mode` | RewardManager |
| `ActionTermCfg` | `mode` | ActionManager |
| `DatasetFunctorCfg` | `mode` | DatasetManager |

### Python Config Example

```python
from embodichain.utils import configclass
from embodichain.lab.gym.envs.managers.cfg import (
    ObservationCfg,
    RewardCfg,
    EventCfg,
    SceneEntityCfg,
)
from embodichain.lab.gym.envs.managers.observations import get_object_pose


@configclass
class MyObsCfg:
    object_pose: ObservationCfg = ObservationCfg(
        func=get_object_pose,
        mode="add",
        name="object/pose",
        params={"entity_cfg": SceneEntityCfg(uid="my_cube")},
    )


@configclass
class MyRewardCfg:
    distance: RewardCfg = RewardCfg(
        func="distance_between_objects",
        weight=0.5,
        params={
            "source_entity_cfg": SceneEntityCfg(uid="cube"),
            "target_entity_cfg": SceneEntityCfg(uid="target"),
        },
    )


@configclass
class MyEventCfg:
    randomize_light: EventCfg = EventCfg(
        func="randomize_light",
        mode="interval",
        interval_step=5,
        params={"light_uid": "main_light"},
    )
```

---

## JSON and YAML Configuration

For RL training and data generation, EmbodiChain uses file-based configs (`.json`, `.yaml`, or `.yml`). The file format mirrors the Python config structure but uses string names instead of direct function references.

Configs are loaded with `embodichain.utils.utility.load_config`, which selects the parser from the file extension. Both formats produce the same in-memory dictionary and are passed to `config_to_cfg()` for environment setup.

Example paths in the repository:

| Use case | JSON example | YAML example |
|---|---|---|
| Gym environment | `embodichain_tasks/configs/gym/cobotmagic.json` | `embodichain_tasks/configs/gym/cobotmagic.yaml` |
| RL training | `embodichain_tasks/configs/agents/rl/basic/cart_pole/train_config.json` | `embodichain_tasks/configs/agents/rl/basic/cart_pole/train_config.yaml` |

When a training config references a gym config (via `trainer.gym_config`), the nested path may also use any supported extension.

### Environment Config (`gym_config.json` / `gym_config.yaml`)

```json
{
    "id": "EmbodiedEnv-v1",
    "num_envs": 4,
    "max_episodes": 100,
    "max_episode_steps": 600,
    "physics_config": {
        "gravity": [0.0, 0.0, -9.81],
        "bounce_threshold": 2.0,
        "enable_ccd": false,
        "length_tolerance": 0.05,
        "speed_tolerance": 0.25
    },
    "render_cfg": {
        "renderer": "auto",
        "spp": 1,
        "tone_mapping_enabled": false,
        "tone_mapping_exposure": 1.0
    },
    "visualization": {
        "backend": "viser",
        "scene_fps": 15.0,
        "sensor_image_fps": 2.0,
        "soft_body_fps": 5.0,
        "env_ids": [0],
        "viser_server": {
            "host": "127.0.0.1",
            "port": 8080
        }
    },
    "robot": {
        "uid": "robot",
        "urdf_cfg": {
            "components": [
                {
                    "component_type": "arm",
                    "urdf_path": "robots/my_robot/my_robot.urdf"
                }
            ]
        }
    },
    "sensor": [
        {
            "uid": "cam_high",
            "type": "StereoCamera",
            "height": 540,
            "width": 960
        }
    ],
    "env": {
        "control_parts": ["arm"],
        "actions": {
            "delta_qpos": {
                "func": "DeltaQposTerm",
                "params": {"scale": 0.1}
            }
        },
        "events": {
            "randomize_table": {
                "func": "randomize_visual_material",
                "mode": "interval",
                "interval_step": 10,
                "params": {"uid": "table"}
            }
        },
        "observations": {
            "obj_pose": {
                "func": "get_object_pose",
                "mode": "add",
                "name": "object/pose",
                "params": {"entity_cfg": {"uid": "cube"}}
            }
        },
        "rewards": {
            "distance": {
                "func": "distance_between_objects",
                "weight": 0.5,
                "params": {
                    "source_entity_cfg": {"uid": "cube"},
                    "target_entity_cfg": {"uid": "target"}
                }
            }
        },
        "dataset": {
            "lerobot": {
                "func": "LeRobotRecorder",
                "mode": "save",
                "params": {
                    "save_path": "/path/to/output",
                    "robot_meta": {"robot_type": "DexforceW1"},
                    "use_videos": true
                }
            }
        },
        "extensions": {
            "success_threshold": 0.1
        }
    }
}
```

The `visualization` section is optional and defaults to
`{"backend": "none"}`. Setting `"backend": "viser"` starts browser
visualization when the environment constructs its `SimulationManager`. The
`--viser*` command-line options override these values for
`embodichain run-env`.

Set `sensor_image_fps` to `null` to capture camera previews once per eligible
simulation step instead of applying a wall-clock FPS limit. `run-env --viser`
uses this step-synchronized mode by default when neither the configuration nor
`--viser-image-fps` supplies a rate.

Keep `viser_server.host` on loopback for remote workers and use SSH port
forwarding unless the service is behind an authenticated gateway. See
[Browser visualization with Viser](../overview/sim/viser_visualization.md) for
the full schema, supported scene content, and deformable-object behavior.

### RL Training Config (`train_config.json` / `train_config.yaml`)

```json
{
    "trainer": {
        "exp_name": "push_cube",
        "seed": 42,
        "device": "cuda:0",
        "iterations": 500,
        "buffer_size": 1024
    },
    "env": {
        "id": "PushCubeRL",
        "cfg": {
            "num_envs": 4,
            "actions": {
                "delta_qpos": {
                    "func": "DeltaQposTerm",
                    "params": {"scale": 0.1}
                }
            }
        }
    },
    "policy": {
        "name": "actor_critic",
        "actor": {
            "type": "mlp",
            "network_cfg": {"hidden_sizes": [256, 256], "activation": "relu"}
        },
        "critic": {
            "type": "mlp",
            "network_cfg": {"hidden_sizes": [256, 256], "activation": "relu"}
        }
    },
    "algorithm": {
        "name": "ppo",
        "cfg": {
            "learning_rate": 0.0001,
            "n_epochs": 10,
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2
        }
    }
}
```

The same structure in YAML:

```yaml
trainer:
  exp_name: push_cube
  seed: 42
  device: cuda:0
  iterations: 500
  buffer_size: 1024
  gym_config: embodichain_tasks/configs/agents/rl/basic/cart_pole/gym_config.yaml
policy:
  name: actor_critic
  actor:
    type: mlp
    network_cfg:
      hidden_sizes: [256, 256]
      activation: relu
algorithm:
  name: ppo
  cfg:
    learning_rate: 0.0001
    batch_size: 64
    gamma: 0.99
```

---

## String-Based Function Resolution

In JSON and YAML configs, functor functions are specified by name (string). EmbodiChain resolves these strings at runtime by searching registered modules. For example:

- `"distance_between_objects"` resolves to `embodichain.lab.gym.envs.managers.rewards.distance_between_objects`
- `"DeltaQposTerm"` resolves to `embodichain.lab.gym.envs.managers.actions.DeltaQposTerm`
- `"get_object_pose"` resolves to `embodichain.lab.gym.envs.managers.observations.get_object_pose`

When writing custom functors, make sure they are imported in the module's `__init__.py` so the resolver can find them.

---

## `SceneEntityCfg` in Config Files

When referencing scene entities in JSON or YAML, use a dictionary with a `uid` key:

```json
{"uid": "my_cube"}
```

This is automatically converted to a `SceneEntityCfg` object at runtime.

---

## Tips

1. **Start from an existing config.** Copy a config file from `embodichain_tasks/configs/gym/` or `embodichain_tasks/configs/agents/rl/` and modify it for your task.
2. **Use Python configs for development.** They provide IDE auto-completion and type checking.
3. **Use JSON or YAML configs for experiments.** YAML is often easier to read for nested structures; JSON remains fully supported.
4. **Validate configs early.** Run your environment with a short episode count to catch config errors before long training runs.
5. **Keep config pairs together.** For action-bank tasks, version `gym_config` and `action_config` together (either format).

---

## See Also

- [Custom Functors Guide](custom_functors.md) — How to write observation, reward, event, and action functors
- [Embodied Environments](../overview/gym/env.md) — Full environment configuration reference
- [Browser Visualization](../overview/sim/viser_visualization.md) — Viser configuration and runtime behavior
- [Tutorial: Modular Environment](../tutorial/modular_env.rst) — Complete example using config-driven setup
- [Tutorial: RL Training](../tutorial/rl.rst) — RL training configuration walkthrough
