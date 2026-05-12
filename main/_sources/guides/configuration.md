# Configuration Guide

EmbodiChain uses a declarative configuration system built on Python dataclasses. This guide explains the key patterns: `@configclass`, `FunctorCfg`, and JSON configuration files.

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
│   └── gpu_memory_config: GPUMemoryCfg
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

## JSON Configuration

For RL training and data generation, EmbodiChain uses JSON config files. The JSON config mirrors the Python config structure but uses string names instead of direct function references.

### Environment Config (`gym_config.json`)

```json
{
    "max_episodes": 100,
    "max_episode_steps": 600,
    "env": {
        "num_envs": 4,
        "sim_cfg": {
            "sim_device": "cuda:0",
            "headless": true
        },
        "robot": {
            "uid": "robot",
            "urdf_cfg": {"fpath": "robots/my_robot/my_robot.urdf"}
        },
        "control_parts": ["arm"],
        "sensor": [
            {
                "uid": "cam_high",
                "type": "StereoCamera",
                "height": 540,
                "width": 960
            }
        ],
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

### RL Training Config (`train_config.json`)

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

---

## String-Based Function Resolution

In JSON configs, functor functions are specified by name (string). EmbodiChain resolves these strings at runtime by searching registered modules. For example:

- `"distance_between_objects"` resolves to `embodichain.lab.gym.envs.managers.rewards.distance_between_objects`
- `"DeltaQposTerm"` resolves to `embodichain.lab.gym.envs.managers.actions.DeltaQposTerm`
- `"get_object_pose"` resolves to `embodichain.lab.gym.envs.managers.observations.get_object_pose`

When writing custom functors, make sure they are imported in the module's `__init__.py` so the resolver can find them.

---

## `SceneEntityCfg` in JSON

When referencing scene entities in JSON, use a dictionary with a `uid` key:

```json
{"uid": "my_cube"}
```

This is automatically converted to a `SceneEntityCfg` object at runtime.

---

## Tips

1. **Start from an existing config.** Copy a config file from `configs/gym/` and modify it for your task.
2. **Use Python configs for development.** They provide IDE auto-completion and type checking.
3. **Use JSON configs for experiments.** They are easier to version, diff, and share.
4. **Validate configs early.** Run your environment with a short episode count to catch config errors before long training runs.
5. **Keep config pairs together.** For action-bank tasks, version `gym_config.json` and `action_config.json` together.

---

## See Also

- [Custom Functors Guide](custom_functors.md) — How to write observation, reward, event, and action functors
- [Embodied Environments](../overview/gym/env.md) — Full environment configuration reference
- [Tutorial: Modular Environment](../tutorial/modular_env.rst) — Complete example using config-driven setup
- [Tutorial: RL Training](../tutorial/rl.rst) — RL training configuration walkthrough
