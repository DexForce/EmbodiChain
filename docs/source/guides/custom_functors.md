# Writing Custom Functors

Functors are the building blocks of EmbodiChain's manager system. They define how observations are computed, rewards are calculated, events are triggered, actions are preprocessed, and datasets are recorded.

This guide explains the two functor styles (function and class), how to register them in manager configs, and provides examples for each functor type.

---

## Functor Basics

Every functor is configured through a `FunctorCfg` object with three fields:

| Field | Type | Description |
|-------|------|-------------|
| `func` | `Callable \| Functor` | The function or class to call. **Required.** |
| `params` | `dict` | Keyword arguments passed to the function. |
| `extra` | `dict` | Optional metadata (e.g., observation shapes). |

The `func` field can be:
- A **function** (callable) — receives the environment as the first argument, plus any `params` as keyword arguments.
- A **class** inheriting from `Functor` — instantiated with `(cfg, env)`, then called via `__call__`.

---

## Function-Style Functors

Function-style functors are plain Python functions. They are stateless and easy to write. Use them when your functor is a simple computation that doesn't need to maintain state between calls.

### General Pattern

```python
def my_functor(env, obs, **kwargs) -> torch.Tensor:
    """Compute something from the environment state.

    Args:
        env: The environment instance.
        obs: The current observation dictionary.
        **kwargs: Additional parameters from FunctorCfg.params.

    Returns:
        A tensor of shape (num_envs, ...).
    """
    # Access environment state
    value = compute_value(env)

    return value
```

The exact signature depends on the functor type (see below).

### Example: Observation Functor

Observation functors receive `(env, obs)` plus any params. They must return a tensor.

```python
from __future__ import annotations
import torch
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.managers.observations import EnvObs
from embodichain.lab.sim.cfg import SceneEntityCfg


def get_object_height(
    env: EmbodiedEnv,
    obs: EnvObs,
    entity_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Get the Z-coordinate (height) of an object.

    Args:
        env: The environment instance.
        obs: The current observation dictionary.
        entity_cfg: Scene entity configuration with the object UID.

    Returns:
        Tensor of shape (num_envs, 1) with the object height.
    """
    obj = env.sim.get_rigid_object(entity_cfg.uid)
    pose = obj.get_local_pose(to_matrix=True)  # (num_envs, 4, 4)
    height = pose[:, 2, 3:4]  # Extract Z from translation
    return height
```

Register it in your environment config:

```python
from embodichain.lab.gym.envs.managers.cfg import ObservationCfg, SceneEntityCfg
from embodichain.utils import configclass


@configclass
class MyObsCfg:
    obj_height: ObservationCfg = ObservationCfg(
        func=get_object_height,
        mode="add",
        name="object/height",
        params={"entity_cfg": SceneEntityCfg(uid="my_cube")},
    )
```

Or in JSON:

```json
"observations": {
    "obj_height": {
        "func": "get_object_height",
        "mode": "add",
        "name": "object/height",
        "params": {"entity_cfg": {"uid": "my_cube"}}
    }
}
```

### Example: Reward Functor

Reward functors receive `(env, obs, action, info)` plus any params. They return a tensor of shape `(num_envs,)`.

```python
import torch
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.sim.cfg import SceneEntityCfg


def target_height_reward(
    env: EmbodiedEnv,
    obs: dict,
    action,
    info: dict,
    entity_cfg: SceneEntityCfg = None,
    target_height: float = 0.5,
) -> torch.Tensor:
    """Reward for lifting an object to a target height.

    Returns:
        Negative distance to the target height. Shape (num_envs,).
    """
    obj = env.sim.get_rigid_object(entity_cfg.uid)
    pose = obj.get_local_pose(to_matrix=True)
    current_height = pose[:, 2, 3]
    return -torch.abs(current_height - target_height)
```

Register it:

```python
from embodichain.lab.gym.envs.managers.cfg import RewardCfg
from embodichain.utils import configclass


@configclass
class MyRewardCfg:
    lift_reward: RewardCfg = RewardCfg(
        func=target_height_reward,
        weight=1.0,
        params={
            "entity_cfg": SceneEntityCfg(uid="my_cube"),
            "target_height": 0.5,
        },
    )
```

---

## Class-Style Functors

Class-style functors inherit from `Functor` and implement `__init__(cfg, env)` and `__call__(...)`. Use them when you need to:

- Maintain state across calls (e.g., caching, counters)
- Perform expensive initialization once
- Implement a `reset()` method for per-episode cleanup

### General Pattern

```python
from embodichain.lab.gym.envs.managers import Functor
from embodichain.lab.gym.envs.managers.cfg import FunctorCfg


class MyFunctor(Functor):
    """A stateful functor."""

    def __init__(self, cfg: FunctorCfg, env):
        super().__init__(cfg, env)
        # Initialize state, buffers, etc.
        self._counter = 0

    def reset(self, env_ids=None):
        """Called on environment reset."""
        self._counter = 0

    def __call__(self, env, obs, **kwargs):
        """Called every step."""
        self._counter += 1
        # Compute and return result
```

### Example: Observation Functor with Caching

```python
from __future__ import annotations
import torch
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.managers import Functor
from embodichain.lab.gym.envs.managers.cfg import FunctorCfg, ObservationCfg
from embodichain.lab.sim.cfg import SceneEntityCfg


class get_object_mass(Functor):
    """Get the mass of a rigid object, with caching.

    Caches the result to avoid repeated queries to the physics engine.
    Cache is cleared on environment reset.
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._cache = {}

    def reset(self, env_ids=None):
        self._cache.clear()

    def __call__(
        self,
        env: EmbodiedEnv,
        obs,
        entity_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        uid = entity_cfg.uid
        if uid in self._cache:
            return self._cache[uid].clone()

        obj = env.sim.get_rigid_object(uid)
        mass = obj.get_mass()  # (num_envs, 1)

        self._cache[uid] = mass.clone()
        return mass
```

### Example: Action Functor

Action functors inherit from `ActionTerm` and implement `process_action`. They transform raw policy actions into robot control commands.

```python
from __future__ import annotations
import torch
from embodichain.lab.gym.envs.managers.actions import ActionTerm
from embodichain.lab.gym.envs.managers.cfg import ActionTermCfg


class DeltaQposTerm(ActionTerm):
    """Delta joint position: current_qpos + scale * action -> target qpos.

    The policy outputs a position offset, which is added to the current
    joint positions to get the target.
    """

    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)
        self._scale = cfg.params.get("scale", 1.0)

    @property
    def input_key(self) -> str:
        return "qpos"

    @property
    def action_dim(self) -> int:
        return len(self._env.active_joint_ids)

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self._scale + self._env.robot.get_qpos()
```

Register it in your gym config file (JSON or YAML):

```json
"actions": {
    "delta_qpos": {
        "func": "DeltaQposTerm",
        "params": {"scale": 0.1}
    }
}
```

---

## Functor Signature Reference

Each functor type has a specific call signature:

### Observation Functors

```python
def my_obs_functor(env, obs, **params) -> torch.Tensor
```

- `env`: The environment instance.
- `obs`: The current observation dictionary.
- Additional params from `ObservationCfg.params`.
- Returns: tensor of shape `(num_envs, ...)`.

Config class: `ObservationCfg` with `mode` (`"add"` or `"modify"`) and `name`.

### Reward Functors

```python
def my_reward_functor(env, obs, action, info, **params) -> torch.Tensor
```

- `env`: The environment instance.
- `obs`: The current observation dictionary.
- `action`: The action taken this step.
- `info`: The info dictionary.
- Additional params from `RewardCfg.params`.
- Returns: tensor of shape `(num_envs,)`.

Config class: `RewardCfg` with `weight` and `mode` (`"add"` or `"replace"`).

### Event Functors

```python
def my_event_functor(env, env_ids, **params) -> None
```

- `env`: The environment instance.
- `env_ids`: The environment IDs affected by this event.
- Additional params from `EventCfg.params`.
- Returns: `None` (events modify the environment in-place).

Config class: `EventCfg` with `mode` (`"startup"`, `"reset"`, or `"interval"`) and `interval_step`.

### Action Functors

```python
class MyActionTerm(ActionTerm):
    def process_action(self, action: torch.Tensor) -> torch.Tensor
```

- `action`: Raw action from the policy, shape `(num_envs, action_dim)`.
- Returns: transformed action tensor.

Config class: `ActionTermCfg` with `mode` (`"pre"` or `"post"`).

### Dataset Functors

Dataset functors handle recording and saving. In most cases you should use the built-in `LeRobotRecorder` rather than writing a custom one.

Config class: `DatasetFunctorCfg` with `mode` (`"save"`).

---

## Using `SceneEntityCfg` in Params

Many functors need to reference scene objects (robots, rigid objects, sensors). Instead of passing string UIDs directly, use `SceneEntityCfg`:

```python
from embodichain.lab.sim.cfg import SceneEntityCfg

params = {
    "entity_cfg": SceneEntityCfg(uid="my_cube"),
}
```

The manager automatically resolves `SceneEntityCfg` objects to the actual simulation entities at runtime.

---

## File Placement

| Functor Type | Recommended Location |
|---|---|
| Observation | `embodichain/lab/gym/envs/managers/observations.py` |
| Reward | `embodichain/lab/gym/envs/managers/rewards.py` |
| Event | `embodichain/lab/gym/envs/managers/events.py` or `embodichain/lab/gym/envs/managers/randomization/` |
| Action | `embodichain/lab/gym/envs/managers/actions.py` |
| Dataset | `embodichain/lab/gym/envs/managers/datasets.py` |

For task-specific functors, place them in the task module file (e.g., alongside the task environment class).

Remember to:
- Add the functor to `__all__` in the module.
- Add the Apache 2.0 license header.
- Use type annotations with `from __future__ import annotations`.

---

## See Also

- [Configuration Guide](configuration.md) — How to set up `@configclass` configs and JSON files
- [Embodied Environments](../overview/gym/env.md) — Full environment architecture
- [Tutorial: Modular Environment](../tutorial/modular_env.rst) — Using functors in a complete environment
