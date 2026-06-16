---
name: add-functor
description: Use when adding a new observation, event, reward, action, dataset, or randomization functor to an EmbodiChain environment
---

# Add Functor

Scaffold a new functor following EmbodiChain's Functor/FunctorCfg pattern.

## When to Use

- User asks to add an observation term, reward function, event handler, action term, dataset functor, or randomizer
- User says "add a reward", "new observation", "create a randomizer", "add event functor"
- Any new function needs to be registered in a manager config

## Determine Functor Type

| Functor Type | Config Class | Module File | Manager | Signature |
|-------------|-------------|-------------|---------|-----------|
| Observation | `ObservationCfg` (extends `FunctorCfg`) | `managers/observations.py` | `ObservationManager` | `(env, obs, entity_cfg, ...) -> Tensor` |
| Reward | `RewardCfg` (extends `FunctorCfg`) | `managers/rewards.py` | `RewardManager` | `(env, obs, action, info, ...) -> Tensor` |
| Event | `EventCfg` (extends `FunctorCfg`) | `managers/events.py` | `EventManager` | `(env, env_ids, ...) -> None` |
| Action | `ActionTermCfg` (extends `FunctorCfg`) | `managers/actions.py` | `ActionManager` | Varies |
| Dataset | `DatasetFunctorCfg` (extends `FunctorCfg`) | `managers/datasets.py` | `DatasetManager` | `(env, ...) -> dict` |
| Randomization | `EventCfg` (randomizations ARE events) | `managers/randomization/<type>.py` | `EventManager` | `(env, env_ids, entity_cfg, ...) -> None` |

## Two Functor Styles

### Function-style (Preferred for Simple Functors)

A plain function with the right signature. Registered via `FunctorCfg(func=my_function, params={...})`.

```python
def my_reward(
    env: EmbodiedEnv,
    obs: dict,
    action: EnvAction,
    info: dict,
    my_param: float = 1.0,       # params become keyword args
) -> torch.Tensor:
    """Short one-line summary.

    Longer description if needed.

    Args:
        env: The environment instance.
        obs: The observation dictionary.
        action: The action taken.
        info: The info dictionary.
        my_param: Description of this parameter.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    # implementation
    return result
```

### Class-style (Required When Functor Has State)

A class inheriting `Functor`, with `__init__(cfg, env)` and `__call__(env, ...)`. Registered via `FunctorCfg(func=MyClass, params={...})`.

```python
class my_randomizer(Functor):
    """One-line summary."""

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        # Extract params and initialize state
        self.entity_cfg: SceneEntityCfg = cfg.params["entity_cfg"]

    def __call__(self, env: EmbodiedEnv, env_ids: torch.Tensor, **kwargs):
        """Apply the randomization.

        Args:
            env: The environment instance.
            env_ids: Target environment IDs.
        """
        # implementation
```

## Steps

### 1. Identify Functor Type and Style

Ask the user:
1. **Which manager?** (observation / reward / event / action / dataset / randomization)
2. **Function or class style?** (function for stateless, class for stateful)
3. **What does it do?** (brief description for naming + docstring)

### 2. Choose the Right Module File

Place the functor in the existing module for its type:

| Type | File |
|------|------|
| Observation | `embodichain/lab/gym/envs/managers/observations.py` |
| Reward | `embodichain/lab/gym/envs/managers/rewards.py` |
| Event | `embodichain/lab/gym/envs/managers/events.py` |
| Action | `embodichain/lab/gym/envs/managers/actions.py` |
| Dataset | `embodichain/lab/gym/envs/managers/datasets.py` |
| Physics randomization | `embodichain/lab/gym/envs/managers/randomization/physics.py` |
| Visual randomization | `embodichain/lab/gym/envs/managers/randomization/visual.py` |
| Spatial randomization | `embodichain/lab/gym/envs/managers/randomization/spatial.py` |
| Geometry randomization | `embodichain/lab/gym/envs/managers/randomization/geometry.py` |

### 3. Write the Functor

Follow the template for function-style or class-style (see above).

Key rules:
- First argument is always `env: EmbodiedEnv` (use `TYPE_CHECKING` guard for the import)
- Use `from __future__ import annotations` at the top
- Use `SceneEntityCfg` for entity references, not raw strings
- For observation functors: add `shape` key to `FunctorCfg.extra` dict
- For randomization functors: second arg is `env_ids: torch.Tensor | list[int]`
- For reward functors: return shape must be `(num_envs,)`

### 4. Update `__all__`

Add the new functor to the module's `__all__` list. If no `__all__` exists, create one.

### 5. Write a Test

Place at `tests/gym/envs/managers/test_<functor_type>.py` (append to existing file if present).

For functors that don't need a live simulation, use mock objects (`MockEnv`, `MockSim`, etc.) following the pattern in `tests/gym/envs/managers/test_reward_functors.py`.

### 6. Run `black`

```bash
black embodichain/lab/gym/envs/managers/<module>.py
black tests/gym/envs/managers/test_<functor_type>.py
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Wrong first argument signature | Observation: `(env, obs, ...)`, Reward: `(env, obs, action, info, ...)`, Event/Randomization: `(env, env_ids, ...)` |
| Importing `EmbodiedEnv` at module level | Use `TYPE_CHECKING` guard to avoid circular imports |
| Forgetting `SceneEntityCfg` for entity refs | Always use `SceneEntityCfg(uid="...")` not bare strings |
| Returning wrong tensor shape | Rewards must return `(num_envs,)`, observations must match declared shape |
| Missing `from __future__ import annotations` | Required in every file |
| Class-style functor not calling `super().__init__` | Always call `super().__init__(cfg, env)` |
| Adding randomizer as standalone | Randomizations ARE events — they go in `randomization/` but use `EventCfg` |

## Quick Reference

| Step | Action |
|------|--------|
| 1 | Identify manager type + function vs class style |
| 2 | Write functor in the correct module file |
| 3 | Update `__all__` in that module |
| 4 | Write test with mocks (no sim needed for most) |
| 5 | Run `black` on changed files |
