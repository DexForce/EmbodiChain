# Design: Episode Observation Support

## Problem

Episode observations (e.g., randomized mass, friction, object UIDs, body scale) are
constant throughout an episode but change between episodes. Currently they are:

- Computed every step (via cached observation functors) and written to the rollout
  buffer at **every time step** — redundant storage of identical values.
- Saved per-frame in LeRobot datasets — same redundancy.
- There is no explicit signal that a given observation is episode-constant, making it
  hard for downstream consumers (RL training, data loading) to treat them differently.

## Goals

1. Add a declarative flag so observation functors can be marked as producing
   episode-constant data.
2. Store episode observations **once per episode** in the rollout buffer (no time
   dimension), instead of `max_episode_steps` times.
3. Save them into LeRobot datasets either per-frame (replicated for compatibility)
   or as episode-level metadata.
4. Keep the design backward-compatible — existing configs without the flag continue
   to work as before.

---

## Design

### 1. Configuration: `is_episode_obs` flag on `ObservationCfg`

**File**: `embodichain/lab/gym/envs/managers/cfg.py`

```python
@configclass
class ObservationCfg(FunctorCfg):
    mode: Literal["modify", "add"] = "modify"
    name: str = MISSING
    is_episode_obs: bool = False  # <-- NEW
    """Whether this observation is constant throughout an episode.

    When True:
    - The observation is computed once during episode reset (after randomization
      events) and stored in a flat per-episode buffer instead of the per-step
      rollout buffer.
    - It is excluded from the per-step observation space and rollout buffer
      time dimension.
    - It is still available for LeRobot dataset saving and RL training.
    """
```

**Only valid with `mode="add"`** — `modify` mode observations already exist in the
observation space and are inherently per-step. Setting `is_episode_obs=True` with
`mode="modify"` raises a `ValueError` during observation manager initialization.

**Example usage**:

```python
observations = ObservationManagerCfg(
    object_physics=ObservationCfg(
        func=get_rigid_object_physics_attributes,
        mode="add",
        name="object_physics",
        is_episode_obs=True,  # <-- mark as episode-constant
        params={"entity_cfg": SceneEntityCfg(uid="cube")},
    ),
    cube_uid=ObservationCfg(
        func=get_object_uid,
        mode="add",
        name="cube_uid",
        is_episode_obs=True,
        params={"entity_cfg": SceneEntityCfg(uid="cube")},
    ),
)
```

---

### 2. ObservationManager changes

**File**: `embodichain/lab/gym/envs/managers/observation_manager.py`

Add a property to query episode-obs functors and a method to compute them:

```python
class ObservationManager(ManagerBase):

    @property
    def has_episode_obs(self) -> bool:
        """Whether any observation functor is marked as episode-constant."""
        return any(
            cfg.is_episode_obs
            for cfgs in self._mode_functor_cfgs.values()
            for cfg in cfgs
        )

    @property
    def episode_obs_functors(self) -> list[tuple[str, ObservationCfg]]:
        """Return (name, cfg) pairs for all episode-obs functors."""
        result = []
        for cfgs in self._mode_functor_cfgs.values():
            for cfg in cfgs:
                if cfg.is_episode_obs:
                    result.append((cfg.name, cfg))
        return result

    def compute_episode_obs(self) -> dict[str, torch.Tensor | TensorDict]:
        """Compute all episode-constant observations.

        Returns:
            Dict mapping observation name to computed tensor/TensorDict.
            Shape: (num_envs, *obs_shape) with no time dimension.
        """
        episode_obs = {}
        for name, cfg in self.episode_obs_functors:
            data = cfg.func(self._env, None, **cfg.params)
            episode_obs[name] = data
        return episode_obs

    def compute(self, obs: EnvObs) -> EnvObs:
        """Modified to skip episode-obs functors from per-step computation."""
        for mode, functor_cfgs in self._mode_functor_cfgs.items():
            for functor_cfg in functor_cfgs:
                # Skip episode observations in per-step compute
                if getattr(functor_cfg, "is_episode_obs", False):
                    continue
                # ... existing logic ...
```

**Validation** (in `_prepare_functors` or `__init__`):
```python
for cfg in all_cfgs:
    if getattr(cfg, "is_episode_obs", False) and cfg.mode == "modify":
        raise ValueError(
            f"ObservationCfg '{cfg.name}': is_episode_obs=True is only "
            f"supported with mode='add', got mode='{cfg.mode}'."
        )
```

---

### 3. Rollout Buffer: flat `episode_obs` section

**File**: `embodichain/lab/gym/utils/gym_utils.py`

Add episode observation buffer initialization alongside the main rollout buffer:

```python
def init_rollout_buffer_from_gym_space(
    obs_space, action_space, max_episode_steps, num_envs, device,
    episode_obs_space=None,  # <-- NEW optional parameter
) -> TensorDict:
    # ... existing buffer creation ...

    buffer_dict = {
        "obs": _init_buffer_from_space(obs_space, num_envs),
        "actions": _init_buffer_from_space(action_space, num_envs),
        "rewards": torch.zeros((num_envs, max_episode_steps), ...),
    }

    # Episode observations: flat [num_envs, *shape], no time dimension
    if episode_obs_space is not None and len(episode_obs_space.spaces) > 0:
        buffer_dict["episode_obs"] = _init_flat_episode_obs_buffer(
            episode_obs_space, num_envs, device
        )

    return TensorDict(buffer_dict, batch_size=[num_envs, max_episode_steps], device=device)
```

Where `_init_flat_episode_obs_buffer` creates tensors of shape `[num_envs, *shape]`
(without the `max_episode_steps` dimension):

```python
def _init_flat_episode_obs_buffer(episode_obs_space, num_envs, device):
    """Create flat episode-obs buffers: [num_envs, *shape] (no time dim)."""
    if isinstance(episode_obs_space, spaces.Dict):
        return TensorDict(
            {k: _init_flat_buffer(v, num_envs, device) for k, v in episode_obs_space.spaces.items()},
            batch_size=[num_envs],
            device=device,
        )
    # Box space
    return torch.zeros((num_envs, *episode_obs_space.shape[1:]), ...)

def _init_flat_buffer(space, num_envs, device):
    if isinstance(space, spaces.Dict):
        return TensorDict(...)
    return torch.zeros((num_envs, *space.shape[1:]), dtype=..., device=device)
```

**Rollout buffer shape summary**:

```
rollout_buffer = TensorDict({
    "obs":              [num_envs, max_steps, *obs_shape],     # per-step (excludes episode obs)
    "actions":          [num_envs, max_steps, *action_shape],
    "rewards":          [num_envs, max_steps],
    "episode_obs": {                                         # NEW: flat, no time dim
        "object_physics": {
            "mass":    [num_envs, 1],
            "friction": [num_envs, 1],
            ...
        },
        "cube_uid":    [num_envs],
    }
})
```

---

### 4. Episode Observation Capture at Reset

**File**: `embodichain/lab/gym/envs/embodied_env.py`

Capture episode observations after reset events (randomization) have been applied:

```python
def _initialize_episode(self, env_ids=None, **kwargs):
    # ... existing logic (save dataset, clear buffers, apply reset events) ...

    # After event_manager.apply(mode="reset", env_ids=env_ids):
    # Reset observation manager (clears caches)
    if self.cfg.observations:
        self.observation_manager.reset(env_ids=env_ids)

    # Capture episode observations AFTER reset events and cache clear
    if self.observation_manager and self.observation_manager.has_episode_obs:
        self._capture_episode_obs(env_ids)

    # ... rest of reset logic ...


def _capture_episode_obs(self, env_ids: Sequence[int] | None) -> None:
    """Compute and store episode-constant observations for the specified environments."""
    if self.rollout_buffer is None or "episode_obs" not in self.rollout_buffer.keys():
        return

    episode_obs_data = self.observation_manager.compute_episode_obs()

    env_ids = list(range(self.num_envs)) if env_ids is None else env_ids
    buffer = self.rollout_buffer["episode_obs"]

    for name, data in episode_obs_data.items():
        if isinstance(data, TensorDict):
            for sub_key, sub_data in data.items():
                buffer[name][sub_key][env_ids].copy_((
                    sub_data[env_ids]
                    if sub_data.shape[0] == self.num_envs
                    else sub_data
                ).to(buffer.device))
        else:
            buffer[name][env_ids].copy_((
                data[env_ids]
                if data.shape[0] == self.num_envs
                else data
            ).to(buffer.device))
```

---

### 5. Exclusion from Per-Step Obs Writing

**File**: `embodichain/lab/gym/envs/embodied_env.py`

In `_write_episode_rollout_step`, the `obs` TensorDict already excludes episode
observations because the observation manager's `compute()` skips `is_episode_obs`
functors. No additional changes needed here — the separation is handled upstream
in the observation manager.

However, the **observation space** used to initialize the rollout buffer must also
exclude episode observations. This is handled by building the episode observation
space separately:

```python
# In EmbodiedEnv.__init__, after observation manager is created:
if self.observation_manager and self.observation_manager.has_episode_obs:
    # Build episode_obs_space from episode-obs functors
    self._episode_obs_space = self._build_episode_obs_space()
else:
    self._episode_obs_space = None

# The main observation space (self.single_observation_space) should exclude
# episode-obs keys. This requires either:
#   (a) Building obs space before adding episode obs, or
#   (b) Removing episode obs keys from the space after construction.
```

**Simpler alternative**: Keep episode observations in the main obs space for
backward compatibility but mark them. The rollout buffer's `obs` section still
has space for them (they'll just be zeros), and the real data lives in
`episode_obs`. During LeRobot conversion, merge both sources.

---

### 6. LeRobot Dataset Saving

**File**: `embodichain/lab/gym/envs/managers/datasets.py`

In `_convert_frame_to_lerobot`, merge episode observations from the flat buffer:

```python
def _convert_frame_to_lerobot(self, obs, action, task, env_id=0) -> Dict:
    frame = {"task": task}

    # ... existing per-frame obs conversion ...

    # Add episode observations (constant across frames, read from flat buffer)
    if "episode_obs" in self._env.rollout_buffer.keys():
        episode_obs = self._env.rollout_buffer["episode_obs"]
        self._add_episode_obs_to_frame(frame, episode_obs, env_id)

    return frame


def _add_episode_obs_to_frame(
    self, frame: Dict, episode_obs: TensorDict, env_id: int
) -> None:
    """Add episode-constant observations to a LeRobot frame.

    These values are the same for every frame in the episode but are included
    in each frame for compatibility with standard training pipelines.
    """
    for key, value in episode_obs.items():
        if isinstance(value, TensorDict):
            for sub_key, sub_value in value.items():
                v = sub_value[env_id].cpu()
                if isinstance(v, torch.Tensor) and v.ndim == 0:
                    v = v.unsqueeze(0)
                frame[f"observation.{key}.{sub_key}"] = v
        else:
            v = value[env_id].cpu()
            if isinstance(v, torch.Tensor) and v.ndim == 0:
                v = v.unsqueeze(0)
            frame[key] = v
```

**Features definition** (`_build_features`): Episode observations are already
handled by the existing `_add_nested_features` logic since they'll appear in the
observation space. The `is_episode_obs` flag only affects storage efficiency,
not the feature schema.

---

### 7. RL Rollout Buffer Support

For RL mode, episode observations can also be useful (e.g., for domain
randomization-aware policies). When an external RL buffer is set via
`set_rollout_buffer()`:

- If the external buffer contains an `"episode_obs"` key, the env populates it
  during reset.
- If it doesn't contain the key, episode observations are silently skipped.
- The RL framework is responsible for deciding how to use episode obs
  (e.g., as conditioning input to the policy).

---

## Summary of Changes

| File | Change |
|------|--------|
| `managers/cfg.py` | Add `is_episode_obs: bool = False` to `ObservationCfg` |
| `managers/observation_manager.py` | Add `has_episode_obs`, `episode_obs_functors`, `compute_episode_obs()`, skip episode obs in `compute()` |
| `gym_utils.py` | Add flat episode-obs buffer initialization in `init_rollout_buffer_from_gym_space` |
| `embodied_env.py` | Add `_capture_episode_obs()` called after reset events |
| `managers/datasets.py` | Read from `episode_obs` buffer in `_convert_frame_to_lerobot` |

## Backward Compatibility

- `is_episode_obs` defaults to `False` — all existing configs work unchanged.
- The rollout buffer only gets the `episode_obs` section when at least one
  functor is marked as episode-constant.
- LeRobot saving falls back gracefully if `episode_obs` is absent.
- RL buffers without `episode_obs` are supported transparently.
