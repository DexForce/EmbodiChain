# Reward Functors

```{currentmodule} embodichain.lab.gym.envs.managers
```

This page lists all available reward functors that can be used with the Reward Manager. Reward functors are configured using {class}`~cfg.RewardCfg` and return scalar reward tensors that are weighted and summed to form the total environment reward.

````{tip}
**Using an AI coding agent?** Use the **`/add-functor`** skill to scaffold a new reward functor with the correct signature (`env, obs, action, info, ...`), module placement in `rewards.py`, and `__all__` export. Use **`/add-test`** to generate mock-based tests.
````

## Distance-Based Rewards

```{list-table} Distance-Based Reward Functors
:header-rows: 1
:widths: 25 75

* - Functor Name
  - Description
* - {func}`~rewards.distance_between_objects`
  - Reward based on distance between two rigid objects. Supports either linear negative distance or exponential Gaussian-shaped reward. Higher when objects are closer.

    ```json
    {"func": "distance_between_objects", "weight": 0.5,
     "params": {"source_entity_cfg": {"uid": "bottle"},
                "target_entity_cfg": {"uid": "cup"},
                "exponential": true, "sigma": 0.2}}
    ```
* - {func}`~rewards.distance_to_target`
  - Reward based on absolute distance to a virtual target pose. Uses target pose stored in env by randomize_target_pose event. Can use exponential or linear reward, and supports XY-only distance.

    ```json
    {"func": "distance_to_target", "weight": 1.0,
     "params": {"entity_cfg": {"uid": "bottle"},
                "target_pose_key": "goal_pose",
                "exponential": true, "sigma": 0.3}}
    ```
* - {func}`~rewards.incremental_distance_to_target`
  - Incremental reward for progress toward a virtual target pose. Rewards getting closer compared to previous timestep. Uses tanh shaping and supports asymmetric weighting for approach vs. retreat.

    ```json
    {"func": "incremental_distance_to_target", "weight": 1.0,
     "params": {"source_entity_cfg": {"uid": "bottle"},
                "target_pose_key": "goal_pose",
                "tanh_scale": 10.0, "positive_weight": 2.0,
                "negative_weight": 0.5, "use_xy_only": true}}
    ```
```

## Alignment Rewards

```{list-table} Alignment Reward Functors
:header-rows: 1
:widths: 25 75

* - Functor Name
  - Description
* - {func}`~rewards.orientation_alignment`
  - Reward rotational alignment between two rigid objects. Uses rotation matrix trace to measure alignment. Ranges from -1 to 1 (1.0 = perfect alignment).

    ```json
    {"func": "orientation_alignment", "weight": 0.5,
     "params": {"source_entity_cfg": {"uid": "bottle"},
                "target_entity_cfg": {"uid": "cup"}}}
    ```
```

## Task-Specific Rewards

```{list-table} Task-Specific Reward Functors
:header-rows: 1
:widths: 25 75

* - Functor Name
  - Description
* - {func}`~rewards.reaching_behind_object`
  - Reward for positioning end-effector behind object for pushing. Encourages reaching a position behind the object along the object-to-goal direction.

    ```json
    {"func": "reaching_behind_object", "weight": 1.0,
     "params": {"robot_uid": "CobotMagic", "part_name": "left_arm",
                "object_cfg": {"uid": "bottle"},
                "target_cfg": {"uid": "cup"}}}
    ```
* - {func}`~rewards.success_reward`
  - Sparse bonus reward when task succeeds. Reads success status from info['success'] which should be set by the environment.

    ```json
    {"func": "success_reward", "weight": 10.0, "params": {}}
    ```
```

## Penalty Rewards

```{list-table} Penalty Reward Functors
:header-rows: 1
:widths: 25 75

* - Functor Name
  - Description
* - {func}`~rewards.joint_velocity_penalty`
  - Penalize high joint velocities to encourage smooth motion. Computes L2 norm of joint velocities and returns negative value as penalty.

    ```json
    {"func": "joint_velocity_penalty", "weight": 0.001,
     "params": {"robot_uid": "CobotMagic", "part_name": "left_arm"}}
    ```
* - {func}`~rewards.action_smoothness_penalty`
  - Penalize large action changes between consecutive timesteps. Encourages smooth control commands. Reads previous action from env.episode_action_buffer.

    ```json
    {"func": "action_smoothness_penalty", "weight": 0.01, "params": {}}
    ```
* - {func}`~rewards.joint_limit_penalty`
  - Penalize robot joints that are close to their position limits. Prevents joints from reaching physical limits. Penalty increases as joints approach limits within a margin.

    ```json
    {"func": "joint_limit_penalty", "weight": 0.01,
     "params": {"robot_uid": "CobotMagic", "part_name": "left_arm",
                "margin": 0.1}}
    ```
```

## Usage Example

```python
from embodichain.lab.gym.envs.managers.cfg import RewardCfg, SceneEntityCfg

# Example: Distance-based reward with exponential shaping
rewards = {
    "approach_object": RewardCfg(
        func="distance_between_objects",
        weight=0.5,
        params={
            "source_entity_cfg": SceneEntityCfg(uid="cube"),
            "target_entity_cfg": SceneEntityCfg(uid="target"),
            "exponential": True,
            "sigma": 0.2,
        },
    ),
}

# Example: Joint velocity penalty
rewards = {
    "joint_velocity_penalty": RewardCfg(
        func="joint_velocity_penalty",
        weight=0.001,
        params={
            "robot_uid": "robot",
            "part_name": "arm",
        },
    ),
}

# Example: Action smoothness penalty
rewards = {
    "action_smoothness": RewardCfg(
        func="action_smoothness_penalty",
        weight=0.01,
        params={},
    ),
}

# Example: Success reward
rewards = {
    "success": RewardCfg(
        func="success_reward",
        weight=10.0,
        params={},
    ),
}

# Example: Incremental distance reward
rewards = {
    "incremental_progress": RewardCfg(
        func="incremental_distance_to_target",
        weight=1.0,
        params={
            "source_entity_cfg": SceneEntityCfg(uid="cube"),
            "target_pose_key": "goal_pose",
            "tanh_scale": 10.0,
            "positive_weight": 2.0,
            "negative_weight": 0.5,
            "use_xy_only": True,
        },
    ),
}
```

## Reward Function Signature

All reward functors follow the same signature:

```python
def reward_functor(
    env: EmbodiedEnv,
    obs: dict,
    action: torch.Tensor | dict,
    info: dict,
    **params,
) -> torch.Tensor:
    """Reward functor.

    Args:
        env: The environment instance.
        obs: Current observation dictionary.
        action: Current action from policy.
        info: Info dictionary from environment.
        **params: Additional parameters from config.

    Returns:
        Reward tensor of shape (num_envs,).
    """
```

The reward manager automatically weights and sums all configured rewards to produce the total reward at each timestep.
