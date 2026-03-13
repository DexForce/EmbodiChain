# Embodied Environments

```{currentmodule} embodichain.lab.gym
```

The {class}`~envs.EmbodiedEnv` is the core environment class in EmbodiChain designed for complex Embodied AI tasks. It adopts a **configuration-driven** architecture, allowing users to define robots, sensors, objects, lighting, and automated behaviors (events) purely through configuration classes, minimizing the need for boilerplate code.

For **Reinforcement Learning** tasks, EmbodiChain provides the **Action Manager** (configured via ``actions`` in {class}`~envs.EmbodiedEnvCfg`), which handles action preprocessing (scaling, IK, delta_qpos, etc.) in a modular, configurable way. RL tasks inherit from {class}`~envs.EmbodiedEnv` directly and use the Action Manager for action processing.

## Core Architecture

EmbodiChain provides a hierarchy of environment classes for different task types:

* **{class}`~envs.BaseEnv`**: Minimal environment for simple tasks with custom simulation logic.
* **{class}`~envs.EmbodiedEnv`**: Feature-rich environment for Embodied AI tasks (IL, custom control). Integrates manager systems:
  * **Scene Management**: Automatically loads and manages robots, sensors, and scene objects.
  * **Event Manager**: Domain randomization, scene setup, and dynamic asset swapping.
  * **Observation Manager**: Flexible observation space extensions.
  * **Dataset Manager**: Built-in support for demonstration data collection.
* **Action Manager**: Configurable action preprocessing for RL tasks (delta_qpos, eef_pose, qvel, etc.), integrated into {class}`~envs.EmbodiedEnv` when ``actions`` is configured.

## Configuration System

The environment is defined by inheriting from {class}`~envs.EmbodiedEnvCfg`. This configuration class serves as the single source of truth for the scene description.

{class}`~envs.EmbodiedEnvCfg` inherits from {class}`~envs.EnvCfg` (the base environment configuration class, sometimes referred to as `BaseEnvCfg`), which provides fundamental environment parameters. The following sections describe both the base class parameters and the additional parameters specific to {class}`~envs.EmbodiedEnvCfg`.

### BaseEnvCfg Parameters

Since {class}`~envs.EmbodiedEnvCfg` inherits from {class}`~envs.EnvCfg`, it includes the following base parameters:

* **num_envs** (int): 
  The number of sub environments (arenas) to be simulated in parallel. Defaults to ``1``.

* **sim_cfg** ({class}`~embodichain.lab.sim.SimulationManagerCfg`): 
  Simulation configuration for the environment, including physics settings, device selection, and rendering options. Defaults to a basic configuration with headless mode enabled.

* **seed** (int | None): 
  The seed for the random number generator. Defaults to ``None``, in which case the seed is not set. The seed is set at the beginning of the environment initialization to ensure deterministic behavior across different runs.

* **sim_steps_per_control** (int): 
  Number of simulation steps per control (environment) step. This parameter determines the relationship between the simulation timestep and the control timestep. For instance, if the simulation dt is 0.01s and the control dt is 0.1s, then ``sim_steps_per_control`` should be 10. This means that the control action is updated every 10 simulation steps. Defaults to ``4``.

* **ignore_terminations** (bool): 
  Whether to ignore terminations when deciding when to auto reset. Terminations can be caused by the task reaching a success or fail state as defined in a task's evaluation function. If set to ``False``, episodes will stop early when termination conditions are met. If set to ``True``, episodes will only stop due to the timelimit, which is useful for modeling tasks as infinite horizon. Defaults to ``False``.

* **max_episode_steps** (int): 
  Maximum number of steps per episode. If set to ``-1``, episodes will not have a step limit and will only end due to success/failure conditions. Defaults to ``300``.

### EmbodiedEnvCfg Parameters

The {class}`~envs.EmbodiedEnvCfg` class exposes the following additional parameters:

* **robot** ({class}`~embodichain.lab.sim.cfg.RobotCfg`): 
  Defines the agent in the scene. Supports loading robots from URDF/MJCF with specified initial state and control mode. This is a required field.

* **control_parts** (List[str]): 
  List of robot part names that are controlled by the environment's action space. This allows for flexible control schemes (e.g., controlling only the left arm or end-effector). Defaults to an empty list, in which case no robot parts are controlled.

* **active_joint_ids** (List[int]): 
  List of joint IDs that are active for control and observation. This is used to filter the robot's full joint state to only the relevant joints for the task. Defaults to an empty list, in which case all joints are considered active.

* **sensor** (List[{class}`~embodichain.lab.sim.sensor.SensorCfg`]): 
  A list of sensors attached to the scene or robot. Common sensors include {class}`~embodichain.lab.sim.sensors.StereoCamera` for RGB-D and segmentation data. Defaults to an empty list.

* **light** ({class}`~envs.EmbodiedEnvCfg.EnvLightCfg`): 
  Configures the lighting environment. The {class}`EnvLightCfg` class contains:
  
  * ``direct``: List of direct light sources (Point, Spot, Directional) affecting local illumination. Defaults to an empty list.
  * ``indirect``: Global illumination settings (Ambient, IBL) - *planned for future release*.

* **rigid_object** (List[{class}`~embodichain.lab.sim.cfg.RigidObjectCfg`]): 
  List of dynamic or kinematic simple bodies. Defaults to an empty list.

* **rigid_object_group** (List[{class}`~embodichain.lab.sim.cfg.RigidObjectGroupCfg`]): 
  Collections of rigid objects that can be managed together. Efficient for many similar objects. Defaults to an empty list.

* **articulation** (List[{class}`~embodichain.lab.sim.cfg.ArticulationCfg`]): 
  List of complex mechanisms with joints (doors, drawers). Defaults to an empty list.

* **background** (List[{class}`~embodichain.lab.sim.cfg.RigidObjectCfg`]): 
  Static or kinematic objects serving as obstacles or landmarks in the scene. Defaults to an empty list.

* **events** (Union[object, None]): 
  Event settings for domain randomization and automated behaviors. Defaults to None, in which case no events are applied through the event manager. Please refer to the {class}`~envs.managers.EventManager` class for more details.

* **observations** (Union[object, None]): 
  Custom observation specifications. Defaults to None, in which case no additional observations are applied through the observation manager. Please refer to the {class}`~envs.managers.ObservationManager` class for more details.

* **dataset** (Union[object, None]): 
  Dataset collection settings. Defaults to None, in which case no dataset collection is performed. Please refer to the {class}`~envs.managers.DatasetManager` class for more details.

* **actions** (Union[object, None]): 
  Action Manager settings for RL tasks. When configured, preprocesses raw policy actions (e.g., delta_qpos, eef_pose) into robot control format. Replaces the legacy RLEnv. Defaults to None. See the {class}`~envs.managers.ActionManager` class for more details.

* **extensions** (Union[Dict[str, Any], None]): 
  Task-specific extension parameters that are automatically bound to the environment instance. This allows passing custom parameters (e.g., ``success_threshold``) without modifying the base configuration class. For action configuration, use the ``actions`` field instead. These parameters are accessible as instance attributes after environment initialization. Defaults to None.

* **filter_visual_rand** (bool): 
  Whether to filter out visual randomization functors. Useful for debugging motion and physics issues when visual randomization interferes with the debugging process. Defaults to ``False``.

* **filter_dataset_saving** (bool): 
  Whether to filter out dataset saving functors. Useful for debugging when dataset saving interferes with the debugging process. Defaults to ``False``.

* **init_rollout_buffer** (bool): 
  Whether to initialize the rollout buffer for data collection. If ``True``, the environment will create a rollout buffer matching the observation/action spaces for episode recording. Defaults to ``False``. If you plan to use the dataset manager for imitation learning, you should set this to ``True`` to enable episode recording.

### Example Configuration

```python
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.utils import configclass

@configclass
class MyTaskEnvCfg(EmbodiedEnvCfg):
    # 1. Define Scene Components
    robot = ...          # Robot configuration
    sensor = [...]       # List of sensors (e.g., Cameras)
    light = ...          # Lighting configuration

    # 2. Define Objects
    rigid_object = [...]       # Dynamic objects (e.g., tools, debris)
    rigid_object_group = [...] # Object groups (efficient for many similar objects)
    articulation = [...]       # Articulated objects (e.g., cabinets)

    # 3. Define Managers
    events = ...         # Event settings (Randomization, etc.)
    observations = ...   # Custom observation spec
    dataset = ...        # Data collection settings

    # 4. Action Manager (for RL tasks)
    actions = ...        # Action preprocessing (e.g., DeltaQposTerm with scale)
    extensions = {       # Task-specific parameters (e.g., success_threshold)
        "success_threshold": 0.1,
    }
```

## Manager Systems

The manager systems in {class}`~envs.EmbodiedEnv` provide modular, configuration-driven functionality for handling complex simulation behaviors. Each manager uses a **functor-based** architecture, allowing you to compose behaviors through configuration without modifying environment code. Functors are reusable functions or classes (inheriting from {class}`~envs.managers.Functor`) that operate on the environment state, configured through {class}`~envs.managers.cfg.FunctorCfg`.

### Event Manager

The Event Manager automates changes in the environment through event functors. Events can be triggered at different stages:

* **startup**: Executed once when the environment initializes. Useful for setting up initial scene properties that don't change during episodes.
* **reset**: Executed every time {meth}`~envs.Env.reset()` is called. Applied to specific environments that need resetting (via ``env_ids`` parameter). This is the most common mode for domain randomization.
* **interval**: Executed periodically every N steps (specified by ``interval_step``, defaults to 10). Can be configured per-environment (``is_global=False``) or globally synchronized (``is_global=True``).

Event functors are configured using {class}`~envs.managers.cfg.EventCfg`. For a complete list of available event functors, please refer to {doc}`event_functors`.

### Observation Manager

While {class}`~envs.EmbodiedEnv` provides default observations organized into two groups:

* **robot**: Contains ``qpos`` (joint positions), ``qvel`` (joint velocities), and ``qf`` (joint forces).
* **sensor**: Contains raw sensor outputs (images, depth, segmentation masks, etc.).

The Observation Manager allows you to extend the observation space with task-specific information. Observations are configured using {class}`~envs.managers.cfg.ObservationCfg` with two operation modes:

* **modify**: Update existing observations in-place. The observation must already exist in the observation dictionary. Useful for normalization, transformation, or filtering existing data. Example: Normalize joint positions to [0, 1] range based on joint limits.
* **add**: Compute and add new observations to the observation space. The observation name can use hierarchical keys separated by ``/`` (e.g., ``"object/fork/pose"``).

For a complete list of available observation functors, please refer to {doc}`observation_functors`.

### Dataset Manager

For Imitation Learning (IL) tasks, the Dataset Manager automates data collection through dataset functors. For a complete list of available dataset functors and their parameters, please refer to {doc}`dataset_functors`. It currently supports:

* **LeRobot Format** (via {class}`~envs.managers.datasets.LeRobotRecorder`):
  Standard format for LeRobot training pipelines. Includes support for task instructions, robot metadata, success flags, and optional video recording.

```{note}
Additional dataset formats (HDF5, Zarr) are planned for future releases.
```

The manager operates in a single mode ``"save"`` which handles both recording and auto-saving:

* **Recording**: On each environment step, observation-action pairs are buffered in memory.
* **Auto-saving**: When ``dones=True`` (episode completion), completed episodes are automatically saved to disk with proper formatting.

**Configuration options include:**
 * ``save_path``: Root directory for saving datasets.
 * ``robot_meta``: Robot metadata dictionary (required for LeRobot format).
 * ``instruction``: Task instruction dictionary.
 * ``use_videos``: Whether to save video recordings of episodes.

The dataset manager is called automatically during {meth}`~envs.Env.step()`, ensuring all observation-action pairs are recorded without additional user code.

## Reinforcement Learning Environment

For RL tasks, EmbodiChain uses the **Action Manager** integrated into {class}`~envs.EmbodiedEnv`:

* **Action Preprocessing**: Configurable via ``actions`` in {class}`~envs.EmbodiedEnvCfg`. Supports DeltaQposTerm, QposTerm, QposNormalizedTerm, EefPoseTerm, QvelTerm, QfTerm. For a complete list of available action terms, please refer to {doc}`action_functors`.
* **Standardized Info Structure**: {class}`~envs.EmbodiedEnv` provides ``compute_task_state``, ``get_info``, and ``evaluate`` for task-specific success/failure and metrics.
* **Episode Management**: Configurable episode length and truncation logic.

### Action Manager Configuration

Configure action preprocessing via the ``actions`` field:

```python
from embodichain.lab.gym.envs.managers import ActionTermCfg, DeltaQposTerm
from embodichain.utils import configclass

@configclass
class MyRLActionCfg:
    delta_qpos: ActionTermCfg = ActionTermCfg(
        func=DeltaQposTerm,
        params={"scale": 0.1}
    )

# In EmbodiedEnvCfg:
actions = MyRLActionCfg()
extensions = {"success_threshold": 0.1}  # Task-specific parameters
```

In JSON config, use the ``actions`` section:

```json
"actions": {
    "delta_qpos": {
        "func": "DeltaQposTerm",
        "params": { "scale": 0.1 }
    }
}
```


## Creating a Custom Task

### For Reinforcement Learning Tasks

Inherit from {class}`~envs.EmbodiedEnv` and implement the task-specific logic. Configure the Action Manager via ``actions`` in your config:

```python
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env

@register_env("MyRLTask-v0")
class MyRLTaskEnv(EmbodiedEnv):
    def __init__(self, cfg: MyTaskEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def compute_task_state(self, **kwargs):
        # Required: Compute task-specific success/failure and metrics
        # Returns: Tuple[success, fail, metrics]
        #   - success: torch.Tensor of shape (num_envs,) with boolean values
        #   - fail: torch.Tensor of shape (num_envs,) with boolean values
        #   - metrics: Dict of metric tensors for logging
        
        is_success = ...  # Compute success condition
        is_fail = torch.zeros_like(is_success)
        metrics = {"distance": ..., "angle_error": ...}
        
        return is_success, is_fail, metrics
```

Configure rewards through the {class}`~envs.managers.RewardManager` in your environment config rather than overriding ``get_reward``. For a complete list of available reward functors, please refer to {doc}`reward_functors`.

### For Imitation Learning Tasks

Inherit from {class}`~envs.EmbodiedEnv` for IL tasks:

```python
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env

@register_env("MyILTask-v0")
class MyILTaskEnv(EmbodiedEnv):
    def __init__(self, cfg: MyTaskEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def create_demo_action_list(self, *args, **kwargs):
        # Required: Generate scripted demonstrations for data collection
        pass

    def is_task_success(self, **kwargs):
        # Required: Define success criteria for filtering successful episodes
        # Returns: torch.Tensor of shape (num_envs,) with boolean values
        return success_tensor

    def get_info(self, **kwargs):
        # Optional: Override to add custom info fields
        info = super().get_info(**kwargs)
        info["custom_metric"] = ...
        return info
```

For a complete example of a modular environment setup, please refer to the {ref}`tutorial_modular_env` tutorial.

## See Also

- {ref}`tutorial_create_basic_env` - Creating basic environments
- {ref}`tutorial_modular_env` - Advanced modular environment setup
- {ref}`tutorial_rl` - Reinforcement learning training guide
- {doc}`/api_reference/embodichain/embodichain.lab.gym.envs` - Complete API reference for EmbodiedEnv and configurations

```{toctree}
:maxdepth: 1

event_functors.md
observation_functors.md
reward_functors.md
action_functors.md
dataset_functors.md
```
