Embodied Environments
=====================

.. currentmodule:: embodichain.lab.gym

The :class:`envs.EmbodiedEnv` is the core environment class in EmbodiChain designed for complex Embodied AI tasks. It adopts a **configuration-driven** architecture, allowing users to define robots, sensors, objects, lighting, and automated behaviors (events) purely through configuration classes, minimizing the need for boilerplate code.

Core Architecture
-----------------

Unlike the standard :class:`envs.BaseEnv`, the :class:`envs.EmbodiedEnv` integrates several manager systems to handle the complexity of simulation:

- **Scene Management**: Automatically loads and manages robots, sensors, and scene objects defined in the configuration.
- **Event Manager**: Handles automated behaviors such as domain randomization, scene setup, and dynamic asset swapping.
- **Observation Manager**: Allows flexible extension of observation spaces without modifying the environment code.
- **Dataset Manager**: Built-in support for collecting demonstration data during simulation steps.
- **Action Bank**: Advanced action composition system supporting action graphs and configurable action primitives (optional, for complex manipulation sequences).

Configuration System
--------------------

The environment is defined by inheriting from :class:`envs.EmbodiedEnvCfg`. This configuration class serves as the single source of truth for the scene description.

.. code-block:: python

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
        
        # 4. Task Extensions
        extensions = {       # Task-specific parameters
            "episode_length": 500,
            "obs_mode": "state",
        }

Key Components
~~~~~~~~~~~~~~

The :class:`envs.EmbodiedEnvCfg` exposes several key fields to compose your environment:

- **robot** (:class:`embodichain.lab.sim.cfg.RobotCfg`): 
  Defines the agent in the scene. Supports loading robots from URDF/MJCF with specified initial state and control mode.
  
- **sensor** (List[:class:`embodichain.lab.sim.cfg.SensorCfg`]): 
  A list of sensors attached to the scene or robot. Common sensors include :class:`embodichain.lab.sim.sensors.StereoCamera` for RGB-D and segmentation data.

- **light** (:class:`envs.EmbodiedEnvCfg.EnvLightCfg`): 
  Configures the lighting environment.
  
  - ``direct``: Direct light sources (Point, Spot, Directional) affecting local illumination.
  - ``indirect``: Global illumination settings (Ambient, IBL) - *planned for future release*.

- **rigid_object** / **rigid_object_group** / **articulation**: 
  Defines the interactive elements.
  
  - ``rigid_object``: For dynamic or kinematic simple bodies.
  - ``rigid_object_group``: Collections of rigid objects that can be managed together.
  - ``articulation``: For complex mechanisms with joints (doors, drawers).

- **background** (List[:class:`embodichain.lab.sim.cfg.RigidObjectCfg`]):
  Static or kinematic objects serving as obstacles or landmarks in the scene.

- **extensions** (Dict[str, Any]): 
  Task-specific extension parameters that are automatically bound to the environment instance. 
  This allows passing custom parameters (e.g., ``episode_length``, ``obs_mode``, ``action_scale``) 
  without modifying the base configuration class. These parameters are accessible via ``self.param_name`` 
  after environment initialization.

- **filter_visual_rand** (bool): 
  Whether to filter out visual randomization functors. Useful for debugging motion and physics issues 
  when visual randomization interferes with the debugging process. Defaults to ``False``.

Manager Systems
---------------

Event Manager
~~~~~~~~~~~~~

The Event Manager automates changes in the environment. Events can be triggered at different stages:

- **startup**: Executed once when the environment initializes.
- **reset**: Executed every time ``env.reset()`` is called.
- **interval**: Executed periodically every N steps.

Common use cases include **Domain Randomization** (randomizing friction, mass, visual textures) and **Asset Swapping** (changing object models on reset).

Observation Manager
~~~~~~~~~~~~~~~~~~~

While :class:`envs.EmbodiedEnv` provides default observations (robot state and raw sensor data), the Observation Manager allows you to inject task-specific information.

You can configure it to compute and add:

  - Object poses relative to the robot.
  - Keypoint positions.
  - Task-specific metrics (e.g., distance to target).

Dataset Manager
~~~~~~~~~~~~~~~

For Imitation Learning (IL) tasks, the Dataset Manager automates data collection. It can:

  - Record full observation-action pairs.
  - Save episodes automatically when environments terminate.
  - Handle data formatting for training pipelines.

Creating a Custom Task
----------------------

To create a new task, inherit from :class:`envs.EmbodiedEnv` and implement the task-specific logic.

.. code-block:: python

    from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
    from embodichain.lab.gym.utils.registration import register_env

    @register_env("MyTask-v0", max_episode_steps=500)
    class MyTaskEnv(EmbodiedEnv):
        def __init__(self, cfg: MyTaskEnvCfg, **kwargs):
            super().__init__(cfg, **kwargs)
            
        def create_demo_action_list(self, *args, **kwargs):
            # Optional: Implement for scripted demonstrations
            # Must set self.action_length = len(action_list) if returning actions
            pass
            
        def is_task_success(self, **kwargs):
            # Optional: Define success criteria (mainly for IL data collection)
            # Returns: torch.Tensor of shape (num_envs,) with boolean values
            return success_tensor
            
        def get_reward(self, obs, action, info):
            # Optional: Override for RL tasks
            # Returns: torch.Tensor of shape (num_envs,)
            return super().get_reward(obs, action, info)
            
        def get_info(self, **kwargs):
            # Optional: Override to add custom info fields
            # Should include "success" and "fail" keys for termination
            info = super().get_info(**kwargs)
            info["custom_metric"] = ...
            return info

For a complete example of a modular environment setup, please refer to the :ref:`tutorial_modular_env` tutorial.

See Also
--------

- :ref:`tutorial_create_basic_env` - Creating basic environments
- :ref:`tutorial_modular_env` - Advanced modular environment setup
- :doc:`/api_reference/embodichain/embodichain.lab.gym.envs` - Complete API reference for EmbodiedEnv and EmbodiedEnvCfg
