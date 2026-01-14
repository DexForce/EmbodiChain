Embodied Environments
=====================

.. currentmodule:: embodichain.lab.gym

The :class:`envs.EmbodiedEnv` is the core environment class in EmbodiChain designed for complex Embodied AI tasks. It adopts a **configuration-driven** architecture, allowing users to define robots, sensors, objects, lighting, and automated behaviors (events) purely through configuration classes, minimizing the need for boilerplate code.

Core Architecture
-----------------

Unlike the standard :class:`envs.BaseEnv`, the :class:`envs.EmbodiedEnv` integrates several manager systems to handle the complexity of simulation:

* **Scene Management**: Automatically loads and manages robots, sensors, and scene objects defined in the configuration.
* **Event Manager**: Handles automated behaviors such as domain randomization, scene setup, and dynamic asset swapping.
* **Observation Manager**: Allows flexible extension of observation spaces without modifying the environment code.
* **Dataset Manager**: Built-in support for collecting demonstration data during simulation steps.
* **Action Bank**: Advanced action composition system supporting action graphs and configurable action primitives (optional, for complex manipulation sequences).

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

* **robot** (:class:`embodichain.lab.sim.cfg.RobotCfg`): 
  Defines the agent in the scene. Supports loading robots from URDF/MJCF with specified initial state and control mode.

* **sensor** (List[:class:`embodichain.lab.sim.cfg.SensorCfg`]): 
  A list of sensors attached to the scene or robot. Common sensors include :class:`~embodichain.lab.sim.sensors.StereoCamera` for RGB-D and segmentation data.

* **light** (:class:`envs.EmbodiedEnvCfg.EnvLightCfg`): 
  Configures the lighting environment.
  
  * ``direct``: Direct light sources (Point, Spot, Directional) affecting local illumination.
  * ``indirect``: Global illumination settings (Ambient, IBL) - *planned for future release*.

* **rigid_object** / **rigid_object_group** / **articulation**: 
  Defines the interactive elements.
  
  * ``rigid_object``: For dynamic or kinematic simple bodies.
  * ``rigid_object_group``: Collections of rigid objects that can be managed together.
  * ``articulation``: For complex mechanisms with joints (doors, drawers).

* **background** (List[:class:`embodichain.lab.sim.cfg.RigidObjectCfg`]): 
  Static or kinematic objects serving as obstacles or landmarks in the scene.

* **extensions** (Dict[str, Any]): 
  Task-specific extension parameters that are automatically bound to the environment instance. This allows passing custom parameters (e.g., ``episode_length``, ``obs_mode``, ``action_scale``) without modifying the base configuration class. These parameters are accessible as instance attributes after environment initialization. For example, if ``extensions = {"episode_length": 500}``, you can access it via ``self.episode_length``.

* **filter_visual_rand** (bool): 
  Whether to filter out visual randomization functors. Useful for debugging motion and physics issues when visual randomization interferes with the debugging process. Defaults to ``False``.

Manager Systems
---------------

The manager systems in :class:`envs.EmbodiedEnv` provide modular, configuration-driven functionality for handling complex simulation behaviors. Each manager uses a **functor-based** architecture, allowing you to compose behaviors through configuration without modifying environment code. Functors are reusable functions or classes (inheriting from :class:`envs.managers.Functor`) that operate on the environment state, configured through :class:`envs.managers.cfg.FunctorCfg`.

Event Manager
~~~~~~~~~~~~~

The Event Manager automates changes in the environment through event functors. Events can be triggered at different stages:

* **startup**: Executed once when the environment initializes. Useful for setting up initial scene properties that don’t change during episodes.
* **reset**: Executed every time ``env.reset()`` is called. Applied to specific environments that need resetting (via ``env_ids`` parameter). This is the most common mode for domain randomization.
* **interval**: Executed periodically every N steps (specified by ``interval_step``, defaults to 10). Can be configured per-environment (``is_global=False``) or globally synchronized (``is_global=True``).

Event functors are configured using :class:`envs.managers.cfg.EventCfg`. Common event functors include:

**Physics Randomization:**
 * ``randomize_rigid_object_mass``: Randomize object masses within a range.
 * ``randomize_rigid_object_friction``: Vary friction coefficients for more robust sim-to-real transfer. *Planned for future release.*
 * ``randomize_rigid_object_restitution``: Adjust bounciness of objects. *Planned for future release.*

**Visual Randomization:**
 * ``randomize_visual_material``: Randomize textures, base colors, and material properties (implemented as a Functor class).
 * ``randomize_light``: Vary light position, color, and intensity.
 * ``randomize_camera_extrinsics``: Randomize camera poses for viewpoint diversity.
 * ``randomize_camera_intrinsics``: Vary focal length and principal point.

**Spatial Randomization:**
 * ``randomize_rigid_object_pose``: Randomize object positions and orientations.
 * ``randomize_robot_eef_pose``: Vary end-effector initial poses.
 * ``randomize_robot_qpos``: Randomize robot joint configurations.

**Asset Management:**
 * ``replace_assets_from_group``: Swap object models from a folder on reset for visual diversity.
 * ``prepare_extra_attr``: Set up additional object attributes dynamically.

Observation Manager
~~~~~~~~~~~~~~~~~~~

While :class:`envs.EmbodiedEnv` provides default observations organized into two groups:

* **robot**: Contains ``qpos`` (joint positions), ``qvel`` (joint velocities), and ``qf`` (joint forces).
* **sensor**: Contains raw sensor outputs (images, depth, segmentation masks, etc.).

The Observation Manager allows you to extend the observation space with task-specific information. Observations are configured using :class:`envs.managers.cfg.ObservationCfg` with two operation modes:

* **modify**: Update existing observations in-place. The observation must already exist in the observation dictionary. Useful for normalization, transformation, or filtering existing data. Example: Normalize joint positions to [0, 1] range based on joint limits.
* **add**: Compute and add new observations to the observation space. The observation name can use hierarchical keys separated by ``/`` (e.g., ``"object/fork/pose"``).

Common observation functors include:

**Pose Computations:**
 * ``get_rigid_object_pose``: Get world poses of objects (returns 4x4 transformation matrices).
 * ``get_sensor_pose_in_robot_frame``: Transform sensor poses to robot coordinate frame.

**Relative Measurements:**
 * ``get_relative_pose``: Compute relative poses between objects or robot parts. *Planned for future release.*
 * ``get_distance``: Calculate distances between entities. *Planned for future release.*

.. note::
   To get robot end-effector poses, you can use the robot's ``compute_fk()`` method directly in your observation functors or task code.

**Keypoint Projections:**
 * ``compute_exteroception``: Project 3D keypoints (affordance poses, robot parts) onto camera image planes. Supports multiple sources: affordance poses from objects (e.g., grasp poses, place poses) and robot control part poses (e.g., end-effector positions).

**Normalization:**
 * ``normalize_robot_joint_data``: Normalize joint positions/velocities based on limits.

Dataset Manager
~~~~~~~~~~~~~~~

For Imitation Learning (IL) tasks, the Dataset Manager automates data collection through dataset functors. It currently supports:

* **LeRobot Format** (via :class:`envs.managers.datasets.LeRobotRecorder`):
  Standard format for LeRobot training pipelines. Includes support for task instructions, robot metadata, success flags, and optional video recording.

.. note::
   Additional dataset formats (HDF5, Zarr) are planned for future releases.

The manager operates in a single mode ``"save"`` which handles both recording and auto-saving:

* **Recording**: On each environment step, observation-action pairs are buffered in memory.
* **Auto-saving**: When ``dones=True`` (episode completion), completed episodes are automatically saved to disk with proper formatting.

**Configuration options include:**
 * ``save_path``: Root directory for saving datasets.
 * ``robot_meta``: Robot metadata dictionary (required for LeRobot format).
 * ``instruction``: Task instruction dictionary.
 * ``use_videos``: Whether to save video recordings of episodes.
 * ``export_success_only``: Filter to save only successful episodes (based on ``info["success"]``).

The dataset manager is called automatically during ``env.step()``, ensuring all observation-action pairs are recorded without additional user code.

Action Bank
~~~~~~~~~~~~

The Action Bank is an advanced action composition system designed for complex manipulation tasks. It enables you to define and compose complex action sequences through **action graphs** and **configurable action primitives**, making it particularly useful for generating expert demonstrations and handling multi-step manipulation sequences.

**Key Concepts:**

* **Action Graph**: A directed graph structure where nodes represent affordances (target poses/states) and edges represent action primitives (trajectory segments) that connect affordances. The graph defines the possible action sequences for completing a task.

* **Action Primitives**: Reusable action functions that generate trajectories between affordances. Each primitive is configured with duration, source affordance, target affordance, and optional parameters.

* **Scopes**: Independent action executors (e.g., ``left_arm``, ``right_arm``, ``left_eef``) that can operate in parallel. Each scope has its own action graph and dimension.

**Usage:**

Action Bank is optional and typically used for:

* Generating expert demonstration trajectories for Imitation Learning
* Defining complex multi-step manipulation sequences
* Composing actions from reusable primitives

To use Action Bank in your environment:

1. Create a custom ActionBank class inheriting from :class:`envs.action_bank.configurable_action.ActionBank`
2. Define node and edge functions using function tags (``@tag_node`` and ``@tag_edge`` decorators)
3. Initialize the action bank in your environment's ``__init__`` method using ``_init_action_bank()``
4. Use ``action_bank.create_action_list()`` to generate action sequences

.. note::
   Action Bank is an advanced feature primarily used for complex manipulation tasks. For simple tasks, you can use standard action spaces without Action Bank.

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

* :ref:`tutorial_create_basic_env` - Creating basic environments
* :ref:`tutorial_modular_env` - Advanced modular environment setup
* :mod:`embodichain.lab.gym.envs` - Complete API reference for EmbodiedEnv and EmbodiedEnvCfg
