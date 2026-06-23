Gym
===

.. currentmodule:: embodichain.lab.gym

The ``embodichain.lab.gym`` module is the environment layer that turns
simulation scenes into robot-learning tasks. It follows the Gymnasium interface
while adding vectorized simulation, declarative scene configuration, manager
systems for task logic, and tensor-based observations and actions for learning
pipelines.

The design mirrors the separation used in modern robot-learning frameworks such
as Isaac Lab: simulation objects live in the scene, task behavior is composed
through managers, and the environment exposes a stable ``reset`` / ``step`` API
to reinforcement learning, imitation learning, data collection, and evaluation
code.

Environment Classes
-------------------

:class:`~embodichain.lab.gym.envs.base_env.BaseEnv` is the common
Gymnasium-compatible foundation. It owns the
:class:`~embodichain.lab.sim.SimulationManager`, manages the number of parallel
environments, configures timing through ``sim_steps_per_control``, tracks
episode length, exposes batched observation and action spaces, and defines the
standard environment lifecycle.

:class:`~embodichain.lab.gym.envs.embodied_env.EmbodiedEnv` builds on
:class:`~embodichain.lab.gym.envs.base_env.BaseEnv` for configuration-driven
embodied tasks. A
single :class:`~embodichain.lab.gym.envs.embodied_env.EmbodiedEnvCfg` declares
the robot, sensors, lights, background objects, interactive objects,
articulations, and manager configs. The environment constructs the simulation
scene from that config and then delegates task-specific behavior to managers and
functors.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Class
     - Role
     - Use it when
   * - :class:`~embodichain.lab.gym.envs.base_env.BaseEnv`
     - Provides Gymnasium compatibility, vectorized arena control, simulation
       timing, spaces, reset, and step bookkeeping.
     - You need a custom environment base with direct control over scene setup
       and task methods.
   * - :class:`~embodichain.lab.gym.envs.base_env.EnvCfg`
     - Configures common environment settings such as ``num_envs``, simulation
       config, seed, control frequency, and episode length.
     - You are defining shared runtime parameters for any environment.
   * - :class:`~embodichain.lab.gym.envs.embodied_env.EmbodiedEnv`
     - Adds declarative scene creation and manager-based actions, events,
       observations, rewards, and dataset recording.
     - You are building robot manipulation, RL, IL, or data-generation tasks.
   * - :class:`~embodichain.lab.gym.envs.embodied_env.EmbodiedEnvCfg`
     - Declares the robot, controlled parts, sensors, objects, articulations,
       lights, managers, and task extension fields.
     - You want the task definition to live primarily in configuration.

Architecture
------------

The Gym module sits above the simulation module and below training or data
collection code:

.. code-block:: text

    RL trainer / IL recorder / evaluation script
        |
        v
    Gymnasium API: reset(), step(action), observation_space, action_space
        |
        v
    BaseEnv
    |-- SimulationManager ownership
    |-- vectorized arenas and timing
    |-- episode state and auto reset
    `-- batched observation/action spaces
        |
        v
    EmbodiedEnv
    |-- scene config: robot, sensors, objects, lights, articulations
    |-- ActionManager: policy action -> robot command
    |-- EventManager: startup, reset, and interval behavior
    |-- ObservationManager: add or modify observation terms
    |-- RewardManager: weighted scalar reward terms
    `-- DatasetManager: episode recording and export

This layering keeps task code small. A task can define only the scene
configuration, reward or termination logic, and any task-specific parameters
while relying on managers for reusable behavior.

Manager and Functor Pattern
---------------------------

Managers are configured collections of functors. A functor is either a function
or a callable class that receives the environment and optional parameters, then
performs one well-scoped operation. The config objects identify the callable and
its parameters, and
:class:`~embodichain.lab.gym.envs.managers.cfg.SceneEntityCfg` resolves named
robots, objects, joints, links, or bodies from the simulation scene.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Manager
     - Responsibility
     - Typical use
   * - Action manager
     - Converts raw policy actions into robot commands such as joint targets,
       velocity commands, force commands, or end-effector targets.
     - RL policies that output normalized or task-space actions.
   * - Event manager
     - Applies startup, reset, and interval behaviors.
     - Domain randomization, object placement, visual changes, and scripted
       scene updates.
   * - Observation manager
     - Adds task-specific observations or modifies existing nested observation
       entries.
     - Object poses, target states, normalized proprioception, sensor-derived
       terms, and keypoint projections.
   * - Reward manager
     - Evaluates weighted reward functors and sums them into the environment
       reward.
     - Distance, alignment, success, smoothness, and penalty terms for RL.
   * - Dataset manager
     - Records episode data through dataset functors.
     - Imitation-learning demonstrations, LeRobot export, and offline dataset
       generation.

Typical Step Flow
-----------------

At runtime, an ``EmbodiedEnv`` step usually follows this high-level sequence:

1. Receive a batched action from a policy, script, or teleoperation source.
2. Use the Action Manager to convert it into robot control targets.
3. Step the simulation for ``sim_steps_per_control`` physics steps.
4. Update sensors and collect base observations from the robot and scene.
5. Apply Observation Manager terms to add or transform observation entries.
6. Evaluate rewards, success, failure, timeout, and reset conditions.
7. Record transition data through the Dataset Manager when configured.
8. Return Gymnasium-compatible ``obs``, ``reward``, ``terminated``,
   ``truncated``, and ``info`` values.

Task Authoring Workflow
-----------------------

For most new tasks, start from
:class:`~embodichain.lab.gym.envs.embodied_env.EmbodiedEnv`:

1. Define an
   :class:`~embodichain.lab.gym.envs.embodied_env.EmbodiedEnvCfg` subclass with
   the robot, objects, sensors, lights, and manager configs.
2. Register the task with ``register_env`` so it can be constructed by ID.
3. Configure actions for RL tasks, or dataset recording for IL tasks.
4. Add observation, reward, and event functors instead of hard-coding reusable
   logic in the environment class.
5. Keep custom environment methods focused on task-specific success, failure,
   reset, or demonstration behavior.

Choosing Where to Start
-----------------------

- Start with :doc:`env` for the full
  :class:`~embodichain.lab.gym.envs.embodied_env.EmbodiedEnv` configuration and
  custom task guide.
- Use :doc:`action_functors` when connecting policy outputs to robot control.
- Use :doc:`event_functors` for reset randomization, visual randomization, and
  scene perturbations.
- Use :doc:`observation_functors` to add task observations without changing base
  environment code.
- Use :doc:`reward_functors` when composing RL reward terms.
- Use :doc:`dataset_functors` when recording demonstrations or exporting
  datasets.

Documentation Quality Notes
---------------------------

Gym documentation should make the runtime contract explicit: tensor shapes,
whether data is batched by ``num_envs``, which manager mode invokes a functor,
and which scene entity each functor expects. Prefer linking to the simulation
overview for asset and sensor details, and keep task pages focused on the
environment lifecycle, manager configuration, and learning interface.

See Also
--------

.. toctree::
   :maxdepth: 1

   env.md
