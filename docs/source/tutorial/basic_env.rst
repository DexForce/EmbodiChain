
.. _tutorial_create_basic_env:

Creating a Basic Environment
============================

.. currentmodule:: embodichain.lab.gym

This tutorial shows you how to create a simple robot learning environment using EmbodiChain's Gym interface. You'll learn how to inherit from the base environment class, set up robots and objects, define actions and observations, and run training scenarios.

The Code
~~~~~~~~

The tutorial corresponds to the ``random_reach.py`` script in the ``scripts/tutorials/gym`` directory.

.. dropdown:: Code for random_reach.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/gym/random_reach.py
      :language: python
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

This tutorial demonstrates how to create a custom RL environment by inheriting from :class:`envs.BaseEnv`. The environment implements a simple reach task where a robot arm tries to reach randomly positioned targets.

Environment Registration
-------------------------

First, we register the environment with the Gymnasium registry using the :func:`utils.registration.register_env` decorator:

.. literalinclude:: ../../../scripts/tutorials/gym/random_reach.py
   :language: python
   :start-at: @register_env("RandomReach-v1", max_episode_steps=100, override=True)
   :end-at: class RandomReachEnv(BaseEnv):

The decorator parameters define:

- **Environment ID**: ``"RandomReach-v1"`` - unique identifier for the environment
- **max_episode_steps**: Maximum steps per episode (100 in this case)
- **override**: Whether to override existing environment with same ID

Environment Initialization
---------------------------

The ``__init__`` method configures the simulation environment and calls the parent constructor:

.. literalinclude:: ../../../scripts/tutorials/gym/random_reach.py
   :language: python
   :lines: 25-46

Key configuration options include:

- **num_envs**: Number of parallel environments to run
- **headless**: Whether to run without GUI (useful for training)
- **device**: Computation device ("cpu" or "cuda")

Robot Setup
------------

The `_setup_robot` method loads and configures the robot for the environment:

.. literalinclude:: ../../../scripts/tutorials/gym/random_reach.py
   :language: python
   :start-at: def _setup_robot(self, **kwargs) -> Robot:
   :end-at: return robot

This method demonstrates:

1. **URDF Loading**: Using data module to access robot URDF files
2. **Robot Configuration**: Setting initial position and joint configuration
3. **Action Space Definition**: Creating action space based on joint limits

The action space is automatically derived from the robot's joint limits, ensuring actions stay within valid ranges.

Scene Preparation
-----------------

The :meth:`_prepare_scene` method adds additional objects to the simulation environment:

.. literalinclude:: ../../../scripts/tutorials/gym/random_reach.py
   :language: python
   :lines: 72-84

In this example, we add a kinematic cube that serves as a visual target. The cube is configured with:

- **No collision**: ``enable_collision=False`` for visualization only
- **Kinematic body**: Can be moved programmatically without physics
- **Custom size**: Small 3cm cube for target visualization
- **initial position**: Initially placed at a fixed location

State Updates
-------------

The `_update_sim_state` method is called at each simulation step to update object states:

.. literalinclude:: ../../../scripts/tutorials/gym/random_reach.py
   :language: python
   :start-at: def _update_sim_state(self, **kwargs) -> None:
   :end-at: self.cube.set_local_pose(pose=pose)

This method randomizes the cube's position. The pose is updated for all parallel environments simultaneously.

Note that this method is called after perform action execution and simulation update but before observation collection. For more details, see :meth:`envs.BaseEnv.step`.

Action Execution
----------------

The `_step_action` method applies actions to the robot:

.. literalinclude:: ../../../scripts/tutorials/gym/random_reach.py
   :language: python
   :start-at: def _step_action(self, action: EnvAction) -> EnvAction:
   :end-at: return action

In this simple environment, actions directly set joint positions. More complex environments might:

- Convert actions to joint torques or velocities
- Apply action filtering or scaling
- Implement inverse kinematics for end-effector control

Observation Extension
---------------------

The default observations include the following keys:

- `robot`: Robot proprioception data (joint positions, velocities, efforts)
- `sensor` (optional): Data from any sensors (e.g., cameras)

The `_extend_obs` method allows you to add custom observations:

.. literalinclude:: ../../../scripts/tutorials/gym/random_reach.py
   :language: python
   :start-at: def _extend_obs(self, obs: EnvObs, **kwargs) -> EnvObs:
   :end-at: return obs

While commented out in this example, you can add custom data like:

- Object positions and orientations
- Distance calculations
- Custom sensor readings
- Task-specific state information

The Code Execution
~~~~~~~~~~~~~~~~~~

To run the environment:

.. code-block:: bash

   cd /path/to/embodichain
   python scripts/tutorials/gym/random_reach.py

You can customize the execution with command-line options:

.. code-block:: bash

   # Run multiple parallel environments
   python scripts/tutorials/gym/random_reach.py --num_envs 4
   
   # Run with GPU acceleration
   python scripts/tutorials/gym/random_reach.py --device cuda
   
   # Run in headless mode (no GUI)
   python scripts/tutorials/gym/random_reach.py --headless

The script demonstrates:

1. **Environment Creation**: Using ``gym.make()`` with custom parameters
2. **Episode Loop**: Running multiple episodes with random actions
3. **Performance Monitoring**: Calculating frames per second (FPS)

Key Features Demonstrated
~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial showcases several important features of EmbodiChain environments:

1. **Gymnasium Integration**: Full compatibility with the Gymnasium API
2. **Parallel Environments**: Running multiple environments simultaneously for efficient training
3. **Robot Integration**: Easy loading and control of robotic systems
4. **Custom Objects**: Adding and manipulating scene objects
5. **Flexible Actions**: Customizable action spaces and execution methods
6. **Extensible Observations**: Adding task-specific observation data
