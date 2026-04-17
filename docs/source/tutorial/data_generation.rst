.. _tutorial_data_generation:

Data Generation
===============

.. currentmodule:: embodichain.lab.gym

This tutorial shows you how to generate synthetic expert demonstration datasets using EmbodiChain's built-in environment launcher and dataset manager. You will learn how to configure LeRobot recording in a JSON file, how the official ``run_env.py`` script builds environments and executes expert trajectories, and how completed episodes are automatically saved to disk.

Overview
~~~~~~~~

EmbodiChain provides a built-in data generation workflow for imitation learning tasks:

- **Gym Configuration**: Describes the scene, robot, sensors, randomization events, observations, and dataset recorder.
- **Action Configuration**: Describes how the task-specific expert trajectory should be generated.
- **Environment Launcher**: Builds the environment directly from configuration files.
- **Expert Policy**: Each task provides ``create_demo_action_list()`` to generate a scripted trajectory.
- **Dataset Manager**: Records observation-action pairs during ``env.step()``.
- **LeRobotRecorder**: Converts completed episodes into LeRobot-compatible datasets, with optional video export.

This workflow is useful when you want to generate large amounts of synthetic data for manipulation tasks without writing a custom rollout loop from scratch.

What This Tutorial Records
--------------------------

This page documents the full path from task configuration to saved dataset:

1. Prepare a task ``gym_config.json``.
2. Prepare an ``action_config.json`` that controls expert action generation.
3. Launch the environment runner.
4. Let the task generate actions through ``create_demo_action_list()``.
5. Execute the actions with ``env.step()``.
6. Let the dataset manager automatically save completed episodes.

The intention is to make the data-generation process easy to reproduce for new tasks.

Example Task: Items Handover and Place
--------------------------------------

As a running example, we use an ``items_handover_place`` task. In a project repository, this usually comes with two files:

- ``configs/items_handover_place/gym_config.json``
- ``configs/items_handover_place/action_config.json``

The gym configuration defines the simulation scene and recording behavior, while the action configuration defines the expert action graph used to solve the task.

The Code
~~~~~~~~

The tutorial corresponds to the ``run_env.py`` script in the ``embodichain/lab/scripts`` directory.

.. dropdown:: Code for run_env.py
   :icon: code

   .. literalinclude:: ../../../embodichain/lab/scripts/run_env.py
      :language: python
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

The launcher builds the environment from configuration, generates expert trajectories, executes them step by step, and relies on the dataset manager to auto-save valid episodes.

Step 1: Prepare the Task Configuration
--------------------------------------

The first input to the pipeline is the task ``gym_config.json``. In the ``items_handover_place`` example, the same file contains scene randomization, observations, dataset recording, and robot/sensor definitions.

The dataset-related part looks like this:

.. code-block:: json

   {
       "env": {
           "dataset": {
               "lerobot": {
                   "func": "LeRobotRecorder",
                   "mode": "save",
                   "params": {
                       "save_path": "/root/workspace/Embodied_Challenge/lerobot_dataset/",
                       "robot_meta": {
                           "robot_type": "CobotMagic",
                           "control_freq": 25,
                           "control_parts": ["left_arm", "left_eef", "right_arm", "right_eef"]
                       },
                       "instruction": {
                           "lang": "Right arm picks up the pen and hands it to left arm, then places it inside the pen holder"
                       },
                       "extra": {
                           "scene_type": "Sim",
                           "task_description": "items_handover_place",
                           "data_type": "sim"
                       },
                       "use_videos": true
                   }
               }
           }
       }
   }

The most important parameters are:

- **save_path**: Where the generated dataset will be written.
- **robot_meta**: Robot metadata such as robot type, control frequency, and active control parts.
- **instruction**: Task language instruction stored together with the dataset.
- **extra**: Additional metadata such as scene type and task description.
- **use_videos**: Whether camera observations should be stored as videos instead of raw images.

In the same example file, the task also defines scene randomization and sensor setup. This is important because the generated data is determined not only by the recorder, but also by how the scene is randomized before each rollout.

In the current implementation, ``LeRobotRecorder`` stores robot state and action features such as ``observation.qpos``, ``observation.qvel``, ``observation.qf``, ``action``, and camera images when sensors are present.

Step 2: Prepare the Action Configuration
----------------------------------------

The second input is the ``action_config.json`` file. This file describes how the task generates its expert trajectory. In the ``items_handover_place`` example, it defines graph scopes such as ``right_arm``, ``left_arm``, ``left_eef``, and ``right_eef``, together with affordance nodes and IK-based transitions.

In practice, this file is what turns a task from “a simulated scene” into “a data generator”. Without it, the launcher can build the environment, but the task cannot produce expert actions.

Step 3: Build the Environment
-----------------------------

The launcher parses command-line arguments, loads ``gym_config.json`` and ``action_config.json``, converts them into environment configuration objects, and creates the environment instance:

.. literalinclude:: ../../../embodichain/lab/scripts/run_env.py
   :language: python
   :start-at: def cli():
   :end-at:     main(args, env, gym_config)

This means the runtime inputs of the whole data-generation pipeline are simply the task config files plus launcher arguments.

Step 4: Generate and Execute Expert Actions
-------------------------------------------

The launcher first asks the task to generate an expert action sequence, then executes each action with ``env.step()``:

.. literalinclude:: ../../../embodichain/lab/scripts/run_env.py
   :language: python
   :start-at: def generate_and_execute_action_list(env, idx, debug_mode, **kwargs):
   :end-at: return True

This function highlights the core data-generation loop:

1. Call ``create_demo_action_list()`` from the environment.
2. Validate that the returned action list is not empty.
3. Execute each action with ``env.step(action)``.
4. Let the environment and dataset manager record the rollout automatically.

For internal users, this is the most important handoff point: the task code is responsible for producing a valid scripted trajectory, while the launcher is responsible for executing and recording it.

Step 5: Validate and Regenerate Failed Rollouts
-----------------------------------------------

The launcher also handles invalid trajectories safely:

.. literalinclude:: ../../../embodichain/lab/scripts/run_env.py
   :language: python
   :start-at: def generate_function(
   :end-at: return True

If a generated action sequence is invalid, the launcher resets the environment with ``save_data=False`` so that broken rollouts are discarded instead of being written into the dataset.

Step 6: Loop Over Episodes
--------------------------

The top-level ``main()`` function runs offline generation for the configured number of episodes:

.. literalinclude:: ../../../embodichain/lab/scripts/run_env.py
   :language: python
   :start-at: def main(args, env, gym_config):
   :end-at:     _, _ = env.reset()

In practice, the most commonly used command-line arguments are:

- **--gym_config**: Path to the task JSON configuration.
- **--action_config**: Optional path to the action-bank configuration.
- **--num_envs**: Number of environments to run in parallel.
- **--device**: Simulation device, such as ``cpu`` or ``cuda``.
- **--arena_space**: Arena size used by the simulation manager.
- **--headless**: Run without GUI for faster generation.
- **--enable_rt**: Enable ray tracing for higher-quality visual observations.
- **--filter_dataset_saving**: Disable dataset saving for debugging.

Using a Project Wrapper
-----------------------

If your tasks, managers, or action banks live outside the core EmbodiChain package, create a thin wrapper launcher that:

1. Imports your task package so custom environments are registered.
2. Extends ``DEFAULT_MANAGER_MODULES`` with your custom manager modules.
3. Delegates execution to ``embodichain.lab.scripts.run_env.main``.

This pattern keeps the data-generation logic in the official launcher while allowing project-specific extensions.

In project repositories such as ``Embodied_Challenge``, the wrapper typically imports the project package, extends ``DEFAULT_MANAGER_MODULES``, and then forwards execution to the official launcher. This keeps the rollout logic centralized while still allowing task-specific managers and action functors.

The Code Execution
~~~~~~~~~~~~~~~~~~

To run data generation with the official launcher:

.. code-block:: bash

   python -m embodichain.lab.scripts.run_env \
       --gym_config path/to/gym_config.json \
       --action_config path/to/action_config.json \
       --arena_space 5.0 \
       --headless \
       --enable_rt

If you maintain a project-specific wrapper for custom tasks, run that wrapper with the same arguments instead.

Outputs
~~~~~~~

After successful execution, completed episodes are saved under the configured ``save_path``. A LeRobot dataset typically contains:

- **data/**: Recorded action and state data.
- **videos/**: Camera observations saved as videos when ``use_videos=True``.
- **meta/**: Dataset metadata such as task information and robot description.

Dataset folders are automatically numbered, which makes it easy to run repeated generations without overwriting previous results. In the current implementation, the recorder also auto-increments dataset names and writes dataset metadata during creation.

In a practical workflow, the output of this stage is the synthesized dataset itself. Later training scripts should consume these saved LeRobot episodes rather than regenerating trajectories every time.

Best Practices
~~~~~~~~~~~~~~

- **Keep the config pair together**: Always version ``gym_config.json`` and ``action_config.json`` together for a task.
- **Use valid scripted policies**: Make sure ``create_demo_action_list()`` returns executable trajectories for the current scene.
- **Enable ``use_videos`` for visual tasks**: This is especially useful for downstream vision-based training.
- **Use ``--headless`` for throughput**: Disable the GUI when generating large datasets.
- **Use ``--enable_rt`` when image quality matters**: Ray tracing improves realism for camera observations.
- **Use ``--filter_dataset_saving`` for debugging**: This is useful when you need to inspect task logic without writing datasets.
- **Discard invalid rollouts**: Keep the default validation logic so failed trajectories are not saved.
