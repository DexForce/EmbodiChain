.. _tutorial_data_generation:

Data Generation
===============

.. currentmodule:: embodichain.lab.gym

This tutorial shows you how to generate synthetic expert demonstration datasets using EmbodiChain's built-in environment launcher and dataset manager. You will learn how to configure LeRobot recording in a JSON file, how the ``run_env.py`` script builds environments and executes expert trajectories, and how completed episodes are automatically saved to disk.

Overview
~~~~~~~~

EmbodiChain provides a built-in data generation workflow for imitation learning tasks:

- **Gym Configuration**: Describes the scene, robot, sensors, randomization events, observations, and dataset recorder.
- **Action Configuration**: Describes how the task-specific expert trajectory should be generated.
- **Environment Launcher**: Builds the environment directly from configuration files.
- **Expert Policy**: Each task provides ``create_demo_action_list()`` to generate a scripted trajectory.
- **Dataset Manager**: Records observation-action pairs during ``env.step()``.
- **LeRobotRecorder**: Converts completed episodes into LeRobot-compatible datasets, with optional video export.

What This Tutorial Records
--------------------------

This page documents the full path from task configuration to saved dataset:

1. Prepare a task ``gym_config.json``.
2. Prepare an ``action_config.json`` that controls expert action generation.
3. Launch the environment runner.
4. Let the task generate actions through ``create_demo_action_list()``.
5. Execute the actions with ``env.step()``.
6. Let the dataset manager automatically save completed episodes.

Example Task: Items Handover and Place
--------------------------------------

As a running example, we use an ``items_handover_place`` task. In a project repository, this usually comes with two files:

- ``configs/items_handover_place/gym_config.json`` The gym configuration defines the simulation scene and recording behavior.
- ``configs/items_handover_place/action_config.json`` The action configuration defines the expert action graph used to solve the task.

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

Important parameters are:

- **save_path**: Where the generated dataset will be written.
- **robot_meta**: Robot metadata such as robot type, control frequency, and active control parts.
- **instruction**: Task language instruction stored together with the dataset.
- **extra**: Additional metadata such as scene type and task description.
- **use_videos**: Whether camera observations should be stored as videos instead of raw images.
 
In the current implementation, ``LeRobotRecorder`` stores robot state and action features such as ``observation.qpos``, ``observation.qvel``, ``observation.qf``, ``action``, and camera images when sensors are present.

Step 2: Prepare the Action Configuration
----------------------------------------

The second input is the ``action_config.json`` file. This file defines the expert action graph used by the task. It is the main configuration entry for scripted trajectory generation. Take ``items_handover_place`` as example, the file is organized around ``scope``, ``node``, ``edge``, and ``sync``.

**Scope Configuration**

.. literalinclude:: ../../../configs/items_handover_place/action_config.json
   :language: json
   :lines: 1-40

This section defines the controllable sub-graphs used by the task:

- **Control groups**: Scopes such as ``right_arm``, ``left_arm``, ``right_eef``, and ``left_eef`` separate arm motion from gripper motion.
- **Initialization**: Each scope specifies how its initial state is obtained, such as ``current_qpos`` or ``given_qpos``.
- **Action dimensions**: The ``dim`` field defines the action dimension for each scope, for example 6 DoF for an arm and 1 DoF for a gripper.

**Node Configuration**

The following excerpts show representative node entries from the real ``action_config.json``:

.. literalinclude:: ../../../configs/items_handover_place/action_config.json
   :language: json
   :lines: 304-364


This section defines how key poses or joint targets are generated:

- **Affordance-driven targets**: Nodes typically start from an affordance source such as an object pose or a previously generated pose.
- **Pose processing**: Intermediate transforms such as offsets and rotations are applied before motion targets are finalized.
- **IK conversion**: For arm scopes, nodes often solve inverse kinematics to convert a pose target into a valid joint target.
- **Cross-scope reuse**: A node in one scope can depend on data produced by another scope, which is common in dual-arm tasks.

**Edge Configuration**

.. literalinclude:: ../../../configs/items_handover_place/action_config.json
   :language: json
   :lines: 1005-1017

.. literalinclude:: ../../../configs/items_handover_place/action_config.json    
   :language: json
   :lines: 1141-1150

This section defines executable transitions between nodes:

- **Motion edges**: Entries such as ``right_up_to_handover`` use ``plan_trajectory`` to move an arm between two node states.
- **Gripper edges**: Entries such as ``right_open0`` use functions like ``execute_open`` or ``execute_close`` to generate gripper actions.
- **Durations**: The ``duration`` field controls how many simulation steps each transition occupies.
- **Execution binding**: The ``name`` field selects which execution function is used for that transition.

**Synchronization**

.. literalinclude:: ../../../configs/items_handover_place/action_config.json
   :language: json
   :lines: 1191-1210

This section defines dependencies between sub-actions:

- **Temporal ordering**: The ``sync`` block enforces that some actions can only start after other actions finish.
- **Cross-scope coordination**: Dependencies commonly connect arm motion and gripper actions across different scopes.
- **Multi-stage execution**: This is how multiple independently configured primitives become one coherent expert rollout.

Together, ``scope`` defines what can be controlled, ``node`` defines target states, ``edge`` defines executable transitions, and ``sync`` defines ordering constraints. This is the core configuration structure that ``create_demo_action_list()`` consumes when generating an expert rollout.

Note: Action bank is not the only way to generate action demos. Depending on the task design, trajectories can also be produced by other scripted generation methods.

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

This function highlights the data-generation loop:

1. Call ``create_demo_action_list()`` from the environment.
2. Validate that the returned action list is not empty.
3. Execute each action with ``env.step(action)``.
4. Let the environment and dataset manager record the rollout automatically.

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

Commonly used command-line arguments are:

- **--gym_config**: Path to the task JSON configuration.
- **--action_config**: Optional path to the action-bank configuration.
- **--num_envs**: Number of environments to run in parallel.
- **--device**: Simulation device, such as ``cpu`` or ``cuda``.
- **--arena_space**: Arena size used by the simulation manager.
- **--headless**: Run without GUI for faster generation.
- **--enable_rt**: Enable ray tracing for higher-quality visual observations.
- **--filter_dataset_saving**: Disable dataset saving for debugging.

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
