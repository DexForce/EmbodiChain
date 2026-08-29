.. _tutorial_data_generation:

Data Generation
===============

.. currentmodule:: embodichain.lab.gym

This tutorial shows how to generate synthetic expert demonstration datasets using EmbodiChain's built-in environment rollout and dataset manager. You will learn how to configure LeRobot recording in a gym config file (``.json``, ``.yaml``, or ``.yml``), how ``run_env.py`` builds an environment from configuration files, and how completed episodes are automatically saved to disk.

Overview
~~~~~~~~

EmbodiChain provides a built-in data generation workflow for imitation-learning and manipulation tasks:

- **Gym Configuration**: Describes the simulated environment, robot, sensors,
  managers, dataset recorder, rollout settings, and Task Program resource paths.
- **Task Program**: Describes a declarative semantic workflow and selects its
  configured scene, robot-skill profile, and runtime policy.
- **Task Program Integration**: Declares trusted scene/profile bindings,
  policies, and allowlisted live services in a sibling YAML file.
- **Environment Rollout**: Builds the environment directly from configuration files and executes offline generation.
- **Expert Policy**: A supported task loads and compiles its Task Program;
  custom environments may still provide ``create_demo_segments()`` directly.
- **Dataset Manager**: Records observation-action pairs during ``env.step()``
  and transactionally commits selected successful or configured failed rows.
- **LeRobotRecorder**: Converts completed episodes into LeRobot-compatible datasets, with optional video export.

What This Tutorial Records
--------------------------

This page documents the full path from task configuration to saved dataset:

1. Prepare a task gym config (e.g. ``gym_config.json`` or ``gym_config.yaml``).
2. Reference a task-local Task Program and integration from the gym config
   when scripted demonstrations are required.
3. Launch the environment rollout with ``run-env``.
4. Let the dataset manager automatically save completed episodes.

Example Task
------------

As a concrete example, this tutorial uses a configuration-defined Task Program
task shipped in the repository:

- ``embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/env.json``
  defines the simulation environment, dataset behavior, and Task Program
  resource paths.
- ``embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task_program/program.yaml``
  declares the semantic pick, transport, pour, and place workflow.
- ``embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task_program/integration.yaml``
  declares the trusted scene/profile bindings and live services.

The Code
~~~~~~~~

The tutorial corresponds to the ``run_env.py`` script in ``embodichain/lab/scripts``.

.. dropdown:: Code for run_env.py
   :icon: code

   .. literalinclude:: ../../../embodichain/lab/scripts/run_env.py
      :language: python
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

The rollout script builds the environment from configuration, generates expert trajectories, executes them step by step, and relies on the dataset manager to auto-save valid episodes.

Step 1: Prepare the Task Configuration
--------------------------------------

The first input to the pipeline is the task gym config file. In the example below, the same file contains rollout settings, scene randomization, observations, dataset recording, and robot or sensor definitions.

The rollout settings include the episode count:

.. literalinclude:: ../../../embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/env.json
   :language: json
   :start-at: "max_episodes":
   :end-before: "env":

The dataset-related part looks like this:

.. literalinclude:: ../../../embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/env.json
   :language: json
   :start-at: "dataset": {
   :end-before: "control_parts":

Important parameters are:

- **max_episodes**: Exact number of persisted per-environment episodes. It is
  not the number of vector batches; a final partial batch commits only the rows
  still needed to reach this count.
- **num_envs**: Maximum number of episodes generated concurrently in one
  vector batch.
- **max_episode_steps**: Maximum number of environment steps per episode.
- **dataset.lerobot.save_failed_episodes**: When ``true``, persist failed or
  truncated attempts that contain frames. It belongs beside ``func`` and
  ``mode``, not inside ``params``. A saved failure counts toward
  ``max_episodes``. The default is ``false``.
- **dataset.lerobot.params.robot_meta**: Robot identity metadata such as robot type. Dataset frequency is derived from the environment timestep.
- **dataset.lerobot.params.instruction**: Task language instruction stored together with the dataset.
- **dataset.lerobot.params.extra**: Additional metadata such as scene type and task description.
- **dataset.lerobot.params.use_videos**: Whether camera observations should be stored as videos.
- **env.control_parts**: Controlled robot parts in the environment.


``LeRobotRecorder`` stores robot state and action features following the LeRobot
format: ``observation.state`` for joint positions, ``action`` for applied
actions, and ``observation.images.{sensor_name}`` for RGB camera images. The
episode-level instruction remains in LeRobot's ``task`` field. Segment-aware
demonstrations additionally store a per-frame ``subtask_index`` and the
corresponding descriptions in ``meta/subtasks.parquet``; precise boundaries
and terminal state are stored in ``annotation.episode_step``,
``annotation.segment_id``, ``annotation.segment_step``,
``annotation.segment_start``, ``annotation.segment_end``,
``annotation.terminated``, and ``annotation.truncated``. The internal expert
``valid`` mask selects the real per-environment prefix but is not exported as
an ``annotation.valid`` column. When a camera also produces
depth or segmentation data, the recorder preserves those arrays: masks are
always stored as exact numeric features under
``observation.mask.{sensor_name}``, while depth is stored either as a numeric
feature (``observation.depth.{sensor_name}``, the default) or as compressed
``gray12le``/HEVC sidecar videos when ``dataset.lerobot.params.depth_video.enable``
is set (issue #424, Path A). Stereo-camera keys use the ``_right`` suffix. The
``use_videos`` option applies only to RGB images; see
:doc:`/overview/gym/dataset_functors` for the depth sidecar configuration.

Step 2: Prepare the Task Program
----------------------------------

The gym config references the ``task_program/`` directory once through
``task_program_dir``. The loader reads its fixed ``integration.yaml`` first and
then validates ``program.yaml`` against that trusted provider assembly. The
program uses registered scene identities and robot resources instead of
embedding simulation code:

.. dropdown:: Task Program for Pour Water
   :icon: code

   .. literalinclude:: ../../../embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task_program/program.yaml
      :language: yaml
      :linenos:

The sibling integration file owns scene bindings, the robot profile, policy
presets, and allowlisted live-service declarations:

.. dropdown:: Task Program Integration for Pour Water
   :icon: code

   .. literalinclude:: ../../../embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task_program/integration.yaml
      :language: yaml
      :linenos:

The workflow first picks ``bottle`` with ``right_manipulator``. It then invokes
the registered held-object transport target, whose pose is resolved relative to
the cup at planning time, performs a signed 75-degree pour, and places the
bottle at its return target. The segment waits for the bottle and cup to settle
and validates the final bottle position before committing the episode.

Step 3: Launch the Environment Rollout
--------------------------------------

The rollout script parses command-line arguments, loads the gym config and its
task-local Task Program, creates the environment instance, and then runs
offline rollout for ``max_episodes`` episodes:

.. literalinclude:: ../../../embodichain/lab/scripts/run_env.py
   :language: python
   :start-at: def cli(
   :end-at:     main(args, env, gym_config)

Each rollout obtains demonstration segments from the compiled Task Program
or a custom environment's ``create_demo_segments()`` implementation. The runner
validates and executes every action with ``env.step(action)``. By default an
invalid rollout is discarded with ``save_data=False`` and retried. If the
dataset functor sets ``save_failed_episodes: true``, a failed or truncated
attempt with recorded frames is committed with ``success=false`` metadata and
counts toward ``max_episodes``. Empty plans and exceptions are still discarded.

``generate_function(num_traj=...)`` no longer controls how many sub-actions
belong to an episode. Direct callers may pass only ``None`` or ``1``; larger
values raise ``ValueError``. Represent repeated subtasks by yielding multiple
``DemoSegment`` objects from ``create_demo_segments()`` instead.

The recommended CLI entrypoint is:

.. code-block:: bash

   embodichain run-env \
       --gym_config embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/env.json \
       --headless

For interactive inspection, you can use preview mode: replace ``--headless`` with ``--preview``.
When ``--preview`` is enabled, the script opens the environment in an interactive debugging mode. This mode is for inspection and does not save datasets.

For a detailed comparison of preview, structured dataset recording, debug-video
recording, trajectory recording, and the three replay modes, see
:doc:`Run Environment </guides/run_env>`.


Useful CLI arguments:

- **--gym_config**: Path to the task config file (``.json``, ``.yaml``, or ``.yml``).
- **--num_envs**: Number of environments to run in parallel. ``run-env`` uses
  as many vector batches as needed and trims the last commit so that the
  persisted episode count equals ``max_episodes``.
- **--device**: Simulation device, such as ``cpu`` or ``cuda``.
- **--headless**: Run without GUI for faster generation.
- **--enable_rt**: Enable ray tracing for higher-quality visual observations.
- **--preview**: Launch the environment in interactive preview mode.
- **--filter_dataset_saving**: Disable dataset saving for debugging.

For the complete CLI argument list, see :doc:`CLI Reference </guides/cli>`.

Outputs
~~~~~~~

After execution, committed episodes are saved under the configured dataset
root. They are successful episodes by default and may include explicitly
configured failed episodes. A LeRobot dataset typically contains:

If no explicit save path is provided and ``EMBODICHAIN_DATASET_ROOT`` is not set, ``LeRobotRecorder`` uses ``~/.cache/embodichain_datasets`` as the default dataset root.

- **data/**: Recorded action and state data.
- **videos/**: Camera observations saved as videos when ``use_videos=True``.
- **meta/**: Dataset metadata such as task information and robot description.

Dataset folders are automatically numbered, which makes it easy to run repeated generations without overwriting previous results.

In a practical workflow, the output of this stage is the synthesized dataset itself. Later training scripts typically consume these saved LeRobot episodes instead of regenerating trajectories each time.


Repeated Pick-and-Place Task Program Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The repository includes a complete repeated pick-and-place example at
``embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/env.json``. It uses
the specified UR5 robot and parallel gripper to pick up the same cube and place
it three times in one episode. Its task-local Task Program is selected by the
gym config, so no second CLI config argument is required.

Each pick/place cycle is one lazy demonstration segment. After placing the
cube, the task waits for its free-fall motion to settle. Only then does it read
the cube's measured pose and plan the next pickup. This avoids assuming that
the previous release position is still the cube's current position.

Generate one episode with:

.. code-block:: bash

   embodichain run-env \
       --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/env.json \
       --headless \
       --device cuda \
       --max_episodes 1

Use ``--device cpu`` instead when the simulator is configured for CPU
execution. The config writes an auto-numbered dataset below:

.. code-block:: text

   outputs/lerobot/task_program/
   `-- ur5_task_program_repeated_pick_place_NNN/

The example has no configured camera sensor and sets ``use_videos`` to
``false``. Its dataset therefore contains robot state, action, task, subtask,
and segment annotations, but no RGB video. Add a sensor and record its image
observation when visual playback of the cube is required.


.. _tutorial_data_generation_preview:

Inspect Recorded LeRobot Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EmbodiChain provides a terminal-oriented structural preview, while LeRobot
provides an interactive Rerun visualization. They are complementary:

.. list-table:: Dataset preview tools
   :header-rows: 1
   :widths: 23 37 40

   * - Tool
     - Best for
     - Important limitation
   * - EmbodiChain terminal preview
     - Verifying frame/timestamp continuity, episode task, subtask mapping,
       segment boundaries, and the EmbodiChain sidecar.
     - Prints summaries and validation results; it does not render images or
       time-series plots.
   * - LeRobot ``lerobot-dataset-viz``
     - Interactively scrubbing standard camera, state, and action streams in
       Rerun.
     - LeRobot 0.4.4 does not plot EmbodiChain's ``subtask_index`` or
       ``annotation.segment_*`` fields.

EmbodiChain terminal preview
----------------------------

Use the parent output directory with ``--latest`` to inspect the most recently
generated auto-numbered dataset:

.. code-block:: bash

   embodichain preview_lerobot_data \
       outputs/lerobot/task_program \
       --latest \
       --episode 0 \
       --expect-segments 3

The command loads the selected episode through LeRobot's official
``LeRobotDataset`` API, then checks:

- required state, action, task, subtask, and segment features;
- contiguous ``frame_index``, ``episode_index``, and
  ``annotation.episode_step`` values;
- timestamps against ``frame_index / fps``;
- one constant episode-level task;
- monotonic contiguous segment ranges, per-segment step counters, and exactly
  one start/end marker per segment;
- final-frame-only ``annotation.terminated`` and ``annotation.truncated``
  values, including agreement with the sidecar;
- stable ``subtask_index`` to description mappings; and
- agreement with ``meta/embodichain_episodes.jsonl`` when the sidecar exists.

A deliberately saved failed episode may still pass these structural checks.
The preview then reports ``Sidecar : success=False``; failure classification is
data, not a malformed-dataset error.

The CLI arguments are:

.. list-table:: EmbodiChain preview arguments
   :header-rows: 1
   :widths: 28 72

   * - Argument
     - Meaning
   * - ``dataset_root``
     - Exact LeRobot dataset root containing ``meta/info.json``. It may instead
       be a parent directory when ``--latest`` is supplied.
   * - ``--latest``
     - Select the newest direct child dataset below ``dataset_root``.
   * - ``--episode N``
     - Inspect episode index ``N``; defaults to episode 0.
   * - ``--expect-segments N``
     - Add an optional assertion that the episode contains exactly ``N``
       contiguous segments. It does not change or generate data. A mismatch
       makes validation fail; omit it when the segment count is not known.

A representative successful result ends with:

.. code-block:: text

   Segments:
     #0: frames [0, 304) (304), subtask_index=0
     #1: frames [304, 608) (304), subtask_index=1
     #2: frames [608, 912) (304), subtask_index=2
   Sidecar : success=True
   [PASS] Dataset structure and segment metadata are consistent.

The exact frame ranges can change with motion planning and settling time. Exit
status 0 means validation passed, status 1 means the dataset loaded but failed
one or more checks, and status 2 means the path, episode, or dataset could not
be loaded.

LeRobot official Rerun preview
------------------------------

The supported LeRobot 0.4.x release installs the official
``lerobot-dataset-viz`` command. Unlike the EmbodiChain preview, its ``--root``
must be the exact auto-numbered dataset directory:

.. code-block:: bash

   lerobot-dataset-viz \
       --repo-id DexForce/ur5_task_program_repeated_pick_place_000 \
       --root outputs/lerobot/task_program/ur5_task_program_repeated_pick_place_000 \
       --mode local \
       --episode-index 0 \
       --num-workers 0

Replace the ``_000`` suffix with the dataset directory produced by your run.
``--repo-id`` remains required for a local dataset and gives the recording its
identifier inside Rerun; supplying ``--root`` makes the CLI load the local
directory. The viewer displays one curve per dimension under ``state`` and
``action`` and displays camera streams when standard LeRobot image features
are present. For the camera-free example above, seeing only the eight UR5 and
gripper state curves plus the eight action curves is expected.

To create a portable Rerun recording without opening a viewer, add
``--save 1`` and ``--output-dir``:

.. code-block:: bash

   lerobot-dataset-viz \
       --repo-id DexForce/ur5_task_program_repeated_pick_place_000 \
       --root outputs/lerobot/task_program/ur5_task_program_repeated_pick_place_000 \
       --episode-index 0 \
       --num-workers 0 \
       --save 1 \
       --output-dir outputs/lerobot/previews

   rerun outputs/lerobot/previews/DexForce_ur5_task_program_repeated_pick_place_000_episode_0.rrd

``--save 1`` disables automatic viewer spawning and writes the ``.rrd`` file;
the second command opens that saved recording. You can validate the container
itself with ``rerun rrd verify path/to/recording.rrd``.

This repository constrains LeRobot to ``>=0.4.4,<0.5`` for Python 3.10 and
3.11 compatibility. Use ``lerobot-dataset-viz --help`` for the flags available
in the installed version; options documented only for later LeRobot releases
may not exist here. See the `official LeRobot dataset visualization guide
<https://huggingface.co/docs/lerobot/en/using_dataset_tools#dataset-visualization>`_
for the upstream workflow.

Best Practices
~~~~~~~~~~~~~~

- **Keep task-local files together**: Version ``env.json``,
  ``task_program/program.yaml``, and ``task_program/integration.yaml`` together
  so their scene, profile, and call IDs stay aligned.
- **Use valid scripted policies**: Preflight the Task Program against its
  configured scene and robot profile before long rollout runs.
- **Use ``--headless`` for throughput**: Disable the GUI when generating large datasets.
- **Use ``--preview`` and ``--filter_dataset_saving`` for debugging**: Inspect task logic without writing datasets.
- **Discard invalid rollouts**: Keep the default validation logic so failed trajectories are not saved.
