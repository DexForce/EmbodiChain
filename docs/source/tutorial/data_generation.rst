.. _tutorial_data_generation:

Expert Data Generation
======================

.. currentmodule:: embodichain.lab.gym

EmbodiChain supports two expert-authoring paradigms for synthetic demonstration
data:

1. **Handwritten trajectories** implement the expert in a registered Python
   task, using ``MotionGenerator`` directly or composing reusable Atomic Skills.
2. **Task Program** declares the task flow in typed YAML and lets the configured
   environment integration lower Semantic Calls to the same Atomic Skills.

The two paradigms differ only in how the expert plan is authored. Both produce
lazy :class:`~embodichain.lab.gym.envs.DemoSegment` objects, and both use the
same Gym executor, ``env.step()`` path, validation rules, dataset manager, and
transactional commit boundary.

Choose an Expert-Authoring Paradigm
-----------------------------------

.. list-table:: Handwritten trajectories and Task Program
   :header-rows: 1
   :widths: 19 38 43

   * - Concern
     - Handwritten trajectory
     - Task Program
   * - Authoring surface
     - A registered Python task module implementing
       ``create_demo_segments()``.
     - ``program.yaml`` plus a trusted integration, reusable physical
       environment, embodiment, execution policy, and runnable task deployment.
   * - Planning level
     - Call ``MotionGenerator`` for task-specific waypoints, or construct typed
       ``ActionInvocation`` values for ``AtomicActionEngine``.
     - Declare Pick, Place, HandOver, registered Semantic Calls, control flow,
       post-policies, and validators; the adapter lowers them to Atomic Skills.
   * - Best fit
     - Custom geometry, unusual control logic, experimental skills, or logic
       that needs unrestricted Python.
     - Reusable object-centric workflows whose task intent should remain
       independent of a particular robot and simulator assembly.
   * - Responsibility
     - The task owns planning, command assembly, lazy segment boundaries, and
       physical validation.
     - The program owns intent; trusted integration and deployment components
       own physical identities, resources, policies, and services.
   * - Environment entry
     - A task-specific registered ``EmbodiedEnv`` subclass.
     - A supported configuration-defined task can use the common
       ``EmbodiedEnv`` without a task-specific Python module.

Use handwritten Python while the behavior itself is still changing rapidly or
does not have a reusable semantic contract. Use Task Program when the behavior
can be expressed as stable semantic operations and should be portable across
compatible embodiments. A common progression is to prototype in Python, move
reusable motion behavior into Atomic Skills, and then expose stable task flows
through Task Program.

The Shared Rollout Contract
---------------------------

The rollout path is deliberately shared:

.. code-block:: text

   registered Python task                 configured Task Program
   create_demo_segments()                 compile + Gym bridge
                \                           /
                 +---- Iterable[DemoSegment]
                                |
                     execute_demo_episode()
                                |
                         env.step(action)
                                |
                  observation/action buffers
                                |
               reset(commit) or reset(discard)
                                |
                       LeRobot dataset

A :class:`~embodichain.lab.gym.envs.DemoSegment` contains an action iterable
and may also contain a stable name, target, instruction, JSON-compatible
metadata, and a validator. Its actions may be lazy: the next segment can be
planned only after the previous segment has executed and changed the scene.

The expert planner must **not** call ``env.step()`` itself. The common executor
owns stepping so observations, actions, terminal signals, segment annotations,
and dataset frames stay causally aligned.

Paradigm 1: Handwritten Trajectories
------------------------------------

A handwritten expert lives beside its registered task entry point, for example:

* ``embodichain_tasks/embodichain_tasks/manipulation/tableware/stack_blocks_two.py``;
* ``embodichain_tasks/embodichain_tasks/manipulation/tableware/blocks_ranking_rgb.py``.

Both examples build a ``MotionGenerator`` backed by TOPPRA and pass it to an
``AtomicActionEngine``. The engine then plans reusable PickUp and Place
invocations while the task retains control over scene queries, command timing,
segment metadata, and success checks.

Direct Motion Generator or Atomic Skills
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are two abstraction levels, not two unrelated planning engines:

* Use ``MotionGenerator.generate()`` directly when the task already owns the
  joint or Cartesian waypoints and only needs a planned joint trajectory. Wrap
  ``PlanResult.positions`` as a ``DemoSegment`` action iterable and combine
  ``PlanResult.success`` with a physical validator. See :doc:`motion_gen` for
  the complete planning API.
* Use ``AtomicActionEngine`` when the behavior matches reusable skills such as
  PickUp, Place, MoveEndEffector, Pour, or HandOver. The engine owns the shared
  motion generator, command profiles, typed goals, and projected semantic
  context. See :doc:`atomic_actions` for the action contracts.

When using ``MotionGenerator`` directly, its planned positions are ordered for
the selected ``control_part``. The actions yielded to the environment must
match its active action space. If the environment controls more joints than
the planned arm, merge the arm plan with hold or gripper commands before
yielding it; do not silently rely on incompatible dimensions.

Build the planning services
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two-block stacking task demonstrates the recommended Atomic Skills setup:

.. dropdown:: Motion Generator and AtomicActionEngine setup
   :icon: code

   .. literalinclude:: ../../../embodichain_tasks/embodichain_tasks/manipulation/tableware/stack_blocks_two.py
      :language: python
      :start-at:     def _initialize_atomic_actions(self) -> None:
      :end-before:     def create_demo_segments(self, **kwargs: Any) -> tuple[DemoSegment]:
      :linenos:

The task constructs the motion generator once, declares command profiles for
the gripper, and reuses the engine for every episode. Object semantics are
explicit inputs to the skills rather than being inferred from simulator names.

Return semantic demonstration segments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``create_demo_segments()`` is the preferred handwritten expert API:

.. literalinclude:: ../../../embodichain_tasks/embodichain_tasks/manipulation/tableware/stack_blocks_two.py
   :language: python
   :start-at:     def create_demo_segments(self, **kwargs: Any) -> tuple[DemoSegment]:
   :end-before:     def _plan_stack(
   :linenos:

This segment keeps three outcomes separate:

* ``actions`` is the controller-command stream consumed by the common runner;
* ``planning_success`` records whether PickUp and Place were planned; and
* ``validator`` checks both planning success and the physical stack after all
  commands and settling actions have executed.

The task creates typed PickUp and Place requests and threads the projected
held-object state from the first plan into the second:

.. dropdown:: Compile the PickUp and Place trajectory
   :icon: code

   .. literalinclude:: ../../../embodichain_tasks/embodichain_tasks/manipulation/tableware/stack_blocks_two.py
      :language: python
      :start-at:         pick_compiled = self._action_engine.compile(
      :end-before:     def _insert_grasp_hold(self, trajectory: torch.Tensor) -> torch.Tensor:
      :linenos:

For a multi-object episode, yield segments lazily. The
``BlocksRankingRGBEnv.create_demo_segments()`` example yields the red-block
segment first, then queries the updated reference-block pose before planning
the blue-block segment. This avoids planning later subtasks against stale
scene state.

Legacy tasks implementing ``create_demo_action_list()`` remain supported and
are wrapped as one segment named ``legacy``. New tasks should implement
``create_demo_segments()`` so subtask boundaries, language, metadata, and
validators remain explicit.

Try the handwritten expert
~~~~~~~~~~~~~~~~~~~~~~~~~~

The shipped stacking config currently defines the scene and rollout but does
not configure a dataset recorder. First smoke-test the expert without writing
data:

.. code-block:: bash

   embodichain run-env \
       --gym_config embodichain_tasks/configs/tasks/manipulation/tableware/stack_blocks_two/env.json \
       --headless \
       --filter_dataset_saving \
       --max_episodes 1

To collect it, copy that config, add the recorder from
`Configure Dataset Recording`_, and run the copied config without
``--filter_dataset_saving``:

.. code-block:: bash

   embodichain run-env \
       --gym_config path/to/stack_blocks_two_recording.json \
       --headless \
       --max_episodes 5

Paradigm 2: Task Program
------------------------

Task Program replaces the task-specific expert method with declarative task
intent. It does not introduce a second rollout or recording API: the configured
adapter compiles the program, creates lazy demonstration segments, and returns
them to the same executor used by handwritten tasks.

The current Pour Water example is a complete configuration-defined task:

* ``env.yaml`` owns the physical scene, environment values, and dataset
  recorder, without any Task Program fields;
* ``task.cobotmagic.yaml`` is the runnable deployment that selects the
  environment, Task Program, embodiment, and execution policy;
* ``program.yaml`` owns embodiment-independent intent and targets;
* ``integration.yaml`` owns task-specific contracts, scene binding, semantic
  defaults, action options, and allowlisted runtime services;
* the embodiment component owns the robot, sensors, endpoints, and compatible
  skill profile.

Compose the deployment
~~~~~~~~~~~~~~~~~~~~~~

The runnable deployment is intentionally thin:

.. literalinclude:: ../../../embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic.yaml
   :language: yaml
   :linenos:

Its selected ``env.yaml`` is reusable outside Task Program because it contains
only physical simulation and ordinary Gym values:

.. dropdown:: Pour Water reusable environment
   :icon: code

   .. literalinclude:: ../../../embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/env.yaml
      :language: yaml
      :linenos:

During ``config_to_cfg()``, component paths are resolved relative to
``task.cobotmagic.yaml``. The resolver expands the physical environment and
embodiment, checks scene-binding UIDs and semantic contracts, and binds trusted
provider identities into the otherwise embodiment-independent program.

Declare the semantic workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The program describes Pick, transport, Pour, and Place without importing
Python callables or naming simulator joints:

.. dropdown:: Pour Water Task Program
   :icon: code

   .. literalinclude:: ../../../embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task_program/program.yaml
      :language: yaml
      :linenos:

The trusted integration maps ``bottle`` and ``cup`` to physical scene objects,
selects the primary manipulator, configures action options, and allowlists the
registered transport and pour lowerers:

.. dropdown:: Pour Water integration
   :icon: code

   .. literalinclude:: ../../../embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task_program/integration.yaml
      :language: yaml
      :linenos:

The program remains provider-independent. Simulation UIDs, grasp affordances,
fixed object-to-end-effector transforms, and live target resolution stay in the
trusted integration. For a detailed deployment walkthrough, see
:doc:`task_program`; for constructing and compiling the language directly in
Python, see :doc:`task_program_python`.

Run the configured Task Program
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pour Water already configures ``LeRobotRecorder`` in ``env.yaml``, so its
runnable deployment can record directly:

.. code-block:: bash

   embodichain run-env \
       --gym_config embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/task.cobotmagic.yaml \
       --headless \
       --max_episodes 1

The Task Program bridge plans and yields actions lazily. It never calls
``env.step()``; stepping, annotations, validation, commit, retry, and discard
remain owned by the shared environment rollout.

Configure Dataset Recording
---------------------------

The expert paradigm and recorder configuration are independent. Add a dataset
functor under ``env.dataset`` in an inline Gym config or in a reusable
``env.yaml`` environment component:

.. code-block:: yaml

   max_episodes: 5
   max_episode_steps: 600

   env:
     dataset:
       lerobot:
         func: LeRobotRecorder
         mode: save
         save_failed_episodes: false
         params:
           save_path: outputs/lerobot/expert_demos
           robot_meta:
             robot_type: CobotMagic
           instruction:
             lang: Pick and place the object
           extra:
             scene_type: tabletop
             task_description: pick_and_place
             data_type: sim
           use_videos: true

Important fields are:

* ``max_episodes`` is the exact number of persisted per-environment episodes,
  not the number of vector batches.
* ``max_episode_steps`` must exceed the longest valid expert execution,
  including gripper holds and settling actions.
* ``save_failed_episodes`` belongs beside ``func`` and ``mode``. It defaults to
  ``false``; when enabled, a failed or truncated attempt with recorded frames
  is committed with ``success=false`` metadata and counts toward
  ``max_episodes``.
* ``params.save_path`` is the parent directory for auto-numbered datasets. If
  omitted, the default is ``~/.cache/embodichain_datasets`` or the value of
  ``EMBODICHAIN_DATASET_ROOT``.
* ``params.use_videos`` controls RGB dataset videos. It has an effect only when
  image observations from configured sensors are present.
* Dataset frequency is derived from ``env.step_dt`` and must be an integer
  number of frames per second for LeRobot.

The real Pour Water recorder block can be inspected directly:

.. literalinclude:: ../../../embodichain_tasks/configs/tasks/manipulation/tableware/pour_water/env.yaml
   :language: yaml
   :start-at: dataset:
   :end-at: control_parts: [left_arm, left_eef, right_arm, right_eef]
   :linenos:

Execution, Validation, and Persistence
--------------------------------------

Without ``--preview`` or ``--replay``, ``embodichain run-env`` performs offline
data generation:

1. Resolve ``create_demo_segments()`` from the handwritten task or configured
   Task Program bridge.
2. Execute every yielded action through ``env.step(action)`` and record the
   resulting transition.
3. Run the segment validator after its action iterable is exhausted.
4. Check episode termination and final task success.
5. Commit selected environment rows with an explicit reset, or discard the
   attempt with ``reset(options={"save_data": False})``.

Failed attempts are discarded and retried by default, up to
``demo_max_attempts`` (default: 3). Empty plans and exceptions are always
discarded because they do not form a complete dataset transaction. With
``save_failed_episodes: true``, a failed or truncated attempt is retained only
when every selected row contains recorded frames.

``num_envs`` controls collection parallelism. If ``max_episodes=10`` and
``num_envs=4``, the runner uses three vector batches and commits only two rows
from the final batch, so it never overshoots the requested episode count.

An episode is the complete task; a segment is one semantic subtask. Do not use
``generate_function(num_traj=...)`` to repeat subtasks: direct callers may pass
only ``None`` or ``1``. Yield multiple ``DemoSegment`` objects instead so the
task owns their order, live-state dependencies, and validation.

Useful modes and options are:

* ``--headless`` disables the GUI for collection throughput.
* ``--preview`` opens interactive inspection and does not save a dataset.
* ``--filter_dataset_saving`` executes the expert while suppressing structured
  dataset writes.
* ``--num_envs`` overrides collection parallelism.
* ``--max_episodes`` overrides the configured episode target.

See :doc:`/guides/run_env` for preview, dataset recording, debug video,
trajectory recording, and replay modes, and :doc:`/guides/cli` for the complete
argument list.

Recorded Data
-------------

``LeRobotRecorder`` creates an auto-numbered dataset directory containing:

* ``data/`` for Parquet action, state, and annotation features;
* ``videos/`` for RGB observations when ``use_videos`` is enabled;
* ``meta/`` for LeRobot metadata, task/subtask mappings, and EmbodiChain episode
  metadata.

The primary fields include ``observation.state``, ``action``, and
``observation.images.{sensor_name}``. Segment-aware episodes additionally
record ``subtask_index`` and ``annotation.segment_*`` boundaries, plus terminal
and truncation annotations. Depth and segmentation observations have their own
numeric or configured sidecar representation; see
:doc:`/overview/gym/dataset_functors` for the complete schema.

.. _tutorial_data_generation_preview:

Inspect Recorded LeRobot Data
-----------------------------

Use EmbodiChain's structural preview on the parent directory of auto-numbered
datasets:

.. code-block:: bash

   embodichain preview_lerobot_data \
       outputs/lerobot/expert_demos \
       --latest \
       --episode 0

For Pour Water without an explicit ``save_path``, use the default parent:

.. code-block:: bash

   embodichain preview_lerobot_data \
       ~/.cache/embodichain_datasets \
       --latest \
       --episode 0 \
       --expect-segments 1

The command validates frame and timestamp continuity, one episode-level task,
subtask mappings, contiguous segment ranges, terminal annotations, and the
EmbodiChain metadata sidecar. ``--expect-segments`` is an optional assertion;
omit it when the expert's segment count is data-dependent.

Use LeRobot's ``lerobot-dataset-viz`` with the exact auto-numbered dataset
directory when interactive Rerun plots and camera playback are needed. The
EmbodiChain preview focuses on structure and annotations; it does not render
images or time-series plots.

Best Practices
--------------

* Keep planning and stepping separate: experts yield actions; the runner calls
  ``env.step()``.
* Validate physical outcomes from live simulator state, not only planner
  success or projected semantic state.
* Yield later subtasks lazily when they depend on object poses changed by
  earlier segments.
* Smoke-test with ``--filter_dataset_saving`` before a long collection run,
  then inspect one committed episode before scaling ``num_envs``.
* Keep each task's ``env.yaml``, ``task.<embodiment>.yaml``, and
  ``task_program/{program,integration}.yaml`` together so physical UIDs,
  contracts, and canonical IDs stay aligned. Keep reusable embodiment and
  execution-policy components under ``configs/components/``.
* Move reusable motion behavior into Atomic Skills instead of copying
  task-local trajectory logic across Python tasks or registered Semantic Calls.
