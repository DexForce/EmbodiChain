embodichain.lab.gym.envs.managers
==========================================

.. automodule:: embodichain.lab.gym.envs.managers

Overview
--------

Managers orchestrate collections of **functors** that run at specific points in
the environment step loop. Each manager owns a typed ``@configclass`` config
whose attributes are :class:`FunctorCfg` (or subclass) instances; at init the
manager resolves every ``func`` reference, validates argument signatures against
``params``, resolves :class:`SceneEntityCfg` targets to scene indices, and
groups functors by mode. The config attribute name becomes the functor's unique
identifier within its manager.

The standard manager types are :class:`ObservationManager` (``compute(obs)``),
:class:`RewardManager` (``compute(obs, action, info)``),
:class:`EventManager` (``apply(mode, env_ids)``, the home of all randomization
functors), :class:`ActionManager` (``process_action(action)`` then
``apply_action()``), and
:class:`DatasetManager` (``step``/``save`` for LeRobot recording).

   .. rubric:: Submodules

   .. autosummary::

      randomization

   .. rubric:: Classes

   .. autosummary::

      FunctorCfg
      ConditionCfg
      RLTaskCfg
      TerminationManager
      SceneEntityCfg
      EventCfg
      ObservationCfg
      RewardCfg
      ActionTermCfg
      DatasetFunctorCfg
      Functor
      ManagerBase
      ~event_manager.EventManager
      ObservationManager
      RewardManager
      ActionManager
      DatasetManager
      ActionTerm
      ActionDescriptor
      ActionTermDescriptor
      ActionTrace
      DefaultJointPositionAction
      EefPoseAction
      JointEffortAction
      JointPositionAction
      JointPositionToLimitsAction
      JointVelocityAction
      ParallelGripperAction
      RelativeEefDlsAction
      RelativeEefPoseAction
      RelativeJointPositionAction
      LeRobotRecorder
      AsyncLeRobotRecorder

   .. rubric:: Functions

   .. autosummary::

      observations.get_rigid_object_pose
      observations.normalize_robot_joint_data
      observations.compute_semantic_mask
      observations.compute_exteroception
      events.replace_assets_from_group
      record.record_camera_data
      rewards.distance_between_objects
      rewards.success_reward
      rewards.distance_to_target
      randomization.visual.randomize_light
      randomization.visual.randomize_camera_intrinsics
      randomization.visual.randomize_visual_material
      randomization.spatial.get_random_pose
      randomization.spatial.randomize_rigid_object_pose
      randomization.spatial.randomize_robot_eef_pose
      randomization.spatial.randomize_robot_qpos
      actions.register_action_contract
      actions.resolve_action_contract
      register_action_contract
      resolve_action_contract

.. currentmodule:: embodichain.lab.gym.envs.managers

Configuration Classes
---------------------

.. autoclass:: FunctorCfg
    :members:
    :exclude-members: __init__, class_type

.. autoclass:: SceneEntityCfg
    :members:
    :exclude-members: __init__, class_type

.. autoclass:: EventCfg
    :members:
    :exclude-members: __init__, class_type

.. autoclass:: ObservationCfg
    :members:
    :exclude-members: __init__, class_type

.. autoclass:: RewardCfg
    :members:
    :exclude-members: __init__, class_type

.. autoclass:: ActionTermCfg
    :members:
    :exclude-members: __init__, class_type

.. autoclass:: DatasetFunctorCfg
    :members:
    :exclude-members: __init__, class_type

Base Classes
------------

.. autoclass:: Functor
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: ManagerBase
    :members:
    :inherited-members:
    :show-inheritance:

Managers
--------

.. autoclass:: EventManager
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: ObservationManager
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: RewardManager
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: ActionManager
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: DatasetManager
    :members:
    :inherited-members:
    :show-inheritance:

Action Terms
------------

.. autoclass:: ActionTerm
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: DefaultJointPositionAction
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: EefPoseAction
    :members:
    :inherited-members:
    :show-inheritance:

``RelativeEefPoseAction`` defaults to the configured arm IK solver. Both backends
validate the solver chain's joint order and map seeds and returned commands
between that order and the configured joint selection. Its optional
``ik_method="dls"`` backend applies one damped Jacobian increment in the arena
frame, rescales the whole increment to ``max_joint_delta``, and clips position
limits, reading current robot limits on every control step. It uses the actual solver
root-link transform, and shifts the end-link Jacobian to the configured TCP
offset on the environment device. This backend approximates a local pose step;
``ik_success`` means a finite controller step and does not assert pose convergence.
The action descriptor records its method, damping, bound, and success semantics.
``joint_delta``, ``joint_step_limited``, and ``joint_limit_clipped_mask`` expose
the applied increment and both forms of limiting; resets clear selected rows.

``RelativeEefDlsAction`` keeps an unbounded xyz/rotation-vector request as one
fixed arena-frame target for the control interval. It recomputes the damped
Jacobian solve from measured joints before each physics substep and clips only
joint-position limits. It shares solver-order mapping and arena/TCP Jacobian
adaptation with the single-step relative controller. Its ``eef_pose.relative_substep_dls@1`` contract selects
this behavior separately from the bounded, single-step relative-pose contract.
``ActionManager.get_substep_callback(num_substeps)`` owns the remaining-step
count, action refresh and inactive-row holds for one control interval.
``EmbodiedEnv`` connects that callback after sensor sampling through the batched
simulation update, preserving intermediate recording and visualization frames.

Configure the same arm and binary gripper terms for expert and policy execution.
Set ``env.expert_trajectory.use_action_manager: true`` to route position-only
``ControllerAction`` commands through ``ActionManager.process_joint_targets``.
The DLS term retains the expert's exact FK target while recording the
corresponding relative request, and keeps the existing expert storage schema.
Completed demo rows retain measured joint holds after conversion and refresh.
The default controller-ready route stays direct. Every configured term must
support joint-target processing.
For binary grippers, ``open_on_zero=False`` makes zero a closing request;
the default retains nonnegative opening requests. The descriptor records this
boundary together with the gripper mode.

.. autoclass:: RelativeEefDlsAction
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: RelativeEefPoseAction
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: JointEffortAction
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: JointPositionAction
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: JointPositionToLimitsAction
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: JointVelocityAction
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: ParallelGripperAction
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: RelativeJointPositionAction
    :members:
    :inherited-members:
    :show-inheritance:

Action Descriptors
------------------

.. autoclass:: ActionDescriptor
    :members:

.. autoclass:: ActionTermDescriptor

.. autoclass:: ActionTrace
    :members:

Observation Manager
-------------------

.. automodule:: embodichain.lab.gym.envs.managers.observation_manager
    :members:
    :no-index:

The manager owns the raw field schema, source aliases and acquisition context.
``capture`` returns the raw boundary and its actual policy view. The existing
host rollout writer fills raw fields and temporal metadata directly;
``bind_rollout_storage`` grants the manager read access to those tensors.
``read_history`` returns owned indexed snapshots and ``snapshot_history`` serves
the export adapter. ``read_history_window`` copies explicit inclusive start/end
slots per selected environment, preserving their order and individual lengths.
It returns the same raw fields with validity, acquisition errors, revisions,
timestamps and timestamp validity; aliases use the existing schema. Values and
masks are independently owned on the requested device. Invalid samples remain
invalid, and unrecorded slots or changed episodes/closed owners reject the read.
This bounded slice performs no physical acquisition and avoids repeated
single-slot calls. ``read_history_step_counts`` returns qualified completed
state-slot boundaries only, without copying field values; zero denotes recorded
initialization. Both APIs check episode and owner lifetime before and after reads.
There is no history instance, allocation engine or separate
bind/seed/append API. Evidence adapters bind roles through ``schema``; numeric
buffer operations perform no entity/quantity lookup.

Expert policy state slots retain T+1 observations; existing rollout consumers
keep their T-transition view. External learner windows preserve their own device
and horizon. ``utils.trajectory_state.export_observation_history`` combines the
single observation schema with a numeric snapshot for compatible output,
without copying another complete state history.

Observation Functions
---------------------

``ObservationCfg.consumers`` defaults to ``("policy",)``. Add functors may
select ``recording`` without expanding the policy space. Evidence adapters
read qualified raw history; there is no separate evidence output route.
Recording terms require an ``ObservationSample`` with an explicit descriptor
and row validity; arbitrary policy tensors cannot become physical evidence by
renaming them. Modify functors remain policy-only.

For example, a single original arena pose can serve policy and recording:

.. code-block:: python

   ObservationCfg(
       func=observations.get_entity_state,
       mode="add",
       name="scene/cube/pose",
       params={"entity_cfg": SceneEntityCfg(uid="cube"), "quantity": "pose"},
       consumers=("policy", "recording"),
       extra={"shape": (7,)},
   )

The descriptor derives the resolved UID, shape, dtype, coordinate frame,
representation and joint identities from the source. Pose values use arena
coordinates and ``(x, y, z, qx, qy, qz, qw)``. Missing entities have a false validity
mask and acquisition reasons; a policy zero-fill is never qualified evidence.
Original channels are retained before in-place policy transforms. Existing
Tensor-returning policy functors keep their signature and execution order.
The manager resolves configured and required replay sources once, including
callable strings and defaults, and reuses matching declarations. The descriptor
cache contains immutable schema values, never live simulation handles.

.. automodule:: embodichain.lab.gym.envs.managers.observations
    :members:

Reward Functions
----------------

.. automodule:: embodichain.lab.gym.envs.managers.rewards
    :members:

Event Functions
---------------

.. automodule:: embodichain.lab.gym.envs.managers.events
    :members:

Recording Functions
-------------------

.. automodule:: embodichain.lab.gym.envs.managers.record
    :members:

Dataset Recording
-----------------

.. automodule:: embodichain.lab.gym.envs.managers.dataset_manager
    :members:

.. automodule:: embodichain.lab.gym.envs.managers.datasets
    :members:

.. automodule:: embodichain.lab.gym.envs.managers.async_datasets
    :members:

Randomization
-------------

.. automodule:: embodichain.lab.gym.envs.managers.randomization

    .. rubric:: Submodules

    .. autosummary::

        physics
        visual
        spatial
        geometry

    Physics
    ~~~~~~~~~~~~~~~~~~~~~~~
    .. automodule:: embodichain.lab.gym.envs.managers.randomization.physics
         :members:

    Visual
    ~~~~~~~~~~~~~~~~~~~~~~~

    .. automodule:: embodichain.lab.gym.envs.managers.randomization.visual
         :members:

    Spatial
    ~~~~~~~~~~~~~~~~~~~~~

    .. automodule:: embodichain.lab.gym.envs.managers.randomization.spatial
         :members:

    Geometry
    ~~~~~~~~~~~~~~~~~~~~~~

    .. automodule:: embodichain.lab.gym.envs.managers.randomization.geometry
         :members:

Configured RL conditions
------------------------

.. currentmodule:: embodichain.lab.gym.envs.managers

.. autoclass:: ConditionCfg
    :members:

.. autoclass:: RLTaskCfg
    :members:

.. autoclass:: TerminationManager
    :members:
