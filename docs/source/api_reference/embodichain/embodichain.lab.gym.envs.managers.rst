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

The five manager types are :class:`ObservationManager` (``compute(obs)``),
:class:`RewardManager` (``compute(obs, action, info)``),
:class:`EventManager` (``apply(mode, env_ids)``, the home of all randomization
functors), :class:`ActionManager` (``process_actions(actions)``), and
:class:`DatasetManager` (``step``/``save`` for LeRobot recording).

   .. rubric:: Submodules

   .. autosummary::

      randomization

   .. rubric:: Classes

   .. autosummary::

      FunctorCfg
      SceneEntityCfg
      EventCfg
      ObservationCfg
      RewardCfg
      ActionTermCfg
      DatasetFunctorCfg
      Functor
      ManagerBase
      EventManager
      ObservationManager
      RewardManager
      ActionManager
      DatasetManager
      ActionTerm
      DeltaQposTerm
      QposTerm
      QposDenormalizedTerm
      QposNormalizedTerm
      EefPoseTerm
      QvelTerm
      QfTerm
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

.. autoclass:: DeltaQposTerm
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: QposTerm
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: QposDenormalizedTerm
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: QposNormalizedTerm
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: EefPoseTerm
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: QvelTerm
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: QfTerm
    :members:
    :inherited-members:
    :show-inheritance:

Observation Functions
---------------------

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
