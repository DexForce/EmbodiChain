embodichain.lab.gym.envs.managers
==========================================

.. automodule:: embodichain.lab.gym.envs.managers

   .. rubric:: Submodules

   .. autosummary::
      
      randomization

   .. rubric:: Classes

   .. autosummary::

      FunctorCfg
      SceneEntityCfg
      EventCfg
      ObservationCfg
      Functor
      ManagerBase
      EventManager
      ObservationManager

   .. rubric:: Functions

   .. autosummary::

      observations.get_rigid_object_pose
      observations.normalize_robot_joint_data
      observations.compute_semantic_mask
      observations.compute_exteroception
      events.replace_assets_from_group
      record.record_camera_data
      randomization.rendering.randomize_light
      randomization.rendering.randomize_camera_intrinsics
      randomization.rendering.randomize_visual_material
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

Observation Functions
--------------------

.. automodule:: embodichain.lab.gym.envs.managers.observations
    :members:

Event Functions
--------------

.. automodule:: embodichain.lab.gym.envs.managers.events
    :members:

Recording Functions
------------------

.. automodule:: embodichain.lab.gym.envs.managers.record
    :members:

Randomization
-------------

.. automodule:: embodichain.lab.gym.envs.managers.randomization

    .. rubric:: Submodules

    .. autosummary::
        
        physics
        visual
        spatial

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
