embodichain.lab.sim.cfg
===================================

.. automodule:: embodichain.lab.sim.cfg

Overview
--------

This module collects the ``@configclass`` configuration objects for everything
that can be spawned into a simulation scene. It covers global simulation
settings (rendering, physics, GPU memory, markers, window recording/camera),
rigid-body / soft-body / cloth physical attributes and their overrides, joint
drive properties, and the per-entity configs consumed by
:class:`~embodichain.lab.sim.sim_manager.SimulationManager` and the object
factory in :mod:`embodichain.lab.sim.utility.sim_utils`.

Entity configs form a small inheritance hierarchy rooted at ``ObjectBaseCfg``
(``LightCfg``, ``RigidObjectCfg``, ``SoftObjectCfg``, ``ClothObjectCfg``,
``ArticulationCfg`` and its ``RobotCfg`` subclass), while ``URDFCfg`` and
``RigidConstraintCfg`` describe multi-component assembly and constraints.

.. rubric:: Classes

.. autosummary::

   RenderCfg
   PhysicsCfg
   MarkerCfg
   WindowRecordCfg
   WindowCameraPoseCfg
   GPUMemoryCfg
   RigidBodyAttributesCfg
   RigidBodyAttributesOverrideCfg
   LinkPhysicsOverrideCfg
   SoftbodyVoxelAttributesCfg
   SoftbodyPhysicalAttributesCfg
   ClothPhysicalAttributesCfg
   JointDrivePropertiesCfg
   ObjectBaseCfg
   LightCfg
   RigidObjectCfg
   SoftObjectCfg
   ClothObjectCfg
   RigidObjectGroupCfg
   RigidConstraintCfg
   URDFCfg
   ArticulationCfg
   RobotCfg
