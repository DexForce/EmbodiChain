embodichain.lab.task_program.integrations.simulation
====================================================

.. automodule:: embodichain.lab.task_program.integrations.simulation
   :members:
   :undoc-members:

   These immutable declarations bind canonical Task Program scene and robot
   contracts to explicitly selected simulation entities and control parts.

   .. autosummary::

      AntipodalGraspAffordanceBinding
      ContainerAffordanceBinding
      ControlPartCommandPreset
      ControlPartEndpointBinding
      ControlPartResourceBinding
      RigidizedArticulationAntipodalGraspBinding
      SimulationArticulationBinding
      SimulationArticulationLinkBinding
      SimulationRigidizedArticulationObjectBinding
      SimulationRigidObjectBinding
      SimulationRobotSkillProfileBinding
      SimulationSceneBinding
      SupportSurfaceAffordanceBinding

Scene preparation and workspace
-------------------------------

Scene proposal contracts remain in ``lab.sim.scene.expansion``. These adapters
connect them to the current robot and the Gym reset lifecycle:

.. toctree::
   :maxdepth: 1

   embodichain.lab.task_program.integrations.simulation.scene_expansion
   embodichain.lab.task_program.integrations.simulation.workspace

Configured planning services
----------------------------

Planning-service declarations bind explicit collision rosters, Place scene
owners and observed cuRobo settings without constructing native resources during
configuration. The whole optional declaration is absent by default; configured
choices enter the registration identity. Native observation capture remains
owned by the simulation factory and its public source stamp, without serialized
callables or task-name dispatch.

.. automodule:: embodichain.lab.task_program.integrations.simulation.planning_services
   :members:

.. automodule:: embodichain.lab.task_program.integrations.simulation.place_scene
   :members:

.. automodule:: embodichain.lab.task_program.integrations.simulation.collision_observation
   :members:
