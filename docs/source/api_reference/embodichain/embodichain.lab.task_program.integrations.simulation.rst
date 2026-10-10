embodichain.lab.task_program.integrations.simulation
====================================================

.. automodule:: embodichain.lab.task_program.integrations.simulation
   :members:
   :no-index:

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

Scene proposal contracts remain in ``lab.sim.scene_expansion``. These adapters
connect them to the current robot and the Gym reset lifecycle:

.. toctree::
   :maxdepth: 1

   embodichain.lab.task_program.integrations.simulation.scene_expansion
   embodichain.lab.task_program.integrations.simulation.workspace
