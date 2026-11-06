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

Pour transport evidence
-----------------------

The ``effect_contract`` keyword on
:class:`~embodichain.lab.task_program.integrations.simulation.pour_transport.PourTransportFactory`
and :class:`~embodichain.lab.task_program.integrations.simulation.pour_transport.PourRegionTransportFactory`
uses :class:`~embodichain.lab.task_program.integrations.simulation.effect_contracts.TransportEffectContract`.
The default retains transport-phase observations. Explicit ``terminal_attachment``
selects terminal attachment verification for staging and return routes, requiring
``MoveHeldObjectOptions.report_retained_attachment=True`` and verified terminal
monitoring in every selected occurrence preset. Geometry, targets and monitoring
thresholds are unchanged; the selected contract participates in registration identity.
