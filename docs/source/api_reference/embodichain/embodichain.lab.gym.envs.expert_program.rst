embodichain.lab.gym.envs.expert_program
=======================================

.. automodule:: embodichain.lab.gym.envs.expert_program

   .. autosummary::

      ExpertProgramCfg
      ExpertProgramIntegrationCfg
      ExpertProgramCompiler
      CompiledProgram
      load_expert_program
      loads_expert_program_json
      parse_expert_program_json
      decode_expert_program
      ExpertProgramEnvironmentMixin
      ExpertProgramEnvironmentAdapter
      SimulationSceneBinding
      SimulationResourceEndpointBinding
      SimulationRobotResourceBinding
      RobotResourceBinding
      ControlPartEndpointBinding
      ControlPartResourceBinding
      SimulationRobotSkillProfileBinding
      SimulationExpertProgramFactory
      SimulationSegmentPolicyPort
      ControlCommandStateEvidenceTracker

.. currentmodule:: embodichain.lab.gym.envs.expert_program

Schema and loading
------------------

The public decoders and file loaders support Expert Program schema versions 1
and 2. Version 2 adds deterministic parallel blocks with explicit barriers.

.. autoclass:: ExpertProgramCfg
   :members:

.. autoclass:: ExpertProgramIntegrationCfg
   :members:

.. autofunction:: load_expert_program

.. autofunction:: loads_expert_program_json

.. autofunction:: parse_expert_program_json

.. autofunction:: decode_expert_program

Compilation and environment integration
---------------------------------------

.. autoclass:: ExpertProgramCompiler
   :members:

.. autoclass:: CompiledProgram
   :members:

.. autoclass:: ExpertProgramEnvironmentMixin
   :members:

.. autoclass:: ExpertProgramEnvironmentAdapter
   :members:

Simulation integration
----------------------

.. autoclass:: SimulationSceneBinding
   :members:

.. autoclass:: SimulationResourceEndpointBinding

.. autoclass:: SimulationRobotResourceBinding

.. autoclass:: RobotResourceBinding
   :members:

.. autoclass:: ControlPartEndpointBinding
   :members:

.. autoclass:: ControlPartResourceBinding
   :members:

.. autoclass:: SimulationRobotSkillProfileBinding
   :members:

.. autoclass:: SimulationExpertProgramFactory
   :members:

.. autoclass:: SimulationSegmentPolicyPort
   :members:

.. autoclass:: ControlCommandStateEvidenceTracker
   :members:
