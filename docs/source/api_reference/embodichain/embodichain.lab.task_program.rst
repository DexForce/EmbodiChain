embodichain.lab.task_program
==============================

.. automodule:: embodichain.lab.task_program
   :members:
   :undoc-members:
   :exclude-members: TaskProgramCfg, TaskProgramIntegrationCfg, PoseCfg, TargetRefCfg,
      CyclicPoseTargetCfg, PickCfg, PlaceCfg, HandOverCfg, RegisteredSemanticCallCfg,
      InvokeCfg, SequenceCfg, RepeatCfg, SegmentCfg, ParallelCfg, BarrierCfg,
      WaitStablePostCfg, ObjectNearTargetValidatorCfg, ObjectLiftedValidatorCfg,
      ObjectRelativePositionValidatorCfg, ObjectUprightValidatorCfg,
      ArticulationJointPositionValidatorCfg, load_task_program, loads_task_program_json,
      parse_task_program_json, decode_task_program, validate_task_program,
      render_config_path, ConfigPath, ConfigPathPart, SceneReferenceRole,
      TaskProgramValidationContext, TaskProgramConfigError, TaskProgramDecodeError,
      TaskProgramValidationError, TaskProgramCompiler, CompiledTaskProgram,
      TaskProgramCompileError

   .. autosummary::

      TaskProgramCfg
      TaskProgramIntegrationCfg
      PoseCfg
      TargetRefCfg
      CyclicPoseTargetCfg
      PickCfg
      PlaceCfg
      HandOverCfg
      RegisteredSemanticCallCfg
      InvokeCfg
      SequenceCfg
      RepeatCfg
      SegmentCfg
      ParallelCfg
      BarrierCfg
      WaitStablePostCfg
      ObjectNearTargetValidatorCfg
      ArticulationJointPositionValidatorCfg
      TaskProgramCompiler
      CompiledTaskProgram
      load_task_program
      loads_task_program_json
      parse_task_program_json
      decode_task_program
      validate_task_program
      render_config_path
      ConfigPath
      ConfigPathPart
      SceneReferenceRole
      TaskProgramValidationContext
      TaskProgramConfigError
      TaskProgramDecodeError
      TaskProgramValidationError
      TaskProgramCompileError

.. currentmodule:: embodichain.lab.task_program

Schema
------

.. Register shared implementation names once beside their public definitions.
   Re-export packages can then index their own names without duplicate
   canonical registrations for the same implementation.

.. py:class:: TaskProgramCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: TaskProgramCfg
   :members:

.. py:class:: TaskProgramIntegrationCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: TaskProgramIntegrationCfg
   :members:

.. py:class:: PoseCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: PoseCfg
   :members:

.. py:class:: TargetRefCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: TargetRefCfg
   :members:

.. py:class:: CyclicPoseTargetCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: CyclicPoseTargetCfg
   :members:

.. py:class:: PickCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: PickCfg
   :members:

.. py:class:: PlaceCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: PlaceCfg
   :members:

.. py:class:: HandOverCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: HandOverCfg
   :members:

.. py:class:: RegisteredSemanticCallCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: RegisteredSemanticCallCfg
   :members:

.. py:class:: InvokeCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: InvokeCfg
   :members:

.. py:class:: SequenceCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: SequenceCfg
   :members:

.. py:class:: RepeatCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: RepeatCfg
   :members:

.. py:class:: SegmentCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: SegmentCfg
   :members:

.. py:class:: ParallelCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: ParallelCfg
   :members:

.. py:class:: BarrierCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: BarrierCfg
   :members:

.. py:class:: WaitStablePostCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: WaitStablePostCfg
   :members:

.. py:class:: ObjectNearTargetValidatorCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: ObjectNearTargetValidatorCfg
   :members:

.. py:class:: ObjectLiftedValidatorCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: ObjectLiftedValidatorCfg
   :members:

.. py:class:: ObjectRelativePositionValidatorCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: ObjectRelativePositionValidatorCfg
   :members:

.. py:class:: ObjectUprightValidatorCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: ObjectUprightValidatorCfg
   :members:

.. py:class:: ArticulationJointPositionValidatorCfg
   :module: embodichain.lab.task_program.language.schema
   :no-typesetting:
   :no-index-entry:

.. autoclass:: ArticulationJointPositionValidatorCfg
   :members:

Loading and validation
----------------------

.. autofunction:: load_task_program

.. autofunction:: loads_task_program_json

.. autofunction:: parse_task_program_json

.. autofunction:: decode_task_program

.. autofunction:: validate_task_program

.. autofunction:: render_config_path

.. autodata:: ConfigPath

.. autodata:: ConfigPathPart

.. autodata:: SceneReferenceRole

.. py:class:: TaskProgramValidationContext
   :module: embodichain.lab.task_program.language.decoder
   :no-typesetting:
   :no-index-entry:

.. autoclass:: TaskProgramValidationContext

.. py:class:: TaskProgramConfigError
   :module: embodichain.lab.task_program.language.decoder
   :no-typesetting:
   :no-index-entry:

.. autoclass:: TaskProgramConfigError

.. py:class:: TaskProgramDecodeError
   :module: embodichain.lab.task_program.language.decoder
   :no-typesetting:
   :no-index-entry:

.. autoclass:: TaskProgramDecodeError

.. py:class:: TaskProgramValidationError
   :module: embodichain.lab.task_program.language.decoder
   :no-typesetting:
   :no-index-entry:

.. autoclass:: TaskProgramValidationError

Compilation
-----------

Compilation is provider-independent. Environment and simulator binding is
documented under :mod:`embodichain.lab.task_program.integrations`; the final
Gym lifecycle bridge is :mod:`embodichain.lab.gym.envs.task_program`.

.. py:class:: TaskProgramCompiler
   :module: embodichain.lab.task_program.compiler.program
   :no-typesetting:
   :no-index-entry:

.. autoclass:: TaskProgramCompiler
   :members:

.. py:class:: CompiledTaskProgram
   :module: embodichain.lab.task_program.compiler.program
   :no-typesetting:
   :no-index-entry:

.. autoclass:: CompiledTaskProgram
   :members:

.. py:class:: TaskProgramCompileError
   :module: embodichain.lab.task_program.compiler.program
   :no-typesetting:
   :no-index-entry:

.. autoclass:: TaskProgramCompileError

Configured deployment composition
---------------------------------

Compose program, integration and execution-policy selections with a resolved
embodiment skill profile through the public loader. It uses the same strict
component decoders as the Gym loader, returns independently owned declarations,
and preserves the integration fingerprint across identical inputs. This step
does not register an environment or create a simulator or live adapter.

The host still resolves physical components and loads the selected program.
After creating its live scene, it can call the returned integration's
``adapter_factory.create_adapter(environment)`` through the existing
``TaskProgramAdapterFactory`` protocol. Configured nominal motion sources retain
their existing static program preflight during composition.

.. currentmodule:: embodichain.lab.task_program.integrations.composition

.. autosummary::

   ConfiguredTaskProgramDeployment
   ConfiguredTaskProgramIntegration
   load_configured_task_program_deployment

.. autoclass:: ConfiguredTaskProgramDeployment
   :members:

.. autoclass:: ConfiguredTaskProgramIntegration
   :members:

.. autofunction:: load_configured_task_program_deployment

Expansion integration
----------------------

Call-scoped expansion remains outside the Task Program language. Runtime
requests identify one grounded Atomic planning call, while the integration
adapter exposes the resulting Atomic plan through the source-neutral motion
template contract.

.. currentmodule:: embodichain.lab.task_program.runtime.expansion

.. autosummary::
   :nosignatures:

   TaskProgramPlanRequest
   TaskProgramPlanTransformFactory
   TaskProgramNominalSourceFactory

.. autoclass:: TaskProgramPlanRequest
   :members:

.. autoclass:: TaskProgramPlanTransformFactory
   :members:

.. autoclass:: TaskProgramNominalSourceFactory
   :members:

.. currentmodule:: embodichain.lab.task_program.integrations.nominal

.. autoclass:: BoundTaskProgramNominalSources
   :members:

.. autoclass:: TracingTaskProgramNominalSources
   :members:

.. currentmodule:: embodichain.lab.task_program.integrations.expansion

.. autosummary::
   :nosignatures:

   TaskProgramSourceAdapter
   TaskProgramExpansionRecord
   CombinedTaskProgramExpansionFactory
   TaskProgramMappedRetimeFactory
   TaskProgramMappedRetimeRecord
   TaskProgramSameGridExpansionFactory
   TaskProgramSameGridExpansionRecord

.. autoclass:: TaskProgramSourceAdapter
   :members:

.. autoclass:: TaskProgramExpansionRecord
   :members:

.. autoclass:: CombinedTaskProgramExpansionFactory
   :members:

.. autoclass:: TaskProgramMappedRetimeFactory
   :members:

.. autoclass:: TaskProgramMappedRetimeRecord
   :members:

.. autoclass:: TaskProgramSameGridExpansionFactory
   :members:

.. autoclass:: TaskProgramSameGridExpansionRecord
   :members:

.. currentmodule:: embodichain.lab.task_program

MLLM frontend
-------------

.. autofunction:: embodichain.agents.mllm.decode_mllm_task_program

.. autofunction:: embodichain.agents.mllm.compile_mllm_task_program
