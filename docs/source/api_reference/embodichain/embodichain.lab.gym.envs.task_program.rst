embodichain.lab.gym.envs.task_program
=====================================

.. automodule:: embodichain.lab.gym.envs.task_program
   :members:
   :no-index:

   This package is the narrow Gym boundary for compiled Task Programs. It
   adapts runtime output to lazy ``DemoSegment`` actions and synchronizes the
   execution clock with ordinary environment steps; language, compilation,
   catalogs, and simulation assembly live under
   :mod:`embodichain.lab.task_program`.

   .. autosummary::

      BufferedGymCommandSink
      EnvironmentStepClock
      EnvironmentStepTimingError
      GymPlanningObservationProvider
      RuntimeCommandFrameEncoder
      RuntimeTransportActionEncoder
      SegmentPostPolicyPort
      SegmentValidatorPort
      TaskProgramBridgeError
      TaskProgramDemoBridge
      UnsupportedRuntimeTransportError

.. currentmodule:: embodichain.lab.gym.envs.task_program

.. autoclass:: TaskProgramDemoBridge
   :members:

.. autoclass:: BufferedGymCommandSink
   :members:

.. autoclass:: EnvironmentStepClock
   :members:

.. autoclass:: GymPlanningObservationProvider
   :members:

.. autoclass:: RuntimeCommandFrameEncoder
   :members:

.. autoclass:: RuntimeTransportActionEncoder

.. autoclass:: SegmentPostPolicyPort

.. autoclass:: SegmentValidatorPort

.. autoclass:: TaskProgramBridgeError

.. autoclass:: EnvironmentStepTimingError

.. autoclass:: UnsupportedRuntimeTransportError
