Policy-owned Task Program progress
==================================

.. currentmodule:: embodichain.lab.task_program.integrations.progress

This adapter binds measured sequential segment declarations to the shared
rule session. The environment advances it after canonical observation acquisition;
reward and termination adapters only read its owned cache. It creates no expert
executor and requires explicitly configured action ownership and reward semantics.

.. autosummary::

   TaskProgramProgress

.. autoclass:: TaskProgramProgress
   :members:
