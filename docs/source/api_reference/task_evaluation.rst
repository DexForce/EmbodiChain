Task observation evaluation
===========================

These measurements and evaluators do not own simulation stepping, reset,
retries or persistence. The initial implementation qualifies instantaneous
local-+Z upright observations only; process and unsupported requirements are
rejected explicitly, and invalid pose rows are unavailable rather than passing.

.. autosummary::

   embodichain.compute.task_predicates.axis_tilt
   embodichain.lab.task_evaluation.UprightTaskEvaluator

.. autofunction:: embodichain.compute.task_predicates.axis_tilt

.. autoclass:: embodichain.lab.task_evaluation.UprightTaskEvaluator
   :members:
