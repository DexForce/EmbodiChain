TaskSpec semantic contracts
===========================

TaskSpec v0.1 separates normative task identity from grounded instances and
candidate or executed solutions. All decoders return detached JSON records,
reject unknown fields and avoid simulator, execution and filesystem ownership.

Task identity includes roles, initial/final conditions, invariants, explicit
temporal sequences and capability requirements. It excludes concrete assets,
robots, plans and trajectories by rejecting them at the template boundary.
Canonical quantities use exact decimal strings in SI units; supported units are
m/cm/mm, s/ms and rad. Role renaming, inverse spatial relations and commutative
sorting are supported; arbitrary logical equivalence is not.

A certificate is a report record, not a verifier. Required checks must exist
and pass with matching input identities, checker/predicate versions and
evidence references. Hosts must verify referenced content and observations.
Unavailable/not-run/unsupported checks cannot be accepted as passes.

Public package interface
------------------------

.. automodule:: embodichain.task_spec
   :members:
   :imported-members:

Canonicalization
----------------

Templates are bounded to six roles, 128 predicate occurrences and logical depth
16. Canonicalization compares all relabelings within equal role declarations,
preserving normative temporal order. The stored semantic hash is excluded only
at the top level, so hashing an already sealed template is stable.

.. automodule:: embodichain.task_spec.canonicalization
   :members:

Records and acceptance
----------------------

Instances own grounded asset/component and observed initial-state references.
Instance bindings use canonical role IDs, obtained through canonical_role_map,
and checkers consume canonical_template against these bindings. Author labels
cannot be used directly: equivalent templates may permute their meanings.
Witnesses own program/integration/policy/constraint and execution references.
An expansion records lineage and invalidated checks without scheduling them.
A certificate binds checks to one template, instance, witness and episode.

.. automodule:: embodichain.task_spec.contracts
   :members:

Expression and vocabulary boundaries
-------------------------------------

Predicate quantities are explicit, dimensional and nonnegative except signed
revolute-joint positions. The coordinate convention is scene +X right, +Y front,
+Z up. The vocabulary describes bounded geometry and attachment observations;
it does not assert installed evaluators, liquid transfer, force support or
continuous-contact proof.

.. automodule:: embodichain.task_spec.expressions
   :members:

.. automodule:: embodichain.task_spec.registry
   :members:

.. automodule:: embodichain.task_spec.validation
   :members:

GenSim candidate graph version
------------------------------

The optional TaskSpec planner entry preserves the legacy candidate step hash
as provenance, emits v2 candidate graphs and retains the v1 path unchanged.
Bundle export and execution reject v2 until final task evaluation is integrated before data
submission and reset. This interface is not a certified rollout path.

.. autodata:: embodichain.gen_sim.task_engine.semantic_graph.TASK_SPEC_GRAPH_SCHEMA
