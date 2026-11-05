Task Engine semantic contracts
==============================

This package provides task understanding and declaration validation independently
of simulation. TaskSpec owns normative identity and evaluation; shared Task
Program, Atomic Skill and Motion modules own execution.

.. automodule:: embodichain.gen_sim.task_engine
   :members:

Task understanding and candidates
---------------------------------

Interpret an explicit caller result into independently validated task candidates, scene requests and success declarations. Model calls occur only when requested through the caller interface.

.. automodule:: embodichain.gen_sim.task_engine.agent
   :members:
   :no-index:

Candidate data contracts
------------------------

Validate finite, detached draft/candidate mappings and their task identities. Canonical hashing belongs to the unchanged declaration format.

.. automodule:: embodichain.gen_sim.task_engine.contracts
   :members:
   :no-index:

Instruction intent
------------------

Decode supported instruction intent through an explicit model caller. Fixed intent declarations can be validated without making a model request.

.. automodule:: embodichain.gen_sim.task_engine.interpretation
   :members:
   :no-index:

Task intention vocabulary
-------------------------

Describe semantic intentions, supported relations and their declared success types; these descriptors provide no physical implementation.

.. automodule:: embodichain.gen_sim.task_engine.ontology
   :members:
   :no-index:

Normative specification generation
----------------------------------

Generate the independent TaskSpec protocol from validated candidates, retaining source-role semantics and stable normative identity.

.. automodule:: embodichain.gen_sim.task_engine.task_spec
   :members:
   :no-index:

Semantic Call graph
-------------------

Validate graph topology, group ownership, dependencies and canonical Semantic Calls through the public Task Program language decoder. This creates no runtime or motion plan.

.. automodule:: embodichain.gen_sim.task_engine.semantic_graph
   :members:

Grounding and binding contracts
-------------------------------

Validate role bindings, scene manifests and binding reports as detached declarations, preserving source candidate and task identity.

.. automodule:: embodichain.gen_sim.task_engine.orchestration.contracts
   :members:

Declared coverage
-----------------

Compare graph calls and lineage with the complete candidate, TaskSpec and role bindings. Covered means declaration coverage; it does not establish execution or physical success.

.. automodule:: embodichain.gen_sim.task_engine.semantic_graph_validation
   :members:
