embodichain.gen_sim.task_engine
================================

The Task Engine owns immutable task intent and semantic planning artifacts. It
does not own action grounding, robot routing, controller policy, or physical
execution.

.. automodule:: embodichain.gen_sim.task_engine

   .. rubric:: Task and graph contracts

   .. autosummary::

      TaskSpec
      TaskInstanceSpec
      SuccessSpec
      SemanticTaskGraph
      SemanticTaskNode
      TaskGroupSpec
      FailurePolicy
      PlannerProvenance
      decode_task_spec
      decode_semantic_task_graph
      task_spec_hash
      semantic_task_graph_hash

   .. rubric:: Schemas and ontology

   .. autosummary::

      TASK_SPEC_SCHEMA
      SEMANTIC_TASK_GRAPH_SCHEMA
      TASK_SPEC_FILENAME
      SEMANTIC_TASK_GRAPH_FILENAME
      TASK_LEVELS
      TASK_TYPES
      REASONING_TYPES
      PLANNER_ROUTES
      FORBIDDEN_SEMANTIC_GRAPH_FIELDS
      RELATIONS
      TRANSPORT_DIRECTIONS
      TERMINAL_BEHAVIORS
      TASK_CONTRACTS
      TaskContract
      task_contract
      task_success_type

embodichain.gen_sim.task_engine.contracts
-----------------------------------------

.. automodule:: embodichain.gen_sim.task_engine.contracts

   .. autosummary::

      TASK_SPEC_SCHEMA
      SEMANTIC_TASK_GRAPH_SCHEMA
      TASK_SPEC_FILENAME
      SEMANTIC_TASK_GRAPH_FILENAME
      TASK_LEVELS
      REASONING_TYPES
      PLANNER_ROUTES
      FORBIDDEN_SEMANTIC_GRAPH_FIELDS
      TaskSpec
      TaskInstanceSpec
      SuccessSpec
      SemanticTaskGraph
      SemanticTaskNode
      TaskGroupSpec
      FailurePolicy
      PlannerProvenance
      decode_task_spec
      decode_semantic_task_graph
      task_spec_hash
      semantic_task_graph_hash

embodichain.gen_sim.task_engine.ontology
----------------------------------------

.. automodule:: embodichain.gen_sim.task_engine.ontology

   .. autosummary::

      RELATIONS
      TRANSPORT_DIRECTIONS
      TERMINAL_BEHAVIORS
      TASK_TYPES
      TASK_CONTRACTS
      TaskContract
      task_contract
      task_success_type
