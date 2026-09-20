embodichain.gen_sim.task_engine
================================

The Task Engine turns instructions and generated-scene evidence into immutable
semantic task graphs.  Its execution workflow delegates Semantic Calls to the
canonical Task Program runtime; it does not ground atomic goals or issue robot
commands itself.

Public facade
-------------

.. automodule:: embodichain.gen_sim.task_engine
   :members:
   :imported-members:
   :no-index:

Interpretation and planning contracts
-------------------------------------

.. automodule:: embodichain.gen_sim.task_engine.agent
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.cli
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.config
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.contracts
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.interpretation
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.ontology
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.state_machine
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.workflow_contracts
   :members:
   :no-index:

Scene orchestration
-------------------

These modules preserve canonical scene identity while separating generated
authoring data, conservative planning evidence, and live simulator bindings.

.. automodule:: embodichain.gen_sim.task_engine.orchestration
   :members:
   :imported-members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.orchestration.artifacts
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.orchestration.contracts
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.orchestration.coordinator
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.orchestration.legacy_scene
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.orchestration.scene_adapter
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.orchestration.scene_source
   :members:
   :no-index:

Scene analysis and inspection
-----------------------------

.. automodule:: embodichain.gen_sim.task_engine.scene
   :members:
   :imported-members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.scene.conservative_graph
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.scene.contracts
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.scene.feasibility
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.scene.final_inspection
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.scene.scene_engine_v1
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.scene_backend
   :members:
   :no-index:

Execution workflow
------------------

The workflow owns run directories, bounded orchestration attempts, and
tensor-free reports.  Physical execution remains behind the configured Task
Program subprocess boundary.

.. automodule:: embodichain.gen_sim.task_engine.run_directory
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.workflow
   :members:
   :no-index:
