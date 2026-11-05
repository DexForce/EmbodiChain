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

Offline semantic graph viewer
-----------------------------

Render an existing semantic graph into one HTML document with embedded styles,
JavaScript, licensed local fonts and SVG interactions. TaskSpec, declaration
coverage, report overlays, thumbnails and translated display labels are optional.
The pure renderer makes no model, network or simulator call. Display data keeps
program outcomes and normative verdicts separate; unavailable evidence is never
converted into success. Report and companion identities must match the graph,
and supplied coverage is recomputed from its original declarations.

``write_semantic_task_graph_html`` writes a caller-selected display outside every
sealed bundle. Original JSON/YAML reports, graph declarations and certificates
remain read-only. Missing thumbnails stay unavailable; supplied scene provenance
selects exact source UIDs rather than guessing replacement assets.

.. automodule:: embodichain.gen_sim.task_engine.semantic_graph_html
   :members:

For an existing declaration file, choose a separate display directory:

.. code-block:: python

   import json
   from pathlib import Path
   from embodichain.gen_sim.task_engine.semantic_graph_html import (
       write_semantic_task_graph_html,
   )

   graph = json.loads(Path("bundle/semantic_task_graph.json").read_text())
   write_semantic_task_graph_html(graph, "run/semantic_graph.html")

.. image:: /_static/task_engine/semantic_graph_viewer.png
   :alt: Offline semantic graph with unexecuted and unavailable evidence states.

Recorded report contract
------------------------

Validate and detach the original finite execution-report shape for read-only
rendering. This validation does not execute a program, acquire physical evidence,
or reclassify a recorded outcome.

.. automodule:: embodichain.gen_sim.task_engine.reporting
   :members:

Optional TaskSpec companion
--------------------------

Preserve valid TaskSpec identities and the explicit closed absence marker.
Absence carries no normative success or certificate.

.. automodule:: embodichain.gen_sim.task_engine.bundle_spec
   :members:
