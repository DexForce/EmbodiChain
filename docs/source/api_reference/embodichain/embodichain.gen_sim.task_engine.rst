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
a compiled React/TypeScript interface, licensed local fonts and React Flow interactions. TaskSpec, declaration
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
---------------------------

Preserve valid TaskSpec identities and the explicit closed absence marker.
Absence carries no normative success or certificate.

.. automodule:: embodichain.gen_sim.task_engine.bundle_spec
   :members:

The frontend build uses the same React/TypeScript/Vite/React Flow versions as
the architecture explorer. Shared viewport controls and theme tokens live in
``web/graph-ui``; each app retains its own data model and layout. The Python
API embeds committed build outputs, so installed offline rendering has no Node
runtime requirement. Vector SVG export serializes the current task layout and
retains the original evidence status rather than generating a new verdict.


Normal source physical declarations
-----------------------------------

These frontend adapters compose PreparedScene, public model descriptions and
shared Scene/Compute services into a new frozen physical declaration. Control
subjects and semantic resource eligibility remain explicit. Asset fitting, source
geometry, physical calibration and reset/cadence declarations retain their
independent owners. Existing snapshots bypass preparation and keep their identity.
This surface neither publishes a runtime bundle nor constructs a simulator.

.. currentmodule:: embodichain.gen_sim.task_engine.control_physical_preparation

.. autoclass:: PreparedControlPhysicalSources
   :members:

.. autofunction:: prepare_source_control_physics

.. currentmodule:: embodichain.gen_sim.task_engine.source_physical_input

.. autoclass:: SourcePhysicalPreparation
   :members:

.. autofunction:: prepare_source_physical_input

Pure program declarations
-------------------------

Project a validated Semantic Call graph into the existing program format using
explicit per-segment acceptance bindings. Calls, targets, stage identities and
ordering remain tied to the graph. Optional acceptance deferral must retain every
original post-policy and validator at the terminal boundary. This builder does
not read a bundle, compose runtime services, plan motion or create a simulator;
callers still perform complete configured deployment preflight.

.. automodule:: embodichain.gen_sim.task_engine.program_declarations
   :members:

Static deployment declarations
------------------------------

Resolve graph relations into ordered placement routes and declared scene
bindings using shared Scene and Compute source geometry. These functions read
frozen source assets without constructing a simulator, robot or motion solver.
Targets remain proposals; runtime effects and TaskSpec acceptance are separate.

.. automodule:: embodichain.gen_sim.task_engine.placement_declarations
   :members:

.. automodule:: embodichain.gen_sim.task_engine.scene_binding_declarations
   :members:

Stability declarations select existing typed Task Program conditions and exact
sensor declarations. They preserve task/resource identities and require callers
to provide the sensors needed by the selected support conditions. They do not
collect contact facts or decide whether a physical objective succeeded.

.. automodule:: embodichain.gen_sim.task_engine.stability_declarations
   :members:

Control and articulation declarations
-------------------------------------

Resolve complete source-control recipes into the existing typed Task Program
bindings, action options and conditions. Authored joint/link inspection belongs
to shared Scene; hand calibration stays with the robot owner. Task Engine retains
logical entities and named presets, while explicit post-policy occurrence tables
select the physical terminal condition. These declarations construct no native
runtime and do not evaluate contact facts or certify physical success.

.. automodule:: embodichain.gen_sim.task_engine.control_declarations
   :members:

.. automodule:: embodichain.gen_sim.task_engine.prismatic_declarations
   :members:

Drawer declarations preserve explicit payload/container relations, per-occurrence
release acceptance and the original acquisition-to-close ordering. Graph
refinement keeps the declared roles, canonical calls and explicit resource
selection; numeric drawer targets and runtime lowering stay with shared owners.

.. automodule:: embodichain.gen_sim.task_engine.drawer_declarations
   :members:

.. automodule:: embodichain.gen_sim.task_engine.drawer_graph_declarations
   :members:

Authored semantic graph refinements
-----------------------------------

Refine a fixed source graph into explicit upright, carrier, handover and return
target declarations. Scene/Compute owns geometry and pose arithmetic; the
frontend retains semantic first-use eligibility, resource transitions, target
identity and graph ordering. Input graphs remain unchanged and identity collisions
reject. These APIs do not construct a skill, planner, simulator or runtime.

.. currentmodule:: embodichain.gen_sim.task_engine.source_graph_declarations

.. autofunction:: refine_source_manipulation_declarations

.. autofunction:: refine_source_return_declarations

.. currentmodule:: embodichain.gen_sim.task_engine.alignment_declarations

.. autofunction:: resolve_alignment_pick_declarations

.. autofunction:: resolve_alignment_lowerer_declarations

.. autofunction:: resolve_handover_profile_declarations

Action policy and continuation declarations
-------------------------------------------

Compose the unchanged semantic program, typed action options, named execution
policies and occurrence-bound continuation declarations. Source selections keep
their explicit profile and resource boundaries; shared defaults remain owned by
Lab. Native Pick and retained/released continuation declarations use complete
compiled occurrence identities rather than execution-cache indices or private
skill mutations. Pour targets remain authored geometric proposals.

Experiment permissions are explicit declarations. Execution completion, selected
acceptance and strict TaskSpec certificates remain separate outcomes; this
frontend neither collects evidence nor upgrades unavailable physical facts.
Configuration defaults and bundle/CLI publication are separate integration owners.

.. automodule:: embodichain.gen_sim.task_engine.semantic_program
   :members:

.. automodule:: embodichain.gen_sim.task_engine.policy_declarations
   :members:

.. automodule:: embodichain.gen_sim.task_engine.action_profile_declarations
   :members:

.. automodule:: embodichain.gen_sim.task_engine.native_pick_declarations
   :members:

.. automodule:: embodichain.gen_sim.task_engine.continuation_declarations
   :members:

.. automodule:: embodichain.gen_sim.task_engine.pour_declarations
   :members:

.. automodule:: embodichain.gen_sim.task_engine.experiment_declarations
   :members:

Workflow configuration and replayable state
-------------------------------------------

Validate the existing workflow/planning/execution configuration and packaged
defaults independently of runtime assembly. Requests refer to an existing source
scene and a separate output directory. Typed stage transitions retain an ordered
event audit that can be replayed without a simulator, planner or bundle loader.

.. automodule:: embodichain.gen_sim.task_engine.config
   :members:

.. automodule:: embodichain.gen_sim.task_engine.workflow_contracts
   :members:

.. automodule:: embodichain.gen_sim.task_engine.state_machine
   :members:

Run-output helpers protect sealed declaration ancestors, preserve cancelled
sealed histories and publish finite JSON without overwriting historical reports
when requested. This file owns output persistence only; workflow scheduling,
execution, report evaluation and bundle publication remain separate consumers.

.. automodule:: embodichain.gen_sim.task_engine.orchestration.workflow_artifacts
   :members:

Semantic planning and existing-scene adaptation
-----------------------------------------------

Plan canonical Semantic Call recipes from explicit task candidates and role
bindings, and adapt existing source scenes through declared manifests and
conservative relationship graphs. Model calls remain behind caller interfaces;
unknown geometry, support and orientation remain explicitly unknown rather than
physical success. Shared Scene/Compute owns source measurements and numeric
geometry; this frontend selects semantic roles and declarations.

These APIs do not construct Atomic Skills, planners or simulators. Scene
finalization may inspect exported files and publish owned audit data. Coordinator,
bundle/CLI execution and runtime report aggregation are separate consumers.

.. automodule:: embodichain.gen_sim.task_engine.semantic_planner
   :members:

.. automodule:: embodichain.gen_sim.task_engine.orchestration.scene_inventory
   :members:

.. automodule:: embodichain.gen_sim.task_engine.orchestration.grounding
   :members:

.. automodule:: embodichain.gen_sim.task_engine.orchestration.scene_source
   :members:

.. automodule:: embodichain.gen_sim.task_engine.orchestration.scene_adapter
   :members:

The scene package re-exports the same static manifest and conservative graph
contracts for callers composing an adapter; their canonical descriptions follow.

.. automodule:: embodichain.gen_sim.task_engine.scene
   :members:
   :no-index:

.. automodule:: embodichain.gen_sim.task_engine.scene.contracts
   :members:

.. automodule:: embodichain.gen_sim.task_engine.scene.conservative_graph
   :members:

.. automodule:: embodichain.gen_sim.task_engine.scene.scene_engine_v1
   :members:

.. automodule:: embodichain.gen_sim.task_engine.scene.final_inspection
   :members:

.. automodule:: embodichain.gen_sim.task_engine.scene_backend
   :members:
