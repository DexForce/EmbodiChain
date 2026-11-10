Generative Simulation
=====================

Prepare existing meshes with the Asset Engine, generate and edit
scenes with Scene Engine, or use Task Engine to interpret an instruction and
photograph into a semantic task graph. Their guides explain the separate asset,
scene, declaration, and execution responsibilities and the handoff to simulation.

Architecture
------------

GenSim lives in ``embodichain.gen_sim``. Its three engines have independent CLI
and Python entry points. Role-bound model services, asset-processing workers, and task
contracts support these workflows; explicit runtime integrations turn their
outputs into physical scenes and executable tasks.

Solid arrows show how engine outputs reach a caller-owned integration. Dashed
arrows show selected service dependencies; each workflow selects the services
it needs.

.. mermaid::

   flowchart TB
       subgraph gensim["GenSim"]
           subgraph engines["Engine workflows"]
               asset["Asset Engine<br/>Prepared assets"]
               scene["Scene Engine<br/>Scene exports"]
               task["Task Engine<br/>Task declarations"]
           end
           subgraph support["Supporting modules"]
               models["GenSimServices<br/>Roles / adapters / ownership"]
               blender["Isolated Blender worker"]
               spec["TaskSpec"]
           end
           asset -.-> models
           scene -.-> models
           task -.-> models
           asset -.-> blender
           scene -.-> blender
           task -.-> spec
       end
       binding["Caller-owned runtime integration<br/>Scene / robot / program bindings<br/>Task Program / Atomic Skills / Gym"]
       asset --> binding
       scene --> binding
       task --> binding
       models ~~~ binding
       blender ~~~ binding
       spec ~~~ binding

Scene Engine reuses Asset Engine's articulated-generation service while owning
its own graph-conditioned mesh preparation and scene layout. The general
``asset-engine`` ingestion path is a separate entry point. Task Engine can plan from
an image and instruction without first running either asset or scene generation.

Executing a task requires explicit scene entities, robot profiles, Semantic
Call bindings, and acceptance checks. Callers can reuse existing assets, scenes,
and task declarations at this boundary. Physical success is established through
measured runtime evidence and the selected task criteria.

Engine modules
--------------

Module paths below are relative to ``embodichain.gen_sim``.

.. list-table::
   :header-rows: 1
   :widths: 22 43 35

   * - Module
     - Responsibility and input
     - Main output
   * - :doc:`Asset Engine <asset_engine>`

       ``asset_engine/``
     - Prepare an existing mesh through SimReady ingestion, normalization,
       semantic/physics annotation, and export. A separate service generates
       articulated assets from image/text observations.
     - Mesh/USD assets, physical and semantic metadata, and asset registry
       entries for ingestion runs; articulated USDCs for service-based generation.
   * - :doc:`Scene Engine <scene_engine>`

       ``scene_engine/``
     - Understand an image, generate assets and a support graph, refine layout,
       and export the scene. Edit an existing export or generate layout variants
       through the corresponding scene workflows.
     - Portable ``scene_export/`` containing assets, poses, and support
       relationships; optional scene USD packages and saved layout variants.
   * - :doc:`Task Engine <task_engine>`

       ``task_engine/``
     - Interpret an instruction and image, derive task requirements, plan
       canonical Semantic Calls, and check declaration coverage and binding needs.
     - Candidates, TaskSpec, image evidence, semantic graph, pending runtime
       bindings, validation reports, and an offline HTML viewer.

Supporting modules
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Role
   * - ``task_spec/``
     - Simulator-independent task requirements, semantic identity, predicate and
       milestone evaluation, and content-addressed caching. It evaluates supplied
       evidence; callers acquire physical observations.
   * - :doc:`Model services <model_services>`

       ``services/``
     - Capability contracts, role/profile configuration, adapter factories,
       lazy run-scoped ownership and safe call records. Existing engine client
       modules retain compatibility import paths.
   * - Blender process boundary
     - ``_blender_runtime.py`` sends file-based jobs to ``_blender_worker.py`` in
       an isolated Python environment. Asset and scene processing select the
       required mesh operations without mixing Blender's native dependencies
       into the main framework process.
   * - ``gradio_ui/``
     - Browser interfaces for asset and scene workflows, previews, and runtime
       tooling. Configuration, callbacks, and session-owned subprocess management
       keep presentation separate from engine and execution implementations.
   * - ``scene_engine/randomization/``
     - Adapt exported assets and support trees to shared scene expansion APIs;
       retain source identities, policies, saved variants, and replay behavior.

Shared numerical algorithms and physical scene operations are owned by
:doc:`Scene Services </overview/sim/scene/index>` and
``embodichain.compute.geometry``. GenSim adapts its assets, coordinate conventions,
and support graph to these services. Motion planning, explicit physics stepping,
observations, and recording belong to the simulation and Gym layers.

Choose a workflow
-----------------

Start with Asset Engine for an existing mesh, Scene Engine for image-based scene
generation or editing, and Task Engine for instruction/image semantic planning.
For execution, continue with :doc:`Task Program </overview/task_program/index>`
and :doc:`scene preparation </overview/sim/scene/workflows>` to bind and validate
the selected assets, scene, robot, and task declarations.

.. toctree::
   :maxdepth: 2

   Model services <model_services.md>
   Asset Engine <asset_engine.md>
   Scene Engine <scene_engine.md>
   Task Engine <task_engine.md>
