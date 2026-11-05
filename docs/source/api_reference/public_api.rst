Public API Supplement
==============
.. This page is maintained by the update-api-docs agent skill.
   docs/scripts/check_api_docs.py reads it but never writes it.

This page lists module exports declared through ``__all__`` that are not
covered by a more focused API-reference page. Prefer curated pages for APIs
that need deeper explanations or examples. Sphinx obtains signatures and
summaries here from the canonical Python docstrings.

embodichain.agents.mllm.task_program
--------------------------------------

Strict MLLM entry points that inject host-owned integration settings before
decoding and compiling the constrained Task Program schema surface.

.. currentmodule:: embodichain.agents.mllm.task_program

.. autosummary::

   compile_mllm_task_program
   decode_mllm_task_program

embodichain.compute.kinematics.ik_path
--------------------------------------

Bounded fixed-time branch search for one Cartesian path. Solver callbacks own
kinematics; this numeric routine preserves serial/batched seed order and accepts
no simulation objects. Failure means the bounded search did not produce a path.

.. currentmodule:: embodichain.compute.kinematics.ik_path

.. autosummary::

   resolve_seeded_ik_path

embodichain.compute.trajectory.limits
-------------------------------------

Per-environment sampled joint-speed checks with explicit arrival intervals.
The result checks finite positions and declared speeds; it does not establish
continuous collision freedom or acceleration compliance.

.. currentmodule:: embodichain.compute.trajectory.limits

.. autosummary::

   joint_velocity_validity

embodichain.data
----------------

The public data namespace exposes dataset resolution, legacy database locations,
and resource metadata. Its asset namespace loads on first access so extension
asset definitions can import the dataset base without circular initialization.

.. currentmodule:: embodichain.data

.. autosummary::

   assets
   EMBODICHAIN_DEFAULT_DATABASE_ROOT
   database_dir
   database_2d_dir
   database_agent_prompt_dir
   database_demo_dir
   EmbodiChainDataset
   DEFAULT_DATA_MODULES
   get_data_class
   get_data_path
   list_default_materials
   get_default_material_info

embodichain.data.assets.demo_assets
-----------------------------------

Downloadable bundles used by standalone manipulation and deformable-body
demos. Each class resolves one versioned archive into the configured
EmbodiChain data cache.

.. currentmodule:: embodichain.data.assets.demo_assets

.. autosummary::

   CoordinatedPlacementAndPickment
   DeformableDemoData
   MultiW1Data
   RubiksCube
   ScoopIceNewEnv
   TabletopFourBases
   Real2SimCarton

``Real2SimCarton`` downloads a checksum-pinned original DROID episode, manual
annotations, two motion-prototype sources and estimated physical proxies. Its
revision-specific cache verifies extracted files. Measured DROID derivatives
retain CC BY 4.0; the bundle is not a generated training dataset or hardware
qualification.

.. autoclass:: Real2SimCarton

embodichain.data.assets.materials
------------------------------------

Material, environment-map and background bundles use the shared asset cache.
``DefaultMaterials`` downloads the pinned 18-preset collection and its auxiliary
soap-film layer; it does not create a simulator World.

.. currentmodule:: embodichain.data.assets.materials

.. autosummary::

   SimResources
   EnvMapHDR
   CocoBackground
   BackgroundTexture
   DefaultMaterials

.. autoclass:: DefaultMaterials
   :members: get_material_path

embodichain.data.assets.planner_assets
--------------------------------------

.. currentmodule:: embodichain.data.assets.planner_assets

.. autosummary::

   download_neural_planner_checkpoint

embodichain.data.assets.policy_assets
-------------------------------------

Resolve one official model ID to a pinned repository snapshot and materialize
its native RUN files inside the model cache. The returned provenance records
the exact repository and revision used for evaluation.

.. currentmodule:: embodichain.data.assets.policy_assets

.. autosummary::

   download_pretrained_policy

.. autofunction:: download_pretrained_policy

embodichain.data.assets.solver_assets
-------------------------------------

.. currentmodule:: embodichain.data.assets.solver_assets

.. autosummary::

   download_neural_ik_checkpoint

embodichain.data.assets.w1_assets
---------------------------------

Versioned W1 URDF archives use the shared data-root and download cache. V021
provides separate body components; V022 and V025 bundle the full robot and
components. V030 packages the standard 20-DOF body without detachable grippers
or wrist-camera attachments. Its component visuals and inertials match the
corrected full-body definition.

.. currentmodule:: embodichain.data.assets.w1_assets

.. autosummary::
   :nosignatures:

   DexforceW1V021
   DexforceW1V022
   DexforceW1V025
   DexforceW1V030
   DexforceW1ChassisV021
   DexforceW1TorsoV021
   DexforceW1EyesV021
   DexforceW1HeadV021
   DexforceW1LeftArmV021
   DexforceW1RightArmV021

.. autoclass:: DexforceW1V021

.. autoclass:: DexforceW1V022

.. autoclass:: DexforceW1V025

.. autoclass:: DexforceW1V030

.. autoclass:: DexforceW1ChassisV021

.. autoclass:: DexforceW1TorsoV021

.. autoclass:: DexforceW1EyesV021

.. autoclass:: DexforceW1HeadV021

.. autoclass:: DexforceW1LeftArmV021

.. autoclass:: DexforceW1RightArmV021

embodichain.data.default_materials
------------------------------------

The wheel includes catalog metadata so these functions work offline without
asset downloads. Both functions are also available from ``embodichain.data``. The 24-preset
catalog combines six built-in parameter definitions and 18 downloadable
resources; metadata identifies their source and download requirement.
Metadata includes compatible renderer names, source revision, dependencies,
auxiliary resources and suggested physical texture dimensions. Dimensions do
not change geometry or UVs.

.. currentmodule:: embodichain.data.default_materials

.. autosummary::

   list_default_materials
   get_default_material_info

.. autofunction:: list_default_materials

.. autofunction:: get_default_material_info

.. currentmodule:: embodichain.data

.. autofunction:: list_default_materials
   :no-index:

.. autofunction:: get_default_material_info
   :no-index:

embodichain.data_pipeline
-------------------------

The configuration submodule derives bounded pipeline settings without starting
an online worker. Importing the data-pipeline package remains lazy.

.. currentmodule:: embodichain.data_pipeline

.. autosummary::

   configuration

embodichain.data_pipeline.configuration
---------------------------------------

Configure ODS from measured generation and training capacities. A versioned
profile bounds selections per trajectory; the planner never promotes provisional
bounds to qualified ones. Runtime metadata constrains cache storage, refill
latency and short-window consumption. The online engine and consumer must enforce
the returned quota and pacing targets. Measurement helpers preserve the units,
common monotonic window and difference between active and wall-clock rates.

.. currentmodule:: embodichain.data_pipeline.configuration

.. autosummary::

   ODSReuseProfileCfg
   ODSRuntimeCfg
   ODSConfiguration
   configure_ods
   ODSRateMeasurementCfg
   ODSThroughputMeasurement
   measure_ods_throughput

.. autoclass:: ODSReuseProfileCfg
   :members:

.. autoclass:: ODSRuntimeCfg
   :members:

.. autoclass:: ODSConfiguration
   :members:

.. autofunction:: configure_ods

.. autoclass:: ODSRateMeasurementCfg
   :members:

.. autoclass:: ODSThroughputMeasurement
   :members:

.. autofunction:: measure_ods_throughput

embodichain.data_pipeline.depth_video
-------------------------------------

.. currentmodule:: embodichain.data_pipeline.depth_video

.. autosummary::

   DEFAULT_DEPTH_MIN
   DEFAULT_DEPTH_MAX
   DEFAULT_DEPTH_SHIFT
   DEFAULT_DEPTH_USE_LOG
   DEFAULT_DEPTH_PIX_FMT
   DEPTH_METER_UNIT
   DEPTH_MILLIMETER_UNIT
   DEPTH_QMAX

embodichain.gen_sim.asset_engine.cli.start
-----------------------------------------------

.. currentmodule:: embodichain.gen_sim.asset_engine.cli.start

.. autosummary::

   cli_ingest_single
   main

embodichain.gen_sim.asset_engine.clients
----------------------------------------

Asset-owned service client for generating and downloading articulated USDC assets.
Close the client after use to release its HTTP session.

.. currentmodule:: embodichain.gen_sim.asset_engine.clients

.. autosummary::

   ArticulatedGenerationClient

embodichain.gen_sim.asset_engine.clients.articulated_generation
---------------------------------------------------------------

HTTP generation client with bounded retry attempts, health checks, and validation
of returned articulation artifacts before publication.

.. currentmodule:: embodichain.gen_sim.asset_engine.clients.articulated_generation

.. autosummary::

   ArticulatedGenerationClient

embodichain.gen_sim.asset_engine.configs
----------------------------------------

Read Asset Engine settings from the shared GenSim dotenv file. Legacy Scene
Engine articulation keys remain accepted as aliases.

.. currentmodule:: embodichain.gen_sim.asset_engine.configs

.. autosummary::

   read_asset_engine_env_values

embodichain.gen_sim.asset_engine.configs.environment
----------------------------------------------------

Resolves requested Asset Engine dotenv keys and reports missing configuration
without mutating process environment variables.

.. currentmodule:: embodichain.gen_sim.asset_engine.configs.environment

.. autosummary::

   read_asset_engine_env_values

embodichain.gen_sim.asset_engine.pipeline
-----------------------------------------

Generate articulated assets from scene-independent asset records. The operation
updates their USDC path and scale metadata after normalization.

.. currentmodule:: embodichain.gen_sim.asset_engine.pipeline

.. autosummary::

   generate_articulated_usdcs

embodichain.gen_sim.asset_engine.pipeline.articulation
------------------------------------------------------

Materializes functional articulated records through the Asset Engine client.
Records provide source observations and coarse scales; the owning scene pipeline
continues to control layout and export.

.. currentmodule:: embodichain.gen_sim.asset_engine.pipeline.articulation

.. autosummary::

   generate_articulated_usdcs

embodichain.gen_sim.asset_engine.utils.geometry_utils
----------------------------------------------------------

File-based OBJ cleanup and simplification through the isolated Blender runtime.
The caller retains ownership of its input and destination asset paths.

.. currentmodule:: embodichain.gen_sim.asset_engine.utils.geometry_utils

.. autosummary::

   process_obj

.. autofunction:: process_obj

embodichain.gen_sim.env
-----------------------

.. currentmodule:: embodichain.gen_sim.env

.. autosummary::

   find_gen_sim_env_file
   get_embodichain_root
   load_gen_sim_env

embodichain.gen_sim.gradio_ui.app_articraft
-------------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_articraft

.. autosummary::

   build_articraft_panel
   cleanup_articraft_session
   configure_articraft_environment
   generate_articraft_asset
   reset_articraft_asset

embodichain.gen_sim.gradio_ui.app_asset_engine
----------------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_asset_engine

.. autosummary::

   build_asset_engine_panel
   cleanup_asset_engine_session
   prepare_asset_input_preview
   reset_simready_asset
   run_simready_asset

embodichain.gen_sim.gradio_ui.app_commands
------------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_commands

.. autosummary::

   build_run_agent_command

embodichain.gen_sim.gradio_ui.app_env
-------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_env

.. autosummary::

   ACTION_ENGINE_VISER_PORT
   ARTICULATION_SERVER_BASE_URL
   ARTICULATION_SERVER_POLL_INTERVAL_S
   ARTICULATION_SERVER_TASK_TIMEOUT_S
   ARTICULATION_SERVER_TIMEOUT_S
   ARTICRAFT_CONDA_ENV
   ARTICRAFT_OUTPUT_ROOT
   ARTICRAFT_REPOSITORY_URL
   ARTICRAFT_ROOT
   DIRECT_NO_PROXY_VALUE
   EMBODICHAIN_ROOT
   GRADIO_AUTH_PASSWORD
   GRADIO_AUTH_USERNAME
   PROXY_ENV_KEYS
   SCENE_ENGINE_VISER_PORT
   SERVER_NAME
   SERVER_PORT
   SIMREADY_OPENAI_API_KEY
   SIMREADY_OPENAI_BASE_URL
   SIMREADY_OPENAI_MODEL
   build_gradio_allowed_paths
   build_gradio_blocked_paths
   configure_direct_network_env
   configure_simready_llm_env
   get_inherited_network_env
   get_gradio_auth
   validate_gradio_artifact_root

embodichain.gen_sim.gradio_ui.app_media
---------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_media

.. autosummary::

   latest_audience_output_video

embodichain.gen_sim.gradio_ui.app_processes
-------------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_processes

.. autosummary::

   SessionProcessRegistry
   build_codex_env
   build_pipeline_env
   build_run_agent_command
   force_stop_all_child_processes
   get_request_session_id
   kill_process_group
   read_process_output
   register_managed_process
   run_agent_cli_supports_robot_profile
   start_pipeline
   terminate_process_group
   redact_sensitive_text

embodichain.gen_sim.gradio_ui.app_services
------------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_services

.. autosummary::

   build_app

embodichain.gen_sim.gradio_ui.app_state
---------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_state

.. autosummary::

   PHASES
   Phase
   RuntimeState
   SessionRuntimeRegistry
   runtime_lock
   runtime_registry
   set_runtime_phase_locked

embodichain.gen_sim.gradio_ui.app_ui
------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_ui

.. autosummary::

   build_app

embodichain.gen_sim.gradio_ui.app_workflows
-------------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_workflows

.. autosummary::

   cleanup_workflow_session
   format_status
   preview_editable_scene
   preview_saved_scene
   refresh_saved_scenes
   reset_scene_engine
   run_action_engine_from_current
   run_scene_edit
   run_scene_engine
   stop_action_engine
   ui_snapshot

embodichain.gen_sim.gradio_ui.gradio_app
----------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.gradio_app

.. autosummary::

   main

embodichain.gen_sim.scene_engine.cli.randomization
--------------------------------------------------

Command-line entry point for grouped layout generation from an existing
Scene Engine workspace. Validates the YAML policy, prepares a reusable
baseline and saves successful candidates with a completion manifest.

.. currentmodule:: embodichain.gen_sim.scene_engine.cli.randomization

.. autosummary::

   main

embodichain.gen_sim.scene_engine.configs.environment
----------------------------------------------------

Read Scene Engine settings from the shared GenSim dotenv file through the common
parser, retaining the existing Scene Engine key names.

.. currentmodule:: embodichain.gen_sim.scene_engine.configs.environment

.. autosummary::

   read_scene_engine_env_values

embodichain.gen_sim.scene_engine.core.scene_edit_plan
-----------------------------------------------------

.. currentmodule:: embodichain.gen_sim.scene_engine.core.scene_edit_plan

.. autosummary::

   SceneEditOperation
   SceneEditPlan

embodichain.gen_sim.scene_engine.errors
---------------------------------------

Identifies Scene Engine service failures separately from input or scene-schema
validation failures.

.. currentmodule:: embodichain.gen_sim.scene_engine.errors

.. autosummary::

   SceneServiceError

embodichain.gen_sim.scene_engine.pipeline
-----------------------------------------

Stage-oriented Scene Engine API. Analyze an image or edit into an in-process
blueprint with a persisted audit document, then materialize its assets and layout.
Materialization copies the supplied scene/graph and closes only clients it owns.

.. currentmodule:: embodichain.gen_sim.scene_engine.pipeline

.. autosummary::

   SCENE_BLUEPRINT_SCHEMA
   SCENE_EDIT_BLUEPRINT_SCHEMA
   SceneBlueprintPackage
   SceneEditBlueprintPackage
   SceneMaterialization
   analyze_edit
   analyze_image
   materialize_blueprint
   materialize_edit

embodichain.gen_sim.scene_engine.pipeline.api
---------------------------------------------

Blueprint value objects and their implementation entry points. Image and edit
blueprints have separate versioned schemas and content identities. Materialization
returns the exported scene configuration; it does not construct a Gym task.

.. currentmodule:: embodichain.gen_sim.scene_engine.pipeline.api

.. autosummary::

   SCENE_BLUEPRINT_SCHEMA
   SCENE_EDIT_BLUEPRINT_SCHEMA
   SceneBlueprintPackage
   SceneEditBlueprintPackage
   SceneMaterialization
   analyze_edit
   analyze_image
   materialize_blueprint
   materialize_edit

embodichain.gen_sim.scene_engine.pipeline.editing.scene_edit_asset_preparation
-------------------------------------------------------------------------------

.. currentmodule:: embodichain.gen_sim.scene_engine.pipeline.editing.scene_edit_asset_preparation

.. autosummary::

   prepare_scene_edit_assets

embodichain.gen_sim.scene_engine.pipeline.utils.usd_scene
---------------------------------------------------------

Entity metadata and deterministic UID indexing for schema-v2 USD scene stages.

.. currentmodule:: embodichain.gen_sim.scene_engine.pipeline.utils.usd_scene

.. autosummary::

   USD_SCENE_SCHEMA
   UsdEntityBinding
   UsdEntityDesc
   UsdSceneBinding
   UsdSceneIndex

embodichain.gen_sim.scene_engine.randomization.adapter
------------------------------------------------------

.. currentmodule:: embodichain.gen_sim.scene_engine.randomization.adapter

.. autosummary::

   scene_layout_input_from_export
   randomize_scene_export
   convert_layouts

embodichain.gen_sim.scene_engine.randomization.baseline
-------------------------------------------------------

Extracts one table work area, group AABBs and root-relative member poses from
rigid GLB exports. A group contains a direct table child and all descendants.
The baseline cache is reused only when source identity, extraction version,
structure and checksum match; preparation leaves the source export untouched.

.. currentmodule:: embodichain.gen_sim.scene_engine.randomization.baseline

.. autosummary::

   PreparedSceneBaseline
   prepare_scene_baseline

embodichain.gen_sim.scene_engine.randomization.cfg
--------------------------------------------------

Loads a closed, versioned YAML policy and resolves the workspace's
``scene_export/`` input and ``scene_randomization_base.yaml`` baseline path.
The workspace is supplied separately through ``output_root``. For example:

.. code-block:: yaml

   schema_version: 1
   seed: 42
   candidate_count: 10
   max_attempts: 1000
   collision_margin_m: 0.02

Generate and save layouts, then cycle the original scene and saved candidates:

.. code-block:: bash

   embodichain scene-randomization --output_root ./workspace --config policy.yaml
   python -m embodichain.gen_sim.scene_engine.randomization.loading \
       --layout-dir ./workspace/scene_layouts/run_001/layouts --interval 3

Version 3 adds hierarchical member XY/yaw sampling on CPU or CUDA:

.. code-block:: yaml

   schema_version: 3
   backend: batched
   device: cuda:0
   randomize_groups: true
   intra_group:
     targets:
       mug_001:
         xy_offset_bounds_m: null
         yaw_delta_range_deg: [-180, 180]

``randomize_groups: false`` holds group roots fixed. Unselected members retain
parent-relative poses and follow moved ancestors. XY bounds, when supplied as
``[[min_x, min_y], [max_x, max_y]]``, bound the original entity origin's offset
in its parent support frame. A null bound uses the complete feasible region.
Yaw increments rotate around the support normal while retaining authored tilt.
Defaults are 2 mm boundary and sibling margins, 1 mm support gap and 1 mm grid
resolution. ``intra_group.support_overrides`` accepts parent UID entries with
a proper rigid ``parent_from_support`` matrix and GeoJSON ``polygon``; overrides
must have mesh support evidence. Invalid polygons are rejected during policy
loading before scene preparation. Ambiguous automatic layers raise an error.
The positive height deviation of the selected near-planar layer is included in
its conservative support height, retaining the configured minimum gap over
shallow raised geometry.

Content-addressed geometry caches leave the original baseline unchanged.
Hierarchical layouts save v3 root XY translations and full selected member
poses relative to their parents, using JSON or packed storage. Replay validates
source, cache, policy, geometry and identity before exposing lazy variants and
does not sample again. The existing v1/v2 group layouts and earlier v3
hierarchical layouts remain readable. New hierarchical proposals use algorithm
version 2: dynamic group centers are sampled from the remaining free rectangular
area without nested retries. Support caches use extraction version 2 and include
a conservative layer height ceiling; older caches retain their recorded height
semantics on replay. Saving compares the on-disk cache against the generated
snapshot before publishing a completed run.

.. currentmodule:: embodichain.gen_sim.scene_engine.randomization.cfg

.. autosummary::

   MemberRandomizationCfg
   SupportOverrideCfg
   IntraGroupRandomizationCfg
   SceneRandomizationCfg
   load_randomization_config

embodichain.gen_sim.scene_engine.randomization.generation
---------------------------------------------------------

Uses the shared seeded procedural generator and XY layout solver, then expands
every group from the same baseline. Successful candidates contain a full
``SceneFragment`` and a ``SceneVariant``. Group-only sampling preserves authored
orientation and height. Hierarchical sampling changes selected members' support
XY/yaw, aligns their geometric bottom to the support gap and propagates subtrees.
Scale and exported collision hull budgets are preserved. Acceptance checks
conservative support coverage, sibling/group AABBs and table bounds; it does not
establish physical stability or task success.

``budget_exhausted`` is a normal result that retains any successful candidates.
The result records actual parameters, proposal attempts and solver outcomes;
generation does not save candidates until ``save_layouts`` is called.

.. currentmodule:: embodichain.gen_sim.scene_engine.randomization.generation

.. autosummary::

   SceneRandomizationResult
   randomize_scene

embodichain.gen_sim.scene_engine.randomization.loading
------------------------------------------------------

Validates a completed run's candidate files against its manifest, policy,
baseline and unchanged source export, then reconstructs every group member's
absolute pose without generating layouts or rewriting caches. Playback loads
assets once and cycles original and candidate poses without advancing physics.
Runs with no successful candidates cannot be previewed.

.. currentmodule:: embodichain.gen_sim.scene_engine.randomization.loading

.. autosummary::

   LoadedLayouts
   load_layouts
   preview_layouts
   main

embodichain.gen_sim.scene_engine.randomization.persistence
----------------------------------------------------------

Saves each result in a distinct ``scene_layouts/run_NNN/`` directory containing
the resolved policy and candidate group centers. Relative references identify
the existing source export and baseline, so those files must remain available
and unchanged. The completion manifest is published last, including for a run
with zero candidates; interrupted writes may leave a run without that manifest.

.. currentmodule:: embodichain.gen_sim.scene_engine.randomization.persistence

.. autosummary::

   save_layouts

embodichain.gen_sim.task_engine.evidence
----------------------------------------

The Task Engine adapter projects frozen role/lineage declarations and explicit
compiled occurrence descriptors into TaskSpec bindings. It verifies source
content identities and retains declared preset/occurrence IDs. Missing compiled
capabilities fail explicitly; projection reads no simulation or physical trace
and decides no predicate outcome.

.. currentmodule:: embodichain.gen_sim.task_engine.evidence

.. autosummary::

   project_placement_evidence_binding
   project_configured_placement_evidence_binding

.. autofunction:: project_placement_evidence_binding

New bundles use ``project_configured_placement_evidence_binding`` with a detached
condition from the configured owner's named settle preset. It requires no
``constraints.json`` or constraints digest and does not write either artifact.
The original projection remains only a frozen B0 source-comparison adapter,
checking that source payload's identity before sharing the same projection.
Neither entry point derives numerical policy values from task or object names.

.. autofunction:: project_configured_placement_evidence_binding

embodichain.gen_sim.task_spec
-----------------------------

Simulator-independent task identity, predicate evaluation and content-addressed
storage. TaskSpec describes roles, initial and final conditions, invariants,
bounded temporal conditions and ordered milestones; callers supply observations.
Missing evidence remains unavailable and cannot produce a certificate.

``TaskSpec.from_dict`` decodes existing instances through the same semantic-hash
validation as JSON documents and returns independent nested values. Use a new
TaskSpec (for example, ``dataclasses.replace``) when intentionally changing its
normative conditions; mutating a nested value invalidates the previous identity.
Valid documents retain their hashes. Task identifiers, instructions and metadata
are provenance rather than normative identity.

.. currentmodule:: embodichain.gen_sim.task_spec

.. autosummary::

   CANONICALIZATION_VERSION
   EVALUATION_STATUSES
   KNOWN_PREDICATES
   MILESTONE_SCHEMA
   PREDICATE_OPERATORS
   TASK_SPEC_SCHEMA
   EvaluationReport
   EvaluationStatus
   MilestoneSpec
   Predicate
   PredicateEvaluation
   PredicateSpec
   RoleSpec
   TaskRequirement
   TaskSpec
   TaskSpecCache
   TaskSpecCacheError
   TaskSpecRegistry
   TaskSpecValidationError
   TaskTemplate
   TemporalCondition
   TemporalSpec
   canonical_hash
   canonical_json
   canonicalize
   evaluate_predicate
   evaluate_task_spec
   evaluate_task_template
   evaluate_temporal_condition
   json_snapshot
   lookup_observation
   validate_task_spec
   validate_task_template

embodichain.gen_sim.task_spec.canonicalization
----------------------------------------------

Deterministic JSON snapshots and SHA-256 identities. Nonfinite numbers,
callables, byte strings and unordered sets are rejected before serialization.

.. currentmodule:: embodichain.gen_sim.task_spec.canonicalization

.. autosummary::

   CANONICALIZATION_VERSION
   canonical_hash
   canonical_json
   canonicalize
   json_snapshot

embodichain.gen_sim.task_spec.contracts
---------------------------------------

Logical role requirements for downstream grounding, including cardinality,
affordances, attributes and declared initial state. No simulator handles are
stored in these contracts.

.. currentmodule:: embodichain.gen_sim.task_spec.contracts

.. autosummary::

   RoleSpec

embodichain.gen_sim.task_spec.evidence
--------------------------------------

Placement evidence binds the frozen serialized-program/TaskSpec digests to exact
release, support and boundary occurrences. Binding schema version 2 also seals
``evidence_format``: new runtime snapshots use ``raw_snapshot`` and historical
B0 re-evaluation explicitly selects ``historical_record``. Existing instances
are re-decoded and their content hash is checked, including nested JSON data.
Older binding versions must be rebuilt; the TaskSpec v0.1 identity is unchanged.

Raw snapshots require the complete ``support.com_patch`` acquisition channel,
including values, validity/errors, timestamp, environment IDs, observation
revision and clock domain. Values must agree with the native measurements.
Adding a ``record`` cannot downgrade raw validation, and reports reject mixed
raw/historical input contracts. Raw sampled-timeline collection is not yet
implemented and cannot be substituted with the COM-patch channel.

This slice supports rigid ``on`` relations with legacy ``native_contact_support``
or the explicitly selected ``native_stable_reference_support/v2`` sampled window.
It rejects revision-1 and full-footprint region declarations instead of evaluating
them through a weaker scope. Import is standard-library-only; numerical proof
re-evaluation lazily loads the shared Torch/NumPy/Shapely implementation.

The verifier distinguishes unavailable acquisition from measured negatives and
checks measured release receipts independently of plan/segment success. Raw Lab
snapshots require an explicit episode binding and carry no accepted/released
conclusion; TaskSpec computes those results. Historical B0 claims are checked
when supplied. Legacy scope remains legacy even when a timeline is present. Its instantaneous
final observation needs the exact measured release and actual final runtime
boundary, not an unexecuted later wait/park policy. This preserves measured final
negatives when execution stops early, while recorded milestones still require
the full successful prefix and their original support receipt.

``evaluate_placement_evidence_report`` preserves goal/final and milestone/recorded
separation, verifies declared execution prefixes and orders exact boundary slots.
Sparse segment snapshots cannot certify complete temporal windows or continuous
invariants. The ordinary TaskSpec evaluator remains the certificate authority.

.. currentmodule:: embodichain.gen_sim.task_spec.evidence

.. autosummary::

   SupportedPlacementEvidenceBinding
   evaluate_supported_placement_evidence
   evaluate_placement_evidence_report

.. autoclass:: SupportedPlacementEvidenceBinding
   :members:

.. autofunction:: evaluate_supported_placement_evidence

.. autofunction:: evaluate_placement_evidence_report

embodichain.gen_sim.task_spec.expressions
-----------------------------------------

A finite vocabulary of observation-backed predicates and bounded temporal
conditions. Evaluations distinguish satisfied, failed, unknown, unavailable and
not-run outcomes; an incomplete always-window cannot certify success.

.. currentmodule:: embodichain.gen_sim.task_spec.expressions

.. autosummary::

   EVALUATION_STATUSES
   KNOWN_PREDICATES
   PREDICATE_OPERATORS
   EvaluationStatus
   Predicate
   PredicateEvaluation
   PredicateSpec
   TemporalCondition
   TemporalSpec
   evaluate_predicate
   evaluate_temporal_condition
   lookup_observation

embodichain.gen_sim.task_spec.registry
--------------------------------------

Validated TaskSpec documents are stored by semantic hash using atomic file
replacement. Reads reject corrupt or mismatched identities. Metadata differences
do not create different semantic identities; existing cache entries are reused
unless replacement is requested.

.. currentmodule:: embodichain.gen_sim.task_spec.registry

.. autosummary::

   TaskSpecCache
   TaskSpecCacheError
   TaskSpecRegistry

embodichain.gen_sim.task_spec.spec
----------------------------------

Normative task models and their canonical JSON form. Milestone dependencies
must form an acyclic graph; each milestone requires its predicates to hold in one
snapshot strictly after its predecessors. ``TaskTemplate`` is an alias of
``TaskSpec`` for callers using the earlier spelling.

.. currentmodule:: embodichain.gen_sim.task_spec.spec

.. autosummary::

   MILESTONE_SCHEMA
   TASK_SPEC_SCHEMA
   MilestoneSpec
   TaskRequirement
   TaskSpec
   TaskTemplate

embodichain.gen_sim.task_spec.validation
----------------------------------------

Validate documents or instances, then evaluate caller-provided snapshots and
trajectories. A certificate is ready only when all required checks are satisfied.
These functions do not acquire physical evidence or execute a task. The template
spellings are aliases of the TaskSpec functions.

.. currentmodule:: embodichain.gen_sim.task_spec.validation

.. autosummary::

   EvaluationReport
   TaskSpecValidationError
   evaluate_task_spec
   evaluate_task_template
   validate_task_spec
   validate_task_template

embodichain.lab.gym.envs.base_env
---------------------------------

.. currentmodule:: embodichain.lab.gym.envs.base_env

.. autosummary::

   BaseEnv
   EnvCfg

embodichain.lab.gym.envs.demo
-----------------------------

.. currentmodule:: embodichain.lab.gym.envs.demo

.. autosummary::

   DEMO_ANNOTATION_KEYS
   DEMO_SCHEMA_VERSION
   DemoExecutionCfg
   DemoEpisodeResult
   DemoOutputMode
   DemoSegment
   DemoSegmentOutcomeKind
   DemoSegmentResult
   execute_demo_episode
   resolve_demo_segments

embodichain.lab.gym.envs.embodied_env
-------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.embodied_env

.. autosummary::

   EmbodiedEnvCfg
   EmbodiedEnv

embodichain.lab.sim.motion.planners.curobo.payload
--------------------------------------------------

Runtime observation values for a fresh cuRobo query. Without payload, queries
support the complete robot environment batch in canonical row order, with a
shared static collision world and identical passive joint locks. Full joint
state and world-to-base transforms belong to the query snapshot. Payload queries
remain B=1; the carried body owns its identity, control part, local spheres and
observed world pose. These values are separate from serialized configuration.
Subsets, reordered environments, and per-environment worlds are unsupported.
An omitted start uses the captured selected joint rows; the motion facade reports
the world as immutable.

The owned motion factory closes backend-local attachments on exit and makes no
skill-phase decisions. Queries require CUDA graphs off, a static snapshot world
and preserved native samples. No-payload exact trajectory checks retain observed
passive locks; arbitrary trajectory checks with payload fail closed because their
separate checker does not own those attachments.

.. currentmodule:: embodichain.lab.sim.motion.planners.curobo.payload

.. autosummary::

   CuroboPayload
   CuroboQuerySnapshot
   curobo_motion_generator

embodichain.lab.sim.scene.collision_sources
-------------------------------------------

Explicit collision descriptors and same-observation poses become owned planner
providers through one caller-supplied world-to-frame transform. No pose getter
is called and no scale is applied twice. The USD link-box reader preserves
connected-component order and its conservative input-envelope policy.

.. currentmodule:: embodichain.lab.sim.scene.collision_sources

.. autosummary::

   CollisionObjectSnapshot
   read_articulation_link_boxes
   snapshot_collision_sources

embodichain.lab.sim.motion.ik
-----------------------------

Stateful adapters connect the numeric branch search to a MotionGenerator and
Robot. Cartesian target sampling preserves every requested waypoint within an
explicit sample budget. Continuous recovery is restricted to one environment
and restores the caller's RNG state without changing requested timing.

.. currentmodule:: embodichain.lab.sim.motion.ik

.. autosummary::

   resolve_continuous_ik_path
   sample_cartesian_targets

embodichain.lab.sim.motion.limits
---------------------------------

Strictly intersect runtime speed limits with the assembled URDF's named joint
caps. Missing or nonpositive authored limits are rejected rather than replaced
with an inferred default.

.. currentmodule:: embodichain.lab.sim.motion.limits

.. autosummary::

   authored_joint_velocity_limits

embodichain.lab.sim.motion.search
---------------------------------

Explicit call-local search ledgers and cooperative wall deadlines. Nested
scopes reuse the outer ledger, timings may overlap, and native work is not
preempted. Hosts choose budgets; no deadline is enabled by default.

.. currentmodule:: embodichain.lab.sim.motion.search

.. autosummary::

   PlanningBudgetExceeded
   SearchCost
   current_search_cost
   search_cost_scope
   record_search
   measure_search

embodichain.lab.sim.scene.expansion.layout
------------------------------------------

.. currentmodule:: embodichain.lab.sim.scene.expansion.layout

.. autosummary::

   SceneLayoutInput
   PreparedSceneRandomization
   SceneRandomizationBatch
   SceneVariantApplicationResult
   scene_layout_input_from_sim
   prepare_scene_randomization
   generate_scene_variants
   apply_scene_variants

embodichain.lab.sim.scene.expansion.persistence
-----------------------------------------------

.. currentmodule:: embodichain.lab.sim.scene.expansion.persistence

.. autosummary::

   LoadedSceneVariants
   save_scene_variants
   load_scene_variants

embodichain.lab.sim.sensors.contact.observation
-----------------------------------------------

Standard sensor observers can collect a selected joint alongside native contact
facts in explicit B=1 windows. Substeps reuse the completed contact sample;
initial prepare acquisitions explicitly request one refresh. Joint channels use
actual simulation time and retain independent validity and acquisition errors.
Only valid contact slots are retained, with native loss diagnostics preserved.
Window generations reject delayed callbacks across reset/reuse; completed copies
remain independent. Integrations own invocation/episode binding and action-effect
interpretation. The observer neither computes success nor advances simulation.

.. currentmodule:: embodichain.lab.sim.sensors.contact.observation

.. autosummary::

   JointContactObservation
   JointContactObserver

.. autoclass:: JointContactObservation
   :members:

.. autoclass:: JointContactObserver
   :members:

embodichain.lab.sim.sensors.contact.sampling
---------------------------------------------------------------------

ContactSensor acquisition into the existing PhysicalEvidenceFrame and
PhysicalEvidenceBatch contracts. The first axis follows explicit native environment
IDs; a valid-contact mask accompanies raw actor pairs, Arena-frame positions and
normals, and native impulses. Per-channel validity distinguishes unsupported
capabilities, nonfinite active data, saturation and source drift from measured
negative observations. Native contact revision remains separate from runtime
observation revision. The Default-only revision reader fails explicitly when its
SDK contract is unavailable.

``sample`` keeps frame tensors on its request's environment-ID device. The
``collect`` provider port retains that same device contract. Host observers may
request CPU frames; ``source_device`` records where the original numerical
observation was produced. The common frame has no invocation field.

.. currentmodule:: embodichain.lab.sim.sensors.contact.sampling

.. autosummary::

   CONTACT_ACTOR_IDS
   CONTACT_VALID_MASK
   CONTACT_NORMALS
   CONTACT_IMPULSES
   CONTACT_POSITIONS
   CONTACT_DISTANCES
   DEFAULT_CHANNELS
   ContactEvidenceSample
   ContactEvidenceSampler
   ContactSourceStamp
   native_contact_revision

Signed contact distances require explicit channel selection. Empty requests
retain the original five ``DEFAULT_CHANNELS``; a missing or nonfinite active
distance keeps that requested channel unavailable without invalidating unrelated
contact channels.

embodichain.lab.sim.sensors.contact.timeline
---------------------------------------------------------------------

Bounded raw substep history attached with
``sensor.add_sampling_observer(name, timeline)``. The host supplies a control
clock returning completed steps and elapsed control time, or None while inactive,
and makes the sensor available in its environment sensor suite. BaseEnv.step
then drives the existing begin/substep hooks; the observer never fetches a second
time or advances physics.

Records retain only packed valid contacts on CPU. Reset clears selected
environments and advances their generations, leaving other rows intact. Loss,
missing source revisions, incomplete intervals and capacity exhaustion remain
explicit. The contact-row budget is a soft threshold checked before each whole sample;
no native sample is truncated. With N environments and per-environment sensor
capacity C, charged rows stay below ``max_contact_rows + N*C``. Once the threshold
is reached, the next sample records unavailable capacity loss. Control-step and
substep caps also bound empty histories.
Frontend code owns actor/semantic label maps, serialized report schema IDs and
any task-success evaluation.

.. currentmodule:: embodichain.lab.sim.sensors.contact.timeline

.. autosummary::

   ContactTimeline
   ContactTimelineCfg

embodichain.lab.sim.sensors.support_observation
-----------------------------------------------

Native support acquisition uses a standard borrowed ``ContactSensor`` and the
public sampling observer lifecycle. Each explicit runtime registry owns one
observer per exact native pair. Replacing native instances under the same UID
requires release and rebind; closing an old observer cannot detach a new owner
registered under the same hook name. Borrowed sensors and bodies are never
destroyed by this registry.

Every physical batch has a real timestamp and a collector-owned revision.
Legacy acquisition uses the simulation clock. Strong frames and results use the
bound environment clock and observed substep offset; their fixed mapping to
simulation time is checked, including warmup offsets. Unobserved advancement
cannot be hidden by rebasing that mapping. The bound clock qualifies support contract
revision 1, including native source identity, topology, episode generation and
the complete substep interval. Legacy metadata stays legacy even when sharing
an observer whose strong clock is bound. Its strong-only sample serial and the
all-acquisition frame revision are separate identities.

``NativeSupportResult.support.valid`` must be checked before using its values.
Acquisition errors invalidate the whole pair's witness batch; selected resets
leave the other rows intact. Positive contact patches may persist through
sleep while both body poses remain within the original tolerance. Missing,
overflowed or changed observations invalidate those witnesses. A complete empty
sample is an instantaneous negative, not proof of absent support throughout a
stability window. Normative task acceptance remains outside this module.

.. currentmodule:: embodichain.lab.sim.sensors.support_observation

.. autosummary::

   NATIVE_SUPPORT_CHANNEL
   SupportControlClock
   NativeSupportResult
   NativeSupportObservation
   SupportObservationRegistry
   qualified_support_heights

.. autoclass:: NativeSupportResult
   :members:

.. autoclass:: NativeSupportObservation
   :members:

.. autoclass:: SupportObservationRegistry
   :members:


Qualified contact-height adaptation remains sensor-owned. It uses only a
positive-load COM-qualified patch and returns the highest point; invalid rows
remain NaN. Pure transformed mesh envelopes live in
``compute.geometry.rigid.support_mesh_envelope`` and establish no contact.

.. autofunction:: qualified_support_heights

embodichain.lab.task_program.integrations.simulation.grip_evidence
------------------------------------------------------------------

Target-specific paired-pad observations using explicit semantic/native object
bindings and control-part/pad-link declarations. NativeGripEvidenceFactory uses
the public control-part provider injection API and preserves its provider ID and
revision. Each provider owns its observer and bounded raw-query trace.

Physical interpretation revision 1 preserves the opposed positive-impulse
criterion and absent-positive-contact observations. Revision 2 requires explicit
opening calibration and opt-in; its one-sided open-hand release observation is
not geometric separation or acquisition evidence. Driver indices and joint names
are supplied by the robot/profile owner, with no model-name or suffix inference.

Host frames receive all normal public ownership/validity checks. Grip arithmetic
still runs on the original sensor device and BinaryEffectObservation returns on
the caller's context device. Raw trace data remains observational; application
serializers and TaskSpec evaluation belong outside this module.

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.grip_evidence

.. autosummary::

   NativeGripEvidenceCfg
   NativeGripEvidenceFactory
   NativeGripEvidenceProvider
   NativeGripObserver


embodichain.lab.sim.scene.mesh_snapshot
---------------------------------------

Capture scaled rigid visual envelopes and ordered articulation collision-input
components through plain UID/link mappings. Geometry stays in owning frames
until supplied observations are projected. Planner roles are neutral metadata;
missing or unsupported input geometry remains unavailable. These snapshots do
not replace native cooked-shape, rest-offset, contact, or CCD evidence.

.. currentmodule:: embodichain.lab.sim.scene.mesh_snapshot

.. autosummary::

   SceneMeshComponent
   SceneMeshSnapshot
   capture_scene_meshes
   iter_scene_components

embodichain.lab.task_program.integrations.simulation.handover
-------------------------------------------------------------

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.handover

.. autosummary::

   ConfiguredHandOverPoseProvider

embodichain.lab.task_program.integrations.simulation.observations
-----------------------------------------------------------------

Bind the physical collector to an initialized environment and its actual bridge
before the first bridge control step. It uses the bridge's policy session and episode
IDs; later episodes, closed resources and superseded current clock bindings are
rejected. Independent runtime owners retain independent state.

Current capture selects the registered pair/scope and uses the native
rest-envelope COM tolerance. Recorded capture preserves the original policy
parameters and reads the exact seeded end-step slots through an explicitly
injected ``ObservationManager``. The collector binds pose fields from its schema,
reads original values and masks, and converts only qualified arena poses.
Omitting observation history keeps current capture available but rejects historical capture. It never resamples current contacts for
a historical pose or falls back to a live pose/previous frame.
The result owns physical metadata and availability only; TaskSpec and Task Engine
are not dependencies of this Lab collector.

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.observations

.. autosummary::

   SimulationSupportObservationCollector

.. autoclass:: SimulationSupportObservationCollector
   :members:

embodichain.lab.task_program.integrations.simulation.stability
--------------------------------------------------------------

Named geometric conditions share ``SimulationTaskProgramRegistration.settle_presets``
with ordinary dynamic settle monitors. ``SimulationStabilityConditionCfg`` owns
its duration, timeout, pose/axis targets, attachment-drift bounds and optional
native-support declaration. Registration snapshots and fingerprints the complete
configuration; names do not select hidden behavior.

``SimulationJointStabilityConditionCfg`` observes an explicitly bound native
joint, with independently declared target tolerance and full-window drift.
The standard port preserves the initial observed robot hold, counts only
acknowledged ticks after the initial sample, and records the original
``joint_position_and_stability`` receipt. This is joint retention evidence;
it does not qualify contact safety. When an execution clock is bound, the
receipt additionally includes the actual final joint acquisition with per-row
validity, environment IDs and a policy-window sample sequence. ``joint_window``
retains sampled coordinates, validity/errors, actual step/timestamp arrays,
native joint identity and compiled policy/segment identity for independent
reassessment. ``monitor_initial_values`` preserves the pre-sample range anchor
as numerical initialization, separately from the clock-verified acquisitions.
A final sample alone does not establish a complete time window.
Without a clock, no timestamped raw evidence is fabricated.

The default ``invalid_observation_policy="abort_batch"`` retains the source
behavior. Explicit ``isolate_rows`` marks the first invalid active row unavailable
for that execution while other rows continue. The receipt separates execution
termination from per-row condition status; timeout, unavailable observation and
cancellation are not interchangeable. Cancellation preserves partial diagnostic
facts but publishes no post-policy success.

``max_observation_samples`` bounds raw retention, independently of the runtime
condition. The default 4096 comfortably holds the 126 samples from a 5-second
wait at a 0.04-second control period. Longer waits may drop old samples, report
``dropped_samples`` and set ``record_complete=false``; a truncated history must
not certify complete coverage. Storage is O(batch times retained samples), and
conversion to JSON occurs after sampling, without per-step device-to-host trace
copies. Runtime row handling and retention settings remain registration-owned;
``to_condition_dict()`` preserves the original normative parameter fields.

Target, tolerance and range drift use native runtime joint units. Translation
and rotation require their own explicitly selected policy values; the default
0.001 is not a universal physical distance or velocity threshold. Robot holds
retain the source observed posture, and neither this monitor nor the policy
advances environment time itself.

Every environment-adapter bridge obtains fresh policy/validator session state,
including execution without nominal sources. A later bridge supersedes the
earlier session's observation ownership instead of replacing its clock. Captured
position holds use the configured Gym action encoder, supplying zero velocity
targets in ``position_velocity`` mode while retaining the original positions.

Supported conditions declare a standard contact sensor through
``SupportSensorDeclarationCfg``. The scene binding supplies its exact two native
rigid UIDs; the runtime factory creates or reuses the sensor and the environment
suite drives the same instance. Incompatible existing UIDs and missing clock,
geometry or support inputs fail before execution. There is no geometry-only
success fallback for native support.

The standard policy port preserves the source time grid: the initial observation
does not count toward duration, ordinary pose drift resets the anchor/window,
and held-object failures remain latched. It yields controller-target holds and
leaves environment stepping, cancellation and task acceptance with their owners.
Incomplete support windows keep their per-row validity/errors and continue holds;
other valid rows can qualify independently. Metadata includes the physical
``support_acquisition`` channel, missing-observation counts and the latest error
for each row. A failed result following missing evidence retains
``support_result_available=false`` unless a complete successful window is later
established. This distinguishes unavailable outcomes from measured negatives.

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.stability

.. autosummary::

   SupportSensorDeclarationCfg
   SimulationStabilityConditionCfg
   SimulationJointStabilityConditionCfg

.. autoclass:: SupportSensorDeclarationCfg
   :members:

.. autoclass:: SimulationStabilityConditionCfg
   :members:

.. autoclass:: SimulationJointStabilityConditionCfg
   :members:

embodichain.lab.gym.envs.episode_context
--------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.episode_context

Row-local recording eligibility follows the buffered episode, independently of
replaceable demo annotations. Synchronous checks retain pending and rejected
state across internal resets, and snapshots own their JSON provenance.

.. autosummary::
   :nosignatures:

   EpisodeContext

.. autoclass:: EpisodeContext
   :members:

embodichain.lab.gym.envs.managers.action_manager
------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.managers.action_manager

.. autosummary::

   ActionTerm
   ActionManager

embodichain.lab.gym.envs.managers.actions
-----------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.managers.actions

.. autosummary::

   DefaultJointPositionAction
   EefPoseAction
   JointEffortAction
   JointPositionAction
   JointPositionToLimitsAction
   JointVelocityAction
   ParallelGripperAction
   RelativeEefPoseAction
   RelativeJointPositionAction

embodichain.lab.gym.envs.managers.action_types
----------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.managers.action_types

.. autosummary::

   ActionDescriptor
   ActionTermDescriptor
   ActionTrace

embodichain.lab.gym.envs.objectives
----------------------------------

.. currentmodule:: embodichain.lab.gym.envs.objectives

Optional measured objectives observe ordered stable placement regions independently
of Task Program completion, segment acceptance and dataset persistence. Progress
advances once per control step; snapshots are pure and partial resets are isolated
per environment. Stable region occupancy does not establish gripper detachment.

.. autosummary::

   ConditionResult
   evaluate_near_target
   evaluate_below_height
   GoalRegionCfg
   OrderedPlacementObjectiveCfg
   MeasuredRigidObjectState
   OrderedPlacementObjective

embodichain.lab.gym.envs.objectives.config
-----------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.objectives.config

Strict declaration decoding resolves a deployment's objective component relative
to its owning configuration and rejects unknown fields and invalid thresholds.

.. autosummary::

   GoalRegionCfg
   OrderedPlacementObjectiveCfg
   decode_objective
   load_objective_component

embodichain.lab.gym.envs.objectives.ordered_placement
---------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.objectives.ordered_placement

The evaluator consumes measured positions and velocities. Historical milestone
progress remains recorded when final-region truth becomes false; returning to the
final region requires the configured stable hold duration again.

.. autosummary::

   MeasuredRigidObjectState
   OrderedPlacementObjective

embodichain.lab.gym.envs.wrapper.replay
---------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.wrapper.replay

.. autosummary::

   ReplayWrapper

embodichain.lab.gym.utils
-------------------------

.. currentmodule:: embodichain.lab.gym.utils

.. autosummary::

   EnvProfiler
   EnvProfilerCfg
   capture_trajectory_state
   restore_trajectory_state

embodichain.lab.gym.utils.gym_utils
-----------------------------------

.. currentmodule:: embodichain.lab.gym.utils.gym_utils

.. autosummary::

   DEFAULT_MANAGER_MODULES
   add_env_launcher_args_to_parser
   assign_data_to_dict
   batch
   build_env_cfg_from_args
   cat_tensor_with_ids
   clip_and_scale_action
   config_to_cfg
   convert_observation_to_space
   dict_array_to_torch_inplace
   fetch_data_from_dict
   flatten_state_dict
   get_dtype_bounds
   get_manager_modules
   init_rollout_buffer_from_config
   init_rollout_buffer_from_gym_space
   map_qpos_to_eef_pose
   merge_args_with_gym_config
   register_manager_modules
   to_cpu_tensor
   to_tensor

embodichain.lab.gym.utils.profiler
----------------------------------

.. currentmodule:: embodichain.lab.gym.utils.profiler

.. autosummary::

   EnvProfilerCfg
   EnvProfiler

embodichain.lab.gym.envs.managers.observation_types
---------------------------------------------------

The observation manager owns source acquisition and the single raw-field schema.
Policy views receive separate tensors before in-place normalization/noise;
ordinary queries do not advance history. The existing rollout writer stores raw observations by canonical field ID.
Indexed reads use those tensors and episode/frame metadata directly, with no
history instance, separate write lifecycle or entity-specific reader. Internal
expert policy observations keep T+1 states; the public rollout remains a
compatible T-transition view and also exposes raw views when enabled. Learner
windows retain their own horizon/device, and reuse does not erase episode data. Contact histories
retain their sensor physics clocks and complete-window contracts.

.. currentmodule:: embodichain.lab.gym.envs.managers.observation_types

.. autosummary::

   ObservationDescriptor
   ObservationSample
   ObservationFrame
   ObservationHistorySnapshot

.. autoclass:: ObservationDescriptor
   :members:

.. autoclass:: ObservationSample
   :members:

.. autoclass:: ObservationFrame
   :members:

.. autoclass:: ObservationHistorySnapshot
   :members:

embodichain.lab.gym.utils.trajectory_state
------------------------------------------

Capture/restore helpers retain the standalone replay surface.
``export_observation_history`` combines numeric history with the observation
manager's schema for file output. Alias state views share the same owned tensor
snapshot. ``trajectory_observation_cfgs`` declares replay state fields as raw observation
functors, preserving full robot/mimic joint order, articulation root states and
rigid velocities. Declaration assembly performs no physical acquisition.
The standard environment recorder uses these functors and shares pose acquisition
with policy ``get_object_pose`` / ``get_rigid_object_pose`` within a boundary.

.. currentmodule:: embodichain.lab.gym.utils.trajectory_state

.. autosummary::

   capture_trajectory_state
   restore_trajectory_state
   trajectory_observation_cfgs
   export_observation_history

.. autofunction:: trajectory_observation_cfgs

.. autofunction:: export_observation_history

embodichain.lab.scripts.analyze_workspace
-----------------------------------------

.. currentmodule:: embodichain.lab.scripts.analyze_workspace

.. autosummary::

   build_sim_cfg
   build_robot_cfg
   build_preset_robot_cfg
   build_analyzer_config
   preview_cache
   parse_args
   main
   cli

embodichain.lab.scripts.preview_asset
-------------------------------------

.. currentmodule:: embodichain.lab.scripts.preview_asset

.. autosummary::

   build_sim_cfg
   cli
   load_assets
   main
   preview

embodichain.lab.scripts.preview_joint_control
---------------------------------------------

.. currentmodule:: embodichain.lab.scripts.preview_joint_control

.. autosummary::

   ArticulationPreviewController

embodichain.lab.scripts.preview_lerobot_data
--------------------------------------------

.. currentmodule:: embodichain.lab.scripts.preview_lerobot_data

.. autosummary::

   EpisodePreview
   SegmentPreview
   build_episode_preview
   cli
   inspect_dataset
   main
   resolve_dataset_root

embodichain.lab.scripts.preview_material
----------------------------------------

.. currentmodule:: embodichain.lab.scripts.preview_material

Query the default catalog without loading simulation dependencies, or render
presets and parameter variants with fixed preview settings and managed cleanup.

.. autosummary::

   main
   cli

.. autofunction:: main

.. autofunction:: cli


embodichain.lab.scripts.preview_robot
-------------------------------------

.. currentmodule:: embodichain.lab.scripts.preview_robot

Load a catalog robot or a standalone configuration and run a native or Viser
preview with managed control-part Gizmos and optional joint controls.

.. autosummary::

   build_robot_cfg
   cli
   main

.. autofunction:: build_robot_cfg

.. autofunction:: cli

.. autofunction:: main


embodichain.lab.scripts.run_task
-------------------------------

.. currentmodule:: embodichain.lab.scripts.run_task

.. autosummary::

   cli
   generate_and_execute_action_list
   generate_function
   main
   preview

embodichain.lab.sim
-------------------

.. currentmodule:: embodichain.lab.sim

.. autosummary::

   VisualMaterialCfg
   VisualMaterial
   VisualMaterialInst
   ReuseSegmentState
   BatchEntity
   SimulationManager
   SimulationManagerCfg
   SIM_CACHE_DIR
   MATERIAL_CACHE_DIR
   CONVEX_DECOMP_DIR
   REACHABLE_XPOS_DIR

embodichain.lab.sim.atomic_actions
----------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions

.. autosummary::

   ActionPlanningServices
   Affordance
   AffordancePoseCandidates
   AffordanceSample
   AffordanceSamplingContext
   AntipodalAffordance
   AssembleAffordance
   BUILTIN_ACTION_TYPES
   CoordinatedPickmentOptions
   CoordinatedPlacementOptions
   DynamicCollisionMode
   EntityState
   EffectVerifier
   GRASP_COMMAND
   HandOverOptions
   InteractionPoints
   MoveEndEffectorOptions
   MoveHeldObjectOptions
   MoveJointsOptions
   ObjectActionGoal
   OPEN_COMMAND
   PickUpOptions
   PlaceOptions
   PoseGoalValue
   PushObject
   PushObjectGoal
   PushObjectOptions
   PushObjectToolCalibration
   RigidObjectSceneProvider
   RigidObjectSceneProviderCfg
   RunnerStepCallback
   SceneProvider
   SceneSnapshotSupplier

embodichain.lab.sim.atomic_actions.affordance
---------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.affordance

.. autosummary::

   Affordance
   AntipodalAffordance
   AxisAlignAffordance
   SlideAffordance
   PressAffordance
   TwistAffordance
   InteractionPoints
   AssembleAffordance

.. automodule:: embodichain.lab.sim.atomic_actions.affordance
   :members:
   :no-index:

embodichain.lab.sim.atomic_actions.affordance_sampling
-------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.affordance_sampling

.. autosummary::

   AffordancePoseCandidates
   AffordanceSample
   AffordanceSamplingContext

.. automodule:: embodichain.lab.sim.atomic_actions.affordance_sampling
   :members:
   :no-index:

embodichain.lab.sim.atomic_actions.bindings
-------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.bindings

.. autosummary::

   ActionBinding
   EndpointBinding
   JointPositionTarget
   RuntimeEndpointTarget

embodichain.lab.sim.atomic_actions.control
------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.control

.. autosummary::

   ActionControlOverrides
   ControlCommand
   ControlPartCommandProfile
   GRASP_COMMAND
   JointPositionCommand
   OPEN_COMMAND
   PARK_COMMAND

embodichain.lab.sim.atomic_actions.core
---------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.core

.. autosummary::

   AtomicAction
   ObjectSemantics
   SkillDescriptor
   resolve_runtime_device

embodichain.lab.sim.atomic_actions.effects
------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.effects

.. autosummary::

   StateDelta

embodichain.lab.sim.atomic_actions.engine
-----------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.engine

.. autosummary::

   AtomicActionEngine

embodichain.lab.sim.atomic_actions.execution
--------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.execution

.. autosummary::

   ExecutionEvent
   ExecutionEventKind
   ExecutionPlanAttempt
   ExecutionSession
   ExecutionStatus
   ExecutionTick

embodichain.lab.sim.atomic_actions.verification
------------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.verification

.. autosummary::

   EffectExpectationResult
   EffectVerificationRequest
   EffectVerificationResult
   HeldObjectGuardRequest
   HeldObjectGuardResult
   PhaseEffectGateRequest
   PhaseEffectGateResult

embodichain.lab.sim.atomic_actions.goals
----------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.goals

.. autosummary::

   ObjectActionGoal
   PoseGoalValue
   SceneEntityPose
   collect_scene_dependencies
   resolve_pose_goal
   validate_pose_goal
   validate_pose_tensor

.. automodule:: embodichain.lab.sim.atomic_actions.goals
   :members:
   :no-index:

embodichain.lab.sim.atomic_actions.invocation
---------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.invocation

.. autosummary::

   ActionInvocation
   ActionOptions
   GoalT
   OptionsT
   PhaseEffectGateRequirement
   ResolvedActionRequest

embodichain.lab.sim.atomic_actions.plans
----------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.plans

.. autosummary::

   ActionPlan
   CompiledTrajectory
   EffectVerificationRequirement
   PlannerDiagnostics
   PlanningFailure
   TimedTrajectory
   TrajectorySegment
   normalize_success_mask

embodichain.lab.sim.atomic_actions.policies
-------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.policies

.. autosummary::

   DynamicCollisionMode
   MotionPolicy
   RecoveryPolicy

embodichain.lab.sim.atomic_actions.primitives
---------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.primitives

.. autosummary::

   BUILTIN_ACTION_TYPES

embodichain.lab.sim.atomic_actions.requirements
-----------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.requirements

.. autosummary::

   BATCH_INVERSE_KINEMATICS_CAPABILITY
   CARTESIAN_POSE_CAPABILITY
   DisjointResourceSlots
   DisjointSlotEndpoints
   FORWARD_KINEMATICS_CAPABILITY
   GRASP_CAPABILITY
   INVERSE_KINEMATICS_CAPABILITY
   JOINT_POSITION_CAPABILITY
   SkillBindingContract
   SkillEndpointRequirement
   SkillResourceSlot

embodichain.lab.sim.atomic_actions.runner
-----------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.runner

.. autosummary::

   CommandAckStatus
   CommandAcknowledgement
   CommandDispatch
   CommandOperation
   CommandSink
   EffectVerifier
   ExecutionClock
   ExecutionRunner
   ExecutionRunnerCfg
   HeldObjectGuardVerifier
   MonotonicExecutionClock
   ObservationProvider
   PhaseEffectGateVerifier
   RunnerStatus
   RunnerStep
   RunnerStepCallback

embodichain.lab.sim.atomic_actions.runtime
------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.runtime

.. autosummary::

   ActionPlanningServices

embodichain.lab.sim.atomic_actions.runtime_commands
---------------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.runtime_commands

.. autosummary::

   EndpointCommand
   JointPositionPayload
   RuntimeCommandFrame
   RuntimeCommandPayload
   TimedCommandSequence

embodichain.lab.sim.atomic_actions.scene
----------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.scene

.. autosummary::

   SceneProvider

embodichain.lab.sim.atomic_actions.sim_adapter
----------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.sim_adapter

.. autosummary::

   create_rigidized_articulation_antipodal_semantics
   create_simulation_atomic_action_engine
   RigidObjectSceneProvider
   RigidObjectSceneProviderCfg
   SceneEntity
   SceneSnapshotSupplier
   SimulationExecutionAdapter

embodichain.lab.sim.atomic_actions.state
----------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.state

.. autosummary::

   EntityState
   HeldObjectState
   PlanningContext
   RobotObservation
   SceneSnapshot
   TaskState

.. automodule:: embodichain.lab.sim.atomic_actions.state
   :members:
   :no-index:

embodichain.lab.sim.atomic_actions.trajectory_ops
-------------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.trajectory_ops

.. autosummary::

   axis_translation_keyframes
   build_joint_plan_states
   build_pose_plan_states
   interpolate_hand_qpos
   interpolate_joint_trajectory
   resolve_joint_target
   resolve_pose_target
   split_three_segments
   to_full_robot_trajectory
   translate_pose_world

embodichain.lab.sim.atomic_actions.transports
---------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.transports

.. autosummary::

   EndpointCommandRouter
   EndpointCommandTransport

embodichain.lab.sim.diff
------------------------

Public bridge from task-defined Newton kinematics and Warp tapes into PyTorch
autograd. It does not advance the Newton dynamics solver.

.. currentmodule:: embodichain.lab.sim.diff

.. autosummary::

   NewtonStepFunc
   tape_context

embodichain.lab.sim.diff.bridge
-------------------------------

.. currentmodule:: embodichain.lab.sim.diff.bridge

.. autosummary::

   NewtonStepFunc
   tape_context

embodichain.lab.sim.cfg.source
-------------------------------

USD source configuration identifies a file and an optional exact prim inside
that file. Source selection is independent of runtime identity, placement and
parent/link attachment. Supply it through ``source`` on a visual, rigid-object
or articulation configuration. Existing mesh and URDF file inputs remain
available; each asset has one file owner.

.. currentmodule:: embodichain.lab.sim.cfg.source

.. autosummary::

   UsdSourceCfg

.. autoclass:: UsdSourceCfg
   :members:

embodichain.lab.sim.cfg.visual_object
-------------------------------------

Render-only geometry or a USD source, and initial visual state.

.. currentmodule:: embodichain.lab.sim.cfg.visual_object

.. autosummary::

   VisualObjectCfg

embodichain.lab.sim.diff.runtime
--------------------------------

.. currentmodule:: embodichain.lab.sim.diff.runtime

.. autosummary::

   NewtonDifferentiableRuntime

embodichain.lab.sim.objects.articulation
----------------------------------------

.. currentmodule:: embodichain.lab.sim.objects.articulation

.. autosummary::

   ArticulationData
   Articulation
   ArticulationJointKinematics


embodichain.lab.sim.objects.constraint
--------------------------------------

.. currentmodule:: embodichain.lab.sim.objects.constraint

.. autosummary::

   RigidConstraint

embodichain.lab.sim.objects.gizmo
---------------------------------

.. currentmodule:: embodichain.lab.sim.objects.gizmo

Native robot targets use DexSim's controller with Newton IK by default.
Set ``GizmoCfg.ik_solver="embodichain"`` to reuse the robot control part's
configured solver, including PinkSolver; Viser uses the same solver adapter.

.. autosummary::

   Gizmo
   GizmoCfg
   create_robot_ik_gizmo_controller

embodichain.lab.sim.objects.rigid_object
----------------------------------------

.. currentmodule:: embodichain.lab.sim.objects.rigid_object

.. autosummary::

   CollisionShapeDesc
   RigidBodyData
   RigidObject
   RigidObjectCfg

embodichain.lab.sim.objects.rigid_object_group
----------------------------------------------

.. currentmodule:: embodichain.lab.sim.objects.rigid_object_group

.. autosummary::

   RigidBodyGroupData
   RigidObjectGroup
   RigidObjectGroupCfg

embodichain.lab.sim.objects.robot
---------------------------------

.. currentmodule:: embodichain.lab.sim.objects.robot

.. autosummary::

   ControlGroup
   Robot


embodichain.lab.sim.objects.visual_object
-----------------------------------------

Batched render-only objects provide pose, visibility and material-assignment operations without physical buffers or collision APIs.

.. currentmodule:: embodichain.lab.sim.objects.visual_object

.. autosummary::

   VisualObject
   VisualObjectCfg

embodichain.lab.sim.physics
---------------------------

Manager-level physics backend selection and lifecycle contracts for the
Default and Newton implementations integrated through DexSim.

.. currentmodule:: embodichain.lab.sim.physics

.. autosummary::

   PhysicsBackend
   DefaultPhysicsBackend
   NewtonPhysicsBackend
   make_physics_backend

embodichain.lab.sim.physics.base
--------------------------------

.. currentmodule:: embodichain.lab.sim.physics.base

.. autosummary::

   PhysicsBackend

embodichain.lab.sim.physics.default
-----------------------------------

.. currentmodule:: embodichain.lab.sim.physics.default

.. autosummary::

   DefaultPhysicsBackend

embodichain.lab.sim.physics.newton
----------------------------------

.. currentmodule:: embodichain.lab.sim.physics.newton

.. autosummary::

   NewtonPhysicsBackend

embodichain.lab.sim.motion.planners.base_planner
------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.planners.base_planner

.. autosummary::

   BasePlannerCfg
   CollisionWorldInfo
   PlanOptions
   BasePlanner
   validate_plan_options

embodichain.lab.sim.motion.planners.curobo.curobo_planner
---------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.planners.curobo.curobo_planner

.. autosummary::

   CuroboAutoGenCfg
   CuroboPlanOptions
   CuroboPlanner
   CuroboPlannerCfg
   CuroboWorldCfg

embodichain.lab.sim.motion.planners.curobo.curobo_yaml
------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.planners.curobo.curobo_yaml

.. autosummary::

   generate_curobo_robot_yaml
   generate_curobo_world_scene
   visualize_curobo_collision_models
   visualize_curobo_robot_collision_model
   visualize_curobo_world_collision_model

embodichain.lab.sim.motion.planners.neural_planner
--------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.planners.neural_planner

.. autosummary::

   NeuralPlanner
   NeuralPlannerCfg
   NeuralPlanOptions

embodichain.lab.sim.motion.planners.bezier
------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.planners.bezier

Differentiable quadratic and quintic Bézier geometry, including
HolisticMotion-compatible waypoint blending and path-constraint projection.
Use ``BezierPath.parameter_at_arc_length(distance)`` to convert a scalar
time law's geometric distances to polynomial parameters. Reuse those parameters
for ``evaluate``, ``arc_tangent``, and ``arc_curvature``; normalized arc length
is generally different from the Bézier parameter. Lookup accuracy is controlled
by ``table_count`` independently of the output sample count.

.. autosummary::

   BezierPath
   bezier_arc_length
   bezier_derivative
   bezier_evaluate
   sample_bezier_path

.. automethod:: BezierPath.parameter_at_arc_length

embodichain.lab.sim.motion.planners.toppra_planner
--------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.planners.toppra_planner

Time-optimal joint-path parameterization. The planner solves timing for
waypoint input; :func:`retime_joint_paths` applies the same parameterization to
paths another planner already produced, such as a closed-loop policy rollout
whose samples carry only nominal timing.

.. autosummary::

   ToppraPlanner
   ToppraPlannerCfg
   ToppraPlanOptions
   retime_joint_paths

embodichain.lab.sim.motion.planners.se3
---------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.planners.se3

SE(3) screw interpolation and constrained Cartesian line trajectories with
explicit twist, acceleration, jerk, and timing outputs.

.. autosummary::

   SE3LineResult
   plan_se3_line

embodichain.lab.sim.motion.planners.trapezoidal_planner
-------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.planners.trapezoidal_planner

.. autosummary::

   TrapezoidalPlanOptions
   TrapezoidalPlanner
   TrapezoidalPlannerCfg

embodichain.utils.warp.kinematics.trapezoidal_warp
--------------------------------------------------

.. currentmodule:: embodichain.utils.warp.kinematics.trapezoidal_warp

Warp-accelerated helpers construct scalar trapezoidal or Double-S motion
profiles and compose their sampled path derivatives into batched joint-space
trajectories.

.. autosummary::

   build_profile_warp
   compose_profile_samples_warp

embodichain.lab.sim.motion.planners.utils
-----------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.planners.utils

.. autosummary::

   TrajectorySampleMethod
   MovePart
   MoveType
   PlanState
   PlanResult
   RejectedMotionCandidate
   normalize_success_mask
   calculate_point_allocations
   interpolate_xpos
   interpolate_xpos_batched

embodichain.lab.sim.robots.dual_arm
-----------------------------------

.. currentmodule:: embodichain.lab.sim.robots.dual_arm

.. autosummary::

   DualArmRobotCfg
   build_dual_arm_cfg
   resolve_mounts

embodichain.lab.sim.sensors.camera
----------------------------------

.. currentmodule:: embodichain.lab.sim.sensors.camera

.. autosummary::

   Camera
   CameraCfg

embodichain.lab.sim.sensors.contact.sensor
------------------------------------------

.. currentmodule:: embodichain.lab.sim.sensors.contact.sensor

.. autosummary::

   ArticulationContactFilterCfg
   ContactSensor
   ContactSensorCfg

embodichain.lab.sim.sim_manager
-------------------------------

.. currentmodule:: embodichain.lab.sim.sim_manager

.. autosummary::

   SIM_CACHE_DIR
   MATERIAL_CACHE_DIR
   CONVEX_DECOMP_DIR
   REACHABLE_XPOS_DIR

embodichain.lab.task_program.semantics.calls
------------------------------------------------

.. currentmodule:: embodichain.lab.task_program.semantics.calls

.. autosummary::

   DeclarativeValue
   HandOver
   Pick
   Place
   PlaceRelationTarget
   RegisteredSemanticCall
   SemanticCallCatalog
   SemanticCallDescriptor
   SemanticCallSpec
   SemanticPose
   builtin_semantic_call_catalog

.. automodule:: embodichain.lab.task_program.semantics.calls
   :members:
   :no-index:

embodichain.lab.task_program.semantics.integration
------------------------------------------------------

.. currentmodule:: embodichain.lab.task_program.semantics.integration

.. autosummary::

   BoundSemanticCall
   LinkedSemanticCall
   PathPart
   SceneEntityManifest
   SceneManifest
   SemanticDiagnostic
   SemanticIntegrationManifest
   SemanticValidationError

.. automodule:: embodichain.lab.task_program.semantics.integration
   :members:
   :no-index:

embodichain.lab.task_program.semantics.effects
--------------------------------------------------

.. automodule:: embodichain.lab.task_program.semantics.effects
   :members:
   :no-index:

embodichain.lab.task_program.semantics.evidence
---------------------------------------------------

.. automodule:: embodichain.lab.task_program.semantics.evidence
   :members:
   :no-index:

embodichain.lab.task_program.semantics.profiles
---------------------------------------------------

.. currentmodule:: embodichain.lab.task_program.semantics.profiles

.. autosummary::

   AmbiguousSkillBindingError
   BoundRobotSkillProfile
   ControlPartEndpoint
   ControlPartEndpointAdapter
   EffectAssurance
   EndpointResolution
   ProfileValidationError
   ResourceEndpoint
   ResourceEndpointAdapter
   ResolvedRobotResource
   ResolvedResourceEndpoint
   ResolvedSkillBinding
   ResourceBinding
   ResourceClaim
   RobotResource
   RobotSkillProfile
   SkillPolicyPreset
   UnsupportedSkillError
   WorkflowRecoveryPolicy

embodichain.lab.task_program.semantics.scene
------------------------------------------------

.. currentmodule:: embodichain.lab.task_program.semantics.scene

.. autosummary::

   AmbiguousSceneAffordanceError
   GRASP_AFFORDANCE_CAPABILITY
   PLACE_IN_AFFORDANCE_CAPABILITY
   PLACE_ON_AFFORDANCE_CAPABILITY
   RegistrySceneProvider
   SceneAffordanceRef
   SceneArticulationRef
   SceneCollisionRole
   SceneCollisionWorldMode
   SceneDynamics
   SceneEntityRef
   SceneEntityMetadata
   SceneEntityRegistration
   SceneEntityStateProvider
   SceneGeometryProvider
   SceneLinkRef
   SceneObjectRef
   SceneRegistry
   UnsupportedSceneAffordanceError

.. automodule:: embodichain.lab.task_program.semantics.scene
   :members:
   :no-index:

embodichain.lab.sim.motion.solvers.base_solver
----------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.solvers.base_solver

Shared solver configuration and runtime contracts for IK, FK, joint limits,
and optional continuous batch IK. Continuous selection defaults to unsupported;
implementations opt in through ``supports_continuous_batch_ik`` and implement
the protected ``_select_continuous_ik_path`` hook for batched candidates.

.. autosummary::

   BaseSolver
   SolverCfg

embodichain.lab.sim.motion.solvers.neural_ik_solver
---------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.solvers.neural_ik_solver

.. autosummary::

   NeuralIKSolverCfg
   NeuralIKSolver

embodichain.lab.sim.motion.solvers.null_space_posture_task
----------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.solvers.null_space_posture_task

.. autosummary::

   NullSpacePostureTask

embodichain.lab.sim.motion.solvers.opw_solver
---------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.solvers.opw_solver

Configuration and runtime solver for analytic OPW forward and inverse
kinematics of compatible six-axis manipulators.

.. autosummary::

   OPWSolver
   OPWSolverCfg

embodichain.lab.sim.motion.solvers.pink_solver
----------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.solvers.pink_solver

.. autosummary::

   PinkSolver
   PinkSolverCfg

embodichain.lab.sim.motion.solvers.srs_solver
---------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.solvers.srs_solver

.. autosummary::

   SRSSolver
   SRSSolverCfg

embodichain.lab.sim.spawn
-------------------------

Translation boundary from EmbodiChain object configs and singleton USD assets
into DexSim Spawn descriptors.

.. currentmodule:: embodichain.lab.sim.spawn

.. autosummary::

   articulation_desc_from_cfg
   articulation_desc_from_usd
   rigid_desc_from_cfg
   visual_desc_from_cfg
   rigid_desc_from_usd
   visual_desc_from_usd
   surface_deformable_desc_from_cfg
   volume_deformable_desc_from_cfg

embodichain.lab.sim.spawn.descriptors
-------------------------------------

.. currentmodule:: embodichain.lab.sim.spawn.descriptors

.. autosummary::

   articulation_desc_from_cfg
   configure_articulation_desc
   rigid_desc_from_cfg
   visual_desc_from_cfg
   surface_deformable_desc_from_cfg
   volume_deformable_desc_from_cfg

embodichain.lab.sim.spawn.scene
-------------------------------

.. currentmodule:: embodichain.lab.sim.spawn.scene

.. autosummary::

   SpawnScene

embodichain.lab.sim.spawn.source
--------------------------------

.. currentmodule:: embodichain.lab.sim.spawn.source

.. autosummary::

   resolve_articulation_source

embodichain.lab.sim.spawn.usd
-----------------------------

.. currentmodule:: embodichain.lab.sim.spawn.usd

.. autosummary::

   articulation_desc_from_usd
   rigid_desc_from_usd
   visual_desc_from_usd

embodichain.lab.sim.utility.dynamic_pybind
----------------------------------------

Install the projective UV helper on DexSim render bodies. The helper projects
vertices across mesh segments and adapts UV storage to the pinned DexSim
0.5.1rc1 binding before writing each segment.

.. currentmodule:: embodichain.lab.sim.utility.dynamic_pybind

.. autosummary::

   set_projective_uv
   init_dynamic_pybind

embodichain.lab.sim.utility.render_utils
----------------------------------------

.. currentmodule:: embodichain.lab.sim.utility.render_utils

.. autosummary::

   select_default_renderer

embodichain.lab.sim.motion.workspace.caches.cache_utils
-------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.caches.cache_utils

.. autosummary::

   clean_all_sessions
   clean_session
   format_size
   get_cache_root
   get_dir_size
   list_sessions
   main
   show_session_info
   show_total_size

embodichain.lab.sim.motion.workspace.caches.results_cache
---------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.caches.results_cache

.. autosummary::

   DEFAULT_RESULTS_CACHE_DIR
   ResultsCache
   compute_cache_key
   serialize_results
   deserialize_results

embodichain.lab.sim.motion.workspace.constraints.base_constraint
----------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.constraints.base_constraint

.. autosummary::

   IConstraintChecker
   BaseConstraintChecker

embodichain.lab.sim.motion.workspace.constraints.workspace_constraint
---------------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.constraints.workspace_constraint

.. autosummary::

   WorkspaceConstraintChecker

embodichain.lab.sim.motion.workspace.samplers.base_sampler
----------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.samplers.base_sampler

.. autosummary::

   ISampler
   BaseSampler

embodichain.lab.sim.motion.workspace.samplers.gaussian_sampler
--------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.samplers.gaussian_sampler

.. autosummary::

   GaussianSampler

embodichain.lab.sim.motion.workspace.samplers.halton_sampler
------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.samplers.halton_sampler

.. autosummary::

   HaltonSampler

embodichain.lab.sim.motion.workspace.samplers.importance_sampler
----------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.samplers.importance_sampler

.. autosummary::

   ImportanceSampler

embodichain.lab.sim.motion.workspace.samplers.iniform_sampler
-------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.samplers.iniform_sampler

.. autosummary::

   UniformSampler

embodichain.lab.sim.motion.workspace.samplers.lhs_sampler
---------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.samplers.lhs_sampler

.. autosummary::

   LatinHypercubeSampler

embodichain.lab.sim.motion.workspace.samplers.random_sampler
------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.samplers.random_sampler

.. autosummary::

   RandomSampler

embodichain.lab.sim.motion.workspace.samplers.sobol_sampler
-----------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.samplers.sobol_sampler

.. autosummary::

   SobolSampler

embodichain.lab.sim.motion.workspace.visualizers.axis_visualizer
----------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.visualizers.axis_visualizer

.. autosummary::

   AxisVisualizer

embodichain.lab.sim.motion.workspace.visualizers.base_visualizer
----------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.visualizers.base_visualizer

.. autosummary::

   IVisualizer
   BaseVisualizer

embodichain.lab.sim.motion.workspace.visualizers.manipulability_visualizer
---------------------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.visualizers.manipulability_visualizer

.. autosummary::

   ManipulabilityColorCfg
   ManipulabilityPointSet
   ManipulabilityNormalization
   ManipulabilityColorMapping
   InspectionSelection
   PointInspection
   ManipulabilityVisualizer
   align_manipulability_scores
   normalize_manipulability
   map_manipulability_colors
   select_inspection_indices
   inspect_points
   translational_manipulability_ellipsoid
   ellipsoid_surface

embodichain.lab.sim.motion.workspace.visualizers.point_cloud_visualizer
-----------------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.visualizers.point_cloud_visualizer

.. autosummary::

   PointCloudVisualizer

embodichain.lab.sim.motion.workspace.visualizers.sphere_visualizer
------------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.visualizers.sphere_visualizer

.. autosummary::

   SphereVisualizer

embodichain.lab.sim.motion.workspace.visualizers.visualizer_factory
-------------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.visualizers.visualizer_factory

.. autosummary::

   VisualizerFactory
   create_visualizer

embodichain.lab.sim.motion.workspace.visualizers.voxel_visualizer
-----------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.motion.workspace.visualizers.voxel_visualizer

.. autosummary::

   VoxelVisualizer

embodichain.lab.visualization.backends
--------------------------------------

.. currentmodule:: embodichain.lab.visualization.backends

.. autosummary::

   VisualizationBackend
   ViserBackend

embodichain.lab.visualization.backends.base
-------------------------------------------

.. currentmodule:: embodichain.lab.visualization.backends.base

.. autosummary::

   VisualizationBackend

embodichain.lab.visualization.backends.viser
--------------------------------------------

.. currentmodule:: embodichain.lab.visualization.backends.viser

.. autosummary::

   ViserBackend

embodichain.lab.visualization.cfg
---------------------------------

.. currentmodule:: embodichain.lab.visualization.cfg

.. autosummary::

   VisualizationCfg
   ViserServerCfg

embodichain.lab.visualization.cli
---------------------------------

.. currentmodule:: embodichain.lab.visualization.cli

.. autosummary::

   add_viser_args_to_parser
   visualization_cfg_from_args

embodichain.lab.visualization.picker
------------------------------------

.. currentmodule:: embodichain.lab.visualization.picker

Browser picking caches triangle geometry and returns the closest node hit by
a world-space ray. The Viser worker pairs this geometry with poses from the
same scene revision before producing a pick command.

.. autosummary::

   ScenePicker

.. autoclass:: ScenePicker
   :members:

embodichain.lab.visualization.protocol
--------------------------------------

.. currentmodule:: embodichain.lab.visualization.protocol

.. autosummary::

   SCHEMA_VERSION
   CameraImage
   CameraImageFrame
   CameraSpec
   DynamicMeshUpdate
   FrameOverlay
   GizmoCommand
   GizmoSpec
   GizmoState
   JointControlCommand
   JointControlProvider
   JointControlSpec
   JointControlState
   MeshGeometry
   PickCommand
   PointCloudOverlay
   SceneFrame
   SceneManifest
   SceneNode
   SceneOverlays
   TargetOverlay
   TrajectoryOverlay
   estimate_camera_image_frame_bytes
   estimate_frame_bytes
   estimate_manifest_bytes
   pose_to_position_wxyz

embodichain.lab.visualization.runtime
-------------------------------------

.. currentmodule:: embodichain.lab.visualization.runtime

.. autosummary::

   GizmoCommandQueue
   JointControlCommandQueue
   LatestFrameQueue
   RuntimeHealth
   RuntimeStats
   VisualizationRuntime

embodichain.lab.visualization.scene_exporter
--------------------------------------------

.. currentmodule:: embodichain.lab.visualization.scene_exporter

.. autosummary::

   CameraImageCaptureResult
   CaptureResult
   SceneExporter
   mesh_geometry_id
   safe_path_component

embodichain.learning.rl.algo.apg
--------------------------------

.. currentmodule:: embodichain.learning.rl.algo.apg

.. autosummary::

   APG
   APGCfg
   complete_discounted_return
   segmented_discounted_return

embodichain.learning.rl.algo.base
---------------------------------

.. currentmodule:: embodichain.learning.rl.algo.base

.. autosummary::

   BaseAlgorithm
   RolloutKind

embodichain.learning.rl.algo.common
-----------------------------------

.. currentmodule:: embodichain.learning.rl.algo.common

.. autosummary::

   compute_gae

embodichain.learning.rl.algo.grpo
---------------------------------

.. currentmodule:: embodichain.learning.rl.algo.grpo

.. autosummary::

   GRPO
   GRPOCfg

embodichain.learning.rl.algo.ppo
--------------------------------

.. currentmodule:: embodichain.learning.rl.algo.ppo

.. autosummary::

   PPO
   PPOCfg

embodichain.learning.rl.collector.base
--------------------------------------

.. currentmodule:: embodichain.learning.rl.collector.base

.. autosummary::

   BaseCollector

embodichain.learning.rl.collector.differentiable
------------------------------------------------

.. currentmodule:: embodichain.learning.rl.collector.differentiable

.. autosummary::

   DifferentiableCollector
   DifferentiableRollout
   DifferentiableTransition

embodichain.learning.rl.collector.sync_collector
------------------------------------------------

.. currentmodule:: embodichain.learning.rl.collector.sync_collector

.. autosummary::

   SyncCollector

embodichain.learning.rl.experimental.newton
-------------------------------------------

.. currentmodule:: embodichain.learning.rl.experimental.newton

.. autosummary::

   NewtonPlanarReachEnv
   NewtonPlanarReachEnvCfg

embodichain.learning.rl.experimental.newton.planar_reach
--------------------------------------------------------

.. currentmodule:: embodichain.learning.rl.experimental.newton.planar_reach

.. autosummary::

   NewtonPlanarReachEnv
   NewtonPlanarReachEnvCfg

embodichain.learning.rl.experimental.newton.train_planar_reach
--------------------------------------------------------------

.. currentmodule:: embodichain.learning.rl.experimental.newton.train_planar_reach

.. autosummary::

   NewtonPlanarReachTrainingCfg
   train_planar_reach

embodichain.learning.rl.gradients
---------------------------------

Row-wise action-adjoint clipping and its rollout-level diagnostics.

.. currentmodule:: embodichain.learning.rl.gradients

.. autosummary::

   BatchedGradientNormStats
   clip_batched_gradient_norm

embodichain.learning.rl.policies.actor_critic
---------------------------------------------

.. currentmodule:: embodichain.learning.rl.policies.actor_critic

.. autosummary::

   ActorCritic

embodichain.learning.rl.policies.actor_only
-------------------------------------------

.. currentmodule:: embodichain.learning.rl.policies.actor_only

.. autosummary::

   ActorOnly

embodichain.learning.rl.policies.policy
---------------------------------------

.. currentmodule:: embodichain.learning.rl.policies.policy

.. autosummary::

   Policy

embodichain.learning.rl.normalization
-------------------------------------

.. currentmodule:: embodichain.learning.rl.normalization

.. autosummary::

   RunningObservationNormalizer

embodichain.learning.rl.utils.optimizer
---------------------------------------

.. currentmodule:: embodichain.learning.rl.utils.optimizer

.. autosummary::

   bind_scheduler_horizon
   build_lr_scheduler
   build_optimizer
   coerce_lr_scheduler_cfg
   coerce_optimizer_cfg
   get_registered_lr_scheduler_names
   get_registered_optimizer_names
   scheduler_needs_horizon

embodichain.toolkits.acd
------------------------

.. currentmodule:: embodichain.toolkits.acd

.. autosummary::

   generate_urdf_collision_convexes

embodichain.toolkits.acd.cli
----------------------------

.. currentmodule:: embodichain.toolkits.acd.cli

.. autosummary::

   main

embodichain.toolkits.acd.urdf_modifider
---------------------------------------

.. currentmodule:: embodichain.toolkits.acd.urdf_modifider

.. autosummary::

   URDFModifider

embodichain.toolkits.graspkit.pg_grasp.antipodal_sampler
--------------------------------------------------------

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.antipodal_sampler

.. autosummary::

   AntipodalSamplerCfg
   AntipodalSampler

embodichain.toolkits.graspkit.pg_grasp.collision_checker
--------------------------------------------------------

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.collision_checker

.. autosummary::

   ConvexCollisionCheckerCfg
   ConvexCollisionChecker

embodichain.toolkits.graspkit.pg_grasp.fixed_geometry
---------------------------------------------------

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.fixed_geometry

.. autosummary::

   FixedGripperCollisionGeometry

embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker
----------------------------------------------------------------

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker

.. autosummary::

   GripperCollisionCfg
   GripperCollisionChecker

embodichain.toolkits.graspkit.pg_grasp.model_geometry
---------------------------------------------------

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.model_geometry

.. autosummary::
   :nosignatures:

   get_parallel_jaw_collision_geometry

embodichain.toolkits.graspkit.scripts.annotate_grasp
----------------------------------------------------

.. currentmodule:: embodichain.toolkits.graspkit.scripts.annotate_grasp

.. autosummary::

   cli

embodichain.toolkits.urdf_assembly.component
--------------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.component

.. autosummary::

   ComponentRegistry
   URDFComponent
   URDFComponentManager

embodichain.toolkits.urdf_assembly.connection
---------------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.connection

.. autosummary::

   URDFConnectionManager

embodichain.toolkits.urdf_assembly.file_writer
----------------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.file_writer

.. autosummary::

   URDFFileWriter

embodichain.toolkits.urdf_assembly.logging_utils
------------------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.logging_utils

.. autosummary::

   URDFAssemblyLogger

embodichain.toolkits.urdf_assembly.mesh
---------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.mesh

.. autosummary::

   URDFMeshManager

embodichain.toolkits.urdf_assembly.sensor
-----------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.sensor

.. autosummary::

   SensorRegistry
   SensorAttachment
   URDFSensorManager

embodichain.toolkits.urdf_assembly.signature
--------------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.signature

.. autosummary::

   URDFAssemblySignatureManager

embodichain.toolkits.urdf_assembly.urdf_assembly_manager
--------------------------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.urdf_assembly_manager

.. autosummary::

   URDFAssemblyManager

embodichain_tasks.classic_control.cart_pole
--------------------------------------------

Cart-pole environment registration under the task-first import path.

.. currentmodule:: embodichain_tasks.classic_control.cart_pole

.. autosummary::

   CartPoleEnv

embodichain_tasks.classic_control.point_mass
---------------------------------------------

Differentiable lightweight point-mass task and learning-environment registration.

.. currentmodule:: embodichain_tasks.classic_control.point_mass

.. autosummary::

   PointMassEnv

embodichain_tasks.configs
-------------------------

.. currentmodule:: embodichain_tasks.configs

.. autosummary::

   get_config_path

embodichain_tasks.manipulation.push_cube
----------------------------------------

.. currentmodule:: embodichain_tasks.manipulation.push_cube

.. autosummary::

   PushCubeEnv

embodichain_tasks.manipulation.repeated_pick_place
--------------------------------------------------

RLinf-facing repeated pick-and-place task registrations with explicit EEF and
joint-position action variants.

.. currentmodule:: embodichain_tasks.manipulation.repeated_pick_place

.. autosummary::


embodichain_tasks.manipulation.tableware.blocks_ranking_size
------------------------------------------------------------

.. currentmodule:: embodichain_tasks.manipulation.tableware.blocks_ranking_size

.. autosummary::

   BlocksRankingSizeEnv

embodichain_tasks.manipulation.tableware.match_object_container
---------------------------------------------------------------

.. currentmodule:: embodichain_tasks.manipulation.tableware.match_object_container

.. autosummary::

   MatchObjectContainerEnv

embodichain_tasks.manipulation.tableware.place_object_drawer
------------------------------------------------------------

.. currentmodule:: embodichain_tasks.manipulation.tableware.place_object_drawer

.. autosummary::

   PlaceObjectDrawerEnv

embodichain_tasks.manipulation.tableware.stack_cups
---------------------------------------------------

.. currentmodule:: embodichain_tasks.manipulation.tableware.stack_cups

.. autosummary::

   StackCupsEnv

embodichain_tasks.special.franka_reach_apg
-------------------------------------------

Differentiable Franka FR3 reach environment that demonstrates the explicit
kinematics route used by analytic policy-gradient experiments.

.. currentmodule:: embodichain_tasks.special.franka_reach_apg

.. autosummary::

   FrankaReachApgEnv

embodichain_tasks.utils.importer
--------------------------------

.. currentmodule:: embodichain_tasks.utils.importer

.. autosummary::

   import_packages

Standalone simulation command-line options
-----------------------------------------

``embodichain.cli.sim`` builds simulation arguments without importing the
simulation runtime. Gym launchers compose these options with configuration
and recording flags. Seed options are opt-in: ``resolve_seed`` resolves ``-1``
to an effective 32-bit seed without changing random streams or deterministic
kernel policy; callers log and apply that seed to their own generators.

.. currentmodule:: embodichain.cli.sim

.. autosummary::

   add_sim_args_to_parser
   add_seed_arg_to_parser
   resolve_seed

.. autofunction:: add_sim_args_to_parser

.. autofunction:: add_seed_arg_to_parser

.. autofunction:: resolve_seed


.. toctree::
   :hidden:

   object_asset_bundles


embodichain.lab.sim.cfg.robot
-----------------------------

Robot construction and complete backend alternatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Robot configurations include optional mounted camera declarations.
Standalone simulation initializes them only when ``enable_sensor`` is true;
Gym uses its environment-wide sensor switch, which defaults to enabled.

.. currentmodule:: embodichain.lab.sim.cfg.robot

.. autosummary::
   :nosignatures:

   RobotCfg
   RobotPresetCfg

.. autoclass:: RobotCfg
   :members: sensors, enable_sensor
   :noindex:

.. autoclass:: RobotPresetCfg
   :noindex:


Asset generalization contracts
------------------------------

.. toctree::
   :maxdepth: 2

   asset_generalization

Real-to-simulation motion prototypes
------------------------------------

.. toctree::
   :maxdepth: 2

   real2sim

embodichain.lab.gym.envs.managers.cfg
-------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.managers.cfg

Manager configuration binds named functors and explicit scene entities. RL condition groups aggregate success with conjunction, failure and truncation with disjunction.

.. autosummary::

   FunctorCfg
   ConditionCfg
   RLTaskCfg
   EventCfg
   ObservationCfg
   SceneEntityCfg
   RewardCfg
   ActionTermCfg
   DatasetFunctorCfg

.. autoclass:: FunctorCfg
   :members:

.. autoclass:: ConditionCfg
   :members:

.. autoclass:: RLTaskCfg
   :members:

.. autoclass:: EventCfg
   :members:

.. autoclass:: ObservationCfg
   :members:

.. autoclass:: SceneEntityCfg
   :members:

.. autoclass:: RewardCfg
   :members:

.. autoclass:: ActionTermCfg
   :members:

.. autoclass:: DatasetFunctorCfg
   :members:


embodichain.lab.gym.envs.managers.conditions
--------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.managers.conditions

Measured manipulation functors bind rigid objects and validate geometric parameters at construction. Calls read live physical state and reset-owned virtual targets. Missing target contexts fail explicitly.

.. autosummary::

   object_near_target
   object_below_height

.. autoclass:: object_near_target
   :members:
   :special-members: __call__

.. autoclass:: object_below_height
   :members:
   :special-members: __call__


embodichain.lab.gym.envs.managers.termination_manager
-----------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.managers.termination_manager

The termination manager combines condition measurements with a host-supplied time limit once per control transition. It owns the cached Gym flags and task acceptance, resets selected rows and publishes owned snapshots without resampling.

.. autosummary::

   TerminationManager

.. autoclass:: TerminationManager
   :members:


embodichain.lab.gym.envs.objectives.conditions
----------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.objectives.conditions

Typed physical predicates provide shared distance and height measurements for RL and Task Program adapters. Invalid evidence never satisfies a condition.

.. autosummary::

   ConditionResult
   evaluate_near_target
   evaluate_below_height

.. autoclass:: ConditionResult
   :members:

.. autofunction:: evaluate_near_target

.. autofunction:: evaluate_below_height

embodichain.lab.sim.evidence
----------------------------

Shared physical observation values. Atomic public imports refer to these same
classes; sensor acquisition never imports Task Program or Atomic implementation.

.. currentmodule:: embodichain.lab.sim.evidence

.. autosummary::

   PhysicalEvidenceRequest
   PhysicalEvidenceBatch
   PhysicalEvidenceFrame

embodichain.compute.geometry.grip_contacts
------------------------------------------

Pure bilateral contact arithmetic on explicitly supplied actor pairs, normals
and impulses. Invalid or ambiguous data remains distinct from a measured negative.

.. currentmodule:: embodichain.compute.geometry.grip_contacts

.. autosummary::

   classify_contacts

embodichain.lab.sim.atomic_actions.grip_evidence
------------------------------------------------

Shared physical grip interpretation consumes a PhysicalEvidenceRequest and
explicit actor bindings, without Semantic Call or Task Program types. Results
are PhysicalEvidenceBatch values; semantic clause adaptation belongs to integrations.

.. currentmodule:: embodichain.lab.sim.atomic_actions.grip_evidence

.. autosummary::

   GripContactObserver
   DriverOpeningBinding

``DriverOpeningBinding.bind(robot)`` validates the explicit joint order before
creating a per-environment reader. It performs no qpos read during binding;
observations require the entire hand row to be finite and retain the strict
``abs(measured - open_qpos) < tolerance`` driver threshold. This calibration and
release interpretation belong to Atomic, not to a separate sensor or reader
factory. Integrations only select and bind the declared calibration.

The former ``sensors.joint_opening`` module and standalone
``create_joint_opening_reader`` export are removed. Declarations sealing the
qualified DriverOpeningBinding type must be rebuilt with its Atomic owner.

.. autoclass:: DriverOpeningBinding
   :members:

Stability execution ownership
----------------------------

A simulation factory with native support owns one unfinished bridge at a time.
``validate_runtime_bridge_creation`` rejects overlap before rebinding clocks;
``bind_runtime_bridge`` records the bridge created by the standard adapter.
``TaskProgramDemoBridge.execution_finished`` requires closed segment iteration
and a consumed/validated or explicitly aborted final segment, with no active
action generator. It is distinct from ``program_completed`` and does not certify
success or acknowledge commands. Non-support factories retain existing assembly.

Factory-returned effect providers retain their public IDs/revisions and guard
all collection through the shared lifetime. The same guard protects live scene
snapshots and drive-target holds. Independently owned past reports remain readable.

The source control grid, stability counters and held-failure latch are unchanged.
CPU policy profiling can run with
``python -m scripts.benchmark.task_program.stability --output /tmp/stability``.
This measures constant-pose policy overhead and serialization separately from
native contact acquisition, physical tasks and GPU execution.
