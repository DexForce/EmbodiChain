Asset generalization
====================

Asset generalization preserves logical task roles while selecting compatible rigid assets.
Source protocols live in data; task requirements remain in Task Program semantics.
Online generation adapters are deferred.

embodichain.data.asset_catalog
------------------------------

Provider-neutral asset records and the candidate/materialization protocol.

.. currentmodule:: embodichain.data.asset_catalog

.. autosummary::

   AssetCandidate
   AssetDescriptor
   AssetProvider
   AssetQuery
   Bounds3D
   PreparationContext
   PreparedAsset

.. autoclass:: embodichain.data.asset_catalog.AssetCandidate
   :members:

.. autoclass:: embodichain.data.asset_catalog.AssetDescriptor
   :members:

.. autoclass:: embodichain.data.asset_catalog.AssetProvider
   :members:

.. autoclass:: embodichain.data.asset_catalog.AssetQuery
   :members:

.. autoclass:: embodichain.data.asset_catalog.Bounds3D
   :members:

.. autoclass:: embodichain.data.asset_catalog.PreparationContext
   :members:

.. autoclass:: embodichain.data.asset_catalog.PreparedAsset
   :members:

embodichain.data.asset_catalog.contracts
----------------------------------------

Immutable JSON-compatible records. Geometry is explicitly normalized to metres and Z-up; prepared identity covers resources and annotations.

.. currentmodule:: embodichain.data.asset_catalog.contracts

.. autosummary::

   Bounds3D
   AssetDescriptor
   AssetCandidate
   PreparedAsset
   AssetQuery
   PreparationContext
   AssetProvider

.. autoclass:: embodichain.data.asset_catalog.contracts.Bounds3D
   :members:

.. autoclass:: embodichain.data.asset_catalog.contracts.AssetDescriptor
   :members:

.. autoclass:: embodichain.data.asset_catalog.contracts.AssetCandidate
   :members:

.. autoclass:: embodichain.data.asset_catalog.contracts.PreparedAsset
   :members:

.. autoclass:: embodichain.data.asset_catalog.contracts.AssetQuery
   :members:

.. autoclass:: embodichain.data.asset_catalog.contracts.PreparationContext
   :members:

.. autoclass:: embodichain.data.asset_catalog.contracts.AssetProvider
   :members:

embodichain.lab.task_program.semantics.asset_requirements
---------------------------------------------------------

Task-owned role requirements and explicit category aliases. Geometric relations require declared usable regions and preserve task semantics.

.. currentmodule:: embodichain.lab.task_program.semantics.asset_requirements

.. autosummary::

   AssetRoleRequirement
   AssetRelationRequirement
   AssetRequirements

.. autoclass:: embodichain.lab.task_program.semantics.asset_requirements.AssetRoleRequirement
   :members:

.. autoclass:: embodichain.lab.task_program.semantics.asset_requirements.AssetRelationRequirement
   :members:

.. autoclass:: embodichain.lab.task_program.semantics.asset_requirements.AssetRequirements
   :members:


embodichain.data.asset_catalog additions
----------------------------------------

Public source adapters and content verification for normalized local asset snapshots.

.. currentmodule:: embodichain.data.asset_catalog

.. autosummary::

   verify_prepared_asset
   DirectoryAssetProvider
   ExplicitAssetProvider

.. autofunction:: embodichain.data.asset_catalog.verify_prepared_asset

.. autoclass:: embodichain.data.asset_catalog.DirectoryAssetProvider
   :members:

.. autoclass:: embodichain.data.asset_catalog.ExplicitAssetProvider
   :members:


embodichain.data.asset_catalog.local additions
----------------------------------------------

Directory and explicit providers retain candidate metadata without claiming task compatibility. Materialization verifies real mesh bounds, local resource dependencies and the content-addressed snapshot. Other providers can emit the same verifiable package inventory.

.. currentmodule:: embodichain.data.asset_catalog.local

.. autosummary::

   DirectoryAssetProvider
   ExplicitAssetProvider
   verify_prepared_asset

.. autoclass:: embodichain.data.asset_catalog.local.DirectoryAssetProvider
   :members:

.. autoclass:: embodichain.data.asset_catalog.local.ExplicitAssetProvider
   :members:

.. autofunction:: embodichain.data.asset_catalog.local.verify_prepared_asset

embodichain.lab.task_program.semantics.asset_matching additions
---------------------------------------------------------------

Bounded deterministic selection applies hard role constraints and upright geometric relations before freezing an asset set. Unknown metadata cannot satisfy required constraints; a resolved set does not certify physical success.

.. currentmodule:: embodichain.lab.task_program.semantics.asset_matching

.. autosummary::

   AssetRejection
   AssetResolutionError
   ResolvedAssetSet
   resolve_asset_set

.. autoclass:: embodichain.lab.task_program.semantics.asset_matching.AssetRejection
   :members:

.. autoclass:: embodichain.lab.task_program.semantics.asset_matching.AssetResolutionError
   :members:

.. autoclass:: embodichain.lab.task_program.semantics.asset_matching.ResolvedAssetSet
   :members:

.. autofunction:: embodichain.lab.task_program.semantics.asset_matching.resolve_asset_set

embodichain.lab.task_program.integrations.asset_materialization additions
-------------------------------------------------------------------------

Materialize standalone configurations and semantic bindings without modifying templates. Recompute supported placement frames and reject unsupported fixed calibrations. Validate packaged content before loading.

.. currentmodule:: embodichain.lab.task_program.integrations.asset_materialization

.. autosummary::

   AssetAdaptationCfg
   MaterializedAssetDeployment
   load_asset_requirements_from_deployment
   materialize_asset_deployment
   validate_materialized_deployment

.. autoclass:: embodichain.lab.task_program.integrations.asset_materialization.AssetAdaptationCfg
   :members:

.. autoclass:: embodichain.lab.task_program.integrations.asset_materialization.MaterializedAssetDeployment
   :members:

.. autofunction:: embodichain.lab.task_program.integrations.asset_materialization.load_asset_requirements_from_deployment

.. autofunction:: embodichain.lab.task_program.integrations.asset_materialization.materialize_asset_deployment

.. autofunction:: embodichain.lab.task_program.integrations.asset_materialization.validate_materialized_deployment

embodichain.lab.gym.envs.asset_generalization additions
-------------------------------------------------------

Prepare bounded asset batches, build fresh single-environment task runtimes, execute through the canonical scene acceptance path, and preserve reset/close recording boundaries. Initial and measured validation callbacks are required; submitted resets are not persistence receipts.

.. currentmodule:: embodichain.lab.gym.envs.asset_generalization

.. autosummary::

   AssetGeneralizationCfg
   AssetEpisodeOutcome
   AssetBatchOutcome
   prepare_asset_batches
   collect_asset_batches

.. autoclass:: embodichain.lab.gym.envs.asset_generalization.AssetGeneralizationCfg
   :members:

.. autoclass:: embodichain.lab.gym.envs.asset_generalization.AssetEpisodeOutcome
   :members:

.. autoclass:: embodichain.lab.gym.envs.asset_generalization.AssetBatchOutcome
   :members:

.. autofunction:: embodichain.lab.gym.envs.asset_generalization.prepare_asset_batches

.. autofunction:: embodichain.lab.gym.envs.asset_generalization.collect_asset_batches
