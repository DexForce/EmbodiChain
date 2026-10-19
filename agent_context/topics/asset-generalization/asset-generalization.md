# Asset generalization

Provider-neutral asset records and candidate/materialization protocols live in
`embodichain/data/asset_catalog/`. Task-owned role requirements and geometric
relations live in `embodichain/lab/task_program/semantics/asset_requirements.py`.
These immutable contracts are independent of external asset services. Provider
implementations own I/O and hashing; construction of a `PreparedAsset` only
validates its declaration and does not certify file integrity or task success.

Geometry is rigid, in metres and the right-handed Z-up asset frame. Bounds of
an object are distinct from annotated usable support/container regions. Unknown
metadata is retained as unknown, and category names do not establish capabilities.
Normalization is an explicit preparation assertion, never inferred from names.

Requirements retain stable program entity IDs, exact property constraints and
explicit category aliases. These contracts do not change program syntax or
instantiate a simulator. Online generation is a separate deferred adapter.

Focused validation: `tests/data/asset_catalog/` and
`tests/lab/task_program/semantics/test_asset_requirements.py`.

## Local sources

`ExplicitAssetProvider` and `DirectoryAssetProvider` read `asset.catalog.json`
packages. A package declares normalized geometry and usable regions; existing
SimReady `asset.json` can supply category/source evidence through an explicit
adapter field, but does not establish task capabilities or units. Unknown raw
files require annotations before preparation. The example manifest lives in
`examples/expansion/asset_catalog/README.md`.

Preparation checks real mesh bounds and file dependencies, then hashes and copies
the complete package into an immutable snapshot. `verify_prepared_asset` is the
common integrity boundary for any provider's prepared output; its package-root,
file-inventory and manifest contract is documented in that function. Changing
source content creates a new identity; changing cached content is an error.

## Task matching

`semantics/asset_matching.py` owns bounded role matching. It resolves explicit
category aliases, checks hard properties/capabilities and dimensions, and prunes
upright support/containment assignments as roles are selected. It never builds
the full Cartesian product. Candidate and search-node budgets are independent;
exhaustion produces structured rejection evidence rather than a relaxed match.
`ResolvedAssetSet` freezes selected content and requirement identity with a seed.
Absolute cache paths are excluded from its identity. Same asset selection does
not establish reachable grasps, settled state or measured task success.

## Deployment materialization

`integrations/asset_materialization.py` reads integration-owned
`asset_requirements`, verifies prepared content and emits standalone deployment
artifacts. It preserves logical entity IDs and derives new Gym/integration
identities. Explicit `./` or `../` resource paths resolve from the component
that declares them; bare paths retain dataset registry semantics.

Physical scene and physics-backend ownership are unchanged. Upright normalized
mesh replacement supports one visual/collision resource and explicit relative
support/container placement. Both placement objects require declared roles and
relations. Absolute placement targets, calibrated grasps/axes, local-pose
overrides, non-unit scale and geometry-specific expansion profiles require
explicit supported adaptation; otherwise they are rejected. Independent objectives are preserved,
not rewritten to make a candidate pass. Container collision must preserve its
annotated cavity; a default convex hull does not qualify.

The materialized manifest binds assets and configuration bytes. Validate it
before constructing environments, including when restoring an artifact without
the optional manifest digest: selected package inventories and referenced local
resources must remain complete. Generated asset paths retain their output owner.
Local resources bundle only their dependency closure under a content digest;
their bytes contribute to deployment identity without source/output path dependence.
Packaged robot/data IDs still require the installed SDK and its normal asset
cache; local selected asset files are bundled.
Focused tests use the production loader in
`tests/gym/envs/task_program/test_asset_materialization.py`.

## Batch host and recording

`lab/gym/envs/asset_generalization.py` owns `AssetGeneralizationCfg`, bounded
provider preparation and sequential collection. `prepare_asset_batches` freezes
one candidate snapshot per role, retains preparation failures in a report and
materializes seeded combinations. `collect_asset_batches` verifies each artifact
before construction and uses the canonical scene-variant executor plus
`SimulationSceneExpansionHost` with no pose changes. Assets are fixed inside a
batch; a new B=1 environment is created for every next deployment.

Both initial and final measured validators are mandatory. Empty/unavailable
checks and incomplete programs cannot save data. Asset provenance is attached
before the ordinary commit/discard reset. Close finalizes recorders and discards
uncommitted buffers; cleanup drains only after successful host frames unwind.
Construction/execution errors preserve their original exception and require
caller-owned deferred cleanup after traceback release.

`record_trajectory=True` explicitly enables accepted-episode trajectory files
in the default collection factory. Each invocation scopes dataset, camera and
trajectory recorder directories beneath a fresh deployment-local run directory.
Custom environment factories own equivalent isolation. Reports distinguish
acceptance, returned reset requests and close finalization; none substitutes
for a per-episode persistence receipt. Normal configuration loading does not
select providers or enable generation.

Examples: `examples/expansion/asset_generalization/run.py` uses direct Cube primitives
for the UR5 Pick/Place baseline, without asset catalogs.
`examples/expansion/asset_generalization/pour_water.py` exercises actual bottle/cup
asset replacement and calibrated CobotMagic pouring-motion acceptance.
Host tests live in `tests/gym/envs/test_asset_generalization.py`; existing scene
expansion tests remain authoritative for canonical execution and reject gates.

Acceptance configurations live beside the examples under
`examples/expansion/asset_generalization/configs/`, with separate primitive
cube/platform and bottle/cup cases. They reuse SDK embodiments/policies without
adding official task deployments. Programs, bindings, asset requirements and
physical environment files remain separate. Ordinary `run-task --gym-config`
loading does not select assets or invoke Providers.

Asset replacement currently belongs to pre-build preparation and batch
collection, not a manager-bound Event functor. `BaseEnv` prepares native topology
before `EmbodiedEnv._init_sim_state` constructs managers and dispatches startup
events. Switching prepared meshes therefore constructs a fresh environment and
binding snapshot. Same-topology pose/material randomization remains event-owned;
there is no pre-build or batch-stage functor adapter for asset preparation yet.


## Calibrated bottle/cup pouring

The `pours_into` role relation checks explicit mouth-radius and aperture evidence
in the descriptor's hashed `provenance.pouring` record. The bottle needs
`pour_source` and `antipodal_grasp`, the cup needs `container`; matching never
interprets a whole bottle as an object that must fit inside the cup.

`AssetPourAdaptationCfg`, exported by the materializer, enables one deliberately
bounded four-segment Pick/Move/Pour/Place adaptation. It requires per-asset mouth
geometry, a proper object-to-TCP grasp transform tied to the exact embodiment
profile, and an explicit object-local tilt axis. The adapter recomputes the
cup-relative staging pose for the existing origin-centered Pour primitive,
creates the semantic return support, and conservatively updates validators.
Pouring position validators explicitly use the live parent's frame so cup yaw
rotates both transport targets and their measured acceptance offsets.
Generic calibration rejection remains in effect outside this declared adapter.

`integrations/simulation/asset_pouring.py` owns `PouringAcceptanceMonitor`.
Hosts sample real object poses after steps and identify the pour segment. Tilt,
whole-mouth projection inside the cup aperture, and positive mouth height must
hold on the same samples. Cup excursions and final bottle return/uprightness
remain separate checks. Missing evidence rejects; rigid-body geometry does not
measure liquid transfer. Default example models are explicitly labelled size
derivatives of existing PourWater assets; external catalogs need their own
calibration and retain the same matching/materialization/acceptance path.
