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
`examples/data/asset_catalog/README.md`.

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
