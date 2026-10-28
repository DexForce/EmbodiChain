# Data assets and path resolution

This topic owns asset paths, dataset-class discovery and download coordination.
The data package exposes its asset namespace lazily so extension assets can
import the shared dataset base before their own classes finish initialization.
Online sampling and demonstration persistence belong to
[data-pipeline](../data-pipeline/data-pipeline.md).

## Entry points

| Request | Owner |
|---|---|
| Asset path lookup | `embodichain/data/dataset.py`: `get_data_path()`, `get_data_class()` |
| Cache roots / mirror prefix | `embodichain/data/constants.py` |
| Locked download and ZIP integrity | `embodichain/data/dataset.py`: `EmbodiChainDataset`, `_dataset_download_lock()` |
| Shared class discovery | `embodichain/data/_asset_registry.py`: core modules and `embodichain.assets` catalogs |
| Robot-owned asset definitions | `extensions/embodichain_robots/_asset_catalog.py` → vendor/model `assets.py` |
| CLI list/download | `embodichain/data/download.py`: `get_registry()`, `download_asset()` |
| Default material catalog / provenance | `embodichain/_material_catalog.py` → `embodichain/data/default_materials.py`; `embodichain/data/default_materials.json` |
| Default material archive | `embodichain/data/assets/materials.py`: `DefaultMaterials` |
| Dataset class visibility | `embodichain/data/__init__.py`, `embodichain/data/assets/__init__.py` |
| Pretrained policy bundles | `embodichain/data/assets/policy_assets.py`: `download_pretrained_policy()` |
| Unified CLI registration | `embodichain/cli/main.py`: `embodichain data` |

## Resolution path

Root constants are evaluated at import time. `EMBODICHAIN_DATA_ROOT` defaults
to `~/.cache/embodichain_data`; `EMBODICHAIN_DATASET_ROOT` defaults to
`~/.cache/embodichain_datasets`. Database storage uses
`~/.cache/embodichain/database`. `EMBODICHAIN_DOWNLOAD_PREFIX` overrides the
base URL for asset archives (default: the hf-mirror.com copy of
`DexForceAI/embodichain_data`). Set overrides before importing consumers.

`get_data_path(path)` resolves in this order:

1. Absolute input is returned unchanged, without existence checks.
2. An existing relative path under the data root is returned.
3. Otherwise, the first path segment selects a class from the shared core and
   extension asset index. Construct it, then append the subpath under `extract_dir`.

Lookup does not verify the final nested file. Check the shared index and installed
entry points for missing IDs. Explicit loaded modules remain available through
`extra_modules` after registered IDs; package re-exports do not register assets.

## Download and registry boundaries

`EmbodiChainDataset` locks `<data_root>/download/.locks/<prefix>.lock` around
both ZIP integrity repair and the Open3D constructor's download/extraction.
Keep the lock outside the directory repair may delete. POSIX uses `flock`;
Windows uses a polled byte lock. Direct Open3D subclasses do not automatically
receive the project's locking/repair behavior.

The `task` category registers `PourCupAssets` runtime meshes and
`PourCupDemoData` historical recordings through `assets/task_assets.py`.
Both use the shared download cache and class-prefixed paths.

The CLI honors an asset class's optional `cache_prefix` when checking readiness
and ensuring extraction; otherwise the class name is the cache entry.
The shared registry merges core category modules and `embodichain.assets`
entry-point mappings. Each extension entry maps an asset ID to `class_path`
(`module:Class`) and `category`; metadata must not import implementations.
Dataset lookup and the download CLI use this same index. Duplicate IDs, including
case-only collisions, fail explicitly. CLI categories come from the merged index;
the public `CATEGORY_MODULES` mapping remains a single-module compatibility view.
Catalogs are cached for the process lifetime; refresh editable metadata and restart
after changing entry points.

Migrated robot assets live in `embodichain_robots`, alongside each model or at
vendor level for shared archives. The legacy `robot_assets`, `w1_assets`,
`so101_assets` and migrated `locomotion_assets` names forward to the same class objects.
Discovery excludes these imported aliases so IDs register once. Keep class names,
asset IDs, URLs, digests and cache prefixes stable across ownership moves.
The six catalog locomotion models register their model-owned archives through
the robot asset catalog. `HumanoidRun` and remaining core robot datasets still
use the core category table, which excludes private helpers. Locomotion archives
keep their existing `robot_assets/<ClassName>.zip` paths. Their common archive
downloader lives in `embodichain/data/_locomotion_asset.py`, outside the eager
legacy asset namespace so asset-first imports do not re-enter model construction.
`download_asset()` returns success or failure; `cmd_download()` finishes the batch
and exits nonzero if any requested asset failed.
`_ensure_extract()` copies non-ZIP downloads into the extract tree
when needed. Solver/planner checkpoint download helpers can have separate paths.

`SO101FollowerAssets` in `embodichain_robots.the_robot_studio.so101.assets` also belongs to the robot CLI
category. It resolves a pinned official URDF and individual STL/reference files,
verifies every SHA256, and repairs dependencies under the shared dataset lock.
Keep its source revision and complete file checksums together when updating it.

ZIP bundles must include directory entries before nested files for Open3D
extraction. Validate new bundles through `get_data_path()` with an empty cache.

`UniversalRobots` selects the V4 archive with corrected UR3/UR3e material
colors. Its versioned `UniversalRobotsV4` cache directory changes source paths
so URDF assembly also refreshes previously copied mesh materials; public
`UniversalRobots/...` asset references remain unchanged.

Demo classes live in `assets/demo_assets.py` and are exported through the existing
asset registry. `TabletopFourBases/scene_export` resolves the four-support rigid
tabletop example from `demo/tabletop_four_bases.zip`. Its portable source contains
only scene configuration, explicit support graph and referenced GLBs; baselines,
support caches, layouts and previews belong to a separate caller-owned workspace.
The registered descriptor respects the configured host and falls back to the
official Hugging Face dataset. The scene tutorial downloads by asset reference
by default and accepts local exports through `--source`.

## Default material resources

`list_default_materials()` and `get_default_material_info()` are exported from
`embodichain.data` and query wheel-packaged metadata without a World or download.
The shared implementation lives in `_material_catalog.py` so CLI catalog queries
can avoid importing the data package's Open3D/download dependencies. Preserve the
public data exports and one catalog implementation. `setup.py` must include
that catalog explicitly. `DefaultMaterials` uses a
revision-specific cache prefix and the shared dataset lock, verifies archive
SHA256 and member digests, and repairs extraction from the verified archive.
Its data CLI registration follows the existing materials category.

The same catalog includes six built-in parameter presets with their own revision,
native parameters and no texture size. Metadata distinguishes `source` and
`requires_download`; only resource entries belong to the `DefaultMaterials`
archive. The runtime resolves built-ins directly from wheel metadata before any
dataset construction. Keep their definitions out of the upstream ZIP checksum.

The original DexSim archive is currently on its internal resource server; an
explicit `EMBODICHAIN_DOWNLOAD_PREFIX` selects the catalog's versioned mirror
path. Do not substitute an unverified public URL or depend on a DexSim source
checkout. Preserve upstream manifests, relative auxiliary texture dependencies
and attribution. The material runtime stages JSON only and does not copy PNGs;
see [visual materials](../simulation-system/materials.md) for renderer, override
and native ownership rules.

## Pretrained policy bundles

`eval-policy --pretrained` reads the official model index at a pinned HF commit,
then downloads only `policies/<model-id>/`. Named revisions resolve once before
both downloads. Files use a revision-specific `local_dir` because RunManifest
rejects paths resolving outside its RUN, including default HF blob symlinks.
The CLI records repository provenance and writes reports outside the model
cache. Robot references still use `get_data_path()` and its existing asset cache.
Task packages must already be installed; downloading weights does not register
new environment implementations. Test the Hub boundary with fake downloads and
validate a real selected-model download separately.

## Failure diagnosis and change sites

| Symptom | Inspect |
|---|---|
| Missing class or CLI item | Shared index, installed asset catalogs, first path segment and alias filtering |
| Download races | Inheritance from `EmbodiChainDataset` and one lock scope spanning repair + extraction |
| Corrupt archive repeatedly retained | `check_zip()` MD5 checks and `is_safe_path()` cleanup guard, particularly with custom roots |
| Returned path is absent | Caller-owned absolute path or unverified nested extracted path |
| Unexpected path for a bare class reference | Bare class names resolve to `extract_dir`; a supplied subpath is appended beneath it |
| Automation sees success after download failure | Verify the caller preserves `cmd_download()`'s nonzero exit status |

Change path policy in `dataset.py`, root defaults in `constants.py`, registry
visibility in `_asset_registry.py`/extension catalogs, and asset URLs/checksums in the
owning asset class. Do not alter a global root to repair one caller's path.

## Focused validation

`tests/data/test_dataset.py` covers concurrent locked downloads. For resolver
changes, exercise absolute, existing-relative and class-prefixed paths plus
missing/bare prefixes and a custom data root. For CLI changes, cover category
visibility and failure exit behavior. Use fake download backends for these
contracts; separately validate actual network downloads when asset content changes.
