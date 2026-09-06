# Data assets and path resolution

This topic owns asset paths, dataset-class discovery and download coordination.
Online sampling and demonstration persistence belong to
[data-pipeline](../data-pipeline/data-pipeline.md).

## Entry points

| Request | Owner |
|---|---|
| Asset path lookup | `embodichain/data/dataset.py`: `get_data_path()`, `get_data_class()` |
| Cache roots / mirror prefix | `embodichain/data/constants.py` |
| Locked download and ZIP integrity | `embodichain/data/dataset.py`: `EmbodiChainDataset`, `_dataset_download_lock()` |
| CLI list/download | `embodichain/data/download.py`: `CATEGORY_MODULES`, `get_registry()`, `download_asset()` |
| Dataset class visibility | `embodichain/data/__init__.py`, `embodichain/data/assets/__init__.py` |
| Unified CLI registration | `embodichain/cli/main.py`: `embodichain data` |

## Resolution path

Root constants are evaluated at import time. `EMBODICHAIN_DATA_ROOT` defaults
to `~/.cache/embodichain_data`; `EMBODICHAIN_DATASET_ROOT` defaults to
`~/.cache/embodichain_datasets`. Database storage uses
`~/.cache/embodichain/database`. Set overrides before importing consumers.

`get_data_path(path)` resolves in this order:

1. Absolute input is returned unchanged, without existence checks.
2. An existing relative path under the data root is returned.
3. Otherwise, the first path segment selects a dataset class in already-loaded
   data modules. Construct it, then append the remaining path under `extract_dir`.

The final nested asset path is not independently verified. A successful lookup
can trigger download/extraction and still name a nonexistent nested file.
Inspect `get_data_class()` and package imports when a newly added class is invisible.

## Download and registry boundaries

`EmbodiChainDataset` locks `<data_root>/download/.locks/<prefix>.lock` around
both ZIP integrity repair and the Open3D constructor's download/extraction.
Keep the lock outside the directory repair may delete. POSIX uses `flock`;
Windows uses a polled byte lock. Direct Open3D subclasses do not automatically
receive the project's locking/repair behavior.

The CLI imports the fixed `CATEGORY_MODULES` and finds `DownloadDataset`
subclasses defined in each module. Adding an asset module alone does not add a
CLI category. `_ensure_extract()` copies non-ZIP downloads into the extract tree
when needed. Solver/planner checkpoint download helpers can have separate paths.

## Failure diagnosis and change sites

| Symptom | Inspect |
|---|---|
| Missing class or CLI item | Package imports, first path segment, `CATEGORY_MODULES` and module ownership filtering |
| Download races | Inheritance from `EmbodiChainDataset` and one lock scope spanning repair + extraction |
| Corrupt archive repeatedly retained | `check_zip()` MD5 checks and `is_safe_path()` cleanup guard, particularly with custom roots |
| Returned path is absent | Caller-owned absolute path or unverified nested extracted path |
| Bare class name fails | The remainder join in `get_data_path()`; do not assume a class-only reference returns the extraction root |
| Automation sees success after download failure | `download_asset()` prints caught errors; inspect aggregate CLI exit handling |

Change path policy in `dataset.py`, root defaults in `constants.py`, registry
visibility in package exports/`download.py`, and asset URLs/checksums in the
owning asset class. Do not alter a global root to repair one caller's path.

## Focused validation

`tests/data/test_dataset.py` covers concurrent locked downloads. For resolver
changes, exercise absolute, existing-relative and class-prefixed paths plus
missing/bare prefixes and a custom data root. For CLI changes, cover category
visibility and failure exit behavior. Use fake download backends for these
contracts; separately validate actual network downloads when asset content changes.
