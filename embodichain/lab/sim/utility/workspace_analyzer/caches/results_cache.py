# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Persistent results cache for :class:`WorkspaceAnalyzer`.

The low-level :class:`DiskCache` / :class:`MemoryCache` caches store raw sample
poses. This module caches the *analysis results* (reachable workspace points,
joint configurations, metrics and metadata) keyed by a content hash of the
inputs that affect the output. It backs
:meth:`WorkspaceAnalyzer._save_to_cache` / :meth:`WorkspaceAnalyzer._load_from_cache`
and lets other applications (e.g. environment data generation) reuse a computed
workspace without re-running sampling + FK/IK.

On-disk layout::

    <cache_dir>/<key>/
        results.npz   # workspace_points, reachable_points, all_points,
                      # joint_configurations, success_rates, reachability_mask
        meta.json     # mode, counts, metrics, analysis_time, config snapshot

The ``results.npz`` file can be loaded directly by other applications via
``np.load``; ``meta.json`` carries the scalar context.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embodichain.utils import logger

__all__ = [
    "DEFAULT_RESULTS_CACHE_DIR",
    "ResultsCache",
    "compute_cache_key",
    "serialize_results",
    "deserialize_results",
]

DEFAULT_RESULTS_CACHE_DIR = os.path.expanduser(
    "~/.cache/embodichain/workspace_analyzer/results"
)

# Tensor/array fields stored in the .npz archive.
_TENSOR_FIELDS = (
    "workspace_points",
    "all_points",
    "reachable_points",
    "joint_configurations",
    "success_rates",
    "reachability_mask",
)

# Scalar/dict fields stored in meta.json (JSON-serializable form).
_META_FIELDS = (
    "mode",
    "num_samples",
    "num_valid",
    "num_reachable",
    "metrics",
    "analysis_time",
    "constraint_statistics",
    "plane_sampling_config",
)


def _to_jsonable(value: Any) -> Any:
    """Convert a value to a JSON-serializable representation.

    Args:
        value: A tensor, numpy array, numpy scalar, or nested container.

    Returns:
        A JSON-serializable representation (lists, floats, strings, ...).
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def compute_cache_key(metadata: dict) -> str:
    """Compute a stable content-hash key for analysis inputs.

    Args:
        metadata: Dictionary of all inputs that affect the analysis output
            (robot identity, mode, sampling, constraints, ...).

    Returns:
        A short hex string (16 chars) used as the cache directory name.
    """
    canonical = json.dumps(_to_jsonable(metadata), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def serialize_results(results: dict) -> tuple[dict, dict]:
    """Split an analysis results dict into array and meta parts.

    Args:
        results: The results dict returned by
            :meth:`WorkspaceAnalyzer.analyze`.

    Returns:
        A tuple ``(arrays, meta)`` where ``arrays`` maps tensor field names to
        numpy arrays and ``meta`` maps scalar field names to JSON-able values.
    """
    arrays: dict[str, np.ndarray] = {}
    for key in _TENSOR_FIELDS:
        value = results.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        arrays[key] = np.asarray(value)

    meta: dict[str, Any] = {}
    for key in _META_FIELDS:
        if key in results:
            meta[key] = _to_jsonable(results[key])
    return arrays, meta


def deserialize_results(arrays: dict, meta: dict) -> dict:
    """Reconstruct an analysis results dict from arrays and metadata.

    Args:
        arrays: Mapping of tensor field names to numpy arrays (from ``np.load``).
        meta: Mapping of scalar field names to JSON-able values (from meta.json).

    Returns:
        A results dict with torch tensors restored, matching the shape produced
        by :meth:`WorkspaceAnalyzer.analyze` for the cached mode.
    """
    results: dict[str, Any] = {}
    for key, value in arrays.items():
        # ``np.load`` may return a 0-d array for scalars; keep arrays as-is.
        results[key] = torch.from_numpy(np.asarray(value))

    mode = meta.get("mode")

    # ``workspace_points`` is an alias for ``all_points`` in Cartesian/plane
    # modes. Restore the alias when the primary array exists but the alias was
    # not stored (it usually is, but be defensive).
    if mode in ("cartesian_space", "plane_sampling"):
        if "workspace_points" not in results and "all_points" in results:
            results["workspace_points"] = results["all_points"]

    for key, value in meta.items():
        results[key] = value

    return results


class ResultsCache:
    """Disk cache for workspace analysis results, keyed by content hash.

    Results are stored under ``<cache_dir>/<key>/`` as ``results.npz`` plus a
    ``meta.json`` sidecar. The key is derived from the analysis inputs so that
    identical configurations reuse the same cache entry.
    """

    RESULTS_FILENAME = "results.npz"
    META_FILENAME = "meta.json"

    def __init__(self, cache_dir: str | os.PathLike | None = None) -> None:
        """Initialize the results cache.

        Args:
            cache_dir: Root directory for cached results. Defaults to
                :data:`DEFAULT_RESULTS_CACHE_DIR`.
        """
        self.cache_dir = Path(cache_dir or DEFAULT_RESULTS_CACHE_DIR)

    def entry_path(self, key: str) -> Path:
        """Get the directory path for a cache entry.

        Args:
            key: Cache key (from :func:`compute_cache_key`).

        Returns:
            Path to the entry directory (not guaranteed to exist).
        """
        return self.cache_dir / key

    def exists(self, key: str) -> bool:
        """Check whether a cache entry exists for the given key.

        Args:
            key: Cache key.

        Returns:
            True if both ``results.npz`` and ``meta.json`` exist.
        """
        entry = self.entry_path(key)
        return (entry / self.RESULTS_FILENAME).is_file() and (
            entry / self.META_FILENAME
        ).is_file()

    def save(
        self,
        key: str,
        results: dict,
        metadata: dict | None = None,
        compression: bool = True,
    ) -> Path:
        """Save analysis results to disk.

        Args:
            key: Cache key.
            results: Results dict from :meth:`WorkspaceAnalyzer.analyze`.
            metadata: Optional key-input metadata to embed in ``meta.json`` for
                traceability.
            compression: If True, compress the ``.npz`` archive.

        Returns:
            Path to the entry directory holding the written files.
        """
        entry = self.entry_path(key)
        entry.mkdir(parents=True, exist_ok=True)

        arrays, meta = serialize_results(results)
        meta = dict(meta)
        meta["cache_key"] = key
        meta["timestamp"] = datetime.now().isoformat(timespec="seconds")
        if metadata is not None:
            meta["inputs"] = _to_jsonable(metadata)

        npz_path = entry / self.RESULTS_FILENAME
        if arrays:
            save_fn = np.savez_compressed if compression else np.savez
            save_fn(str(npz_path), **arrays)
        else:
            # No tensor fields (e.g. empty workspace); write an empty archive
            # so the entry is still marked complete.
            np.savez(str(npz_path))

        meta_path = entry / self.META_FILENAME
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.log_info(
            f"Saved workspace results to cache: {entry} "
            f"({len(arrays)} arrays, key={key})"
        )
        return entry

    def load(self, key: str) -> dict | None:
        """Load analysis results from disk.

        Args:
            key: Cache key.

        Returns:
            Reconstructed results dict, or None if the entry does not exist or
            cannot be read.
        """
        entry = self.entry_path(key)
        npz_path = entry / self.RESULTS_FILENAME
        meta_path = entry / self.META_FILENAME
        if not npz_path.is_file() or not meta_path.is_file():
            return None

        try:
            with np.load(str(npz_path)) as npz:
                arrays = {key_: npz[key_] for key_ in npz.files}
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError) as e:
            logger.log_warning(f"Failed to read results cache {entry}: {e}")
            return None

        results = deserialize_results(arrays, meta)
        logger.log_info(f"Loaded workspace results from cache: {entry} (key={key})")
        return results

    def list_entries(self) -> list[dict]:
        """List all cache entries with summary info.

        Returns:
            A list of dicts (``key``, ``path``, ``size_bytes``, ``meta``) sorted
            newest-first by modification time.
        """
        if not self.cache_dir.is_dir():
            return []

        entries = []
        for child in self.cache_dir.iterdir():
            if not child.is_dir():
                continue
            meta_path = child / self.META_FILENAME
            if not meta_path.is_file():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                meta = {}
            entries.append(
                {
                    "key": child.name,
                    "path": str(child),
                    "size_bytes": _dir_size(child),
                    "modified": meta.get("timestamp", ""),
                    "meta": meta,
                }
            )
        entries.sort(key=lambda e: e["modified"], reverse=True)
        return entries


def _dir_size(path: Path) -> int:
    """Recursively compute the size of a directory in bytes."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except OSError:
        pass
    return total
