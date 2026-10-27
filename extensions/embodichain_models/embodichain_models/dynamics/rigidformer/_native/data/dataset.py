# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Derived from RigidFormer at 160fe0852c51cffae7b7440452b58d0b49db6df6.
# See the model family's THIRD_PARTY_LICENSES for the complete license.

"""
Multi-dataset data loader for multi-body dynamics.
Loads sequences from multiple dataset directories, with train/val/test
membership defined by a split file.
"""

from __future__ import annotations

import os
import json
import time
import concurrent.futures as futures
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Tuple, Optional
import glob
from pathlib import Path
import random
from tqdm import tqdm
import sys
from functools import lru_cache
import logging


from .object_scale import ObjectScale, check_object_counts

from .scene_frames import frame_dirs as scene_frame_dirs, is_packed, load_frame
from ._index import _SceneWindows
import zlib

logger = logging.getLogger(__name__)

# Padding value for positions: zero contributes nothing to
# pooling/aggregation operations.
PADDING_VALUE = 0.0


# Global LRU cache for frequently loaded numpy arrays
# (~50000 entries at ~5KB each, roughly 250MB).
@lru_cache(maxsize=256)
def _cached_numpy_load(filepath: str) -> np.ndarray:
    """Cached numpy array loading. Uses filepath as cache key."""
    return np.load(filepath)


def cached_load_npy(path: Path) -> np.ndarray:
    """Load numpy array with LRU caching."""
    return _cached_numpy_load(str(path))


def check_scene_set(config: dict) -> str:
    """data.scene_set must name the scene set the split file declares ('benchmark' when either is absent).

    'complete' splits (data_generation/build_complete_splits.py) leave out the MOVi scenes whose metadata omits simulated
    bodies; 'benchmark' is the published scene list. Returns the scene set.
    """
    want = (config.get("data", {}) or {}).get("scene_set", "benchmark")
    if want not in ("benchmark", "complete"):
        raise ValueError(
            f"data.scene_set must be 'benchmark' or 'complete'; got {want!r}"
        )
    with open(config["data"]["split_file"]) as f:
        have = json.load(f).get("scene_set", "benchmark")
    if have != want:
        raise ValueError(
            f"data.scene_set is {want!r} but {config['data']['split_file']} declares scene_set {have!r}"
        )
    return want


class MultiObjectDynamics(Dataset):
    """MOVi multi-object dataset with per-object vertices and per-frame padding+mask.

    Expects dataset directories like movi_* with frames frame_00000/... and files:
      - meshes/obj_i_faces.npy (per-object faces, static per dataset)
      - frame_xxxxx/obj_i_pos.npy (per-object vertices in world space)
    """

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 3,
        stride: int = 1,
        split: str = "train",
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
        split_file: Optional[str] = None,
        fast_index: bool = True,
        skip_first_frames: int = 0,
        skip_step_size: int = 1,
        max_vertices_per_object: Optional[int] = None,
        max_objects_per_frame: Optional[int] = None,
        anchor_k: Optional[int] = None,
        if_phys_param: bool = False,
        **kwargs,
    ):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.stride = stride
        self.split = split
        self.skip_first_frames = skip_first_frames
        self.skip_step_size = skip_step_size
        self.max_vertices_per_object = max_vertices_per_object
        self.max_objects_per_frame = max_objects_per_frame
        self.if_phys_param = if_phys_param  # Strict mode for physics parameters

        # Point Cloud Mode configuration
        # When enabled, load point cloud files (obj_i_pc{N}_pos.npy) instead of mesh vertices
        if_point_cloud_config = kwargs.get("if_point_cloud", {})
        self.point_cloud_enable = if_point_cloud_config.get("enable", False)
        if self.point_cloud_enable:
            # the point-cloud path validates neither per-frame point counts nor per-resolution padding (a frame with a
            # missing point was accepted; resolutions above the first were truncated to it), and train.py does not
            # forward the setting: refused until implemented and validated
            raise NotImplementedError(
                "point-cloud inputs (if_point_cloud.enable) are not supported by this loader"
            )
        self.point_cloud_train_resolution = if_point_cloud_config.get(
            "train_resolution", [512]
        )
        self.point_cloud_eval_resolution = if_point_cloud_config.get(
            "eval_resolution", [512]
        )

        # Ensure resolutions are lists
        if isinstance(self.point_cloud_train_resolution, int):
            self.point_cloud_train_resolution = [self.point_cloud_train_resolution]
        if isinstance(self.point_cloud_eval_resolution, int):
            self.point_cloud_eval_resolution = [self.point_cloud_eval_resolution]

        # Select resolution based on split
        if self.point_cloud_enable:
            if split == "train":
                self.point_cloud_resolutions = self.point_cloud_train_resolution
            else:
                self.point_cloud_resolutions = self.point_cloud_eval_resolution
            print(f"[{split}] Point Cloud Mode ENABLED:")
            print(f"  - Resolutions: {self.point_cloud_resolutions}")
            print(f"  - Data will be loaded from: frame_XXXXX/obj_i_pc{{N}}_pos.npy")
        else:
            self.point_cloud_resolutions = None

        # Anchor config (Nq): prefer explicit arg, else env ANCHOR_K, else default 32
        if anchor_k is None:
            env_anchor = os.environ.get("ANCHOR_K", "").strip()
            if env_anchor:
                try:
                    anchor_k = int(env_anchor)
                except Exception:
                    raise ValueError(
                        f"Invalid ANCHOR_K environment value: '{env_anchor}' (expected int)"
                    )
            else:
                anchor_k = 32
        if anchor_k <= 0:
            raise ValueError(f"anchor_k must be positive; got {anchor_k}")
        self.anchor_k = int(anchor_k)

        # Object Permutation Augmentation (train only, at least 2 objects)
        augmentation_config = kwargs.get("augmentation", {})
        # The random-walk / independent noise blocks were never applied on this path (the dataset never stored them and
        # the helpers expect body1_/body2_ arrays, not MOVi's obj_positions): enabling them is an error, not a no-op.
        for _blk in ("random_walk", "independent"):
            for _cfg in (augmentation_config, kwargs.get("noise_config") or {}):
                if ((_cfg or {}).get(_blk) or {}).get("enable", False):
                    raise NotImplementedError(
                        f"data.augmentation.{_blk}.enable: true is not implemented for the MOVi loader "
                        f"(it would be silently ignored); set it to false"
                    )
        # Anchor selection rule. 'index0' starts FPS at vertex 0 (order dependent);
        # 'centroid_farthest' starts at the vertex farthest from the centroid with lexicographic
        # tie-breaks, so the anchors depend on the vertex set only.
        self.anchor_fps_start = str(kwargs.get("anchor_fps_start", "index0"))
        # Order-independent FPS ordering of every object's vertices (cached as obj_i_fpsorder_cf.npy);
        # the encoder's hierarchical levels take the first 50/25/12.5/... percent of this ordering.
        self.fps_order = bool(kwargs.get("fps_order", False))
        # data.object_scale_file: apply each object's metadata size to the stored vertices (src/data/object_scale.py)
        _scale_file = kwargs.get("object_scale_file")
        self._object_scale = ObjectScale(_scale_file) if _scale_file else None
        if self._object_scale is not None:
            # the folder name is compared without regard to case
            _ds_real = os.path.basename(os.path.realpath(str(data_path)))
            if (self._object_scale.converted or "").lower() != _ds_real.lower():
                raise ValueError(
                    f"data.object_scale_file {_scale_file} was built for '{self._object_scale.converted}', "
                    f"but the dataset path resolves to '{_ds_real}'"
                )
        if self._object_scale is not None:
            print(
                f"[{split}] Object scale ACTIVE: vertices scaled about the object origin by the metadata size ({_scale_file})"
            )
        # data.partial_view: partial point clouds of a simulated depth camera instead of the mesh vertices
        # (src/data/partial_view.py)
        _pv = kwargs.get("partial_view") or {}
        self._partial = None
        if _pv.get("enable", False):
            if self._object_scale is None:
                raise ValueError(
                    "data.partial_view needs data.object_scale_file for the objects' metadata sizes"
                )
            self._partial = PartialView(_pv, Path(__file__).resolve().parents[2])
            if self._partial.min_points < self.anchor_k:
                raise ValueError(
                    f"data.partial_view.min_points ({self._partial.min_points}) must be at least "
                    f"anchor_k ({self.anchor_k})"
                )
            self._partial_scenes = {}
            print(f"[{split}] {self._partial.describe(train=split == 'train')}")
        self._fps_order_by_dataset = {}
        if self.anchor_fps_start not in ("index0", "centroid_farthest"):
            raise ValueError(
                f"data.anchor_fps_start must be 'index0' or 'centroid_farthest'; got '{self.anchor_fps_start}'"
            )
        if self.anchor_fps_start != "index0":
            print(
                f"[{split}] Anchor FPS start rule: {self.anchor_fps_start} (anchor cache suffix '_cf')"
            )
        # Anchor resampling augmentation (train split only): with probability `prob` the FPS anchors
        # of every object are recomputed from a random start vertex.
        resample_config = augmentation_config.get("anchor_resampling", {})
        self.anchor_resample_prob = (
            float(resample_config.get("prob", 0.5))
            if resample_config.get("enable", False)
            else 0.0
        )
        if self.anchor_resample_prob > 0 and split == "train":
            print(
                f"[{split}] Anchor resampling augmentation enabled: prob={self.anchor_resample_prob:.2f}"
            )
        permute_config = augmentation_config.get("object_permutation", {})
        self.permute_enabled = permute_config.get("enable", False) and split == "train"
        self.permute_prob = permute_config.get("prob_apply", 0.5)
        self.permute_debug = permute_config.get("if_debug", False)
        if self.permute_enabled:
            print(
                f"[{split}] Object permutation augmentation enabled: prob={self.permute_prob:.2f}, debug={self.permute_debug}"
            )

        # Load the dataset split file (required; splits are never auto-generated).
        if not split_file:
            raise ValueError(
                "split_file is required. Set 'data.split_file' in the config to a split JSON, "
                "e.g. one of the dataset_split_*.json files in the repository root."
            )

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                "Check the 'data.split_file' path in the config. Split files "
                "(dataset_split_*.json) are provided in the repository root; new splits "
                "can be written for data produced by data_generation/movi_to_sim_data.py."
            )

        with open(split_file, "r") as f:
            s = json.load(f)
        names = (
            s["train_datasets"]
            if split == "train"
            else s["val_datasets"] if split == "val" else s["test_datasets"]
        )
        self.dataset_dirs = [self.data_path / n for n in names]
        if self._object_scale is not None:
            check_object_counts(self._object_scale, self.data_path, names)

        # Build sequences
        self.all_sequences = (
            []
        )  # each: { 'dataset': name, 'frames': [paths], 'frame_ids': [int] }
        if kwargs.get("compact_index", False):
            self.all_sequences = _SceneWindows(
                random_sampling=bool(
                    (kwargs.get("random_step_size") or {}).get("enable", False)
                )
            )

        # DDP-aware: only show progress bar on rank 0
        import torch.distributed as dist

        show_progress = True
        if dist.is_available() and dist.is_initialized():
            show_progress = dist.get_rank() == 0

        for ds in tqdm(
            self.dataset_dirs,
            desc=f"Indexing {split} datasets",
            unit="ds",
            disable=not show_progress,
        ):
            frames = scene_frame_dirs(ds)
            if self.skip_first_frames > 0:
                frames = frames[self.skip_first_frames :]
            if self.skip_step_size > 1:
                frames = frames[:: self.skip_step_size]
            if isinstance(self.all_sequences, _SceneWindows):
                self.all_sequences.add_scene(
                    ds.name, frames, sequence_length, self.stride
                )
                continue
            for start in range(0, len(frames) - sequence_length + 1, self.stride):
                seq = frames[start : start + sequence_length]
                if len(seq) == sequence_length:
                    self.all_sequences.append(
                        {
                            "dataset": ds.name,
                            "frames": [p.as_posix() for p in seq],
                            "frame_ids": [
                                int(Path(p).name.split("_")[-1]) for p in seq
                            ],
                        }
                    )

        # Precompute/load per-object anchor indices and faces for each dataset from
        # the reference frame_00000; cached in memory.
        self._anchors_by_dataset = {}
        self._faces_by_dataset = {}

        # For point cloud mode, use the first resolution for __init__ validation
        init_pc_resolution = (
            self.point_cloud_resolutions[0] if self.point_cloud_enable else None
        )

        for ds in self.dataset_dirs:
            ds_name = ds.name
            frame0 = ds / "frame_00000"
            if not (frame0.exists() or is_packed(ds)):
                raise FileNotFoundError(
                    f"Missing reference frame for dataset '{ds_name}': {frame0}. "
                    f"Expected a 'frame_00000' directory (or frames.npy) under {ds}."
                )

            # For point cloud mode, load point clouds for validation
            # For mesh mode, load mesh vertices
            ref_objs = self._load_frame_objects(
                frame0, pc_resolution=init_pc_resolution
            )
            if len(ref_objs) == 0:
                raise RuntimeError(
                    f"Empty reference frame in {frame0}: no per-object vertex files found."
                )
            anchors_dir = ds / "anchors"
            anchors_dir.mkdir(parents=True, exist_ok=True)
            ds_anchor_list = []
            ds_faces_list = []
            ds_order_list = []
            for i, V in enumerate(ref_objs):
                nv = V.shape[0]
                if (
                    self.max_vertices_per_object is not None
                    and nv > self.max_vertices_per_object
                ):
                    raise ValueError(
                        f"Dataset '{ds_name}' reference object {i} has {nv} vertices, "
                        f"exceeding max_vertices_per_object={self.max_vertices_per_object}.\n"
                        f"Either reduce vertex count in dataset or increase data.max_vertices_per_object in config."
                    )

                # Point cloud mode: anchors are unused, so use sequential placeholder
                # indices rather than mesh anchors (whose indices may be out of range).
                if self.point_cloud_enable:
                    idx = np.arange(min(self.anchor_k, nv), dtype=np.int32)
                    if len(idx) < self.anchor_k:
                        idx = np.pad(
                            idx, (0, self.anchor_k - len(idx)), mode="constant"
                        )
                    ds_anchor_list.append(idx)
                    # Create empty faces (not used in point cloud mode)
                    ds_faces_list.append(np.zeros((0, 3), dtype=np.int64))
                    continue

                # Mesh vertex mode: standard anchor and faces loading.
                if nv < self.anchor_k:
                    raise ValueError(
                        f"Dataset '{ds_name}' object {i} has only {nv} vertices (< anchor_k={self.anchor_k}). "
                        f"Reduce ANCHOR_K or increase vertex cap (max_vertices_per_object)."
                    )
                _suffix = "" if self.anchor_fps_start == "index0" else "_cf"
                anchor_path = (
                    anchors_dir / f"obj_{i}_anchors_fps{self.anchor_k}{_suffix}.npy"
                )
                if anchor_path.exists():
                    idx = np.load(anchor_path)
                    if idx.ndim != 1 or idx.shape[0] != self.anchor_k:
                        raise ValueError(
                            f"Invalid anchor file shape: {anchor_path} (expected ({self.anchor_k},))"
                        )
                    if idx.dtype.kind not in ("i", "u"):
                        raise ValueError(
                            f"Invalid anchor file dtype: {anchor_path} (expected integer)"
                        )
                    if int(idx.max()) >= nv or int(idx.min()) < 0:
                        raise ValueError(
                            f"Anchor indices in {anchor_path} out of range [0,{nv-1}] for object {i}"
                        )
                    if np.unique(idx).shape[0] != idx.shape[0]:
                        raise ValueError(
                            f"Anchor cache {anchor_path} repeats a vertex: {idx.tolist()}"
                        )
                    ds_anchor_list.append(idx.astype(np.int32, copy=False))
                else:
                    # Compute deterministic FPS anchors on the STORED (unscaled) frame-0 vertices, as
                    # pregen_cf_anchors.py does: FPS is invariant to the size scaling only in exact arithmetic (float32
                    # rounding of the scaled coordinates breaks the exact ties of symmetric meshes differently)
                    V_ref = load_frame(ds, 0)[i].astype(np.float32, copy=False)
                    idx = self._farthest_point_sample_np(
                        V_ref, self.anchor_k, rule=self.anchor_fps_start
                    )
                    # Atomic write: concurrent DDP ranks must never read a partially written cache.
                    tmp_path = anchor_path.with_name(
                        f"{anchor_path.stem}.{os.getpid()}.tmp.npy"
                    )
                    np.save(tmp_path, idx.astype(np.int32, copy=False))
                    os.replace(tmp_path, anchor_path)
                    ds_anchor_list.append(idx.astype(np.int32, copy=False))
                if self.fps_order:
                    order_path = anchors_dir / f"obj_{i}_fpsorder_cf.npy"
                    if order_path.exists():
                        order = np.load(order_path)
                        if (
                            order.ndim != 1
                            or order.shape[0] != nv
                            or order.dtype.kind not in ("i", "u")
                        ):
                            raise ValueError(
                                f"Invalid FPS-order cache {order_path}: shape {order.shape}, expected ({nv},)"
                            )
                        if not np.array_equal(np.sort(order), np.arange(nv)):
                            raise ValueError(
                                f"FPS-order cache {order_path} is not a permutation of the {nv} vertices"
                            )
                    else:
                        # stored (unscaled) frame-0 vertices, as pregen_fps_order.py does (see the anchors above)
                        order = self._farthest_point_sample_np(
                            load_frame(ds, 0)[i].astype(np.float32, copy=False),
                            nv,
                            rule="centroid_farthest",
                        )
                        tmp_path = order_path.with_name(
                            f"{order_path.stem}.{os.getpid()}.tmp.npy"
                        )
                        np.save(tmp_path, order.astype(np.int32, copy=False))
                        os.replace(tmp_path, order_path)
                    ds_order_list.append(order.astype(np.int32, copy=False))

                # Load faces for this object (static per dataset), and filter to nv cap
                faces_path = ds / "meshes" / f"obj_{i}_faces.npy"
                if not faces_path.exists():
                    raise FileNotFoundError(
                        f"Missing faces file for dataset '{ds_name}' object {i}: {faces_path}"
                    )
                faces = np.load(faces_path)
                if faces.ndim != 2 or faces.shape[1] != 3:
                    raise ValueError(
                        f"Invalid faces array at {faces_path}: shape={faces.shape}, expected (F,3)"
                    )
                # every face must index this object's vertices (the vertex count is never capped: an object above
                # max_vertices_per_object is an error), so an out-of-range face is corrupt data, not a face to drop
                if faces.dtype.kind not in ("i", "u"):
                    raise ValueError(
                        f"Faces of {faces_path} have dtype {faces.dtype}, expected integers"
                    )
                faces = faces.astype(np.int64, copy=False)
                if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= nv):
                    raise ValueError(
                        f"Faces of {faces_path} index vertices outside [0, {nv - 1}]"
                    )
                ds_faces_list.append(faces)
            self._anchors_by_dataset[ds_name] = ds_anchor_list
            self._faces_by_dataset[ds_name] = ds_faces_list
            if self.fps_order:
                self._fps_order_by_dataset[ds_name] = ds_order_list

        # Cache reference frames (frame_00000) per dataset to avoid repeated loading.
        self._reference_frames_cache = {}

        # Per-dataset caches: frame lists (avoids glob in __getitem__), frame IDs
        # (avoids string parsing per sample), and per-object/max vertex counts.
        self._all_frames_by_dataset = {}
        self._frame_id_cache = {}
        self._max_nv_by_dataset = {}
        self._mesh_nv_by_dataset = {}

        for ds in self.dataset_dirs:
            ds_name = ds.name
            all_frames = scene_frame_dirs(ds)
            self._all_frames_by_dataset[ds_name] = all_frames
            for fp in all_frames:
                frame_id = int(fp.name.split("_")[-1])  # "frame_00123" -> 123
                self._frame_id_cache[(ds_name, fp.name)] = frame_id
            # Frames are indexed by position: a missing or extra frame directory would silently change the time step
            # and misalign the ground truth (vis_traj.npy, which the evaluator indexes by frame number).
            _ids = [self._frame_id_cache[(ds_name, fp.name)] for fp in all_frames]
            _T_traj = np.load(ds / "vis_traj.npy", mmap_mode="r").shape[1]
            if _ids != list(range(len(_ids))) or len(_ids) != _T_traj:
                raise ValueError(
                    f"{ds}: frame directories must be frame_00000 .. frame_{_T_traj - 1:05d} without gaps "
                    f"(vis_traj.npy has {_T_traj} frames); found {len(_ids)} directories, "
                    f"first gap at {next((i for i, f in enumerate(_ids) if f != i), len(_ids))}"
                )
            # Load frame_00000 once to record per-object vertex counts
            # (at point cloud resolution when enabled).
            frame0 = ds / "frame_00000"
            ref_objs_data = self._load_frame_objects(
                frame0, pc_resolution=init_pc_resolution
            )
            nv_list = [obj.shape[0] for obj in ref_objs_data]
            self._mesh_nv_by_dataset[ds_name] = nv_list
            self._max_nv_by_dataset[ds_name] = max(nv_list) if nv_list else 0

        # Load physics parameters (mass, friction, restitution) per dataset.
        # With if_phys_param=True the file and every object entry must exist.
        self._physics_by_dataset = {}
        for ds in self.dataset_dirs:
            ds_name = ds.name
            physics_path = ds / "physics_params.json"
            num_objs = len(self._anchors_by_dataset[ds_name])

            if self.if_phys_param:
                if not physics_path.exists():
                    raise FileNotFoundError(
                        f"Physics parameters required but file missing: {physics_path}\n"
                        f"Dataset: {ds_name} | if_phys_param=True requires physics_params.json\n"
                        f"Please generate physics parameters or set if_phys_param=False"
                    )

                # Parse errors propagate deliberately.
                with open(physics_path, "r") as f:
                    physics_params = json.load(f)

                phys_list = []
                for i in range(num_objs):
                    obj_key = f"obj_{i}"
                    if obj_key not in physics_params:
                        raise KeyError(
                            f"Missing physics parameters for {obj_key} in {physics_path}\n"
                            f"Dataset: {ds_name} | Found objects: {list(physics_params.keys())}\n"
                            f"Required: obj_0 through obj_{num_objs-1}"
                        )

                    p = physics_params[obj_key]
                    if "mass" not in p:
                        raise KeyError(
                            f"Missing 'mass' field for {obj_key} in {physics_path}"
                        )
                    if "friction" not in p:
                        raise KeyError(
                            f"Missing 'friction' field for {obj_key} in {physics_path}"
                        )
                    if "restitution" not in p:
                        raise KeyError(
                            f"Missing 'restitution' field for {obj_key} in {physics_path}"
                        )

                    phys_list.append(
                        [
                            float(p["mass"]),
                            float(p["friction"]),
                            float(p["restitution"]),
                        ]
                    )

                self._physics_by_dataset[ds_name] = np.array(
                    phys_list, dtype=np.float32
                )  # (Nobj, 3)
            else:
                # Permissive mode: substitute defaults for missing entries.
                if physics_path.exists():
                    with open(physics_path, "r") as f:
                        physics_params = json.load(f)
                    phys_list = []
                    for i in range(num_objs):
                        obj_key = f"obj_{i}"
                        if obj_key in physics_params:
                            p = physics_params[obj_key]
                            phys_list.append(
                                [
                                    float(p.get("mass", 1.0)),
                                    float(p.get("friction", 0.5)),
                                    float(p.get("restitution", 0.5)),
                                ]
                            )
                        else:
                            phys_list.append([1.0, 0.5, 0.5])
                    self._physics_by_dataset[ds_name] = np.array(
                        phys_list, dtype=np.float32
                    )
                else:
                    # File missing entirely: default values for all objects.
                    self._physics_by_dataset[ds_name] = np.array(
                        [[1.0, 0.5, 0.5]] * num_objs, dtype=np.float32
                    )

        # Multi-timescale training configuration; step sizes are expressed in
        # normalized time units.
        random_step_size_config = kwargs.get("random_step_size", {})
        self.random_step_size_enable = random_step_size_config.get("enable", False)
        self.random_step_sizes = random_step_size_config.get(
            "step_sizes", [1, 2, 5, 10]
        )
        self.random_start_frame = random_step_size_config.get(
            "random_start_frame", True
        )
        self.sampling_weights = random_step_size_config.get("sampling_weights", None)

        if self.random_step_size_enable:
            if (
                not isinstance(self.random_step_sizes, list)
                or len(self.random_step_sizes) == 0
            ):
                raise ValueError(
                    f"random_step_size.step_sizes must be a non-empty list, got: {self.random_step_sizes}"
                )

            for ss in self.random_step_sizes:
                if not isinstance(ss, int) or ss <= 0:
                    raise ValueError(
                        f"All step_sizes must be positive integers, got: {self.random_step_sizes}"
                    )

            if self.sampling_weights is None:
                # Default to uniform sampling.
                self.sampling_weights = [1.0 / len(self.random_step_sizes)] * len(
                    self.random_step_sizes
                )
            else:
                if len(self.sampling_weights) != len(self.random_step_sizes):
                    raise ValueError(
                        f"sampling_weights length ({len(self.sampling_weights)}) must match "
                        f"step_sizes length ({len(self.random_step_sizes)})"
                    )

                weights_sum = sum(self.sampling_weights)
                if abs(weights_sum - 1.0) > 1e-6:
                    raise ValueError(
                        f"sampling_weights must sum to 1.0, got sum={weights_sum:.6f}\n"
                        f"Weights: {self.sampling_weights}"
                    )

            print(f"\n{'='*60}")
            print(f"[{split}] Multi-timescale training ENABLED:")
            print(f"  - step_sizes: {self.random_step_sizes}")
            print(f"  - sampling_weights: {self.sampling_weights}")
            print(f"  - random_start_frame: {self.random_start_frame}")
            print(f"  - if_debug: {random_step_size_config.get('if_debug', False)}")
            print(f"{'='*60}\n")

            # Every dataset must have enough frames for every configured step size.
            print(
                f"[{split}] Validating dataset frame counts for multi-timescale training..."
            )
            insufficient_datasets = []
            for ds in self.dataset_dirs:
                frames = scene_frame_dirs(ds)
                if self.skip_first_frames > 0:
                    frames = frames[self.skip_first_frames :]
                available_frames = len(frames)

                for step_size in self.random_step_sizes:
                    frames_needed = (self.sequence_length - 1) * step_size + 1
                    if available_frames < frames_needed:
                        insufficient_datasets.append(
                            {
                                "dataset": ds.name,
                                "available": available_frames,
                                "needed": frames_needed,
                                "step_size": step_size,
                                "sequence_length": self.sequence_length,
                            }
                        )

            if insufficient_datasets:
                error_msg = f"[{split}] Frame validation failed: {len(insufficient_datasets)} datasets have insufficient frames:\n"
                for info in insufficient_datasets[:5]:
                    error_msg += (
                        f"  - {info['dataset']}: available={info['available']}, "
                        f"needed={info['needed']} (step_size={info['step_size']}, "
                        f"sequence_length={info['sequence_length']})\n"
                    )
                if len(insufficient_datasets) > 5:
                    error_msg += (
                        f"  ... and {len(insufficient_datasets) - 5} more datasets\n"
                    )
                error_msg += (
                    f"\nSolutions:\n"
                    f"  1. Reduce max step_size in config: data.random_step_size.step_sizes\n"
                    f"  2. Reduce sequence_length in config: data.sequence_length\n"
                    f"  3. Filter out datasets with insufficient frames\n"
                    f"  4. Reduce skip_first_frames in config: data.skip_first_frames (currently {self.skip_first_frames})"
                )
                raise ValueError(error_msg)

            print(
                f"[{split}] All {len(self.dataset_dirs)} datasets have sufficient frames for step_sizes {self.random_step_sizes}"
            )
        else:
            # Disabled: use the fixed skip_step_size
            print(
                f"[{split}] Multi-timescale training disabled, using skip_step_size={self.skip_step_size}"
            )

    def __len__(self) -> int:
        return len(self.all_sequences)

    def get_cache_stats(self) -> dict:
        """Return cache statistics for monitoring."""
        return {
            "numpy_cache_info": _cached_numpy_load.cache_info()._asdict(),
            "reference_frames_cached": len(self._reference_frames_cache),
            "reference_frames_datasets": list(self._reference_frames_cache.keys()),
        }

    @staticmethod
    def clear_numpy_cache():
        """Clear the global numpy LRU cache to free memory."""
        _cached_numpy_load.cache_clear()
        logger.info("Cleared numpy LRU cache")

    def _load_frame_objects(
        self, frame_dir: Path, pc_resolution: Optional[int] = None
    ) -> List["np.ndarray"]:
        """
        Load per-object vertex/point data for a single frame.

        Args:
            frame_dir: Path to frame directory
            pc_resolution: If provided, load point cloud at this resolution instead of mesh vertices

        Returns:
            List of (Ni, 3) arrays, one per object
        """
        # Data exceeding the configured limits is rejected rather than truncated.
        objs = []
        idx = 0

        if pc_resolution is not None:
            if self._object_scale is not None:
                raise NotImplementedError(
                    "data.object_scale_file is not supported with point-cloud inputs"
                )
            # Point cloud mode: load from obj_i_pc{N}_pos.npy
            while (frame_dir / f"obj_{idx}_pc{pc_resolution}_pos.npy").exists():
                pc_file = frame_dir / f"obj_{idx}_pc{pc_resolution}_pos.npy"
                V = cached_load_npy(pc_file)  # (N_points, 3)

                # Point clouds have a fixed size; no vertex limit check needed.
                objs.append(V.astype("float32", copy=False))
                idx += 1

            # Mesh vertices exist but the requested point cloud resolution does not.
            if idx == 0 and (frame_dir / "obj_0_pos.npy").exists():
                raise FileNotFoundError(
                    f"Point cloud files not found at resolution {pc_resolution} in {frame_dir}.\n"
                    f"Expected: obj_i_pc{pc_resolution}_pos.npy\n"
                    f"Mesh vertices exist but point cloud mode is enabled.\n"
                    f"Please use a dataset with pre-computed point clouds."
                )
        else:
            # Mesh vertex mode: load from obj_i_pos.npy (packed layout: the same arrays from frames.npy)
            packed = (
                load_frame(frame_dir.parent, int(frame_dir.name.split("_")[-1]))
                if is_packed(frame_dir.parent)
                else None
            )
            while (
                (idx < len(packed))
                if packed is not None
                else (frame_dir / f"obj_{idx}_pos.npy").exists()
            ):
                V = (
                    packed[idx]
                    if packed is not None
                    else cached_load_npy(frame_dir / f"obj_{idx}_pos.npy")
                )  # (Ni, 3)
                if self._object_scale is not None:
                    V = self._object_scale.apply(frame_dir / f"obj_{idx}_pos.npy", V)

                if (
                    self.max_vertices_per_object is not None
                    and V.shape[0] > self.max_vertices_per_object
                ):
                    raise ValueError(
                        f"Object {idx} in {frame_dir} has {V.shape[0]} vertices, "
                        f"exceeding max_vertices_per_object={self.max_vertices_per_object}.\n"
                        f"Either reduce vertex count in dataset or increase data.max_vertices_per_object in config."
                    )

                objs.append(V.astype("float32", copy=False))
                idx += 1

        if (
            self.max_objects_per_frame is not None
            and len(objs) > self.max_objects_per_frame
        ):
            raise ValueError(
                f"Frame {frame_dir} has {len(objs)} objects, "
                f"exceeding max_objects_per_frame={self.max_objects_per_frame}.\n"
                f"Either reduce object count in dataset or increase data.max_objects_per_frame in config."
            )

        return objs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        info = self.all_sequences[idx]

        # Point Cloud Mode: Sample resolution for this sample
        pc_resolution = None
        if self.point_cloud_enable:
            pc_resolution = np.random.choice(self.point_cloud_resolutions)

        # Multi-timescale training: sample a step size for this sequence.
        if self.random_step_size_enable:
            step_size = np.random.choice(
                self.random_step_sizes, p=self.sampling_weights
            )

            ds_name = info["dataset"]
            all_frames = self._all_frames_by_dataset[ds_name]

            # Apply skip_first_frames
            if self.skip_first_frames > 0:
                all_frames = all_frames[self.skip_first_frames :]

            # Total frames spanned by the sequence, e.g. sequence_length=4 with
            # step_size=10 uses frames [0, 10, 20, 30] and spans 31 frames.
            student_reach = (self.sequence_length - 1) * step_size
            total_frames_needed = student_reach + 1

            # Frame counts were validated in __init__; assert as a sanity check.
            assert (
                len(all_frames) >= total_frames_needed
            ), f"Unexpected: insufficient frames in {ds_name} (should have been caught in __init__)"

            # Select a start frame within the valid range.
            if self.random_start_frame:
                max_start_frame = len(all_frames) - total_frames_needed
                start_frame = np.random.randint(0, max_start_frame + 1)
            else:
                # Deterministic start frame derived from idx (for reproducibility).
                max_start_frame = len(all_frames) - total_frames_needed
                start_frame = idx % (max_start_frame + 1)

            # Frame sequence [start_frame, start_frame + step_size, ...].
            frame_indices = [
                start_frame + i * step_size for i in range(self.sequence_length)
            ]
            frame_paths = [all_frames[i] for i in frame_indices]
            frame_ids = np.array(
                [self._frame_id_cache[(ds_name, fp.name)] for fp in frame_paths],
                dtype=np.int32,
            )

            selected_step_size = step_size
        else:
            # Fixed step size: use pre-built sequences with skip_step_size.
            ds_name = info["dataset"]
            frame_paths = [Path(p) for p in info["frames"]]
            frame_ids = np.asarray(info.get("frame_ids", []), dtype=np.int32)
            selected_step_size = self.skip_step_size

        if self._partial is not None:
            return self._partial_item(ds_name, frame_ids, selected_step_size)

        # Load per-frame objects (ragged); point clouds are loaded instead of mesh
        # vertices when pc_resolution is set.
        frames_objs: List[List[np.ndarray]] = [
            self._load_frame_objects(fp, pc_resolution=pc_resolution)
            for fp in frame_paths
        ]
        if pc_resolution is None:
            # every frame holds the reference frame's objects with the same vertex counts (a missing or truncated file
            # would otherwise become padding or a shorter object without notice)
            _ref_nv = self._mesh_nv_by_dataset[ds_name]
            for fp, fo in zip(frame_paths, frames_objs):
                if [V.shape[0] for V in fo] != _ref_nv:
                    raise ValueError(
                        f"{fp}: per-object vertex counts {[V.shape[0] for V in fo]} differ from frame_00000 "
                        f"{_ref_nv}"
                    )
        num_objs = max(len(f) for f in frames_objs)

        if self.max_vertices_per_object is not None:
            max_nv = int(self.max_vertices_per_object)
        else:
            max_nv = self._max_nv_by_dataset[ds_name]

        T = len(frames_objs)

        # Allocate padded arrays.
        obj_positions = np.full(
            (T, num_objs, max_nv, 3), PADDING_VALUE, dtype="float32"
        )
        valid_obj_mask = np.zeros((num_objs,), dtype="bool")
        valid_vtx_mask = np.zeros((T, num_objs, max_nv), dtype="bool")

        for t, fo in enumerate(frames_objs):
            for i, V in enumerate(fo):
                nv = min(V.shape[0], max_nv)
                obj_positions[t, i, :nv, :] = V[:nv]
                valid_obj_mask[i] = True
                valid_vtx_mask[t, i, :nv] = True

        # Reference pose is always the first frame of the dataset (global frame_00000),
        # cached per dataset. Point cloud mode includes the resolution in the cache key.
        cache_key = f"{ds_name}_pc{pc_resolution}" if pc_resolution else ds_name
        if cache_key not in self._reference_frames_cache:
            ds_root = self.data_path / ds_name
            frame0 = ds_root / "frame_00000"
            if not (frame0.exists() or is_packed(ds_root)):
                raise FileNotFoundError(
                    f"Missing reference frame for dataset '{ds_name}': {frame0}. "
                    f"Expected a 'frame_00000' directory (or frames.npy) under {ds_root}."
                )
            ref_objs = self._load_frame_objects(frame0, pc_resolution=pc_resolution)
            if len(ref_objs) == 0:
                raise RuntimeError(
                    f"Empty reference frame in {frame0}: no per-object vertex files found."
                )
            self._reference_frames_cache[cache_key] = ref_objs
        else:
            ref_objs = self._reference_frames_cache[cache_key]
        # Every frame in the sequence must contain all reference objects, each stored
        # as a non-empty (N, 2|3) array. Point cloud mode is already validated inside
        # _load_frame_objects, so it is skipped here, and so is the packed layout (frames.npy holds every object of
        # every frame by construction; scene_frames checks its shape against frames_counts.npy).
        expected_num_objs = len(ref_objs)
        if pc_resolution is None and not is_packed(self.data_path / ds_name):
            for t, fdir in enumerate(frame_paths):
                for i_ref in range(expected_num_objs):
                    npy_path = fdir / f"obj_{i_ref}_pos.npy"
                    if not npy_path.exists():
                        raise RuntimeError(
                            f"Missing expected file for dataset '{ds_name}': {npy_path} "
                            f"(t={t}, obj={i_ref}). Ensure all frames contain the same set of objects as frame_00000."
                        )
                    try:
                        arr_chk = np.load(npy_path)
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to load file: {npy_path} (t={t}, obj={i_ref}). Error: {e}"
                        )
                    if not (
                        arr_chk.ndim == 2
                        and arr_chk.shape[1] in (2, 3)
                        and arr_chk.shape[0] > 0
                    ):
                        raise RuntimeError(
                            f"Invalid/empty vertices at {npy_path}: shape={arr_chk.shape} "
                            f"(t={t}, obj={i_ref})."
                        )
        # Build the reference array with fixed Nv (pad/trim to max_nv).
        ref_obj_positions = np.full(
            (num_objs, max_nv, 3), PADDING_VALUE, dtype="float32"
        )
        for i, V in enumerate(ref_objs[:num_objs]):
            nv = min(V.shape[0], max_nv)
            ref_obj_positions[i, :nv, :] = V[:nv]

        # Load anchors (per object) for this dataset (computed in __init__)
        if ds_name not in self._anchors_by_dataset:
            raise KeyError(
                f"Missing anchors for dataset '{ds_name}'. Initialization may have failed."
            )
        ds_anchor_list = self._anchors_by_dataset[ds_name]
        if len(ds_anchor_list) < num_objs:
            raise ValueError(
                f"Anchors count ({len(ds_anchor_list)}) < objects in sequence ({num_objs}) for dataset '{ds_name}'."
            )
        anchor_idx = np.zeros((num_objs, self.anchor_k), dtype=np.int32)
        for i in range(num_objs):
            idx_i = ds_anchor_list[i]
            if idx_i.shape[0] != self.anchor_k:
                raise ValueError(
                    f"Anchor length mismatch for dataset '{ds_name}' obj {i}: got {idx_i.shape[0]}, expected {self.anchor_k}"
                )
            anchor_idx[i] = idx_i
        vtx_rank = None
        if self.fps_order:
            # rank[i, v] = position of vertex v in object i's FPS ordering (padding: max_nv)
            vtx_rank = np.full((num_objs, max_nv), max_nv, dtype=np.int32)
            ds_order_list = self._fps_order_by_dataset[ds_name]
            for i in range(num_objs):
                order_i = ds_order_list[i]
                nv_i = min(order_i.shape[0], max_nv)
                vtx_rank[i, order_i[:nv_i]] = np.arange(nv_i, dtype=np.int32)
        if (
            self.split == "train"
            and self.anchor_resample_prob > 0
            and not self.point_cloud_enable
            and np.random.rand() < self.anchor_resample_prob
        ):
            _before = anchor_idx.copy()
            for i in range(num_objs):
                nv_i = min(ref_objs[i].shape[0], max_nv)
                if nv_i >= self.anchor_k:
                    anchor_idx[i] = self._farthest_point_sample_np(
                        ref_objs[i][:nv_i].astype(np.float32, copy=False),
                        self.anchor_k,
                        rule=self.anchor_fps_start,
                        start=int(np.random.randint(nv_i)),
                    )
            if not getattr(self, "_anchor_resample_logged", False):
                self._anchor_resample_logged = True
                print(
                    f"[train] Anchor resampling APPLIED (pid {os.getpid()}): obj0 {_before[0].tolist()} -> {anchor_idx[0].tolist()} "
                    f"(rule={self.anchor_fps_start})",
                    flush=True,
                )

        # Object permutation must be applied before faces are gathered so that
        # faces_list[i] matches the permuted obj_i. Only faces for the current
        # sample's objects are taken.
        faces_list = self._faces_by_dataset[ds_name][:num_objs]
        if self.permute_debug:
            print(
                f"[object permutation] dataset={ds_name}, num_objs={num_objs}, faces_list length={len(faces_list)}, total_faces={len(self._faces_by_dataset[ds_name])}"
            )
        perm = None  # Set only if permutation is applied
        if self.permute_enabled and np.random.random() < self.permute_prob:
            if num_objs >= 2:
                # Shuffle all objects; there is no ground mesh to exclude.
                perm = np.arange(num_objs)
                np.random.shuffle(perm)

                if self.permute_debug:
                    print(
                        f"[object permutation] dataset={ds_name}, idx={idx}, num_objs={num_objs}"
                    )
                    print(f"  Original order: {list(range(num_objs))}")
                    print(f"  Shuffled order: {perm.tolist()}")
                    print(
                        f"  Mapping: {' '.join([f'{i}->{perm[i]}' for i in range(num_objs)])}"
                    )

                # Permute all object-indexed data.
                obj_positions = obj_positions[:, perm]  # (T, Nobj, Nv, 3)
                ref_obj_positions = ref_obj_positions[perm]  # (Nobj, Nv, 3)
                ref_objs = [ref_objs[i] for i in perm]  # Permute ref_objs list
                # Permute frames_objs (list of lists)
                frames_objs = [[fo[i] for i in perm] for fo in frames_objs]
                valid_obj_mask = valid_obj_mask[perm]  # (Nobj,)
                valid_vtx_mask = valid_vtx_mask[:, perm]  # (T, Nobj, Nv)
                anchor_idx = anchor_idx[perm]  # (Nobj, Nq)
                if vtx_rank is not None:
                    vtx_rank = vtx_rank[perm]  # (Nobj, Nv)
                # Permute faces_list
                faces_list = [faces_list[i] for i in perm]

        # time_step_id: first global frame id of this sequence (scalar)
        ts_id_val = int(frame_ids[0]) if frame_ids.size > 0 else -1

        # Get physics parameters for this dataset (permuted if augmentation applied)
        phys_params = self._physics_by_dataset[ds_name]  # (Nobj_full, 3)
        phys_params = phys_params[:num_objs]  # Trim to current sample's object count
        if perm is not None:
            phys_params = phys_params[
                perm
            ]  # Apply permutation to match permuted objects

        sample = {
            "obj_positions": torch.from_numpy(obj_positions),
            "valid_obj_mask": torch.from_numpy(valid_obj_mask),
            "valid_vtx_mask": torch.from_numpy(valid_vtx_mask),
            "ref_obj_positions": torch.from_numpy(ref_obj_positions),
            "dt": torch.tensor(1.0, dtype=torch.float32),
            "frame_ids": (
                torch.from_numpy(frame_ids)
                if frame_ids.size > 0
                else torch.tensor([], dtype=torch.int32)
            ),
            "time_step_id": torch.tensor(ts_id_val, dtype=torch.int64),
            "anchor_idx": torch.from_numpy(anchor_idx),
            **(
                {"vtx_rank": torch.from_numpy(vtx_rank)} if vtx_rank is not None else {}
            ),
            "dataset_name": ds_name,
            "faces_list": faces_list,  # Store permuted faces (list of numpy arrays)
            "permutation": (
                perm if perm is not None else np.arange(num_objs)
            ),  # Store permutation order (identity if not permuted)
            "phys_params": torch.from_numpy(
                phys_params
            ),  # (Nobj, 3): [mass, friction, restitution]
            "skip_step_size": torch.tensor(
                selected_step_size, dtype=torch.float32
            ),  # Time step size for temporal encoding (randomly sampled when multi-timescale enabled)
            "is_point_cloud": pc_resolution
            is not None,  # Whether this sample uses point cloud data
        }

        if pc_resolution is not None:
            sample["point_cloud_resolution"] = pc_resolution

        # Apply noise augmentation (train split only)
        if getattr(self, "split", "train") == "train" and getattr(
            self, "noise_config", None
        ):
            rw_cfg = self.noise_config.get("random_walk", {})
            if rw_cfg.get("enable", False):
                start_epoch = rw_cfg.get("start_epoch", 0)
                cur_epoch = getattr(self, "current_epoch", 0)
                if cur_epoch >= start_epoch:
                    sample, noise_info = apply_random_walk_noise_to_batch(
                        sample, self.noise_config, is_training=True
                    )
                    sample["noise_info"] = noise_info
            elif self.noise_config.get("independent", {}).get("enable", False):
                sample = apply_independent_noise(
                    sample, self.noise_config, is_training=True
                )

        return sample

    def set_epoch(self, epoch: int):
        """Set current epoch for noise scheduling."""
        self.current_epoch = int(epoch)

    @staticmethod
    def _farthest_point_sample_np(
        vertices: "np.ndarray",
        k: int,
        rule: str = "index0",
        start: Optional[int] = None,
    ) -> "np.ndarray":
        """Deterministic farthest point sampling on an (N,3) array. Returns indices of shape (k,).

        rule='index0': start at vertex 0, ties resolved by the lowest index.
        rule='centroid_farthest': start at the vertex farthest from the centroid and resolve every
        tie by the lexicographically smallest coordinates, so the result depends on the vertex set
        only, not on its ordering. An explicit `start` overrides the rule's start vertex.
        """
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Expected vertices shape (N,3); got {vertices.shape}")
        n = vertices.shape[0]
        if k > n:
            raise ValueError(f"FPS requested k={k} but only N={n} vertices available")
        V = vertices.astype(np.float64, copy=False)
        lex = rule == "centroid_farthest"

        def argmax(d):
            if not lex:
                return int(np.argmax(d))
            cand = np.flatnonzero(d >= d.max() - 1e-9 * max(1.0, abs(float(d.max()))))
            if cand.shape[0] == 1:
                return int(cand[0])
            order = np.lexsort((V[cand, 2], V[cand, 1], V[cand, 0]))
            return int(cand[order[0]])

        selected = np.empty((k,), dtype=np.int64)
        dists = np.full((n,), np.inf, dtype=np.float64)
        if start is not None:
            selected[0] = int(start)
        elif lex:
            selected[0] = argmax(((V - V.mean(axis=0)) ** 2).sum(axis=1))
        else:
            selected[0] = 0
        last = V[selected[0]]
        for i in range(1, k):
            delta = V - last
            dists = np.minimum(dists, (delta * delta).sum(axis=1))
            # Never re-select a vertex: meshes may contain coincident duplicates, whose distance to an
            # already selected vertex is exactly zero. Excluding selected indices keeps the output a
            # permutation; duplicates are appended last (ties resolved by the rule above).
            dists[selected[:i]] = -np.inf
            idx = argmax(dists)
            selected[i] = idx
            last = V[idx]
        return selected.astype(np.int32, copy=False)


def make_collate_fn(
    max_objects_per_frame: Optional[int] = None,
    max_vertices_per_object: Optional[int] = None,
):
    """Factory function to create a collate_fn with fixed padding dimensions.

    Args:
        max_objects_per_frame: Fixed maximum number of objects per frame (None = dynamic)
        max_vertices_per_object: Fixed maximum vertices per object (None = dynamic)

    Returns:
        Configured collate function
    """

    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching sequences."""
        # Detect multi-object sample: obj_positions present with per-sample padding
        if "obj_positions" in batch[0]:
            # Determine dimensions
            T = batch[0]["obj_positions"].shape[0]

            # Use fixed padding if configured, otherwise compute from batch
            if max_objects_per_frame is not None:
                max_nobj = int(max_objects_per_frame)
                # Validate that no sample exceeds the fixed limit
                for s in batch:
                    _, nobj, _, _ = s["obj_positions"].shape
                    if nobj > max_nobj:
                        raise ValueError(
                            f"Sample has {nobj} objects, exceeding max_objects_per_frame={max_nobj}. "
                            f"Increase data.max_objects_per_frame in config or reduce object count in dataset."
                        )
            else:
                # Dynamic padding
                max_nobj = 0
                for s in batch:
                    _, nobj, _, _ = s["obj_positions"].shape
                    if nobj > max_nobj:
                        max_nobj = nobj

            if max_vertices_per_object is not None:
                max_nv = int(max_vertices_per_object)
                # Validate that no sample exceeds the fixed limit
                for s in batch:
                    _, _, nv, _ = s["obj_positions"].shape
                    if nv > max_nv:
                        raise ValueError(
                            f"Sample has {nv} vertices, exceeding max_vertices_per_object={max_nv}. "
                            f"Increase data.max_vertices_per_object in config or reduce vertex count in dataset."
                        )
            else:
                # Dynamic padding
                max_nv = 0
                for s in batch:
                    _, _, nv, _ = s["obj_positions"].shape
                    if nv > max_nv:
                        max_nv = nv

            B = len(batch)
            device = batch[0]["obj_positions"].device

            # Allocate padded tensors.
            obj_positions = torch.full(
                (B, T, max_nobj, max_nv, 3),
                PADDING_VALUE,
                dtype=torch.float32,
                device=device,
            )
            valid_obj_mask = torch.zeros((B, max_nobj), dtype=torch.bool, device=device)
            valid_vtx_mask = torch.zeros(
                (B, T, max_nobj, max_nv), dtype=torch.bool, device=device
            )
            ref_obj_positions = torch.full(
                (B, max_nobj, max_nv, 3),
                PADDING_VALUE,
                dtype=torch.float32,
                device=device,
            )
            # Optional anchors, padded per object to max_nobj with fixed Nq.
            # anchor_idx pads with index 0; anchor_mask pads with False.
            has_anchors = "anchor_idx" in batch[0]
            has_rank = "vtx_rank" in batch[0]
            if has_rank:
                vtx_rank = torch.full(
                    (B, max_nobj, max_nv), int(max_nv), dtype=torch.int64, device=device
                )
            if has_anchors:
                anchor_k = int(batch[0]["anchor_idx"].shape[1])
                anchor_idx = torch.zeros(
                    (B, max_nobj, anchor_k), dtype=torch.int64, device=device
                )
                anchor_mask = torch.zeros(
                    (B, max_nobj, anchor_k), dtype=torch.bool, device=device
                )

            # Fill per-sample
            for b, s in enumerate(batch):
                t_, nobj, nv, _ = s["obj_positions"].shape
                obj_positions[b, :, :nobj, :nv, :] = s["obj_positions"]
                valid_obj_mask[b, :nobj] = s["valid_obj_mask"]
                valid_vtx_mask[b, :, :nobj, :nv] = s["valid_vtx_mask"]
                ref_obj_positions[b, :nobj, :nv, :] = s["ref_obj_positions"]
                if has_rank:
                    if "vtx_rank" not in s:
                        raise KeyError(
                            "Batch sample missing 'vtx_rank' while others provide it"
                        )
                    r = s["vtx_rank"].to(device=device, dtype=torch.int64)
                    vtx_rank[b, :nobj, :nv] = torch.where(
                        r[:, :nv] >= nv,
                        torch.full_like(r[:, :nv], int(max_nv)),
                        r[:, :nv],
                    )
                if has_anchors:
                    if "anchor_idx" not in s:
                        raise KeyError(
                            "Batch sample missing 'anchor_idx' while others provide it"
                        )
                    if s["anchor_idx"].dim() != 2:
                        raise ValueError(
                            f"anchor_idx must be (Nobj, Nq); got {tuple(s['anchor_idx'].shape)}"
                        )
                    Nq = int(s["anchor_idx"].shape[1])
                    if Nq != anchor_k:
                        raise ValueError(
                            f"Inconsistent anchor_k across batch: {Nq} vs {anchor_k}"
                        )
                    if s["anchor_idx"].shape[0] != nobj:
                        raise ValueError(
                            f"anchor_idx Nobj ({s['anchor_idx'].shape[0]}) != sample Nobj ({nobj})"
                        )
                    anchor_idx[b, :nobj, :] = s["anchor_idx"].to(
                        device=device, dtype=torch.int64
                    )
                    # With strict loader checks, all anchors are valid; set mask True
                    anchor_mask[b, :nobj, :] = True

            collated = {
                "obj_positions": obj_positions,
                "valid_obj_mask": valid_obj_mask,
                "valid_vtx_mask": valid_vtx_mask,
                "ref_obj_positions": ref_obj_positions,
                "dt": torch.stack([s["dt"] for s in batch], dim=0),
            }
            if has_anchors:
                collated["anchor_idx"] = anchor_idx
                collated["anchor_mask"] = anchor_mask
            if has_rank:
                collated["vtx_rank"] = vtx_rank

            # Handle phys_params (Nobj, 3) - pad to max_nobj with PADDING_VALUE
            if "phys_params" in batch[0]:
                phys_params_padded = torch.full(
                    (B, max_nobj, 3), PADDING_VALUE, dtype=torch.float32
                )
                for b, s in enumerate(batch):
                    nobj = s["phys_params"].shape[0]
                    phys_params_padded[b, :nobj, :] = s["phys_params"]
                collated["phys_params"] = phys_params_padded

            # Handle skip_step_size (scalar) - can be stacked directly
            if "skip_step_size" in batch[0]:
                collated["skip_step_size"] = torch.stack(
                    [s["skip_step_size"] for s in batch], dim=0
                )

            # Pass through any remaining keys; tensors are stacked and must agree in shape.
            for key in batch[0].keys():
                if key in collated:
                    continue
                if isinstance(batch[0][key], torch.Tensor):
                    collated[key] = torch.stack(
                        [sample[key] for sample in batch], dim=0
                    )
                else:
                    collated[key] = [sample[key] for sample in batch]

            return collated

    return collate_fn
