# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Derived from RigidFormer at 160fe0852c51cffae7b7440452b58d0b49db6df6.
# See the model family's THIRD_PARTY_LICENSES for the complete license.

"""Per-frame vertices of a converted scene, in either layout written by data_generation/movi_to_sim_data.py:

    per-file  <scene>/frame_<t>/obj_<i>_pos.npy   one array per object and frame (T x N files)
    packed    <scene>/frames.npy                  (T, sum_i Nv_i, 3): the same arrays, concatenated per frame
              <scene>/frames_counts.npy           (N,): the vertex count of every object

Both hold the same float32 values (the stored, unscaled vertices R(q) v_asset + p). The packed layout needs two files
per scene instead of ~5,000 (MOVi-B: 1,200 x 2 files instead of 3.8 million), so it fits on any local disk or a
network file system. The frame paths <scene>/frame_<t> keep naming the frames in both layouts; with the packed layout
they are names only (no folders exist).
"""

from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np

_OPEN_LIMIT = 128  # packed scenes kept memory-mapped per process
_open = OrderedDict()  # scene dir -> (frames memmap, offsets)


def clear_cache(root: Path) -> None:
    """Forget packed scenes under a newly verified source root."""
    root = Path(root).resolve()
    for key in list(_open):
        if Path(key).resolve().is_relative_to(root):
            del _open[key]


def is_packed(scene_dir: Path) -> bool:
    return (Path(scene_dir) / "frames.npy").is_file()


def _packed(scene_dir: Path):
    key = str(scene_dir)
    if key in _open:
        _open.move_to_end(key)
        return _open[key]
    frames = np.load(Path(scene_dir) / "frames.npy", mmap_mode="r")
    counts = np.load(Path(scene_dir) / "frames_counts.npy").astype(np.int64)
    if (
        frames.ndim != 3
        or frames.shape[2] != 3
        or frames.shape[1] != int(counts.sum())
        or (counts <= 0).any()
    ):
        raise ValueError(
            f"{scene_dir}: frames.npy {frames.shape} does not hold the vertex counts {counts.tolist()}"
        )
    entry = (frames, np.concatenate([[0], np.cumsum(counts)]))
    _open[key] = entry
    while len(_open) > _OPEN_LIMIT:
        _open.popitem(last=False)
    return entry


def num_frames(scene_dir: Path) -> int:
    if is_packed(scene_dir):
        return int(_packed(scene_dir)[0].shape[0])
    return len(frame_dirs(scene_dir))


def frame_dirs(scene_dir: Path) -> List[Path]:
    """frame_<t> paths of a scene, t = 0 .. T-1 (per-file layout: the existing folders, sorted)."""
    scene_dir = Path(scene_dir)
    if is_packed(scene_dir):
        return [
            scene_dir / f"frame_{t:05d}" for t in range(_packed(scene_dir)[0].shape[0])
        ]
    return sorted(p for p in scene_dir.glob("frame_*") if p.is_dir())


def load_frame(scene_dir: Path, t: int) -> List[np.ndarray]:
    """The stored vertices (Nv_i, 3) of every object at frame t, float32 (copies for the packed layout)."""
    scene_dir = Path(scene_dir)
    if is_packed(scene_dir):
        frames, off = _packed(scene_dir)
        if not 0 <= t < frames.shape[0]:
            raise IndexError(
                f"{scene_dir}: frame {t} outside 0 .. {frames.shape[0] - 1}"
            )
        row = np.asarray(frames[t])
        return [row[off[i] : off[i + 1]].copy() for i in range(len(off) - 1)]
    frame_dir = scene_dir / f"frame_{t:05d}"
    out = []
    while (frame_dir / f"obj_{len(out)}_pos.npy").exists():
        out.append(np.load(frame_dir / f"obj_{len(out)}_pos.npy"))
    if not out:
        raise FileNotFoundError(f"{frame_dir}: no obj_<i>_pos.npy")
    return out
