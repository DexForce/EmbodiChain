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

"""Prepare pinned official RigidFormer assets and bounded packed scene files."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import numpy as np

DATA_REVISION = "e78514b479b11acfa5d23d28f1211d7830646ff9"
MODEL_REVISION = "8f2c0a3a157b64e4b310c64a01eee6e4e7e25af6"
WEIGHT_SHA256 = {
    "movi_sphere": "ffb0efc6904b5a63504a5005bb6c4df2e79e8de20641dc205885593747fcf7d9",
    "movi_a": "db2e560f132933e2169ec9f90330e6b0595b3c1656684ca7f31adf95b0bc3d19",
    "movi_b": "7db1dacf8850939a6398028fea71d9d58e75eebd7fc5079e75bd8475e6acf1ec",
}
FOLDERS = {"movi_sphere": "MOVi-Sphere", "movi_a": "MOVi-A", "movi_b": "MOVi-B"}
WEIGHTS = {
    "movi_sphere": "rigidformer_movisphere.pth",
    "movi_a": "rigidformer_movia.pth",
    "movi_b": "rigidformer_movib.pth",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(
    cache: Path, dataset: str, *, assets: str = "data"
) -> tuple[Path, Path | None]:
    from huggingface_hub import snapshot_download, hf_hub_download

    if assets not in ("data", "weights", "all"):
        raise ValueError("assets must be data, weights or all")
    raw = cache / "dataset"
    if assets in ("data", "all"):
        raw = Path(
            snapshot_download(
                "frankzydou/RigidFormer-MOVi",
                repo_type="dataset",
                revision=DATA_REVISION,
                local_dir=raw,
                allow_patterns=[
                    "manifest.json",
                    f"data/{dataset}/*",
                    "assets/kubasic/*",
                ],
            )
        )
    weight = None
    if assets in ("weights", "all"):
        weight = Path(
            hf_hub_download(
                "frankzydou/RigidFormer",
                WEIGHTS[dataset],
                revision=MODEL_REVISION,
                local_dir=cache / "model",
            )
        )
        if sha256(weight) != WEIGHT_SHA256[dataset]:
            raise ValueError("Official checkpoint checksum differs")
    return raw, weight


def _rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    columns = [
        "scene",
        "split",
        "num_frames",
        "frame_rate",
        "num_objects",
        "shape",
        "size",
        "mass",
        "friction",
        "restitution",
        "positions",
        "quaternions",
        "recovered",
    ]
    return pq.read_table(path, columns=columns).to_pylist()


def prepare(raw: Path, output: Path, dataset: str, *, limit_per_split: int = 0) -> Path:
    """Write author-compatible packed vertices without storing videos."""
    import trimesh
    from embodichain_models.dynamics.rigidformer._native.data.dataset import (
        MultiObjectDynamics,
    )

    if dataset not in FOLDERS or limit_per_split < 0:
        raise ValueError("Unknown dataset or negative scene limit")
    out = output / FOLDERS[dataset]
    if out.exists():
        raise FileExistsError(f"Prepare into a fresh dataset directory: {out}")
    original = json.loads((raw / "manifest.json").read_text())
    needed = {
        k: v
        for k, v in original["files"].items()
        if k.startswith(f"data/{dataset}/") or k.startswith("assets/")
    }
    selected = sorted((raw / "data" / dataset).glob("*.parquet"))
    if not selected:
        raise FileNotFoundError("No official numeric scenes found")
    selected_names = {str(p.relative_to(raw)) for p in selected}
    for rel, info in needed.items():
        if not limit_per_split or rel.startswith("assets/") or rel in selected_names:
            if not (raw / rel).is_file() or sha256(raw / rel) != info["sha256"]:
                raise ValueError(f"Official raw data checksum differs: {rel}")
    meshes = {}
    out.mkdir(parents=True)
    splits = {
        "scene_set": "complete",
        "train_datasets": [],
        "val_datasets": [],
        "test_datasets": [],
    }
    scales = {
        "converted": FOLDERS[dataset],
        "scales": {},
        "fingerprint": {},
        "radius0": {},
    }
    totals = {"train": 0, "validation": 0, "test": 0}
    files = {}
    try:
        for path in selected:
            split = path.name.split("-")[0]
            if split not in totals or (
                limit_per_split and totals[split] >= limit_per_split
            ):
                continue
            for rec in _rows(path):
                if rec["split"] != split:
                    raise ValueError("Scene split does not match its official shard")
                if limit_per_split and totals[split] >= limit_per_split:
                    break
                name = rec["scene"]
                folder = out / name
                folder.mkdir()
                (folder / "anchors").mkdir()
                (folder / "meshes").mkdir()
                pos = np.asarray(rec["positions"], dtype=np.float64)
                quat = np.asarray(rec["quaternions"], dtype=np.float64)
                if (
                    pos.shape != (rec["num_frames"], rec["num_objects"], 3)
                    or quat.shape != (*pos.shape[:2], 4)
                    or rec["frame_rate"] != 240
                ):
                    raise ValueError("Official pose layout or frame rate differs")
                shapes = []
                for shape in rec["shape"]:
                    if shape not in meshes:
                        mesh = trimesh.load(
                            raw
                            / "assets"
                            / "kubasic"
                            / shape
                            / "collision_geometry.obj",
                            process=False,
                        )
                        if not isinstance(mesh, trimesh.Trimesh):
                            raise ValueError("Expected one collision mesh")
                        meshes[shape] = mesh
                    shapes.append(meshes[shape])
                counts = np.array([len(m.vertices) for m in shapes], dtype=np.int64)
                frames = np.lib.format.open_memmap(
                    folder / "frames.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(pos), int(counts.sum()), 3),
                )
                for t in range(len(pos)):
                    row = []
                    for i, mesh in enumerate(shapes):
                        transform = trimesh.transformations.quaternion_matrix(
                            quat[t, i]
                        )
                        transform[:3, 3] = pos[t, i]
                        row.append(
                            trimesh.transform_points(mesh.vertices, transform).astype(
                                np.float32
                            )
                        )
                    frames[t] = np.concatenate(row, axis=0)
                frames.flush()
                np.save(folder / "frames_counts.npy", counts)
                traj = np.concatenate((pos, quat), axis=-1).astype(np.float32)[None]
                np.save(folder / "vis_traj.npy", traj)
                physics = {}
                radii = []
                offsets = np.r_[0, np.cumsum(counts)]
                for i, mesh in enumerate(shapes):
                    physics[f"obj_{i}"] = {
                        k: float(rec[k][i])
                        for k in ("mass", "friction", "restitution", "size")
                    }
                    np.save(
                        folder / "meshes" / f"obj_{i}_faces.npy",
                        np.asarray(mesh.faces, dtype=np.int32),
                    )
                    geometry = np.asarray(frames[0, offsets[i] : offsets[i + 1]])
                    order = MultiObjectDynamics._farthest_point_sample_np(
                        geometry, int(counts[i]), rule="centroid_farthest"
                    )
                    np.save(folder / "anchors" / f"obj_{i}_fpsorder_cf.npy", order)
                    np.save(
                        folder / "anchors" / f"obj_{i}_anchors_fps4_cf.npy", order[:4]
                    )
                    radii.append(
                        float(
                            np.linalg.norm(
                                geometry.astype(np.float64) - pos[0, i], axis=1
                            ).max()
                        )
                    )
                (folder / "physics_params.json").write_text(
                    json.dumps(physics, sort_keys=True, indent=2)
                )
                (folder / "conversion.json").write_text(
                    json.dumps(
                        {
                            "num_frames": len(pos),
                            "frame_rate": 240,
                            "complete": True,
                            "vertices": "unscaled asset vertices; size sidecar required",
                            "source_revision": DATA_REVISION,
                            "recovered": rec["recovered"],
                        },
                        sort_keys=True,
                    )
                )
                scales["scales"][name] = [float(v) for v in rec["size"]]
                scales["fingerprint"][name] = pos[0, 0].tolist()
                scales["radius0"][name] = radii
                key = "val_datasets" if split == "validation" else f"{split}_datasets"
                splits[key].append(name)
                totals[split] += 1
                del frames
                for f in sorted(folder.rglob("*")):
                    if f.is_file():
                        files[str(f.relative_to(out))] = sha256(f)
        if not totals["train"] or (
            not limit_per_split
            and any(v != n for v, n in zip(totals.values(), (960, 120, 120)))
        ):
            raise ValueError(f"Official scene counts differ: {totals}")
        (out / "splits.json").write_text(json.dumps(splits, sort_keys=True, indent=2))
        (out / "object_scales.json").write_text(
            json.dumps(scales, sort_keys=True, indent=2)
        )
        for f in ("splits.json", "object_scales.json"):
            files[f] = sha256(out / f)
        manifest = {
            "format": "embodichain.rigidformer.packed.v1",
            "dataset": dataset,
            "data_revision": DATA_REVISION,
            "scope": "full" if not limit_per_split else "debug-subset",
            "scene_counts": totals,
            "files": files,
            "licenses": {"scenes": "CC-BY-4.0", "meshes": "CC-BY-SA-4.0"},
        }
        tmp = out / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, sort_keys=True, indent=2))
        tmp.replace(out / "manifest.json")
    except BaseException as exc:
        (out / "failure.json").write_text(
            json.dumps({"type": type(exc).__name__, "message": str(exc)})
        )
        raise
    return out
