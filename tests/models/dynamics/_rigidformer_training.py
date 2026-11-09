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

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from embodichain_models.dynamics.rigidformer.data import MOViSequenceSource
from ._rigidformer_fixtures import tiny_cfg


def make_packed(root: Path, frames_count: int = 120) -> Path:
    folder = root / "scene_0"
    folder.mkdir(parents=True)
    (folder / "anchors").mkdir()
    (folder / "meshes").mkdir()
    shape = np.array(
        [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], dtype=np.float32
    )
    frames = np.stack(
        [
            np.concatenate(
                [shape + [t * 0.001, 0, 1.0], shape + [t * 0.001, 1.0, 1.0]], axis=0
            )
            for t in range(frames_count)
        ]
    ).astype(np.float32)
    np.save(folder / "frames.npy", frames)
    np.save(folder / "frames_counts.npy", np.array([4, 4], dtype=np.int64))
    traj = np.zeros((1, frames_count, 2, 7), dtype=np.float32)
    traj[0, :, :, 3] = 1
    for t in range(frames_count):
        traj[0, t, :, :3] = [[t * 0.001, 0, 1.0], [t * 0.001, 1.0, 1.0]]
    np.save(folder / "vis_traj.npy", traj)
    physics = {
        f"obj_{i}": {"mass": 1.0, "friction": 0.8, "restitution": 0.3, "size": size}
        for i, size in enumerate((0.7, 1.4))
    }
    (folder / "physics_params.json").write_text(json.dumps(physics))
    for i in range(2):
        np.save(
            folder / "meshes" / f"obj_{i}_faces.npy",
            np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32),
        )
        # All four tetrahedron vertices are anchors; the stored order is an
        # explicit fixture input, included in the manifest like converted data.
        order = np.array([1, 2, 3, 0], dtype=np.int32)
        np.save(folder / "anchors" / f"obj_{i}_fpsorder_cf.npy", order)
        np.save(folder / "anchors" / f"obj_{i}_anchors_fps4_cf.npy", order)
    splits = {
        "scene_set": "complete",
        "train_datasets": ["scene_0"],
        "val_datasets": [],
        "test_datasets": [],
    }
    (root / "splits.json").write_text(json.dumps(splits))
    p0 = traj[0, 0, :, :3].astype(np.float64)
    radius = [
        float(
            np.linalg.norm(
                frames[0, i * 4 : (i + 1) * 4].astype(np.float64) - p0[i], axis=1
            ).max()
        )
        for i in range(2)
    ]
    (root / "scales.json").write_text(
        json.dumps(
            {
                "converted": root.name,
                "scales": {"scene_0": [0.7, 1.4]},
                "fingerprint": {"scene_0": p0[0].tolist()},
                "radius0": {"scene_0": radius},
            }
        )
    )
    manifest = {
        "files": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file()
        }
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def source(root, *, random_step=True, **kwargs):
    sampling = {
        "sequence_length": 8,
        "skip_first_frames": 2,
        "random_step_size": {
            "enable": random_step,
            "step_sizes": [1, 5, 10],
            "random_start_frame": True,
            "sampling_weights": [0.34, 0.33, 0.33],
        },
        "anchor_k": 4,
        "anchor_fps_start": "centroid_farthest",
        "fps_order": True,
        "max_objects_per_frame": 2,
        "augmentation": {},
    }
    return MOViSequenceSource(
        root, root / "splits.json", root / "scales.json", data_config=sampling, **kwargs
    )


def recovery_config(root: Path, output: Path, device: str) -> dict:
    """Use four windows per scene to cross epochs and warmup in six updates."""
    architecture = tiny_cfg().architecture
    return {
        "model": {"architecture": architecture},
        "data": {
            "root": str(root),
            "split_file": str(root / "splits.json"),
            "scale_file": str(root / "scales.json"),
            "sampling": {
                "sequence_length": 8,
                "skip_first_frames": 2,
                "max_objects_per_frame": 2,
                "random_step_size": {"enable": False},
                "augmentation": {},
            },
        },
        "training": {"max_epochs": 3, "scheduler": {"warmup_epochs": 2}},
        "precision": "fp32",
        "trainer": {
            "device": device,
            "batch_size": 1,
            "total_updates": 6,
            "seed": 42,
            "checkpoint_interval": 0,
            "output_dir": str(output),
        },
    }


def assert_exact(actual, expected) -> None:
    """Check exact resumed tensor, optimizer, RNG and participant state."""
    import torch

    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_exact(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_exact(a, b)
    else:
        assert actual == expected
