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

"""Private numeric LeRobot transport for the PickPlace learning experiment.

The official SDK writes and reads the dataset. After integrity validation the
learner samples resident CPU tensors, preserving the existing causal-history
sampler. This measures offline learning behavior and startup loading cost;
it does not measure a LeRobot data loader's per-minibatch throughput.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.data_pipeline._ods_pick_place_benchmark import (
    FrozenCorpus,
    _content_manifest,
    _validate_tensors,
    load_corpus,
)

__all__ = ["export_lerobot_corpus", "load_lerobot_corpus"]

_FORMAT = "ods-pick-place-lerobot-corpus-v1"
_SIDECAR = "meta/embodichain_ods_corpus.json"


def _validate_fps(schema: dict[str, Any], fps: int) -> None:
    if type(fps) is not int or fps < 1:
        raise ValueError("fps must be a positive integer")
    step_dt = schema.get("action", {}).get("step_dt")
    if step_dt is not None and (
        not isinstance(step_dt, (float, int))
        or isinstance(step_dt, bool)
        or not math.isfinite(step_dt)
        or not math.isclose(step_dt, 1.0 / fps, rel_tol=1e-6, abs_tol=1e-9)
    ):
        raise ValueError("LeRobot fps must match the policy action step_dt")


def export_lerobot_corpus(
    source: Path | str,
    destination: Path | str,
    *,
    repo_id: str = "local/ods-pick-place",
    fps: int = 30,
) -> dict[str, Any]:
    """Export a validated numeric corpus through the official LeRobot SDK.

    Only valid frames are written; the sidecar preserves original padded row
    positions and hashes, including empty episodes and holes in valid masks.
    The destination is published only after an exact SDK read-back check.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    source_manifest = json.loads(Path(str(source) + ".manifest.json").read_text())
    schema = source_manifest["schema"]
    _validate_fps(schema, fps)
    corpus = load_corpus(source, expected_schema=schema)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (corpus.observations.shape[-1],),
            "names": schema["observation"]["names"],
        },
        "action": {
            "dtype": "float32",
            "shape": (corpus.actions.shape[-1],),
            "names": schema["action"]["names"],
        },
        "ods.provenance": {
            "dtype": "int64",
            "shape": (4,),
            "names": ["run_id", "episode_id", "source_frame_index", "generation"],
        },
    }
    sidecar = {
        "format": _FORMAT,
        "repo_id": repo_id,
        "fps": fps,
        "lerobot_version": importlib.metadata.version("lerobot"),
        "source_manifest": source_manifest,
        "sampling": "resident_cpu_after_official_lerobot_row_load",
        "per_minibatch_lerobot_io": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    staging = temporary / "dataset"
    dataset = None
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=staging,
            robot_type="franka_panda",
            use_videos=False,
        )
        task = "Pick up the cube and place it in the fixed goal region."
        for row in range(corpus.valid.shape[0]):
            steps = corpus.valid[row].nonzero().flatten().tolist()
            if not steps:
                continue
            for step in steps:
                # A packed provenance vector avoids SDK 0.4.4's singleton
                # vector/scalar Arrow mismatch with recent NumPy versions.
                provenance = torch.zeros(4, dtype=torch.int64)
                provenance[:3] = corpus.frame_ids[row, step]
                if corpus.generations is not None:
                    provenance[3] = corpus.generations[row, step]
                frame = {
                    "observation.state": corpus.observations[row, step].clone(),
                    "action": corpus.actions[row, step].clone(),
                    "ods.provenance": provenance,
                    "task": task,
                }
                dataset.add_frame(frame)
            dataset.save_episode()
        dataset.finalize()
        dataset = None
        (staging / _SIDECAR).write_text(
            json.dumps(sidecar, indent=2, allow_nan=False) + "\n"
        )
        restored = load_lerobot_corpus(staging, expected_schema=schema)
        if restored.manifest["content_id"] != corpus.manifest["content_id"]:
            raise ValueError("LeRobot round-trip changed policy supervision")
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    finally:
        if dataset is not None:
            dataset.finalize()
        shutil.rmtree(temporary, ignore_errors=True)
    return sidecar


def load_lerobot_corpus(
    path: Path | str, *, expected_schema: dict[str, Any]
) -> FrozenCorpus:
    """Read every valid SDK row once and validate its original numeric hashes.

    This intentionally requires the showcase sidecar. Unrelated LeRobot
    demonstrations cannot silently acquire the task's observation/action
    semantics. Subsequent sequence sampling uses resident CPU tensors.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    path = Path(path)
    sidecar = json.loads((path / _SIDECAR).read_text())
    if sidecar.get("format") != _FORMAT:
        raise ValueError("Unsupported PickPlace LeRobot corpus format")
    manifest = sidecar["source_manifest"]
    if manifest.get("schema") != expected_schema:
        raise ValueError("LeRobot observation/action schema mismatch")
    _validate_fps(expected_schema, sidecar["fps"])
    tensor_specs = manifest["tensors"]
    dtypes = {
        "observations": torch.float32,
        "actions": torch.float32,
        "valid": torch.bool,
        "frame_ids": torch.int64,
        "generations": torch.int64,
    }
    if set(tensor_specs) - set(dtypes):
        raise ValueError("Unexpected numeric field in LeRobot sidecar")
    tensors = {
        key: torch.zeros(spec["shape"], dtype=dtypes[key])
        for key, spec in tensor_specs.items()
    }
    dataset = LeRobotDataset(sidecar["repo_id"], root=path, download_videos=False)
    if dataset.fps != sidecar["fps"]:
        raise ValueError("LeRobot SDK fps differs from the source sidecar")
    expected_frames = sum(
        len(episode["valid_steps"]) for episode in manifest["episodes"]
    )
    if len(dataset) != expected_frames:
        raise ValueError("LeRobot frame count differs from the source manifest")
    cursor = 0
    sdk_episode = 0
    for row, episode in enumerate(manifest["episodes"]):
        steps = episode["valid_steps"]
        if not steps:
            continue
        for sdk_frame, step in enumerate(steps):
            item = dataset[cursor]
            if (
                item["episode_index"].item() != sdk_episode
                or item["frame_index"].item() != sdk_frame
            ):
                raise ValueError("LeRobot episode/frame order changed")
            for key, shape, dtype in (
                (
                    "observation.state",
                    (tensors["observations"].shape[-1],),
                    torch.float32,
                ),
                ("action", (tensors["actions"].shape[-1],), torch.float32),
                ("ods.provenance", (4,), torch.int64),
            ):
                if item[key].shape != shape or item[key].dtype != dtype:
                    raise ValueError(f"LeRobot row {key} dtype/shape mismatch")
            if "generations" not in tensors and item["ods.provenance"][3].item() != 0:
                raise ValueError("Unexpected generation in the LeRobot provenance")
            tensors["observations"][row, step] = item["observation.state"]
            tensors["actions"][row, step] = item["action"]
            tensors["frame_ids"][row, step] = item["ods.provenance"][:3]
            tensors["valid"][row, step] = True
            if "generations" in tensors:
                tensors["generations"][row, step] = item["ods.provenance"][3]
            cursor += 1
        sdk_episode += 1
    _validate_tensors(tensors, expected_schema)
    actual = _content_manifest(tensors, expected_schema)
    if any(manifest.get(key) != value for key, value in actual.items()):
        raise ValueError("LeRobot numeric content/hash mismatch")
    loaded_manifest = copy.deepcopy(manifest)
    loaded_manifest["storage"] = {
        "format": "lerobot",
        "loader": "lerobot.datasets.lerobot_dataset.LeRobotDataset",
        "lerobot_version": importlib.metadata.version("lerobot"),
        "sampling": sidecar["sampling"],
        "per_minibatch_lerobot_io": False,
        "root": str(path.resolve()),
    }
    return FrozenCorpus(tensors, loaded_manifest)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/ods-pick-place")
    args = parser.parse_args()
    result = export_lerobot_corpus(args.source, args.output, repo_id=args.repo_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _main()
