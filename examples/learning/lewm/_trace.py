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

"""Owned image, action, RNG and physical-state evidence for LeWM evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


def _snapshot(view: Any) -> list[dict[str, Any]]:
    from embodichain.lab.sim.scene import capture_state

    state = capture_state(view)
    return [
        {
            name: getattr(entity, name)
            for name in (
                "kind",
                "uid",
                "body_type",
                "joint_names",
                "pose",
                "velocity",
                "qpos",
                "qvel",
                "target_qpos",
                "target_qvel",
                "qf",
            )
        }
        for entity in state.batches[0]
    ]


def _pixels(obs: Any) -> np.ndarray:
    return obs["sensor"]["main_cam"]["color"][0, ..., :3].detach().cpu().numpy().copy()


class TraceWriter:
    """Write complete per-episode evidence, keeping only one episode in memory."""

    def __init__(self, path: str | Path, metadata: dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_name(self.path.name + ".tmp")
        if self.path.exists() or self.temporary.exists():
            raise FileExistsError(f"trace output already exists: {self.path}")
        self.file = h5py.File(self.temporary, "x")
        self.file.attrs["format"] = "embodichain.lewm.inference_trace.v1"
        self.file.attrs["metadata"] = json.dumps(
            metadata, sort_keys=True, allow_nan=False
        )
        self.active: dict[str, Any] | None = None

    def begin(
        self,
        kind: str,
        seed: int,
        goal: np.ndarray,
        goal_latent: torch.Tensor,
        goal_position: torch.Tensor,
        obs: Any,
        view: Any,
    ) -> None:
        if self.active is not None or kind not in {"lewm", "random"}:
            raise ValueError("trace episode is already active or policy is invalid")
        self.active = {
            "kind": kind,
            "seed": seed,
            "goal": goal.copy(),
            "goal_latent": goal_latent.detach().cpu().numpy().copy(),
            "goal_position": goal_position.detach().cpu().numpy().copy(),
            "pixels": [_pixels(obs)],
            "states": [_snapshot(view)],
            "actions": [],
            "distance": [],
            "success": [],
            "terminated": [],
            "truncated": [],
            "rng": [],
            "history": [],
            "cost": [],
        }

    def add(
        self,
        action: torch.Tensor,
        obs: Any,
        info: Any,
        terminated: Any,
        truncated: Any,
        view: Any,
        *,
        rng: dict[str, Any] | None = None,
        history: torch.Tensor | None = None,
        cost: torch.Tensor | None = None,
    ) -> None:
        if self.active is None:
            raise ValueError("trace has no active episode")
        row = self.active
        row["actions"].append(action.detach().cpu().numpy().copy()[0])
        row["pixels"].append(_pixels(obs))
        row["states"].append(_snapshot(view))
        row["distance"].append(float(info["metrics"]["distance_to_goal"][0]))
        row["success"].append(bool(info["success"][0]))
        row["terminated"].append(bool(torch.as_tensor(terminated).any()))
        row["truncated"].append(bool(torch.as_tensor(truncated).any()))
        row["rng"].append(json.dumps(rng, sort_keys=True) if rng is not None else "")
        row["history"].append(
            None if history is None else history.detach().cpu().numpy().copy()[0]
        )
        row["cost"].append(float("nan") if cost is None else float(cost[0]))

    def commit(self) -> None:
        if self.active is None:
            raise ValueError("trace has no active episode")
        row = self.active
        group = self.file.require_group(row["kind"]).create_group(str(row["seed"]))
        group.create_dataset("goal", data=row["goal"], compression="lzf")
        group.create_dataset("goal_latent", data=row["goal_latent"])
        group.create_dataset("goal_position", data=row["goal_position"])
        group.create_dataset(
            "pixels",
            data=np.stack(row["pixels"]),
            chunks=(1, *row["pixels"][0].shape),
            compression="lzf",
        )
        for key in (
            "actions",
            "distance",
            "success",
            "terminated",
            "truncated",
            "cost",
        ):
            group.create_dataset(key, data=np.asarray(row[key]))
        strings = h5py.string_dtype("utf-8")
        group.create_dataset(
            "states",
            data=[json.dumps(x, allow_nan=False) for x in row["states"]],
            dtype=strings,
        )
        group.create_dataset("rng", data=row["rng"], dtype=strings)
        valid = np.asarray(
            [value is not None for value in row["history"]], dtype=np.bool_
        )
        group.create_dataset("has_plan", data=valid)
        template = next((value for value in row["history"] if value is not None), None)
        if template is not None:
            group.create_dataset(
                "history_latent",
                data=np.stack(
                    [
                        np.zeros_like(template) if value is None else value
                        for value in row["history"]
                    ]
                ),
            )
        self.file.flush()
        self.active = None

    def finish(self) -> dict[str, Any]:
        if self.active is not None:
            raise ValueError("cannot finalize an incomplete trace episode")
        self.file.attrs["complete"] = True
        self.file.close()
        self.temporary.replace(self.path)
        digest = hashlib.sha256()
        with self.path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return {"path": str(self.path.resolve()), "sha256": digest.hexdigest()}

    def abort(self) -> None:
        self.file.close()
