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

import base64
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
        *,
        history_pixels: np.ndarray | None = None,
        history_actions: np.ndarray | None = None,
    ) -> None:
        if self.active is not None or kind not in {"lewm", "random", "zero"}:
            raise ValueError("trace episode is already active or policy is invalid")
        if (history_pixels is None) != (history_actions is None):
            raise ValueError("trace history requires both observations and actions")
        if history_pixels is not None and (
            history_pixels.dtype != np.uint8
            or history_pixels.ndim != 4
            or history_actions.ndim != 2
            or len(history_pixels) != len(history_actions) + 1
            or not np.array_equal(history_pixels[-1], _pixels(obs))
        ):
            raise ValueError("trace history must end at the actual initial observation")
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
            "initial_history_pixels": (
                None if history_pixels is None else history_pixels.copy()
            ),
            "initial_history_actions": (
                None if history_actions is None else history_actions.copy()
            ),
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
        for key in ("initial_history_pixels", "initial_history_actions"):
            if row[key] is not None:
                group.create_dataset(key, data=row[key], compression="lzf")
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


@torch.no_grad()
def check_trace(
    export: str, path: str, device: str, max_plans: int = 10
) -> dict[str, Any]:
    """Recompute recorded visual histories and MPC actions from saved RNG states."""
    from embodichain_models.dynamics.lewm import (
        LeWMCEMCfg,
        LeWMCEMPlanner,
        load_lewm,
        preprocess_rgb,
    )

    if max_plans < 1:
        raise ValueError("max_plans must be positive")
    model, manifest = load_lewm(export, device)
    checks = []
    atol, rtol = 1e-5, 1e-4
    with h5py.File(path, "r") as file:
        if file.attrs.get(
            "format"
        ) != "embodichain.lewm.inference_trace.v1" or not file.attrs.get(
            "complete", False
        ):
            raise ValueError(
                "inference trace is incomplete or has an unsupported format"
            )
        metadata = json.loads(file.attrs["metadata"])
        if metadata["weights_sha256"] != manifest["weights_sha256"]:
            raise ValueError("trace and model weights differ")
        threshold = metadata.get("success_threshold")
        physical_count = success_mismatches = 0
        max_distance_error = 0.0
        for kind in file:
            for episode in file[kind].values():
                states = episode["states"].asstr()
                distances = episode["distance"][:]
                successes = episode["success"][:]
                if len(states) != len(distances) + 1 or len(successes) != len(
                    distances
                ):
                    raise ValueError(
                        "trace physical states and metrics have inconsistent lengths"
                    )
                goal_xy = episode["goal_position"][:2]
                for index, reported_distance in enumerate(distances):
                    state = json.loads(states[index + 1])
                    cubes = [entity for entity in state if entity["uid"] == "cube"]
                    if len(cubes) != 1:
                        raise ValueError("trace physical state has no unique cube")
                    distance = float(
                        np.linalg.norm(np.asarray(cubes[0]["pose"][:2]) - goal_xy)
                    )
                    max_distance_error = max(
                        max_distance_error, abs(distance - reported_distance)
                    )
                    if threshold is not None:
                        success_mismatches += int(
                            bool(successes[index]) != (distance < threshold)
                        )
                    physical_count += 1
        physical_passed = (
            physical_count > 0
            and max_distance_error <= 1e-4
            and success_mismatches == 0
        )
        cfg = LeWMCEMCfg(**metadata["cem"])
        low = torch.as_tensor(metadata["action_low"], device=device)
        high = torch.as_tensor(metadata["action_high"], device=device)
        plan_locations = [
            (seed, int(index))
            for seed in sorted(file["lewm"], key=int)
            for index in np.flatnonzero(file["lewm"][seed]["has_plan"][:])
        ]
        if not plan_locations:
            raise ValueError("trace contains no recorded MPC decisions")
        selected: dict[str, list[int]] = {}
        for index in np.linspace(
            0, len(plan_locations) - 1, min(max_plans, len(plan_locations)), dtype=int
        ):
            seed, step = plan_locations[index]
            selected.setdefault(seed, []).append(step)
        for seed, selected_steps in selected.items():
            episode = file["lewm"][seed]
            actions = episode["actions"]
            pixels = episode["pixels"]
            if len(pixels) != len(actions) + 1:
                raise ValueError("trace pixels and actions have inconsistent lengths")
            goal = torch.as_tensor(episode["goal_latent"][:], device=device)
            goal_pixels = torch.as_tensor(episode["goal"][:], device=device)[None, None]
            reconstructed_goal = model.encode(
                preprocess_rgb(goal_pixels, model.cfg.image_size)
            )[:, 0]
            planner = LeWMCEMPlanner(model, low, high, cfg, seed=int(seed))
            h = model.cfg.history_size
            block = model.cfg.action_block
            raw_dim = model.cfg.action_dim // block
            prefix_pixels = episode.get("initial_history_pixels")
            prefix_actions = episode.get("initial_history_actions")
            if (prefix_pixels is None) != (prefix_actions is None):
                raise ValueError("trace prehistory observations/actions differ")
            prefix_length = 0 if prefix_actions is None else len(prefix_actions)
            if prefix_pixels is not None and (
                len(prefix_pixels) != prefix_length + 1
                or not np.array_equal(prefix_pixels[-1], pixels[0])
            ):
                raise ValueError("trace prehistory does not end at the initial frame")
            all_pixels = (
                pixels[:]
                if prefix_pixels is None
                else np.concatenate((prefix_pixels[:-1], pixels[:]))
            )
            all_actions = (
                actions[:]
                if prefix_actions is None
                else np.concatenate((prefix_actions[:], actions[:]))
            )
            plan_steps = np.flatnonzero(episode["has_plan"][:])
            for index in selected_steps:
                current = prefix_length + index
                first = current - (h - 1) * block
                if first < 0:
                    raise ValueError("planned action has insufficient recorded history")
                recorded_history = torch.as_tensor(
                    episode["history_latent"][index], device=device
                )[None]
                context = torch.as_tensor(
                    all_pixels[first : current + 1 : block], device=device
                )[None]
                reconstructed_history = model.encode(
                    preprocess_rgb(context, model.cfg.image_size)
                )
                past = torch.as_tensor(
                    all_actions[first:current].reshape(h - 1, model.cfg.action_dim),
                    device=device,
                )[None]
                saved_rng = json.loads(episode["rng"].asstr()[index])
                rng_state = np.frombuffer(
                    base64.b64decode(saved_rng["torch_generator_state_b64"]),
                    dtype=np.uint8,
                ).copy()
                planner.generator.set_state(torch.from_numpy(rng_state))
                sequence, cost = planner.plan_sequence(recorded_history, past, goal)
                raw_sequence = sequence.reshape(1, cfg.horizon * block, raw_dim)
                following = plan_steps[plan_steps > index]
                end = int(following[0]) if len(following) else len(actions)
                if end - index > raw_sequence.shape[1]:
                    raise ValueError(
                        "trace executes more actions than the recorded plan"
                    )
                action = raw_sequence[0, : end - index]
                expected_action = torch.as_tensor(actions[index:end], device=device)
                expected_cost = torch.as_tensor(
                    episode["cost"][index], device=device, dtype=cost.dtype
                )
                checks.append(
                    {
                        "seed": int(seed),
                        "step": int(index),
                        "executed_actions_checked": int(end - index),
                        "goal_max_abs_error": float(
                            (reconstructed_goal - goal).abs().max()
                        ),
                        "history_max_abs_error": float(
                            (reconstructed_history - recorded_history).abs().max()
                        ),
                        "action_max_abs_error": float(
                            (action - expected_action).abs().max()
                        ),
                        "cost_abs_error": float((cost[0] - expected_cost).abs()),
                        "passed": bool(
                            torch.allclose(
                                reconstructed_goal, goal, atol=atol, rtol=rtol
                            )
                            and torch.allclose(
                                reconstructed_history,
                                recorded_history,
                                atol=atol,
                                rtol=rtol,
                            )
                            and torch.allclose(
                                action, expected_action, atol=atol, rtol=rtol
                            )
                            and torch.allclose(
                                cost[0], expected_cost, atol=atol, rtol=rtol
                            )
                        ),
                    }
                )
                if len(checks) >= max_plans:
                    break
            if len(checks) >= max_plans:
                break
    if not checks:
        raise ValueError("trace contains no recorded MPC decisions")
    return {
        "passed": physical_passed and all(row["passed"] for row in checks),
        "plans_checked": len(checks),
        "atol": atol,
        "rtol": rtol,
        "weights_sha256": manifest["weights_sha256"],
        "trace": str(Path(path).resolve()),
        "physical_state_check": {
            "passed": physical_passed,
            "steps_checked": physical_count,
            "distance_max_abs_error": max_distance_error,
            "success_verified": threshold is not None,
            "success_mismatches": success_mismatches if threshold is not None else None,
        },
        "checks": checks,
    }
