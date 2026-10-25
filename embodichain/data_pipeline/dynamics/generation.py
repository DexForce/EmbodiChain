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

"""Generate trajectory shards using Newton's built-in analytical solvers.

This module never imports DexSim or installs a custom physics solver.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from embodichain.compute.dynamics import quat_rotate
from embodichain.data_pipeline.dynamics.shards import DynamicsShardWriter
from embodichain.utils import configclass
from ._reference_assets import reference_profile
from ._newton import _NewtonTeacher

__all__ = ["DynamicsGenerationCfg", "generate_dataset"]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _sample(
    robot: str,
    seed: int,
    horizon: int,
    motion_mode: str,
    initial_states: str,
    effort_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one stable trajectory ID independently of collector batch size."""
    profile = reference_profile(robot)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    q, qd = torch.tensor(profile["default_q"]), torch.tensor(profile["default_qd"])
    if initial_states == "sample":
        if robot == "cartpole":
            q = (2 * torch.rand(2, generator=rng) - 1) * torch.tensor([3.5, torch.pi])
            qd = (2 * torch.rand(2, generator=rng) - 1) * 10
        else:
            q[:3] = torch.tensor([-10.0, -10.0, 1.0]) + torch.rand(
                3, generator=rng
            ) * torch.tensor([20.0, 20.0, 0.3])
            angle = (2 * torch.rand((), generator=rng) - 1) * (torch.pi / 2)
            axis = 2 * torch.rand(3, generator=rng) - 1
            axis = axis / axis.norm().clamp_min(1e-12)
            q[3:7] = torch.cat(
                (axis * torch.sin(angle / 2), torch.cos(angle / 2).reshape(1))
            )
            low = torch.tensor(
                [-40.0, 30.0, -40.0, -100.0, -40.0, -100.0, -40.0, 30.0]
            ) * (torch.pi / 180)
            high = torch.tensor(
                [40.0, 100.0, 40.0, -30.0, 40.0, -30.0, 40.0, 100.0]
            ) * (torch.pi / 180)
            q[7:] = low + torch.rand(8, generator=rng) * (high - low)
            qd[:3] = quat_rotate(q[3:7], 2 * torch.rand(3, generator=rng) - 1)
            qd[3:6] = quat_rotate(
                q[3:7], (2 * torch.rand(3, generator=rng) - 1) * torch.pi
            )
            qd[6:] = (2 * torch.rand(8, generator=rng) - 1) * (2 * torch.pi)
    elif initial_states != "default":
        raise ValueError("initial_states must be sample or default.")
    efforts = torch.zeros((horizon, profile["qd_dim"]))
    if motion_mode == "random_effort":
        limits = torch.tensor(profile["effort_limits"]) * effort_scale
        sampled = (2 * torch.rand((horizon, len(limits)), generator=rng) - 1) * limits
        efforts[:, 6:] = sampled if robot == "ant" else efforts[:, 6:]
        if robot == "cartpole":
            efforts = sampled
    elif motion_mode != "passive":
        raise ValueError("motion_mode must be passive or random_effort.")
    return q, qd, efforts


def _positive_integer(name: str, value: Any, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, not {value!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@configclass
class DynamicsGenerationCfg:
    """Validated analytical sampling settings; paths belong to the caller."""

    robot: str = "cartpole"
    output_dir: str = "outputs/dynamics/data"
    device: str = "cpu"
    num_envs: int = 64
    trajectories: int = 10000
    horizon: int = 100
    seed: int = 0
    teacher_solver: dict[str, Any] | None = None
    num_substeps: int = 10
    trajectories_per_shard: int = 64
    resume: bool = False
    motion_mode: str = "random_effort"
    split_seed: int = 0
    initial_states: str = "sample"
    max_attempts: int | None = None
    effort_scale: float = 1.0
    physics_profile: str = "native"
    use_cuda_graph: bool = True

    def __post_init__(self) -> None:
        for name in (
            "num_envs",
            "trajectories",
            "horizon",
            "num_substeps",
            "trajectories_per_shard",
        ):
            _positive_integer(name, getattr(self, name))
        for name in ("seed", "split_seed"):
            _positive_integer(name, getattr(self, name), 0)
        if self.max_attempts is not None:
            _positive_integer("max_attempts", self.max_attempts, self.trajectories)
        if type(self.resume) is not bool or type(self.use_cuda_graph) is not bool:
            raise TypeError("resume and use_cuda_graph must be boolean")
        if (
            isinstance(self.effort_scale, bool)
            or not isinstance(self.effort_scale, (float, int))
            or not math.isfinite(self.effort_scale)
            or self.effort_scale < 0
        ):
            raise ValueError("effort_scale must be finite and nonnegative")
        if (
            self.robot not in {"cartpole", "ant"}
            or self.motion_mode not in {"passive", "random_effort"}
            or self.initial_states not in {"sample", "default"}
        ):
            raise ValueError("Unsupported robot, motion_mode or initial_states")
        if self.physics_profile not in {"native", "warp_reference"}:
            raise ValueError("Unsupported reference physics profile")
        if self.teacher_solver is not None and not isinstance(
            self.teacher_solver, Mapping
        ):
            raise TypeError("teacher_solver must be a mapping")
        if not isinstance(self.output_dir, (str, Path)) or not str(self.output_dir):
            raise ValueError("output_dir must be a nonempty path")


def generate_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    """Generate committed finite trajectories with bounded attempts and staging.

    Config validation happens before simulator creation. Each candidate has an
    independent seed. Resume advances only past examined candidates, including
    a final partial batch; unused padded worlds never acquire trajectory IDs.

    Args:
        config: Settings accepted by :class:`DynamicsGenerationCfg`.

    Returns:
        Actual committed trajectory/transition counts, rejected candidate count,
        rejection evidence path and elapsed wall time.
    """
    cfg = DynamicsGenerationCfg(**dict(config))
    cfg.validate()
    output = Path(cfg.output_dir)
    progress_path = output / "generation-progress.json"
    started = time.perf_counter()
    with _NewtonTeacher(
        cfg.robot,
        device=cfg.device,
        num_envs=cfg.num_envs,
        solver=cfg.teacher_solver,
        num_substeps=cfg.num_substeps,
        physics_profile=cfg.physics_profile,
        use_cuda_graph=cfg.use_cuda_graph,
    ) as host:
        manifest = dict(host.manifest)
        manifest["generation"] = {
            "horizon": cfg.horizon,
            "seed": cfg.seed,
            "motion_mode": cfg.motion_mode,
            "initial_states": cfg.initial_states,
            "effort_scale": cfg.effort_scale,
            "num_envs": cfg.num_envs,
        }
        with DynamicsShardWriter(
            output,
            manifest,
            cfg.trajectories_per_shard,
            resume=cfg.resume,
            split_seed=cfg.split_seed,
        ) as writer:
            progress = writer.manifest.get("generation_progress")
            if progress is None:
                # Older datasets used a separate progress file. New collections
                # recover from the manifest even if that compatibility report
                # was not written, or still describes an earlier boundary.
                progress = (
                    json.loads(progress_path.read_text())
                    if cfg.resume and progress_path.exists()
                    else {"next_candidate": 0, "rejected": 0, "rejections": []}
                )
            progress.setdefault("rejections", [])
            accepted = len(writer.committed_ids)
            if accepted > cfg.trajectories:
                raise ValueError(
                    "Requested trajectory count is smaller than committed data"
                )
            first = max(
                [int(value.rsplit(":", 1)[-1]) + 1 for value in writer.committed_ids]
                or [0]
            )
            cursor = max(first, int(progress["next_candidate"]))
            progress.update(next_candidate=cursor, accepted=accepted)
            # Keep the current candidate boundary beside pending shard state.
            # add_trajectory may commit automatically, so update this same
            # mapping before adding each accepted trajectory.
            writer.manifest["generation_progress"] = progress
            writer.commit()
            maximum = (
                cfg.max_attempts
                if cfg.max_attempts is not None
                else cfg.trajectories * 10
            )
            while accepted < cfg.trajectories:
                if cursor >= maximum:
                    raise RuntimeError(
                        f"Generation attempt limit reached: {accepted}/{cfg.trajectories} valid trajectories, {progress['rejected']} rejected"
                    )
                candidates = list(range(cursor, min(cursor + cfg.num_envs, maximum)))
                samples = [
                    _sample(
                        cfg.robot,
                        cfg.seed + i,
                        cfg.horizon,
                        cfg.motion_mode,
                        cfg.initial_states,
                        cfg.effort_scale,
                    )
                    for i in candidates
                ]
                samples.extend([samples[-1]] * (cfg.num_envs - len(samples)))
                q, qd, force = [
                    torch.stack([row[i] for row in samples]).to(host.device)
                    for i in range(3)
                ]
                host.reset(q, qd)
                states_list, contacts_list = [], []
                with torch.no_grad():
                    for index in range(cfg.horizon):
                        state, contacts = host.snapshot()
                        states_list.append(state)
                        contacts_list.append(contacts)
                        host.step(force[:, index])
                    state, _ = host.snapshot()
                    states_list.append(state)
                states = {
                    key: torch.stack([value[key] for value in states_list], dim=1)
                    .cpu()
                    .numpy()
                    for key in states_list[0]
                }
                transitions = {
                    key: torch.stack([value[key] for value in contacts_list], dim=1)
                    .cpu()
                    .numpy()
                    for key in contacts_list[0]
                }
                transitions["joint_f"] = force.cpu().numpy()
                transitions["gravity_dir"] = (
                    host.gravity_dir[:, None].expand(-1, cfg.horizon, -1).cpu().numpy()
                )
                for row, candidate in enumerate(candidates):
                    if accepted >= cfg.trajectories:
                        break
                    cursor = candidate + 1
                    progress["next_candidate"] = cursor
                    finite = all(
                        np.isfinite(v[row]).all()
                        for v in (*states.values(), *transitions.values())
                    )
                    bounded = (
                        finite
                        and np.max(np.abs(states["q"][row])) < 1e5
                        and np.max(np.abs(states["qd"][row])) < 1e5
                    )
                    if not bounded:
                        progress["rejected"] += 1
                        progress["rejections"].append(
                            {
                                "id": f"{cfg.seed}:{candidate}",
                                "sampling_seed": cfg.seed + candidate,
                                "reason": (
                                    "nonfinite"
                                    if not finite
                                    else "state_magnitude_limit"
                                ),
                            }
                        )
                        continue
                    progress["accepted"] = accepted + 1
                    writer.add_trajectory(
                        f"{cfg.seed}:{candidate}",
                        {k: v[row] for k, v in states.items()},
                        {k: v[row] for k, v in transitions.items()},
                        metadata={"sampling_seed": cfg.seed + candidate},
                    )
                    accepted += 1
                writer.commit()
    # Preserve the historical evidence-file layout as a derived completion
    # report. It is no longer a second recovery transaction.
    _write_json(progress_path, progress)
    result = {
        "status": "completed",
        "dataset": str(output.resolve()),
        "trajectories": accepted,
        "transitions": accepted * cfg.horizon,
        "rejected": progress["rejected"],
        "rejection_evidence": str(progress_path.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output / "generation-result.json", result)
    return result
