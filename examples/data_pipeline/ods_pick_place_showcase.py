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

"""Train and compare lightweight PickPlace BC with live ODS or offline data.

Run from the repository root:
    python examples/data_pipeline/ods_pick_place_showcase.py --sim-device cuda:0
Use --smoke for a small engineering check, not a learning qualification.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import queue
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if __package__ in {None, ""}:
    # Direct file execution must resolve this checkout and its bundled tasks,
    # even when an editable installation points at another worktree.
    _repository = Path(__file__).resolve().parents[2]
    sys.path[:0] = [
        str(path)
        for path in (
            _repository,
            _repository / "extensions",
            _repository / "extensions" / "embodichain_tasks",
        )
        if str(path) not in sys.path
    ]

from embodichain.data_pipeline.engine.data import (
    OnlineDataEngine,
    OnlineDataEngineCfg,
    OnlineDataEngineState,
)

from examples.data_pipeline._ods_pick_place_policy import ChunkBC

TASK_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "extensions/embodichain_tasks/configs/tasks/manipulation/pick_place/task.franka.yaml"
)


def policy_observations(obs: Any, mode: str = "physical") -> torch.Tensor:
    """Pack current measured state only; never read demonstration annotations."""
    if mode == "geometry":
        from embodichain.utils.math import matrix_from_quat

        tcp = obs["tcp_pose"]
        cube = obs["cube_pose"]
        quaternion = F.normalize(cube[..., 3:], dim=-1)
        identity = torch.zeros_like(quaternion)
        identity[..., 3] = 1
        # History padding has a zero payload, including its quaternion. Convert
        # that neutral payload to a finite rotation before masking it downstream.
        quaternion = torch.where(
            quaternion.norm(dim=-1, keepdim=True) == 0, identity, quaternion
        )
        rotation = matrix_from_quat(quaternion)
        return torch.cat(
            (
                obs["robot"]["qpos"],
                obs["robot"]["qvel"],
                tcp[..., :3, 3],
                tcp[..., :3, :2].transpose(-1, -2).flatten(-2),
                cube[..., :3],
                rotation[..., :3, :2].transpose(-1, -2).flatten(-2),
                obs["goal_position"],
                cube[..., :3] - tcp[..., :3, 3],
                obs["goal_position"] - cube[..., :3],
                obs["cube_velocity"],
            ),
            dim=-1,
        ).float()
    return torch.cat(
        (
            obs["robot"]["qpos"],
            obs["robot"]["qvel"],
            obs["cube_pose"],
            obs["goal_position"],
        ),
        dim=-1,
    ).float()


def _canonical_observations(obs: torch.Tensor) -> torch.Tensor:
    """Give equivalent xyzw quaternions the same bounded representation."""
    quaternion = F.normalize(obs[..., 21:25], dim=-1)
    pivot = quaternion.gather(-1, quaternion.abs().argmax(-1, keepdim=True))
    quaternion = torch.where(pivot < 0, -quaternion, quaternion)
    return torch.cat((obs[..., :21], quaternion, obs[..., 25:]), dim=-1)


def _observation_statistics(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Freeze training-only statistics with feature-specific physical scales."""
    values = _canonical_observations(values)
    floors = values.new_tensor(
        [0.1] * 7
        + [0.02] * 2
        + [0.5] * 7
        + [0.02] * 2
        + [0.05] * 3
        + [1.0] * 4
        + [0.1] * 3
    )
    mean, std = values.mean(0), values.std(0).clamp_min(floors)
    # Unit quaternions already have a physical scale and need no centering.
    mean[21:25], std[21:25] = 0, 1
    return mean, std


def _geometry_statistics(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    floors = values.new_tensor(
        [0.1] * 7
        + [0.02] * 2
        + [0.5] * 7
        + [0.02] * 2
        + [0.05] * 3
        + [1.0] * 6
        + [0.05] * 3
        + [1.0] * 6
        + [0.1] * 3
        + [0.05] * 6
        + [0.1] * 3
        + [0.5] * 3
    )
    mean, std = values.mean(0), values.std(0, correction=0).clamp_min(floors)
    mean[21:27], std[21:27] = 0, 1
    mean[30:36], std[30:36] = 0, 1
    return mean, std


class MLPBC(nn.Module):
    """Two-layer state BC in normalized pose-request or joint-residual space."""

    observation_mode = "geometry"

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        *,
        arm_command_mode: str = "pose",
    ) -> None:
        super().__init__()
        width = 7 if arm_command_mode == "pose" else 8
        if arm_command_mode not in {"pose", "residual"}:
            raise ValueError("MLP actions must be pose requests or joint residuals")
        if (
            mean.shape != (51,)
            or std.shape != (51,)
            or low.shape != (width,)
            or high.shape != low.shape
        ):
            raise ValueError(
                "MLP requires 51 measured features and matching action bounds"
            )
        if (
            not all(torch.isfinite(v).all() for v in (mean, std, low, high))
            or (std <= 0).any()
            or (high <= low).any()
        ):
            raise ValueError("Invalid MLP normalization or action bounds")
        self.arm_command_mode = arm_command_mode
        for name, value in (("mean", mean), ("std", std), ("low", low), ("high", high)):
            self.register_buffer(name, value.clone())
        self.net = nn.Sequential(
            nn.Linear(51, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, width),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(((obs - self.mean) / self.std).clamp(-10, 10))

    def action(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        request = self(obs).clamp(-1, 1)
        if self.arm_command_mode == "pose":
            return request, None
        arm = (obs[..., :7] + 0.25 * request[..., :7]).clamp(
            self.low[:7], self.high[:7]
        )
        finger = self.low[7:] + (request[..., 7:] + 1) * 0.5 * (
            self.high[7:] - self.low[7:]
        )
        return torch.cat((arm, finger), dim=-1), None

    def loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = actions
        if self.arm_command_mode == "residual":
            target = torch.cat(
                (
                    (actions[..., :7] - obs[..., :7]) / 0.25,
                    2
                    * (actions[..., 7:] - self.low[7:])
                    / (self.high[7:] - self.low[7:])
                    - 1,
                ),
                dim=-1,
            )
        error = (self(obs) - target).square()
        if loss_mask is not None:
            error = error[loss_mask]
        return error[..., :-1].mean(), error[..., -1].mean()


class GeometryRecurrentBC(MLPBC):
    """Causal measured-geometry fallback with normalized pose-request MSE."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> None:
        super().__init__(mean, std, low, high, arm_command_mode="pose")
        del self.net
        self.gru = nn.GRU(51, 128, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 7))

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict unclamped requests from full causal measured-state sequences."""
        normalized = ((obs - self.mean) / self.std).clamp(-10, 10)
        features, hidden = self.gru(normalized, hidden)
        return self.head(features), hidden

    def action(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one measured-state step and carry its recurrent state."""
        requests, hidden = self(obs[:, None], hidden)
        return requests[:, 0].clamp(-1, 1), hidden

    def loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Supervise valid requests while backpropagating through their prefixes."""
        if loss_mask is not None and (
            loss_mask.shape != obs.shape[:2]
            or loss_mask.dtype != torch.bool
            or not loss_mask.any()
        ):
            raise ValueError("loss_mask must select supervised sequence steps")
        predictions, _ = self(obs)
        error = (predictions - actions).square()
        if loss_mask is not None:
            error = error[loss_mask]
        return error[..., :6].mean(), error[..., 6].mean()


class RecurrentBC(nn.Module):
    """Small recurrent policy with frozen observation and action normalization."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        *,
        observation_mode: str = "physical",
        arm_command_mode: str = "absolute",
    ) -> None:
        super().__init__()
        if (
            mean.shape != (28,)
            or std.shape != mean.shape
            or low.shape != (8,)
            or high.shape != low.shape
        ):
            raise ValueError(
                "Expected 28 observation features and 8 active action joints"
            )
        if (
            not all(torch.isfinite(t).all() for t in (mean, std, low, high))
            or (std <= 0).any()
            or (high <= low).any()
        ):
            raise ValueError("Invalid normalization or action bounds")
        if observation_mode not in {"legacy", "physical"}:
            raise ValueError("Unknown policy observation mode")
        if arm_command_mode not in {"absolute", "residual"}:
            raise ValueError("Unknown arm command mode")
        self.observation_mode = observation_mode
        self.arm_command_mode = arm_command_mode
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())
        self.register_buffer("low", low.clone())
        self.register_buffer("high", high.clone())
        self.gru = nn.GRU(28, 128, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 8), nn.Tanh()
        )

    def forward(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.observation_mode == "physical":
            obs = _canonical_observations(obs)
        normalized = (obs - self.mean) / self.std
        if self.observation_mode == "physical":
            normalized = normalized.clamp(-10, 10)
        features, hidden = self.gru(normalized, hidden)
        prediction = self.head(features)
        if self.arm_command_mode == "residual":
            # Learn the controller lead rather than reconstructing measured qpos.
            # The environment still receives bounded absolute joint positions.
            # A normalized residual of +/-1 adds at most +/-0.25 radians.
            span = self.high[:7] - self.low[:7]
            current = 2 * (obs[..., :7] - self.low[:7]) / span - 1
            arm = (current + prediction[..., :7] * (0.5 / span)).clamp(-1, 1)
            prediction = torch.cat((arm, prediction[..., 7:]), dim=-1)
        return prediction, hidden

    def action(
        self, obs: torch.Tensor, hidden: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized, hidden = self(obs[:, None], hidden)
        return self.low + (normalized[:, 0] + 1) * 0.5 * (self.high - self.low), hidden

    def loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        burn_in: int = 8,
        *,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if loss_mask is not None:
            if loss_mask.shape != obs.shape[:2] or not loss_mask.any():
                raise ValueError("loss_mask must select supervised sequence steps")
            # Full causal prefixes reconstruct the same state as online inference.
            # Gradients through history also teach long contact/dwell transitions.
            prediction, _ = self(obs)
            target = 2 * (actions - self.low) / (self.high - self.low) - 1
            error = F.smooth_l1_loss(prediction, target, reduction="none")[loss_mask]
            return error[..., :7].mean(), error[..., 7:].mean()
        if not 0 <= burn_in < obs.shape[1]:
            raise ValueError("burn_in must leave at least one supervised step")
        hidden = None
        if burn_in:
            with torch.no_grad():
                _, hidden = self(obs[:, :burn_in])
        prediction, _ = self(obs[:, burn_in:], hidden)
        target = 2 * (actions[:, burn_in:] - self.low) / (self.high - self.low) - 1
        error = F.smooth_l1_loss(prediction, target, reduction="none")
        return error[..., :7].mean(), error[..., 7:].mean()


def load_task_config(
    device: str, action_mode: str = "joint", pose_ik: str = "solver"
) -> dict[str, Any]:
    from embodichain.utils.utility import load_config
    from embodichain.lab.gym.utils._component_composition import _resolve_gym_components

    config = _resolve_gym_components(
        load_config(TASK_CONFIG), base_dir=TASK_CONFIG.parent
    ).config
    config.update(
        device=device,
        gpu_id=int(device.split(":")[-1]) if device.startswith("cuda:") else 0,
        headless=True,
        renderer="no-render",
        enable_sensor=False,
        num_envs=1,
    )
    if action_mode == "pose":
        config["env"].setdefault("extensions", {})["pick_place_action_mode"] = "pose"
        config["env"]["actions"] = {
            "arm": {
                "contract": "eef_pose.relative@1",
                "params": {
                    "part_name": "arm",
                    "translation_scale": 0.02,
                    "rotation_scale": 0.1,
                    "ik_method": pose_ik,
                },
            },
            "eef": {
                "contract": "gripper.parallel@1",
                "params": {
                    "part_name": "eef",
                    "command_mode": "continuous",
                    "lower_command": {".*": 0.0},
                    "upper_command": {".*": 0.04},
                },
            },
        }
    return config


def save_checkpoint(
    path: Path,
    model: RecurrentBC | MLPBC | GeometryRecurrentBC | ChunkBC,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    run_id: int,
    *,
    sampling_rng_state: torch.Tensor | None = None,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {path}")
    state = {
        "model": model.state_dict(),
        "policy_config": {
            "type": (
                "chunk_pose"
                if isinstance(model, ChunkBC)
                else (
                    "gru_pose"
                    if isinstance(model, GeometryRecurrentBC)
                    else "mlp" if isinstance(model, MLPBC) else "gru"
                )
            ),
            "observation_mode": model.observation_mode,
            "arm_command_mode": model.arm_command_mode,
            "schema": getattr(model, "schema", None),
            **(
                {
                    "prediction_horizon": model.prediction_horizon,
                    "action_horizon": model.action_horizon,
                    "observation_noise": model.observation_noise,
                }
                if isinstance(model, ChunkBC)
                else {}
            ),
        },
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": vars(args),
        "run_id": run_id,
        "torch_rng": torch.get_rng_state(),
        "sampling_rng": sampling_rng_state,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": (
            {"train": torch.cuda.get_rng_state(args.train_device)}
            if torch.cuda.is_initialized()
            and str(getattr(args, "train_device", "cpu")).startswith("cuda:")
            else None
        ),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_policy(
    path: str, device: str
) -> tuple[RecurrentBC | MLPBC | GeometryRecurrentBC | ChunkBC, dict[str, Any]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = state["model"]
    policy_config = state.get("policy_config", {})
    if policy_config.get("type") == "chunk_pose":
        if (
            policy_config.get("observation_mode") != "geometry"
            or policy_config.get("arm_command_mode") != "pose"
        ):
            raise ValueError(
                "Chunk BC requires geometry observations and pose requests"
            )
        model = ChunkBC(
            weights["mean"],
            weights["std"],
            weights["target_mean"],
            weights["target_std"],
            policy_config["prediction_horizon"],
            policy_config["action_horizon"],
            policy_config["observation_noise"],
        ).to(device)
        model.load_state_dict(weights)
        model.schema = policy_config.get("schema")
        model.eval()
        return model, state
    if policy_config.get("type") == "gru_pose":
        if (
            policy_config.get("observation_mode") != "geometry"
            or policy_config.get("arm_command_mode") != "pose"
        ):
            raise ValueError(
                "gru_pose requires geometry observations and pose requests"
            )
        model = GeometryRecurrentBC(
            weights["mean"], weights["std"], weights["low"], weights["high"]
        ).to(device)
        model.load_state_dict(weights)
        model.schema = policy_config.get("schema")
        model.eval()
        return model, state
    if policy_config.get("type") == "mlp":
        model = MLPBC(
            weights["mean"],
            weights["std"],
            weights["low"],
            weights["high"],
            arm_command_mode=policy_config["arm_command_mode"],
        ).to(device)
        model.load_state_dict(weights)
        model.schema = policy_config.get("schema")
        model.eval()
        return model, state
    model = RecurrentBC(
        weights["mean"],
        weights["std"],
        weights["low"],
        weights["high"],
        observation_mode=state.get("policy_config", {}).get(
            "observation_mode", "legacy"
        ),
        arm_command_mode=state.get("policy_config", {}).get(
            "arm_command_mode", "absolute"
        ),
    ).to(device)
    model.load_state_dict(weights)
    model.eval()
    return model, state


def _evaluate_worker(
    config: dict[str, Any],
    requests: Any,
    results: Any,
    output: str,
    viewer: bool,
    port: int,
    video: bool,
) -> None:
    """Own the evaluation simulator and run only requested immutable policies."""
    env = None
    try:
        import gymnasium as gym
        from embodichain.lab.gym.utils.registration import discover_task_packages
        from embodichain.lab.gym.utils.gym_utils import config_to_cfg
        from embodichain.lab.gym.envs.demo import execute_demo_episode

        discover_task_packages()
        config = copy.deepcopy(config)
        if viewer:
            config["visualization"] = {
                "backend": "viser",
                "allow_commands": False,
                "viser_server": {"host": "127.0.0.1", "port": port},
            }
        if video:
            config["enable_sensor"] = True
            config["renderer"] = "hybrid"
            for sensor in config.get("sensor", []):
                sensor.update(width=320, height=240)
                sensor["intrinsics"] = [244.08325, 244.08325, 160.0, 120.0]
        cfg = config_to_cfg(config)
        cfg.init_rollout_buffer = False
        cfg.filter_dataset_saving = True
        env = gym.make(config["id"], cfg=cfg)
        raw = env.unwrapped
        raw._replay_no_auto_reset = True
        results.put(
            {
                "ready": True,
                "low": raw.single_action_space.low.tolist(),
                "high": raw.single_action_space.high.tolist(),
                "joints": list(raw.expert_action_spec.joint_names),
                "action_terms": (
                    [d.to_dict() for d in raw.action_manager.descriptors]
                    if raw.action_manager is not None
                    else []
                ),
            }
        )
        while True:
            request = requests.get()
            if request is None:
                break
            kind = request["kind"]
            model = None
            model_step = "expert"
            if kind != "qualify":
                model, checkpoint_state = load_policy(
                    request["checkpoint"], config["device"]
                )
                schema = getattr(model, "schema", None)
                if schema is not None:
                    actual = (
                        [d.to_dict() for d in raw.action_manager.descriptors]
                        if raw.action_manager is not None
                        else []
                    )
                    if (
                        schema["action"]["terms"] != actual
                        or schema["observation"]["version"] != "pick_place.geometry@2"
                    ):
                        raise ValueError(
                            "Checkpoint observation/controller schema does not match evaluation environment"
                        )
                model_step = str(checkpoint_state["step"])
            if viewer:
                raw.sim.stop_visualization()
                raw.sim.sim_config.visualization.viser_server.label = (
                    f'ODS {request["name"]} / step {model_step}'
                )
                raw.sim.start_visualization()
            records = []
            for ordinal, seed in enumerate(request["seeds"]):
                reset_options = {"save_data": False}
                if "initial_positions" in request:
                    reset_options["cube_xy"] = [
                        request["initial_positions"][ordinal][:2]
                    ]
                obs, _ = env.reset(seed=seed, options=reset_options)
                if kind == "qualify":
                    result = execute_demo_episode(env)
                    success = bool(result.completed and result.all_success)
                    info = raw.get_info()
                    length = result.length
                else:
                    hidden = None
                    initial_delta = None
                    trace = []
                    writer = None
                    if video and ordinal < 3 and request.get("record", False):
                        import imageio.v2 as imageio

                        path = Path(output) / f'{request["name"]}_seed_{seed}.mp4'
                        writer = imageio.get_writer(
                            path, fps=30, codec="libx264", macro_block_size=None
                        )
                    try:
                        for length in range(1, 601):
                            with torch.inference_mode():
                                action, hidden = model.action(
                                    policy_observations(obs, model.observation_mode),
                                    hidden,
                                )
                            if request.get("trace", False):
                                trace.append(
                                    {
                                        "observations": policy_observations(
                                            obs, model.observation_mode
                                        ).cpu(),
                                        "actions": action.cpu(),
                                        "qpos": obs["robot"]["qpos"].cpu(),
                                        "cube_pose": obs["cube_pose"].cpu(),
                                        "tcp_pose": obs["tcp_pose"].cpu(),
                                    }
                                )
                            if (
                                initial_delta is None
                                and model.arm_command_mode != "pose"
                            ):
                                initial_delta = float(
                                    (action[:, :7] - obs["robot"]["qpos"][:, :7])
                                    .abs()
                                    .max()
                                )
                            obs, _, terminated, truncated, info = env.step(action)
                            if initial_delta is None:
                                initial_delta = float(action[:, :6].abs().max())
                            if writer is not None:
                                frame = (
                                    obs["sensor"]["cam_high"]["color"][0, ..., :3]
                                    .cpu()
                                    .numpy()
                                )
                                writer.append_data(frame)
                            if bool(
                                torch.as_tensor(terminated).any()
                                or torch.as_tensor(truncated).any()
                            ):
                                break
                        success = bool(info["success"][0])
                        if trace:
                            torch.save(
                                {
                                    key: torch.cat([frame[key] for frame in trace])
                                    for key in trace[0]
                                },
                                Path(output)
                                / f'{request["name"]}_seed_{seed}_trace.pt',
                            )
                    finally:
                        if writer is not None:
                            writer.close()
                metrics = info["metrics"]
                records.append(
                    {
                        "seed": seed,
                        "success": success,
                        "lifted": bool(metrics["lifted"][0]),
                        "goal_error": float(metrics["goal_error"][0]),
                        "length": length,
                        **(
                            {"initial_arm_target_delta": initial_delta}
                            if kind != "qualify"
                            else {}
                        ),
                    }
                )
                print(
                    f'{request["name"]}: {ordinal + 1}/{len(request["seeds"])} seed={seed} success={success}',
                    flush=True,
                )
            report = {
                "name": request["name"],
                "checkpoint": request.get("checkpoint"),
                "policy_step": model_step,
                "records": records,
                "success_rate": sum(r["success"] for r in records) / len(records),
                "lift_rate": sum(r["lifted"] for r in records) / len(records),
                "goal_error": sum(r["goal_error"] for r in records) / len(records),
                "episode_length": sum(r["length"] for r in records) / len(records),
            }
            if kind != "qualify":
                report["initial_arm_target_delta"] = sum(
                    r["initial_arm_target_delta"] for r in records
                ) / len(records)
            (Path(output) / f'{request["name"]}.json').write_text(
                json.dumps(report, indent=2)
            )
            results.put(report)
    except BaseException:
        results.put({"error": traceback.format_exc()})
    finally:
        if env is not None:
            env.unwrapped.close(exit_process=False)
            from embodichain.lab.sim import SimulationManager

            SimulationManager.flush_cleanup_queue()


def _receive(
    process: mp.Process, results: Any, timeout: float = 1800
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = results.get(timeout=0.2)
        except queue.Empty:
            if not process.is_alive():
                raise RuntimeError(f"Evaluation worker exited with {process.exitcode}")
            continue
        if "error" in result:
            raise RuntimeError(result["error"])
        return result
    raise TimeoutError("Evaluation worker timed out")


def _sample(
    engine: OnlineDataEngine, batch: int, length: int, *, with_history: bool = False
) -> tuple[Any, float]:
    started = time.monotonic()
    while True:
        try:
            sample = (
                engine.sample_batch_with_history(batch, length, start_probability=0.25)
                if with_history
                else engine.sample_batch(batch, length)
            )
            return sample, time.monotonic() - started
        except RuntimeError as error:
            if (
                engine.state is not OnlineDataEngineState.READY
                or "No unlocked valid chunk" not in str(error)
            ):
                raise
            if time.monotonic() - started >= 30:
                raise TimeoutError("No eligible ODS window for 30 seconds") from error
            time.sleep(0.02)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument("--train-device", default="cuda:0")
    p.add_argument("--output", type=Path, default=Path("/tmp/ods-pick-place-showcase"))
    p.add_argument(
        "--mode",
        choices=(
            "online",
            "frozen",
            "offline-32",
            "offline-full",
            "offline-matched",
            "serial",
        ),
        default="online",
    )
    p.add_argument(
        "--policy",
        choices=("chunk-pose", "mlp-pose", "mlp-joint", "gru-pose", "gru"),
        default="chunk-pose",
    )
    p.add_argument("--pose-ik", choices=("solver", "dls"), default="dls")
    p.add_argument("--prediction-horizon", type=int, default=64)
    p.add_argument("--action-horizon", type=int, default=16)
    p.add_argument("--observation-noise", type=float, default=0.01)
    p.add_argument(
        "--stage",
        choices=("showcase", "overfit-one", "overfit-four"),
        default="showcase",
    )
    p.add_argument(
        "--offline-data",
        type=Path,
        help="Numeric .pt corpus or a showcase-exported LeRobot dataset directory",
    )
    p.add_argument("--export-data", type=Path)
    p.add_argument("--record-batches", action="store_true")
    p.add_argument("--batch-manifest", type=Path)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--qualification-scenes", type=int, default=40)
    p.add_argument(
        "--qualification-report",
        type=Path,
        help="Reuse a matching 40-scene qualification with source/config checks",
    )
    p.add_argument("--dev-scenes", type=int, default=20)
    p.add_argument("--final-test-scenes", type=int, default=100)
    p.add_argument("--wall-time-budget", type=float, default=None)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument(
        "--arm-command-mode",
        choices=("absolute", "residual"),
        default="absolute",
        help="Predict absolute arm targets or a bounded +/-0.25 rad lead from measured qpos.",
    )
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--updates-per-round", type=int, default=5000)
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--producer-seed", type=int, default=None)
    p.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Optional independent CPU sampling RNG; omitted preserves legacy global stream",
    )
    p.add_argument("--producer-envs", type=int, default=None)
    p.add_argument("--dev-seed-base", "--dev-seed-start", type=int, default=20000)
    p.add_argument("--test-seed-base", type=int, default=None)
    p.add_argument(
        "--usage-schedule", choices=("fixed", "anchor_budget"), default="fixed"
    )
    p.add_argument("--anchor-reuse", type=float, default=32.0)
    p.add_argument("--refill-prior-seconds", type=float, default=30.0)
    p.add_argument("--max-generated-episodes", type=int, default=None)
    p.add_argument("--usage-statistics", action="store_true")
    p.add_argument(
        "--auto-config-input",
        type=Path,
        help="Two measured throughputs, reuse profile and runtime/binding metadata JSON",
    )
    p.add_argument(
        "--defer-evaluation",
        action="store_true",
        help="Save checkpoints while training; evaluate only after producer shutdown",
    )

    p.add_argument("--resume", type=Path)
    p.add_argument("--viser-port", type=int, default=8080)
    p.add_argument("--tensorboard-port", type=int, default=6006)
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--qualification-only", action="store_true")
    args = p.parse_args()
    if args.auto_config_input is not None:
        from examples.data_pipeline._ods_pick_place_configuration import (
            apply_configuration,
        )

        try:
            apply_configuration(args)
        except (TypeError, ValueError, OSError) as error:
            p.error(str(error))
    if args.batch_size is None:
        args.batch_size = 64 if args.policy.startswith("gru") else 256
    if args.buffer_size is None:
        args.buffer_size = 32
    if args.producer_envs is None:
        args.producer_envs = 1
    if args.smoke:
        args.rounds, args.updates_per_round, args.buffer_size, args.batch_size = (
            2,
            5,
            2,
            4,
        )
        args.qualification_scenes, args.dev_scenes, args.final_test_scenes = 2, 2, 2
    if args.stage != "showcase":
        args.mode = "frozen"
        args.buffer_size = 1 if args.stage == "overfit-one" else 4
        args.rounds = 1
        args.final_test_scenes = 0
    if args.policy == "gru" and (
        args.stage != "showcase" or args.mode not in {"online", "frozen"}
    ):
        p.error(
            "Legacy GRU supports online/frozen showcase runs; use an MLP for new diagnostic/benchmark modes"
        )
    if (
        args.eval_every < 1
        or args.qualification_scenes < 1
        or args.dev_scenes < 1
        or args.final_test_scenes < 0
    ):
        p.error("Evaluation counts must be positive (final test may be disabled)")
    if args.wall_time_budget is not None and (
        not np.isfinite(args.wall_time_budget) or args.wall_time_budget <= 0
    ):
        p.error("Wall-time budget must be finite and positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("Learning rate must be finite and positive")
    if not 1 <= args.action_horizon <= args.prediction_horizon:
        p.error("Action horizon must be within the positive prediction horizon")
    if not np.isfinite(args.observation_noise) or args.observation_noise < 0:
        p.error("Observation noise must be finite and nonnegative")
    if min(args.rounds, args.updates_per_round, args.buffer_size, args.batch_size) < 1:
        p.error("Training counts must be positive")
    if args.producer_seed is None:
        args.producer_seed = args.seed
    if args.test_seed_base is None:
        args.test_seed_base = 90000 if args.smoke else 50000
    if args.producer_envs < 1 or args.buffer_size % args.producer_envs:
        p.error("Producer environments must divide the positive cache capacity")
    if args.max_generated_episodes is not None and (
        args.max_generated_episodes < args.buffer_size
        or args.max_generated_episodes % args.buffer_size
    ):
        p.error("Accepted episode budget must be a positive multiple of cache capacity")
    if not np.isfinite(args.anchor_reuse) or args.anchor_reuse <= 0:
        p.error("Anchor reuse must be finite and positive")
    if not np.isfinite(args.refill_prior_seconds) or args.refill_prior_seconds <= 0:
        p.error("Refill lead-time prior must be finite and positive")
    if args.usage_schedule == "anchor_budget":
        args.usage_statistics = True
        if args.mode != "online" or args.policy != "chunk-pose" or args.offline_data:
            p.error(
                "Anchor-budget scheduling currently requires live online chunk-pose"
            )
        if args.max_generated_episodes is None:
            p.error("Anchor-budget runs require --max-generated-episodes")
        if args.buffer_size < 2 * args.producer_envs:
            p.error("Anchor-budget cache must hold at least two producer batches")
        if args.export_data:
            p.error(
                "Anchor-budget archival export is not supported; use lifecycle/usage records"
            )
    if args.usage_statistics and args.policy not in {
        "chunk-pose",
        "mlp-pose",
        "mlp-joint",
    }:
        p.error("Usage accounting supports chunk or frame policies")
    if args.policy == "gru" and (
        args.producer_envs != 1
        or args.sampling_seed is not None
        or args.producer_seed != args.seed
        or args.dev_seed_base != 20000
        or args.test_seed_base != (90000 if args.smoke else 50000)
        or args.defer_evaluation
        or args.max_generated_episodes is not None
    ):
        p.error("New experiment controls require the nonlegacy training runner")
    if args.defer_evaluation and args.stage != "showcase":
        p.error("Deferred evaluation is only supported for the showcase stage")
    dev = set(range(args.dev_seed_base, args.dev_seed_base + args.dev_scenes))
    test = set(range(args.test_seed_base, args.test_seed_base + args.final_test_scenes))
    if dev & test:
        p.error("Development and final-test seeds must be disjoint")
    return args


def main() -> None:
    args = _parse_args()
    if args.policy != "gru":
        from examples.data_pipeline._ods_pick_place_training import run

        run(args)
        return
    # Let worker teardown drain resources and publish errors before process exit.
    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            "Use a fresh output directory; resume also writes to a new run"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = load_task_config(args.sim_device)
    config["seed"] = args.seed
    (args.output / "task_config.json").write_text(json.dumps(config, indent=2))
    from importlib.metadata import version

    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "dexsim": version("dexsim-engine"),
    }
    try:
        environment["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=TASK_CONFIG.parents[5], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        environment["commit"] = None
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2))
    (args.output / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )
    context = mp.get_context("spawn")
    requests, results = context.Queue(), context.Queue()
    evaluator = context.Process(
        target=_evaluate_worker,
        args=(
            config,
            requests,
            results,
            str(args.output),
            not args.no_viewer,
            args.viser_port,
            not args.no_video,
        ),
    )
    engine = None
    tensorboard = None
    writer = None
    try:
        evaluator.start()
        ready = _receive(evaluator, results)
        requests.put(
            {
                "kind": "qualify",
                "name": "expert_qualification",
                "seeds": list(range(10000, 10000 + (2 if args.smoke else 40))),
            }
        )
        qualification = _receive(evaluator, results)
        required = 1.0 if args.smoke else 0.95
        if qualification["success_rate"] < required:
            raise RuntimeError(
                f'Expert qualification failed: {qualification["success_rate"]:.1%}'
            )
        if args.qualification_only:
            return
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(args.output / "events"), flush_secs=5)
        if not args.no_viewer:
            tensorboard = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tensorboard.main",
                    "--logdir",
                    str(args.output / "events"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.tensorboard_port),
                ],
                stdout=(args.output / "tensorboard.log").open("w"),
                stderr=subprocess.STDOUT,
            )
            print(
                f"Viser: http://127.0.0.1:{args.viser_port}\nTensorBoard: http://127.0.0.1:{args.tensorboard_port}",
                flush=True,
            )
        engine = OnlineDataEngine(
            OnlineDataEngineCfg(
                gym_config=config,
                state_dim=9,
                action_dim=8,
                buffer_size=args.buffer_size,
                max_episode_steps=600,
                buffer_device="cpu",
                refill_policy="manual",
                include_provenance=True,
                initialization_timeout=1800,
            )
        )
        engine.start()
        initial = engine.shared_buffer.clone()
        values = policy_observations(initial["obs"])[initial["valid"]]
        mean, std = _observation_statistics(values)
        model = RecurrentBC(
            mean,
            std,
            torch.tensor(ready["low"]),
            torch.tensor(ready["high"]),
            arm_command_mode=args.arm_command_mode,
        ).to(args.train_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        step = 0
        if args.resume:
            model, state = load_policy(str(args.resume), args.train_device)
            if model.observation_mode != "physical":
                raise ValueError(
                    "Legacy checkpoints retain the old observation contract; "
                    "start a fresh run to train the corrected policy."
                )
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            optimizer.load_state_dict(state["optimizer"])
            step = state["step"]
            torch.set_rng_state(state["torch_rng"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            if state["cuda_rng"] is not None and args.train_device.startswith("cuda:"):
                if isinstance(state["cuda_rng"], dict):
                    torch.cuda.set_rng_state(
                        state["cuda_rng"]["train"], args.train_device
                    )
                else:
                    source_device = state.get("config", {}).get(
                        "train_device", args.train_device
                    )
                    torch.cuda.set_rng_state(
                        state["cuda_rng"][int(source_device.split(":")[-1])],
                        args.train_device,
                    )
            args.arm_command_mode = model.arm_command_mode
            (args.output / "config.json").write_text(
                json.dumps(vars(args), default=str, indent=2)
            )
        (args.output / "data_manifest.json").write_text(
            json.dumps(
                {
                    "run_id": engine.run_id,
                    "joints": ready["joints"],
                    "resume_data": "fresh ODS source; shared buffer is not restored",
                },
                indent=2,
            )
        )
        baseline = args.output / f"initial_step_{step}.pt"
        save_checkpoint(baseline, model, optimizer, step, args, engine.run_id)
        dev_seeds = list(range(20000, 20000 + (2 if args.smoke else 20)))
        requests.put(
            {
                "kind": "student",
                "name": "initial_dev",
                "checkpoint": str(baseline),
                "seeds": dev_seeds,
                "record": True,
            }
        )
        initial_report = _receive(evaluator, results)
        for name in (
            "success_rate",
            "lift_rate",
            "goal_error",
            "episode_length",
            "initial_arm_target_delta",
        ):
            writer.add_scalar(f"eval/{name}", initial_report[name], step)
        initial_parameters = {
            name: value.detach().cpu().clone()
            for name, value in model.named_parameters()
        }
        best_key, best_path = (
            (
                initial_report["success_rate"],
                initial_report["lift_rate"],
                -initial_report["goal_error"],
            ),
            baseline,
        )
        consumed: dict[int, int] = {}
        supervised: dict[int, int] = {}
        round_sources = []
        for round_index in range(args.rounds):
            generation = 1
            if round_index and args.mode == "online":
                generation = engine.request_refill()
            round_supervised: dict[int, int] = {}
            overlap_updates = (
                args.updates_per_round // 4
                if round_index and args.mode == "online"
                else 0
            )
            model.train()
            for update_index in range(args.updates_per_round):
                update_started = time.monotonic()
                generation_wait = 0.0
                if update_index == overlap_updates:
                    engine.wait_for_generation(generation, timeout=1800)
                    generation_wait = time.monotonic() - update_started
                batch, waited = _sample(engine, args.batch_size, 24, with_history=True)
                waited += generation_wait
                generations, counts = batch["ods_generation"][batch["valid"]].unique(
                    return_counts=True
                )
                for version, count in zip(generations.tolist(), counts.tolist()):
                    consumed[version] = consumed.get(version, 0) + count
                    writer.add_scalar(
                        f"data/consumed_generation_{version}", consumed[version], step
                    )
                versions, counts = batch["ods_generation"][batch["loss_mask"]].unique(
                    return_counts=True
                )
                for version, count in zip(versions.tolist(), counts.tolist()):
                    supervised[version] = supervised.get(version, 0) + count
                    round_supervised[version] = round_supervised.get(version, 0) + count
                    writer.add_scalar(
                        f"data/supervised_generation_{version}",
                        supervised[version],
                        step,
                    )
                if update_index >= overlap_updates and versions.tolist() != [
                    generation
                ]:
                    raise RuntimeError(
                        "A completed refill returned an unexpected generation"
                    )
                obs = policy_observations(batch["obs"]).to(args.train_device)
                actions = batch["actions"].to(args.train_device)
                arm, hand = model.loss(
                    obs, actions, loss_mask=batch["loss_mask"].to(args.train_device)
                )
                optimizer.zero_grad(set_to_none=True)
                (arm + 5 * hand).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
                writer.add_scalar("train/arm_loss", arm.item(), step)
                writer.add_scalar("train/gripper_loss", hand.item(), step)
                writer.add_scalar("data/wait_seconds", waited, step)
                writer.add_scalar(
                    "data/history_frames",
                    int((batch["valid"] & ~batch["loss_mask"]).sum()),
                    step,
                )
                writer.add_scalar(
                    "data/trained_frames_per_second",
                    args.batch_size * 24 / max(time.monotonic() - update_started, 1e-9),
                    step,
                )
            checkpoint = args.output / f"round_{round_index + 1}_step_{step}.pt"
            save_checkpoint(checkpoint, model, optimizer, step, args, engine.run_id)
            requests.put(
                {
                    "kind": "student",
                    "name": f"round_{round_index + 1}_dev",
                    "checkpoint": str(checkpoint),
                    "seeds": dev_seeds,
                }
            )
            report = _receive(evaluator, results)
            for name in (
                "success_rate",
                "lift_rate",
                "goal_error",
                "episode_length",
                "initial_arm_target_delta",
            ):
                writer.add_scalar(f"eval/{name}", report[name], step)
            stats = engine.get_stats()
            writer.add_scalar(
                "ods/generated_frames_per_second",
                stats["generated_frames"] / max(stats["production_seconds"], 1e-9),
                step,
            )
            for name, value in stats.items():
                if name != "run_id":
                    writer.add_scalar(f"ods/{name}", value, step)
            writer.flush()
            selection_key = (
                report["success_rate"],
                report["lift_rate"],
                -report["goal_error"],
            )
            if selection_key > best_key:
                best_key, best_path = selection_key, checkpoint
            round_sources.append(
                {
                    "round": round_index + 1,
                    "generation": generation,
                    "overlap_updates": overlap_updates,
                    "supervised_frames": round_supervised,
                }
            )
            print(
                f"round={round_index + 1} step={step} success={report['success_rate']:.1%}",
                flush=True,
            )
        test_seeds = list(range(30000, 30000 + (2 if args.smoke else 100)))
        initial_test_rate = 0.0
        for name, path in (("initial_test", baseline), ("final_test", best_path)):
            requests.put(
                {
                    "kind": "student",
                    "name": name,
                    "checkpoint": str(path),
                    "seeds": test_seeds,
                    "record": True,
                }
            )
            final = _receive(evaluator, results)
            if name == "initial_test":
                initial_test_rate = final["success_rate"]
        summary = {
            "smoke": args.smoke,
            "mode": args.mode,
            "policy_config": {
                "observation_mode": model.observation_mode,
                "arm_command_mode": model.arm_command_mode,
            },
            "consumed_generations": consumed,
            "supervised_generations": supervised,
            "round_sources": round_sources,
            "sequence_contract": "complete causal history; 24 supervised steps; 25% episode-start windows",
            "best_checkpoint": str(best_path),
            "final_success_rate": final["success_rate"],
            "initial_success_rate": initial_test_rate,
            "parameters_updated": any(
                not torch.equal(initial_parameters[name], value.detach().cpu())
                for name, value in model.named_parameters()
            ),
            "training_step": step,
            "learning_target_met": not args.smoke
            and final["success_rate"] >= 0.8
            and final["success_rate"] > initial_test_rate,
            "stats": engine.get_stats(),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(
            f'Final student success: {final["success_rate"]:.1%}; initial: {initial_test_rate:.1%}; learning target met: {summary["learning_target_met"]}',
            flush=True,
        )
    finally:
        cleanup_error = None
        if engine is not None:
            try:
                engine.stop()
            except BaseException as error:
                cleanup_error = error
        if writer is not None:
            writer.close()
        if evaluator.pid is not None:
            requests.put(None)
            evaluator.join(30)
            if evaluator.is_alive():
                evaluator.terminate()
                evaluator.join(10)
            if evaluator.is_alive():
                evaluator.kill()
                evaluator.join()
        if tensorboard is not None:
            tensorboard.terminate()
            try:
                tensorboard.wait(10)
            except subprocess.TimeoutExpired:
                tensorboard.kill()
                tensorboard.wait()
        requests.close()
        results.close()
        requests.join_thread()
        results.join_thread()
        if cleanup_error is not None and sys.exc_info()[0] is None:
            raise cleanup_error


if __name__ == "__main__":
    main()
