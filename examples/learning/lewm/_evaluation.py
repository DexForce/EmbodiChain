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

"""Held-out latent prediction and paired visual MPC evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embodichain_models.dynamics.lewm import (
    LeWMCEMCfg,
    LeWMCEMPlanner,
    LeWMSequenceSource,
    load_lewm,
    preprocess_rgb,
)

from ._sim import SPLIT_SEEDS, close_env, goal_image, make_env, reset


def _check_contract(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    for key in (
        "action_dim",
        "action_semantics",
        "step_dt",
        "action_scale",
        "action_clip",
        "camera",
        "environment",
        "physics",
    ):
        if expected.get(key) != actual.get(key):
            raise ValueError(f"model and evaluation {key} contracts differ")


@torch.no_grad()
def evaluate_offline(
    export: str, corpus: str, split: str, device: str, batch_size: int = 16
) -> dict[str, Any]:
    model, manifest = load_lewm(export, device)
    h = model.cfg.history_size
    source = LeWMSequenceSource(corpus, split, h, horizon=5)
    _check_contract(manifest["metadata"]["data_contract"], source.metadata)
    used = set(manifest["metadata"]["train_seeds"])
    if split == "test":
        used.update(manifest["metadata"]["validation_seeds"])
    if source.seeds & used:
        raise ValueError("evaluation episode seeds overlap model fitting or selection")
    error, baseline = torch.zeros(5, device=device), torch.zeros(5, device=device)
    sums, squares = torch.zeros(model.cfg.embed_dim, device=device), torch.zeros(
        model.cfg.embed_dim, device=device
    )
    count = 0
    for batch in source.iter_batches(batch_size):
        inputs = batch.to(device).inputs
        pixels, actions = inputs["pixels"], inputs["actions"]
        embeddings = model.encode(preprocess_rgb(pixels, model.cfg.image_size))
        history, targets = embeddings[:, :h], embeddings[:, h:]
        predicted = model.rollout(
            history, actions[:, : h - 1], actions[:, None, h - 1 :]
        )[:, 0]
        if not torch.isfinite(predicted).all() or not torch.isfinite(targets).all():
            raise FloatingPointError("nonfinite held-out embeddings or predictions")
        error += (predicted - targets).square().mean(-1).sum(0)
        baseline += (history[:, -1:, :] - targets).square().mean(-1).sum(0)
        sums += targets.sum((0, 1))
        squares += targets.square().sum((0, 1))
        count += len(pixels)
    mean = sums / (count * 5)
    std = (squares / (count * 5) - mean.square()).clamp_min(0).sqrt()
    return {
        "split": split,
        "source_id": source.source_id,
        "windows": count,
        "prediction_mse": {str(t): float(error[t - 1] / count) for t in (1, 3, 5)},
        "persistence_mse": {str(t): float(baseline[t - 1] / count) for t in (1, 3, 5)},
        "beats_persistence": {
            str(t): bool(error[t - 1] < baseline[t - 1]) for t in (1, 3, 5)
        },
        "embedding_std_mean": float(std.mean()),
        "embedding_std_min": float(std.min()),
        "weights_sha256": manifest["weights_sha256"],
        "nonfinite": 0,
    }


@torch.no_grad()
def evaluate_sim(
    export: str,
    device: str,
    episodes: int = 50,
    cfg: LeWMCEMCfg | None = None,
    split: str = "test",
    start_index: int = 0,
) -> dict[str, Any]:
    if episodes < 1 or start_index < 0 or split not in {"validation", "test"}:
        raise ValueError("invalid episode count or evaluation split")
    model, manifest = load_lewm(export, device)
    env, config = make_env(device)
    raw = env.unwrapped
    actual = {
        "action_dim": raw.single_action_space.shape[0],
        "action_semantics": "joint_position.relative@1",
        "step_dt": raw.step_dt,
        "action_scale": 0.1,
        "action_clip": 1.0,
        "camera": config["sensor"][0],
        "environment": "PushCubeRL",
        "physics": config["physics"],
    }
    try:
        _check_contract(manifest["metadata"]["data_contract"], actual)
        used = set(manifest["metadata"]["train_seeds"])
        if split == "test":
            used.update(manifest["metadata"]["validation_seeds"])
        if used.intersection(
            range(
                SPLIT_SEEDS[split] + start_index,
                SPLIT_SEEDS[split] + start_index + episodes,
            )
        ):
            raise ValueError(
                "simulation evaluation seeds overlap model fitting or selection"
            )
    except Exception:
        close_env(env)
        raise
    records: dict[str, list[dict[str, Any]]] = {"lewm": [], "random": []}
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for kind in records:
            for index in range(episodes):
                seed = SPLIT_SEEDS[split] + start_index + index
                obs, rng = reset(env, seed)
                initial_cube = (
                    raw.sim.get_rigid_object("cube").get_local_pose()[0].cpu().tolist()
                )
                initial_qpos = raw.robot.get_qpos()[0].cpu().tolist()
                initial_goal = raw.goal_pose[0, :3, 3].cpu().tolist()
                target = goal_image(env)
                with raw.sim.render_frame():
                    obs = raw.get_obs()
                goal = model.encode(
                    preprocess_rgb(
                        torch.from_numpy(target)[None, None].to(device),
                        model.cfg.image_size,
                    )
                )[:, 0]
                images = [obs["sensor"]["main_cam"]["color"][..., :3].clone()]
                past: list[torch.Tensor] = []
                info = raw.get_info()
                length, latency, cycle_latency = 0, [], []
                terminated = truncated = False
                # Collect real stationary history in both arms of the comparison.
                for _ in range(model.cfg.history_size - 1):
                    action = torch.zeros((1, model.cfg.action_dim), device=device)
                    obs, _, terminated, truncated, info = env.step(action)
                    images.append(obs["sensor"]["main_cam"]["color"][..., :3].clone())
                    past.append(action)
                    length += 1
                    if bool(
                        torch.as_tensor(terminated).any()
                        or torch.as_tensor(truncated).any()
                    ):
                        break
                history = model.encode(
                    preprocess_rgb(
                        torch.stack(images, dim=1),
                        model.cfg.image_size,
                    )
                )
                planner = LeWMCEMPlanner(
                    model,
                    torch.as_tensor(raw.single_action_space.low, device=device),
                    torch.as_tensor(raw.single_action_space.high, device=device),
                    cfg,
                    seed=seed,
                )
                while length < raw.max_episode_steps and not bool(
                    torch.as_tensor(terminated).any()
                    or torch.as_tensor(truncated).any()
                ):
                    if kind == "lewm":
                        if torch.device(device).type == "cuda":
                            torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        cycle_started = started
                        past_tensor = (
                            torch.stack(past, dim=1)
                            if past
                            else history.new_zeros((1, 0, model.cfg.action_dim))
                        )
                        action, _ = planner.plan(history, past_tensor, goal)
                        if torch.device(device).type == "cuda":
                            torch.cuda.synchronize(device)
                        latency.append(time.perf_counter() - started)
                    else:
                        action = torch.as_tensor(
                            rng.uniform(-1, 1, size=(1, model.cfg.action_dim)),
                            device=device,
                            dtype=torch.float32,
                        )
                    obs, _, terminated, truncated, info = env.step(action)
                    next_frame = (
                        obs["sensor"]["main_cam"]["color"][..., :3].clone().unsqueeze(1)
                    )
                    next_latent = model.encode(
                        preprocess_rgb(next_frame, model.cfg.image_size)
                    )
                    history = torch.cat((history[:, 1:], next_latent), dim=1)
                    past = (
                        (past + [action])[-(model.cfg.history_size - 1) :]
                        if model.cfg.history_size > 1
                        else []
                    )
                    length += 1
                    if kind == "lewm":
                        if torch.device(device).type == "cuda":
                            torch.cuda.synchronize(device)
                        cycle_latency.append(time.perf_counter() - cycle_started)
                records[kind].append(
                    {
                        "seed": seed,
                        "success": bool(info["success"][0]),
                        "distance": float(info["metrics"]["distance_to_goal"][0]),
                        "steps": length,
                        "planning_seconds": latency,
                        "cycle_seconds": cycle_latency,
                        "initial_cube": initial_cube,
                        "initial_qpos": initial_qpos,
                        "initial_goal": initial_goal,
                    }
                )
                print(
                    f"{kind} {index+1}/{episodes}: success={records[kind][-1]['success']} distance={records[kind][-1]['distance']:.4f}",
                    flush=True,
                )
        for planned, random in zip(records["lewm"], records["random"]):
            if planned["seed"] != random["seed"] or any(
                not np.allclose(planned[key], random[key], atol=1e-5, rtol=0)
                for key in ("initial_cube", "initial_qpos", "initial_goal")
            ):
                raise RuntimeError(
                    "paired evaluations did not start from the same scene"
                )
        summary = {}
        for kind, rows in records.items():
            times = [value for row in rows for value in row["planning_seconds"]]
            cycles = [value for row in rows for value in row["cycle_seconds"]]
            summary[kind] = {
                "successes": sum(row["success"] for row in rows),
                "success_rate": sum(row["success"] for row in rows) / len(rows),
                "mean_final_distance": sum(row["distance"] for row in rows) / len(rows),
                "planning_median_seconds": float(np.median(times)) if times else None,
                "planning_p95_seconds": (
                    float(np.percentile(times, 95)) if times else None
                ),
                "cycle_median_seconds": float(np.median(cycles)) if cycles else None,
                "cycle_p95_seconds": (
                    float(np.percentile(cycles, 95)) if cycles else None
                ),
            }
        return {
            "passed": summary["lewm"]["success_rate"]
            > summary["random"]["success_rate"],
            "split": split,
            "episodes": episodes,
            "start_index": start_index,
            "summary": summary,
            "records": records,
            "weights_sha256": manifest["weights_sha256"],
            "device": device,
            "gpu": (
                torch.cuda.get_device_name(device)
                if torch.device(device).type == "cuda"
                else None
            ),
            "torch": torch.__version__,
            "step_dt": raw.step_dt,
            "success_threshold": raw.success_threshold,
            "gpu_peak_bytes": (
                torch.cuda.max_memory_allocated(device)
                if torch.device(device).type == "cuda"
                else None
            ),
            "cem": (cfg or LeWMCEMCfg()).to_dict(),
        }
    finally:
        close_env(env)


def merge_sim_reports(paths: list[str], episodes: int = 50) -> dict[str, Any]:
    """Merge disjoint paired test shards while preserving the full gate."""
    if not paths or episodes < 1:
        raise ValueError("provide reports and a positive expected episode count")
    shards = [json.loads(Path(path).read_text()) for path in paths]
    first = shards[0]
    for shard in shards:
        if any(
            shard.get(key) != first.get(key)
            for key in (
                "split",
                "weights_sha256",
                "cem",
                "step_dt",
                "success_threshold",
            )
        ):
            raise ValueError(
                "simulation shards have different model or evaluation contracts"
            )
    records = {
        kind: sorted(
            (row for shard in shards for row in shard["records"][kind]),
            key=lambda row: row["seed"],
        )
        for kind in ("lewm", "random")
    }
    expected = list(
        range(SPLIT_SEEDS[first["split"]], SPLIT_SEEDS[first["split"]] + episodes)
    )
    for kind in records:
        if [row["seed"] for row in records[kind]] != expected:
            raise ValueError(
                "simulation shards must cover each expected seed exactly once"
            )
    for planned, random in zip(records["lewm"], records["random"]):
        if any(
            not np.allclose(planned[key], random[key], atol=1e-5, rtol=0)
            for key in ("initial_cube", "initial_qpos", "initial_goal")
        ):
            raise ValueError("simulation shard pairs have different initial states")
    summary = {}
    for kind, rows in records.items():
        plans = [value for row in rows for value in row["planning_seconds"]]
        cycles = [value for row in rows for value in row["cycle_seconds"]]
        summary[kind] = {
            "successes": sum(row["success"] for row in rows),
            "success_rate": sum(row["success"] for row in rows) / episodes,
            "mean_final_distance": sum(row["distance"] for row in rows) / episodes,
            "planning_median_seconds": float(np.median(plans)) if plans else None,
            "planning_p95_seconds": float(np.percentile(plans, 95)) if plans else None,
            "cycle_median_seconds": float(np.median(cycles)) if cycles else None,
            "cycle_p95_seconds": float(np.percentile(cycles, 95)) if cycles else None,
        }
    return {
        "passed": summary["lewm"]["success_rate"] > summary["random"]["success_rate"],
        "episodes": episodes,
        "split": first["split"],
        "weights_sha256": first["weights_sha256"],
        "cem": first["cem"],
        "step_dt": first["step_dt"],
        "success_threshold": first["success_threshold"],
        "summary": summary,
        "records": records,
        "shards": [
            {
                key: shard[key]
                for key in (
                    "device",
                    "gpu",
                    "torch",
                    "gpu_peak_bytes",
                    "start_index",
                    "episodes",
                )
            }
            for shard in shards
        ],
    }
