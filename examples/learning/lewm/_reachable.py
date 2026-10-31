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

"""Measured future-image goals and paired raw/blocked MPC execution."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from embodichain_models.dynamics.lewm import (
    LeWMCEMCfg,
    LeWMCEMPlanner,
    load_lewm,
    preprocess_rgb,
)
from ._sim import (
    SPLIT_SEEDS,
    close_env,
    environment_contract,
    frame,
    guided_action,
    make_env,
    reset,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_pair(
    positions: np.ndarray,
    admissible: np.ndarray,
    offset: int,
    history: int,
    threshold: float,
    margin: float,
) -> int | None:
    """Select the first admissible, nontrivial measured future displacement."""
    if offset < 1 or history < 0 or margin <= 0 or len(positions) != len(admissible):
        raise ValueError("invalid reachable-pair selection contract")
    for start in range(history, len(positions) - offset):
        distance = float(
            np.linalg.norm(positions[start + offset, :2] - positions[start, :2])
        )
        if admissible[start] and distance > threshold + margin:
            return start
    return None


def prepare_goals(
    output: str,
    device: str,
    *,
    episodes: int = 20,
    start_index: int = 100,
    max_proposals: int = 200,
    goal_offset: int = 25,
    history_steps: int = 10,
    margin: float = 0.005,
    split: str = "validation",
) -> dict[str, Any]:
    """Collect measured prefixes and future goals; geometry only selects queries."""
    from embodichain.lab.sim.scene import SceneView
    from ._trace import _snapshot

    if (
        split not in {"validation", "test"}
        or min(episodes, max_proposals, goal_offset) < 1
        or history_steps < 0
        or start_index < 0
        or max_proposals < episodes
        or start_index + max_proposals >= 100_000
    ):
        raise ValueError("invalid reachable-goal collection budget")
    path = Path(output)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env, config = make_env(device)
    raw = env.unwrapped
    view = SceneView(raw.sim)
    accepted, audit = [], []
    try:
        with h5py.File(temporary, "x") as evidence:
            if goal_offset + history_steps > raw.max_episode_steps:
                raise ValueError("future goal and history exceed the rollout budget")
            metadata = {
                "environment_contract": environment_contract(env, config),
                "split": split,
                "goal_offset": goal_offset,
                "history_steps": history_steps,
                "minimum_initial_distance": raw.success_threshold + margin,
                "success_threshold": raw.success_threshold,
                "purpose": "measured reachable image goals; geometry is not a policy input",
            }
            evidence.attrs["format"] = "embodichain.lewm.reachable_pairs.v1"
            evidence.attrs["metadata"] = json.dumps(metadata)
            for proposal in range(max_proposals):
                seed = SPLIT_SEEDS[split] + start_index + proposal
                obs, rng = reset(env, seed)
                images, actions, states, positions, valid = [frame(obs)], [], [], [], []
                phase = 0

                def capture() -> None:
                    state = _snapshot(view)
                    states.append(json.dumps(state, allow_nan=False))
                    positions.append(
                        raw.sim.get_rigid_object("cube")
                        .get_local_pose()[0, :3]
                        .cpu()
                        .numpy()
                        .copy()
                    )
                    qpos = raw.robot.get_qpos()
                    limits = raw.robot.get_qpos_limits()
                    valid.append(
                        bool(
                            (
                                (qpos >= limits[..., 0] - 1e-4)
                                & (qpos <= limits[..., 1] + 1e-4)
                            ).all()
                        )
                    )

                capture()
                for _ in range(raw.max_episode_steps):
                    action, phase, _ = guided_action(env, phase, rng)
                    obs, *_ = raw.step(action)
                    actions.append(action[0].cpu().numpy().copy())
                    images.append(frame(obs))
                    capture()
                start = select_pair(
                    np.asarray(positions),
                    np.asarray(valid),
                    goal_offset,
                    history_steps,
                    raw.success_threshold,
                    margin,
                )
                row = {
                    "seed": seed,
                    "accepted": start is not None,
                    "max_cube_displacement_m": float(
                        np.linalg.norm(
                            np.asarray(positions)[:, :2] - positions[0][:2], axis=-1
                        ).max()
                    ),
                }
                if start is not None:
                    goal_step = start + goal_offset
                    group = evidence.create_group(str(seed))
                    group.attrs.update(start_step=start, goal_step=goal_step)
                    group.create_dataset(
                        "pixels",
                        data=np.stack(images[: goal_step + 1]),
                        compression="lzf",
                        chunks=(1, *images[0].shape),
                    )
                    group.create_dataset("actions", data=np.stack(actions[:goal_step]))
                    group.create_dataset(
                        "states",
                        data=states[: goal_step + 1],
                        dtype=h5py.string_dtype("utf-8"),
                    )
                    group.create_dataset(
                        "cube_position", data=np.asarray(positions[: goal_step + 1])
                    )
                    group.create_dataset("goal_position", data=positions[goal_step])
                    row.update(
                        start_step=start,
                        initial_distance_m=float(
                            np.linalg.norm(
                                positions[goal_step][:2] - positions[start][:2]
                            )
                        ),
                    )
                    accepted.append(seed)
                audit.append(row)
                evidence.attrs["audit"] = json.dumps(audit)
                evidence.flush()
                print(
                    json.dumps(
                        {
                            "proposal": proposal + 1,
                            "accepted_count": len(accepted),
                            **row,
                        }
                    ),
                    flush=True,
                )
                if len(accepted) == episodes:
                    break
            if len(accepted) != episodes:
                raise RuntimeError(
                    f"only {len(accepted)}/{episodes} nontrivial reachable pairs within the proposal budget"
                )
            evidence.attrs["complete"] = True
        temporary.replace(path)
        return {
            "path": str(path.resolve()),
            "sha256": _sha(path),
            "metadata": metadata,
            "seeds": accepted,
            "audit": audit,
        }
    finally:
        close_env(env)


def _summarize(records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate physical outcomes and completed planning-call latency."""
    result = {}
    for kind, rows in records.items():
        latency = [value for row in rows for value in row["planning_seconds"]]
        result[kind] = {
            "successes": sum(row["success"] for row in rows),
            "success_rate": sum(row["success"] for row in rows) / len(rows),
            "mean_final_distance": float(np.mean([row["distance"] for row in rows])),
            "planning_median_seconds": float(np.median(latency)) if latency else None,
        }
    return result


@torch.no_grad()
def evaluate_reachable(
    exports: dict[str, str],
    pairs_path: str,
    output: str,
    device: str,
    *,
    start_index: int = 0,
    episodes: int = 5,
    horizon: int = 5,
    receding_horizon: int = 1,
    max_steps: int = 50,
    sampling_distribution: str = "bounds",
    action_selection: str = "best",
) -> dict[str, Any]:
    """Compare models from one owned scene snapshot and a measured future goal.

    Frozen queries supply raw reference actions and future RGB observations.
    Reference actions are replayed and reachability is checked in the new world.
    Controller starts restore an owned snapshot captured in that same world;
    serialized SceneState owner/binding tokens are never transplanted.
    """
    import time
    from embodichain.lab.sim.scene import SceneView, capture_state, restore_state
    from ._diagnostics import _physical_error
    from ._evaluation import _check_contract
    from ._trace import TraceWriter

    if (
        not exports
        or min(episodes, horizon, receding_horizon, max_steps) < 1
        or start_index < 0
        or receding_horizon > horizon
    ):
        raise ValueError("invalid reachable evaluation budget")
    if any(
        not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name)
        for name in exports
    ):
        raise ValueError("model labels require lowercase ASCII names")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    models = {name: load_lewm(path, device) for name, path in exports.items()}
    cfg = LeWMCEMCfg(
        horizon=horizon,
        sampling_distribution=sampling_distribution,
        action_selection=action_selection,
    )
    path = Path(pairs_path)
    pair_hash = _sha(path)
    writers, results, checks = {}, {}, []
    env, config = make_env(device)
    raw = env.unwrapped
    view = SceneView(raw.sim)
    actual_contract = environment_contract(env, config)
    try:
        if max_steps > raw.max_episode_steps:
            raise ValueError(
                "evaluation budget exceeds the configured environment limit"
            )
        with h5py.File(path, "r") as evidence:
            if evidence.attrs.get(
                "format"
            ) != "embodichain.lewm.reachable_pairs.v1" or not evidence.attrs.get(
                "complete", False
            ):
                raise ValueError("reachable queries are incomplete or unsupported")
            metadata = json.loads(evidence.attrs["metadata"])
            _check_contract(metadata["environment_contract"], actual_contract)
            keys = sorted(evidence, key=int)[start_index : start_index + episodes]
            if len(keys) != episodes:
                raise ValueError("reachable query slice is incomplete")
            raw_dim = raw.single_action_space.shape[0]
            for name, (model, manifest) in models.items():
                block = model.cfg.action_block
                if model.cfg.action_dim != block * raw_dim:
                    raise ValueError(
                        "model action width differs from raw controller blocks"
                    )
                if (model.cfg.history_size - 1) * block > metadata["history_steps"]:
                    raise ValueError("query history is shorter than the model contract")
                used = set(manifest["metadata"]["train_seeds"])
                if metadata["split"] == "test":
                    used.update(manifest["metadata"]["validation_seeds"])
                if used.intersection(map(int, keys)):
                    raise ValueError("reachable query seeds overlap model fitting")
                _check_contract(
                    manifest["metadata"]["data_contract"],
                    actual_contract | {"action_block": block},
                )
                low = np.tile(raw.single_action_space.low, block)
                high = np.tile(raw.single_action_space.high, block)
                writers[name] = TraceWriter(
                    root / f"{name}.h5",
                    {
                        "environment_contract": actual_contract
                        | {"action_block": block},
                        "render_contract_verified": True,
                        "weights_sha256": manifest["weights_sha256"],
                        "cem": cfg.to_dict(),
                        "action_low": low.tolist(),
                        "action_high": high.tolist(),
                        "raw_action_dim": raw_dim,
                        "success_threshold": raw.success_threshold,
                        "split": metadata["split"],
                        "device": device,
                        "torch": torch.__version__,
                        "pairs_sha256": pair_hash,
                        "goal_source": "measured_future",
                        "receding_horizon": receding_horizon,
                        "max_steps": max_steps,
                    },
                )
                results[name] = {
                    "weights_sha256": manifest["weights_sha256"],
                    "action_block": block,
                    "model_step_dt": raw.step_dt * block,
                    "planning_span_seconds": raw.step_dt * block * horizon,
                    "cem": cfg.to_dict(),
                    "records": {"lewm": [], "random": [], "zero": []},
                }
            for key in keys:
                seed = int(key)
                group = evidence[key]
                start = int(group.attrs["start_step"])
                goal_step = int(group.attrs["goal_step"])
                reference = group["actions"][:]
                target = group["pixels"][goal_step]
                goal_position = group["goal_position"][:]
                obs, _ = reset(env, seed)
                history_pixels = [frame(obs)]
                for control in reference[:start]:
                    obs, *_ = raw.step(torch.as_tensor(control, device=device)[None])
                    history_pixels.append(frame(obs))
                initial = capture_state(view)
                initial_cube = (
                    raw.sim.get_rigid_object("cube").get_local_pose()[0].cpu().tolist()
                )
                initial_distance = float(
                    np.linalg.norm(np.asarray(initial_cube[:2]) - goal_position[:2])
                )
                if initial_distance <= metadata["minimum_initial_distance"]:
                    raise RuntimeError("query became trivial during reference replay")
                for control in reference[start:goal_step]:
                    raw.step(torch.as_tensor(control, device=device)[None])
                replay_cube = (
                    raw.sim.get_rigid_object("cube")
                    .get_local_pose()[0, :2]
                    .cpu()
                    .numpy()
                )
                replay_distance = float(np.linalg.norm(replay_cube - goal_position[:2]))
                if replay_distance >= raw.success_threshold:
                    raise RuntimeError(
                        f"query {seed} reference replay does not reach the saved goal: {replay_distance}"
                    )
                checks.append(
                    {
                        "seed": seed,
                        "reference_replay_distance_m": replay_distance,
                        "reference_reaches_goal": True,
                        "initial_distance_m": initial_distance,
                    }
                )
                prefix_pixels = np.stack(
                    history_pixels[-metadata["history_steps"] - 1 :]
                )
                prefix_actions = reference[start - metadata["history_steps"] : start]
                for name, (model, manifest) in models.items():
                    block, h = model.cfg.action_block, model.cfg.history_size
                    writer = writers[name]
                    for kind in ("lewm", "random", "zero"):
                        _, rng = reset(env, seed)
                        for control in reference[:start]:
                            raw.step(torch.as_tensor(control, device=device)[None])
                        restore_state(view, initial, qpos_tolerance=1e-4)
                        error = _physical_error(initial, capture_state(view))
                        if error > 1e-4:
                            raise RuntimeError(
                                f"query start round-trip failed: {error}"
                            )
                        restored_cube = (
                            raw.sim.get_rigid_object("cube")
                            .get_local_pose()[0]
                            .cpu()
                            .tolist()
                        )
                        restored_qpos = raw.robot.get_qpos()[0].cpu().tolist()
                        raw.elapsed_steps.zero_()
                        raw.goal_pose[0, :3, 3] = torch.as_tensor(
                            goal_position, device=device
                        )
                        for _ in range(32):
                            with raw.sim.render_frame():
                                obs = raw.get_obs()
                        pixels = [image.copy() for image in prefix_pixels]
                        pixels[-1] = frame(obs)
                        actions = [control.copy() for control in prefix_actions]
                        goal = model.encode(
                            preprocess_rgb(
                                torch.from_numpy(target).to(device)[None, None],
                                model.cfg.image_size,
                            )
                        )[:, 0]
                        writer.begin(
                            kind,
                            seed,
                            target,
                            goal,
                            raw.goal_pose[0, :3, 3],
                            obs,
                            view,
                            history_pixels=np.stack(pixels),
                            history_actions=prefix_actions,
                        )
                        planner = LeWMCEMPlanner(
                            model,
                            np.tile(raw.single_action_space.low, block),
                            np.tile(raw.single_action_space.high, block),
                            cfg,
                            seed=seed,
                        )
                        length, latency, step = 0, [], raw.get_info()
                        while length < max_steps:
                            saved_rng = encoded = cost = None
                            if kind == "lewm":
                                context = np.stack(
                                    pixels[-(h - 1) * block - 1 :: block]
                                )
                                past = (
                                    np.asarray(actions[-(h - 1) * block :])
                                    if h > 1
                                    else np.empty((0, raw_dim), np.float32)
                                )
                                encoded = model.encode(
                                    preprocess_rgb(
                                        torch.from_numpy(context).to(device)[None],
                                        model.cfg.image_size,
                                    )
                                )
                                past_tensor = torch.from_numpy(
                                    past.reshape(h - 1, model.cfg.action_dim)
                                ).to(device)[None]
                                saved_rng = {
                                    "torch_generator_state_b64": base64.b64encode(
                                        planner.generator.get_state()
                                        .cpu()
                                        .numpy()
                                        .tobytes()
                                    ).decode("ascii")
                                }
                                torch.cuda.synchronize(device)
                                started = time.perf_counter()
                                sequence, cost = planner.plan_sequence(
                                    encoded, past_tensor, goal
                                )
                                torch.cuda.synchronize(device)
                                latency.append(time.perf_counter() - started)
                                queue = sequence[0, :receding_horizon].reshape(
                                    -1, raw_dim
                                )
                            elif kind == "zero":
                                queue = torch.zeros((1, raw_dim), device=device)
                            else:
                                saved_rng = {
                                    "numpy_bit_generator": copy.deepcopy(
                                        rng.bit_generator.state
                                    )
                                }
                                queue = torch.as_tensor(
                                    rng.uniform(-1, 1, size=(1, raw_dim)),
                                    device=device,
                                    dtype=torch.float32,
                                )
                            stopped = False
                            for offset, action in enumerate(queue):
                                obs, _, terminated, truncated, step = raw.step(
                                    action[None]
                                )
                                pixels.append(frame(obs))
                                actions.append(action.cpu().numpy().copy())
                                writer.add(
                                    action[None],
                                    obs,
                                    step,
                                    terminated,
                                    truncated,
                                    view,
                                    rng=saved_rng if offset == 0 else None,
                                    history=encoded if offset == 0 else None,
                                    cost=cost if offset == 0 else None,
                                )
                                length += 1
                                stopped = bool(
                                    torch.as_tensor(terminated).any()
                                    or torch.as_tensor(truncated).any()
                                )
                                if stopped or length == max_steps:
                                    break
                            if stopped:
                                break
                        writer.commit()
                        row = {
                            "seed": seed,
                            "success": bool(step["success"][0]),
                            "distance": float(step["metrics"]["distance_to_goal"][0]),
                            "steps": length,
                            "planning_seconds": latency,
                            "initial_cube": restored_cube,
                            "initial_qpos": restored_qpos,
                            "initial_goal": goal_position.tolist(),
                            "start_max_abs_error": error,
                            "initial_distance": initial_distance,
                        }
                        results[name]["records"][kind].append(row)
                        print(
                            json.dumps(
                                {
                                    "model": name,
                                    "policy": kind,
                                    **{
                                        key: value
                                        for key, value in row.items()
                                        if key
                                        not in (
                                            "planning_seconds",
                                            "initial_qpos",
                                            "initial_cube",
                                            "initial_goal",
                                        )
                                    },
                                }
                            ),
                            flush=True,
                        )
            for name, result in results.items():
                result["summary"] = _summarize(result["records"])
                result["trace"] = writers[name].finish()
            writers.clear()
        return {
            "passed": False,
            "acceptance_eligible": False,
            "protocol": "measured_reachable_future",
            "pairs_path": str(path.resolve()),
            "pairs_sha256": pair_hash,
            "seeds": list(map(int, keys)),
            "split": metadata["split"],
            "receding_horizon": receding_horizon,
            "max_steps": max_steps,
            "goal_offset": metadata["goal_offset"],
            "render": actual_contract["render"],
            "reference_checks": checks,
            "models": results,
        }
    finally:
        for writer in writers.values():
            writer.abort()
        close_env(env)
