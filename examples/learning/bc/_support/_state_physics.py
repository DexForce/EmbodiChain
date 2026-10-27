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

"""Collect, replay and evaluate state BC on the current PickPlace task.

Each invocation owns its physical simulator. Qualification and replay save a
protocol before creating a simulator; evaluation uses an immutable checkpoint.
No expert is constructed during student execution.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from importlib import metadata, util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from embodichain_models.policy.bc import ChunkBCExecutor, ChunkBCModel, ChunkBCModelCfg

from ._pick_place import TARGET_SEMANTICS, decode_request
from ._state_data import (
    ACTION_CONTRACTS as _STATE_ACTION_CONTRACTS,
    ACTION_SEMANTICS,
    OBSERVATION_SCHEMA,
    STEP_DT,
    validate_environment,
)

ACTION_CONTRACTS = list(_STATE_ACTION_CONTRACTS)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Unfinished prior write at {temporary}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _current_config(device: str) -> dict[str, Any]:
    from examples.ods.simple_bc_demo.config import load_task_config

    config = load_task_config(device, pose_ik="dls")
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    validate_environment(config)


def _source_identity() -> dict[str, Any]:
    """Bind actual loaded code and configs for a clean checkout or installation."""
    module_names = {
        "examples/learning/bc/_support/_pick_place.py": "examples.learning.bc._support._pick_place",
        "examples/learning/bc/_support/_state_data.py": "examples.learning.bc._support._state_data",
        "examples/learning/bc/_support/_trace.py": "examples.learning.bc._support._trace",
        "examples/ods/simple_bc_demo/environment.py": "examples.ods.simple_bc_demo.environment",
        "examples/ods/simple_bc_demo/observations.py": "examples.ods.simple_bc_demo.observations",
        "examples/ods/simple_bc_demo/config.py": "examples.ods.simple_bc_demo.config",
        "extensions/embodichain_models/embodichain_models/policy/bc/model.py": "embodichain_models.policy.bc.model",
        "embodichain/learning/sl/learner.py": "embodichain.learning.sl.learner",
        "embodichain/learning/sl/trainer.py": "embodichain.learning.sl.trainer",
        "embodichain/lab/gym/envs/demo.py": "embodichain.lab.gym.envs.demo",
        "embodichain/lab/gym/utils/gym_utils.py": "embodichain.lab.gym.utils.gym_utils",
    }
    paths = {
        "examples/learning/bc/_support/_state_physics.py": Path(__file__).resolve()
    }
    for label, module in module_names.items():
        spec = util.find_spec(module)
        if spec is None or spec.origin is None:
            raise ValueError(f"Cannot locate actual runtime module {module}")
        paths[label] = Path(spec.origin).resolve()
    from examples.ods.simple_bc_demo.config import TASK_CONFIG

    for name in ("env.yaml", "task.yaml", "franka.yaml"):
        paths[f"examples/ods/simple_bc_demo/configs/{name}"] = TASK_CONFIG.parent / name
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("A bound runtime source or task configuration is missing")
    root = Path(__file__).resolve().parents[4]
    commit = None
    if (root / ".git").exists():
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip():
            raise ValueError(
                "Physical experiments require a clean frozen source checkout"
            )
        origin = "git-checkout"
    else:
        origin = "installed-distribution"
    try:
        version = metadata.version("embodichain")
    except metadata.PackageNotFoundError:
        if commit is None:
            raise ValueError(
                "Installed execution requires EmbodiChain distribution metadata"
            )
        version = None
    return {
        "source_origin": origin,
        "source_commit": commit,
        "distribution_version": version,
        "source_paths": {name: str(path) for name, path in paths.items()},
        "source_sha256": {name: _sha(path) for name, path in paths.items()},
    }


def _verify_sources(protocol: dict[str, Any]) -> None:
    current = _source_identity()
    for key in (
        "source_origin",
        "source_commit",
        "distribution_version",
        "source_paths",
        "source_sha256",
    ):
        if current[key] != protocol[key]:
            raise ValueError(
                f"Runtime source identity changed during {protocol['kind']}: {key}"
            )


def _scene_panel(seed_base: int, count: int) -> tuple[list[dict[str, Any]], str]:
    if (
        type(seed_base) is not int
        or seed_base < 0
        or type(count) is not int
        or count < 1
    ):
        raise ValueError("Scene seed and count must be nonnegative and positive")
    generator = torch.Generator(device="cpu").manual_seed(seed_base)
    positions = (
        torch.tensor([-0.42, -0.08])
        + (torch.rand(count, 2, generator=generator) - 0.5) * 0.08
    )
    scenes = [
        {"ordinal": i, "seed": seed_base + i, "cube_xy": positions[i].tolist()}
        for i in range(count)
    ]
    return scenes, hashlib.sha256(positions.numpy().tobytes()).hexdigest()


def _config_on_device(config: dict[str, Any], device: str) -> dict[str, Any]:
    value = deepcopy(config)
    value.update(
        device=device,
        gpu_id=int(device.split(":")[-1]) if device.startswith("cuda:") else 0,
        num_envs=1,
        headless=True,
        renderer="no-render",
        enable_sensor=False,
    )
    value["max_episode_steps"] = 600
    _validate_config(value)
    return value


def _open_env(config: dict[str, Any], *, recording: bool) -> Any:
    import gymnasium as gym
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import discover_task_packages

    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    discover_task_packages()
    cfg = config_to_cfg(config)
    cfg.init_rollout_buffer = recording
    cfg.filter_dataset_saving = True
    cfg.max_episode_steps = 600
    if recording:
        cfg.rollout_action_semantics = "policy_request"
    env = gym.make(config["id"], cfg=cfg)
    env.unwrapped._replay_no_auto_reset = True
    return env


def _close_env(env: Any | None) -> None:
    from embodichain.lab.sim import SimulationManager

    try:
        if env is not None:
            env.unwrapped.close(exit_process=False)
    finally:
        SimulationManager.flush_cleanup_queue()


def _reset(env: Any, scene: dict[str, Any]) -> Any:
    torch.manual_seed(scene["seed"])
    obs, _ = env.reset(
        seed=scene["seed"], options={"save_data": False, "cube_xy": [scene["cube_xy"]]}
    )
    return obs


def _freeze(
    output: Path, kind: str, config: dict[str, Any], **values: Any
) -> dict[str, Any]:
    identity = _source_identity()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu = None
    if visible and visible.split(",")[0]:
        try:
            gpu = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    visible.split(",")[0],
                    "--query-gpu=index,uuid,name,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            gpu = "unavailable"
    protocol = {
        "version": "state-bc-physics-v1",
        "kind": kind,
        **identity,
        "environment": config,
        "task_success": "existing PickPlace predicate: lift, goal, open, withdraw and 0.3s stable",
        "action_contracts": ACTION_CONTRACTS,
        "step_dt": STEP_DT,
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_timing": "pre_action",
        "observation_timing_version": "pickplace-preaction-policy-request@1",
        "termination_policy": "stop on first termination/truncation; no automatic reset",
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": visible,
            "physical_gpu": gpu,
            "logical_device": config["device"],
        },
        **values,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "environment.json", config)
    _write(output / "protocol.json", protocol)
    _write(output / "protocol-sha256.json", {"sha256": _sha(output / "protocol.json")})
    return protocol


def _result(raw: Any) -> tuple[bool, bool, dict[str, Any]]:
    success, failure, metrics = raw.compute_task_state()
    return (
        bool(success[0]),
        bool(failure[0]),
        {
            "lifted": bool(metrics["lifted"][0]),
            "goal_error": float(metrics["goal_error"][0]),
        },
    )


def _expert_episode(env: Any, ordinal: int) -> tuple[Any | None, str | None]:
    from embodichain.lab.gym.envs.demo import execute_demo_episode

    try:
        return execute_demo_episode(env, episode_index=ordinal), None
    except RuntimeError as error:
        if str(error) not in {
            "PickPlace grasp planning failed",
            "PickPlace placement planning failed",
        }:
            raise
        return None, str(error)


def qualify(
    *, output: Path, seed_base: int, scenes: int, device: str
) -> dict[str, Any]:
    """Execute independent expert scenes with the existing task predicate."""
    panel, panel_hash = _scene_panel(seed_base, scenes)
    config = _current_config(device)
    protocol = _freeze(
        output,
        "expert-qualification",
        config,
        seed_base=seed_base,
        scenes=panel,
        scene_digest=panel_hash,
    )
    env = None
    records = []
    start = time.monotonic()
    try:
        env = _open_env(config, recording=False)
        raw = env.unwrapped
        for scene in panel:
            _reset(env, scene)
            before_qpos = raw.robot.get_qpos()[0].detach().cpu().tolist()
            result, planning_error = _expert_episode(env, scene["ordinal"])
            success, failure, metrics = _result(raw)
            record = {
                **scene,
                "initial_robot_qpos": before_qpos,
                "physical_success": success,
                "demo_success": (
                    bool(result.success[0]) if result is not None else False
                ),
                "completed": (
                    bool(result.completed_by_env[0]) if result is not None else False
                ),
                "failure": failure,
                "length": (
                    int(result.lengths[0])
                    if result is not None
                    else int(raw._elapsed_steps[0])
                ),
                "terminal_reason": (
                    result.terminal_reasons[0]
                    if result is not None
                    else "planning_failed"
                ),
                "planning_error": planning_error,
                **metrics,
            }
            record["success"] = (
                record["physical_success"]
                and record["demo_success"]
                and record["completed"]
            )
            records.append(record)
            _write(output / "progress.json", {"records": records}, replace=True)
            print(
                json.dumps(
                    {
                        "kind": "qualify",
                        "scene": scene["ordinal"],
                        "success": record["success"],
                    }
                ),
                flush=True,
            )
    finally:
        _close_env(env)
    _verify_sources(protocol)
    report = {
        "status": "complete",
        "protocol_sha256": _sha(output / "protocol.json"),
        "records": records,
        "scenes": scenes,
        "successes": sum(int(record["success"]) for record in records),
        "success_rate": sum(int(record["success"]) for record in records) / scenes,
        "qualification_passed": sum(int(record["success"]) for record in records)
        >= math.ceil(scenes * 0.95),
        "wall_seconds": time.monotonic() - start,
        "closed": True,
    }
    _write(output / "report.json", report)
    return report


def replay(
    *, manifest: Path, output: Path, episodes: int, device: str
) -> dict[str, Any]:
    """Replay original per-frame requests and report visible initial-state parity."""
    if type(episodes) is not int or episodes < 1:
        raise ValueError("Replay episode count must be a positive integer")
    import h5py
    import numpy as np
    from examples.ods.simple_bc_demo.observations import policy_observations

    manifest = manifest.resolve()
    data = json.loads(manifest.read_text())
    if (
        data.get("complete") is not True
        or data.get("action_contracts") != ACTION_CONTRACTS
        or data.get("action_semantics") != "policy_request"
        or data.get("step_dt") != STEP_DT
    ):
        raise ValueError("Replay requires a complete compatible policy-request corpus")
    selected = data["episodes"][:episodes]
    if len(selected) != episodes or any(
        not record.get("physical_success") for record in selected
    ):
        raise ValueError(
            "Replay panel must contain the declared physically successful episodes"
        )
    for record in selected:
        if _sha(manifest.parent / record["path"]) != record["sha256"]:
            raise ValueError("Replay episode content differs from the manifest")
    config = _current_config(device)
    protocol = _freeze(
        output,
        "demonstration-replay",
        config,
        manifest=str(manifest),
        manifest_sha256=_sha(manifest),
        episodes=selected,
        visible_initial_abs_tolerance=1e-4,
    )
    env = None
    records = []
    start = time.monotonic()
    try:
        env = _open_env(config, recording=False)
        raw = env.unwrapped
        for ordinal, episode in enumerate(selected):
            with h5py.File(manifest.parent / episode["path"], "r") as source:
                geometry = np.asarray(source["geometry"][:], dtype=np.float32)
                actions = np.asarray(source["actions"][:], dtype=np.float32)
            if (
                geometry.shape != (episode["length"], 51)
                or actions.shape != (episode["length"], 7)
                or not np.isfinite(geometry).all()
                or not np.isfinite(actions).all()
                or np.abs(actions).max() > 1.000001
            ):
                raise ValueError(
                    "Replay geometry/actions are not aligned finite policy requests"
                )
            scene = {
                "ordinal": ordinal,
                "seed": episode["seed"],
                "cube_xy": episode["cube_xy"],
            }
            observation = _reset(env, scene)
            first = policy_observations(observation)[0].detach().cpu().numpy()
            initial_error = float(np.max(np.abs(first - geometry[0])))
            maximum = initial_error
            mean_error_sum = 0.0
            success = False
            failure = False
            terminated = False
            truncated = False
            steps = 0
            for t, action in enumerate(actions):
                actual = policy_observations(observation)[0].detach().cpu().numpy()
                delta = np.abs(actual - geometry[t])
                maximum = max(maximum, float(delta.max()))
                mean_error_sum += float(delta.mean())
                observation, _, terminated_t, truncated_t, _ = env.step(
                    torch.from_numpy(action).to(raw.device).unsqueeze(0)
                )
                steps = t + 1
                success, failure, metrics = _result(raw)
                terminated = bool(torch.as_tensor(terminated_t).any())
                truncated = bool(torch.as_tensor(truncated_t).any())
                if success or failure or terminated or truncated:
                    break
            record = {
                **scene,
                "candidate": episode["candidate"],
                "recorded_row": episode.get("row"),
                "recorded_steps": len(actions),
                "executed_steps": steps,
                "initial_visible_max_abs": initial_error,
                "initial_visible_match": initial_error <= 1e-4,
                "observation_max_abs": maximum,
                "observation_mean_abs": mean_error_sum / steps if steps else None,
                "physical_success": success,
                "failure": failure,
                "terminated": terminated,
                "truncated": truncated,
                **metrics,
            }
            records.append(record)
            _write(output / "progress.json", {"records": records}, replace=True)
            print(
                json.dumps(
                    {
                        "kind": "replay",
                        "scene": ordinal,
                        "success": success,
                        "initial_visible_match": record["initial_visible_match"],
                    }
                ),
                flush=True,
            )
    finally:
        _close_env(env)
    _verify_sources(protocol)
    if _sha(manifest) != protocol["manifest_sha256"] or any(
        _sha(manifest.parent / record["path"]) != record["sha256"]
        for record in selected
    ):
        raise ValueError("Replay manifest or episode content changed during evaluation")
    report = {
        "status": "complete",
        "protocol_sha256": _sha(output / "protocol.json"),
        "records": records,
        "episodes": episodes,
        "successes": sum(int(record["physical_success"]) for record in records),
        "initial_visible_matches": sum(
            int(record["initial_visible_match"]) for record in records
        ),
        "reuse_qualified": len(records) == episodes
        and all(
            record["physical_success"] and record["initial_visible_match"]
            for record in records
        ),
        "comparison_scope": "visible initial qpos/qvel/TCP/cube/goal state; latent simulator and controller state are not captured; deviations alone do not establish a controller defect",
        "wall_seconds": time.monotonic() - start,
        "closed": True,
    }
    _write(output / "report.json", report)
    return report


def collect(
    *, output: Path, seed_base: int, episodes: int, max_attempts: int, device: str
) -> dict[str, Any]:
    """Collect successful numeric demonstrations without a historical checkout."""
    import h5py
    import numpy as np
    from examples.ods.simple_bc_demo.observations import policy_observations

    if (
        type(episodes) is not int
        or episodes < 1
        or type(max_attempts) is not int
        or max_attempts < episodes
    ):
        raise ValueError(
            "Episode and attempt counts must be positive integers with enough attempts"
        )
    panel, panel_hash = _scene_panel(seed_base, max_attempts)
    config = _current_config(device)
    protocol = _freeze(
        output,
        "collect",
        config,
        seed_base=seed_base,
        requested_episodes=episodes,
        max_attempts=max_attempts,
        scenes=panel,
        scene_digest=panel_hash,
    )
    env = None
    records, attempts = [], []
    start = time.monotonic()
    try:
        env = _open_env(config, recording=True)
        raw = env.unwrapped
        if raw.rollout_action_semantics != "policy_request":
            raise ValueError("State collection requires raw policy-request recording")
        for scene in panel:
            initial = _reset(env, scene)
            initial_geometry = policy_observations(initial)[0].detach().cpu().clone()
            result, planning_error = _expert_episode(env, scene["ordinal"])
            physical, failure, metrics = _result(raw)
            accepted = (
                physical
                and result is not None
                and bool(result.success[0])
                and bool(result.completed_by_env[0])
            )
            attempt = {
                **scene,
                "physical_success": physical,
                "demo_success": (
                    bool(result.success[0]) if result is not None else False
                ),
                "completed": (
                    bool(result.completed_by_env[0]) if result is not None else False
                ),
                "failure": failure,
                "success": accepted,
                "length": (
                    int(result.lengths[0])
                    if result is not None
                    else int(raw._elapsed_steps[0])
                ),
                "terminal_reason": (
                    result.terminal_reasons[0]
                    if result is not None
                    else "planning_failed"
                ),
                "planning_error": planning_error,
                **metrics,
            }
            if accepted:
                length = int(raw.rollout_steps[0])
                recorded = raw.rollout_buffer[0, :length].to("cpu")
                geometry = policy_observations(recorded["obs"]).numpy()
                actions = recorded["actions"].numpy()
                if (
                    length < 2
                    or geometry.shape != (length, 51)
                    or actions.shape != (length, 7)
                    or not np.isfinite(geometry).all()
                    or not np.isfinite(actions).all()
                    or np.abs(actions).max() > 1.000001
                ):
                    raise ValueError(
                        "Collected data are not aligned finite state/action records"
                    )
                if not torch.allclose(
                    torch.from_numpy(geometry[0]), initial_geometry, rtol=0, atol=1e-5
                ):
                    raise ValueError(
                        "Recorded first frame is not the initial pre-action state"
                    )
                path = output / f"episode-{len(records):06d}.h5"
                if path.exists():
                    raise FileExistsError(f"Refusing to replace {path}")
                with h5py.File(path, "x") as target:
                    target.create_dataset("geometry", data=geometry)
                    target.create_dataset("actions", data=actions)
                records.append(
                    {
                        **attempt,
                        "candidate": scene["ordinal"],
                        "row": 0,
                        "path": path.name,
                        "sha256": _sha(path),
                        "physical_success": True,
                    }
                )
            attempts.append(attempt)
            _write(
                output / "attempts.json",
                {"records": attempts, "accepted": len(records)},
                replace=True,
            )
            print(
                json.dumps(
                    {
                        "kind": "collect",
                        "candidate": scene["ordinal"],
                        "accepted": len(records),
                        "success": accepted,
                    }
                ),
                flush=True,
            )
            if len(records) == episodes:
                break
    finally:
        _close_env(env)
    _verify_sources(protocol)
    complete = len(records) == episodes
    result = {
        "version": 1,
        "complete": complete,
        "reference_commit": protocol["source_commit"],
        "requested_episodes": episodes,
        "episodes": records,
        "cameras": [],
        "geometry_width": 51,
        "action_width": 7,
        "step_dt": STEP_DT,
        "action_semantics": "policy_request",
        "action_contracts": ACTION_CONTRACTS,
        "observation_timing": "pre_action",
        "observation_timing_version": "pickplace-preaction-policy-request@1",
        "physics": "default",
        "source_id": "state-bc-pickplace-current-v1",
        "protocol_sha256": _sha(output / "protocol.json"),
        "wall_seconds": time.monotonic() - start,
    }
    _write(output / "manifest.json", result)
    if not complete:
        raise RuntimeError(
            f"Only {len(records)}/{episodes} physically successful episodes collected"
        )
    return result


def _policy(checkpoint: Path, device: str) -> tuple[ChunkBCExecutor, dict[str, Any]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    signature = state["signature"]
    weights = state["learner"]["model"]
    model = ChunkBCModel(ChunkBCModelCfg(**weights["_extra_state"]["config"]))
    recipe = signature.get("state_recipe")
    if (
        signature.get("observation_schema") != OBSERVATION_SCHEMA
        or signature.get("action_contracts") != ACTION_CONTRACTS
        or signature.get("step_dt") != STEP_DT
        or model.cfg.observation_dim != 51
        or model.cfg.cameras
    ):
        raise ValueError("State checkpoint observation/controller contract differs")
    if recipe == "maniskill_action":
        expected = (1, 1, 7, ACTION_SEMANTICS, 1)
    elif recipe == "legacy_chunk":
        expected = (2, 64, 10, TARGET_SEMANTICS, 16)
    else:
        raise ValueError("Unknown state BC recipe")
    actual = (
        model.cfg.history_length,
        model.cfg.prediction_horizon,
        model.cfg.target_dim,
        model.cfg.target_semantics,
        signature.get("action_horizon"),
    )
    if actual != expected:
        raise ValueError("State checkpoint model, target or action horizon differs")
    _validate_config(signature["environment"])
    model.load_state_dict(weights)
    model.to(device=device).eval()
    return ChunkBCExecutor(model, action_horizon=expected[-1]), state


def evaluate(
    *, checkpoint: Path, output: Path, seed_base: int, scenes: int, device: str
) -> dict[str, Any]:
    """Evaluate a state checkpoint and record actual episode-end outcomes."""
    import numpy as np
    from examples.ods.simple_bc_demo.observations import policy_observations

    panel, panel_hash = _scene_panel(seed_base, scenes)
    output = output.resolve()
    if not output.name or output.exists():
        raise FileExistsError("Evaluation output must be a new file path")
    artifact = output.with_name(output.name + ".artifacts")
    if artifact.exists():
        raise FileExistsError("Evaluation artifact directory already exists")
    checkpoint = checkpoint.resolve()
    executor, state = _policy(checkpoint, device)
    config = _config_on_device(state["signature"]["environment"], device)
    protocol = _freeze(
        artifact,
        "student-evaluation",
        config,
        checkpoint=str(checkpoint),
        checkpoint_sha256=_sha(checkpoint),
        seed_base=seed_base,
        scenes=panel,
        scene_digest=panel_hash,
        policy_step=state["learner"]["num_updates"],
    )
    env = None
    records = []
    start = time.monotonic()
    trace_transfer_seconds = 0.0
    trace_write_seconds = 0.0
    policy_calls = 0

    def count_forward(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal policy_calls
        policy_calls += 1

    hook = executor.model.register_forward_hook(count_forward)
    try:
        env = _open_env(config, recording=False)
        raw = env.unwrapped
        for scene in panel:
            observation = _reset(env, scene)
            executor.reset()
            success_once = False
            fail_once = False
            first_success = None
            first_failure = None
            trace_before, trace_targets, trace_actions, trace_after = [], [], [], []
            trace_status = []
            geometry = policy_observations(observation).to(device)
            calls_before_scene = policy_calls
            target_index = 0
            for length in range(1, 601):
                calls_before = policy_calls
                target = executor.action(geometry[:, :51])
                if not torch.isfinite(target).all():
                    raise FloatingPointError("State BC produced a non-finite action")
                action = (
                    target.clamp(-1, 1)
                    if state["signature"]["state_recipe"] == "maniskill_action"
                    else decode_request(target, geometry)
                )
                replanned = policy_calls != calls_before
                target_index = 0 if replanned else target_index + 1
                trace_before.append(geometry[0].detach().clone())
                trace_targets.append(target[0].detach().clone())
                trace_actions.append(action[0].detach().clone())
                observation, _, terminated_t, truncated_t, _ = env.step(
                    action.to(raw.device)
                )
                geometry = policy_observations(observation).to(device)
                trace_after.append(geometry[0].detach().clone())
                success_t, failure_t, raw_metrics = raw.compute_task_state()
                status = (
                    torch.stack(
                        (
                            success_t[0].float(),
                            failure_t[0].float(),
                            raw_metrics["lifted"][0].float(),
                            raw_metrics["goal_error"][0].float(),
                            raw._stable_steps[0].float(),
                            torch.as_tensor(terminated_t, device=raw.device)
                            .any()
                            .float(),
                            torch.as_tensor(truncated_t, device=raw.device)
                            .any()
                            .float(),
                        )
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
                success, failure, lifted = (
                    bool(status[0]),
                    bool(status[1]),
                    bool(status[2]),
                )
                metrics = {"lifted": lifted, "goal_error": status[3]}
                stable_steps = int(status[4])
                terminated, truncated = bool(status[5]), bool(status[6])
                success_once |= success
                fail_once |= failure
                if success and first_success is None:
                    first_success = length
                if failure and first_failure is None:
                    first_failure = length
                trace_status.append(
                    {
                        "step": length,
                        "success": success,
                        "failure": failure,
                        "goal_error": metrics["goal_error"],
                        "lifted": metrics["lifted"],
                        "stable_steps": stable_steps,
                        "terminated": terminated,
                        "truncated": truncated,
                        "replanned": replanned,
                        "target_index": target_index,
                    }
                )
                if terminated or truncated:
                    break
            transfer_started = time.monotonic()
            values = (
                torch.cat(
                    (
                        torch.stack(trace_before),
                        torch.stack(trace_targets),
                        torch.stack(trace_actions),
                        torch.stack(trace_after),
                    ),
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            trace_transfer_seconds += time.monotonic() - transfer_started
            target_size = executor.model.cfg.target_dim
            trace = []
            for index, status in enumerate(trace_status):
                before = values[index, :51]
                predicted = values[index, 51 : 51 + target_size]
                action = values[index, 51 + target_size : 58 + target_size]
                after = values[index, 58 + target_size :]
                cube, tcp, goal = after[27:30], after[18:21], after[36:39]
                distance = float(np.linalg.norm(cube - tcp))
                conditions = {
                    "lifted": status["lifted"],
                    "near_goal": status["goal_error"] <= 0.03,
                    "goal_height": abs(float(cube[2] - goal[2])) <= 0.012,
                    "opened": bool(np.min(after[7:9]) >= 0.035),
                    "withdrawn": distance >= 0.10,
                    "linear_stable": bool(np.linalg.norm(after[45:48]) < 0.05),
                    "angular_stable": bool(np.linalg.norm(after[48:51]) < 0.2),
                    "finite": bool(
                        np.isfinite(after[27:36]).all()
                        and np.isfinite(after[45:51]).all()
                    ),
                }
                if status["success"]:
                    phase = "success"
                elif conditions["near_goal"]:
                    phase = (
                        "stabilize"
                        if conditions["opened"] and conditions["withdrawn"]
                        else "withdraw" if conditions["opened"] else "release"
                    )
                elif status["lifted"]:
                    phase = "transport"
                else:
                    phase = "grasp" if distance <= 0.08 else "approach"
                trace.append(
                    {
                        **status,
                        "phase": phase,
                        "conditions": conditions,
                        "pre_action_observation": before.tolist(),
                        "network_target": predicted.tolist(),
                        "executed_action": action.tolist(),
                        "post_action_observation": after.tolist(),
                    }
                )
            trace_started = time.monotonic()
            trace_path = artifact / f"scene-{scene['ordinal']:03d}-trace.json"
            _write(
                trace_path,
                {
                    "scene": scene,
                    "timing": "before action / target / action / after action",
                    "records": trace,
                },
            )
            trace_write_seconds += time.monotonic() - trace_started
            task_success = success_once and (
                first_failure is None or first_success < first_failure
            )
            record = {
                **scene,
                "success": task_success,
                "success_once": success_once,
                "success_at_end": success,
                "fail_once": fail_once,
                "first_success_step": first_success,
                "first_failure_step": first_failure,
                "length": length,
                "terminated": terminated,
                "truncated": truncated,
                "policy_forward_calls": policy_calls - calls_before_scene,
                **metrics,
                "trace_path": str(trace_path.resolve()),
                "trace_sha256": _sha(trace_path),
            }
            records.append(record)
            _write(artifact / "progress.json", {"records": records}, replace=True)
            print(
                json.dumps(
                    {
                        "kind": "evaluate",
                        "scene": scene["ordinal"],
                        "success": task_success,
                    }
                ),
                flush=True,
            )
        if getattr(raw, "_expert_engine", None) is not None:
            raise RuntimeError("Student execution unexpectedly initialized an expert")
    finally:
        hook.remove()
        _close_env(env)
    _verify_sources(protocol)
    if _sha(checkpoint) != protocol["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during physical evaluation")
    report = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha(checkpoint),
        "policy_step": state["learner"]["num_updates"],
        "state_recipe": state["signature"]["state_recipe"],
        "protocol_sha256": _sha(artifact / "protocol.json"),
        "seed": seed_base,
        "scenes": scenes,
        "scene_digest": panel_hash,
        "records": records,
        "success_rate": sum(int(record["success"]) for record in records) / scenes,
        "success_once": sum(int(record["success_once"]) for record in records) / scenes,
        "success_at_end": sum(int(record["success_at_end"]) for record in records)
        / scenes,
        "fail_once": sum(int(record["fail_once"]) for record in records) / scenes,
        "goal_error": sum(record["goal_error"] for record in records) / scenes,
        "lift_rate": sum(int(record["lifted"]) for record in records) / scenes,
        "expert_initialized": False,
        "termination_policy": "end_on_first_termination_or_truncation; no_auto_reset",
        "trace_transfer_seconds": trace_transfer_seconds,
        "trace_write_seconds": trace_write_seconds,
        "trace_storage": "per scene; tensors transferred as one bounded batch at episode end",
        "policy_forward_calls": policy_calls,
        "wall_seconds": time.monotonic() - start,
        "closed": True,
    }
    _write(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    environment = commands.add_parser("environment")
    environment.add_argument("--output", type=Path, required=True)
    qualification = commands.add_parser("qualify")
    qualification.add_argument("--output-dir", type=Path, required=True)
    qualification.add_argument("--seed-base", type=int, default=20700000)
    qualification.add_argument("--scenes", type=int, default=40)
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--manifest", type=Path, required=True)
    replay_parser.add_argument("--output-dir", type=Path, required=True)
    replay_parser.add_argument("--episodes", type=int, default=32)
    collection = commands.add_parser("collect")
    collection.add_argument("--output-dir", type=Path, required=True)
    collection.add_argument("--seed-base", type=int, default=20800000)
    collection.add_argument("--episodes", type=int, default=32)
    collection.add_argument("--max-attempts", type=int, default=96)
    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("--checkpoint", type=Path, required=True)
    evaluator.add_argument("--output", type=Path, required=True)
    evaluator.add_argument("--seed-base", type=int, required=True)
    evaluator.add_argument("--scenes", type=int, required=True)
    for command in (environment, qualification, replay_parser, collection, evaluator):
        command.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "environment":
        _write(args.output, _current_config(args.device))
    elif args.command == "qualify":
        qualify(
            output=args.output_dir,
            seed_base=args.seed_base,
            scenes=args.scenes,
            device=args.device,
        )
    elif args.command == "replay":
        replay(
            manifest=args.manifest,
            output=args.output_dir,
            episodes=args.episodes,
            device=args.device,
        )
    elif args.command == "collect":
        collect(
            output=args.output_dir,
            seed_base=args.seed_base,
            episodes=args.episodes,
            max_attempts=args.max_attempts,
            device=args.device,
        )
    else:
        evaluate(
            checkpoint=args.checkpoint,
            output=args.output,
            seed_base=args.seed_base,
            scenes=args.scenes,
            device=args.device,
        )


if __name__ == "__main__":
    main()
