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
import queue
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
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

from examples.data_pipeline._ods_pick_place_policy import ChunkBC

TASK_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "extensions/embodichain_tasks/configs/tasks/manipulation/pick_place/task.franka.yaml"
)


def policy_observations(
    obs: Any, *, valid_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Pack measured geometry, canonicalizing optional padding after conversion."""
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
    values = torch.cat(
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
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool or valid_mask.shape != values.shape[:-1]:
            raise ValueError("Geometry valid_mask must match observation batch axes")
        # Rotation conversion maps a zero quaternion to identity. Restore the
        # same zero feature padding used by resident/offline context sampling.
        values.masked_fill_(~valid_mask[..., None], 0)
    return values


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


def load_task_config(device: str, pose_ik: str = "dls") -> dict[str, Any]:
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
    model: ChunkBC,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    run_id: int,
    *,
    sampling_rng_state: torch.Tensor | None = None,
) -> None:
    """Save immutable model, optimizer, normalization and random-state data."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {path}")
    state = {
        "model": model.state_dict(),
        "policy_config": {
            "type": "chunk_pose",
            "observation_mode": model.observation_mode,
            "arm_command_mode": model.arm_command_mode,
            "schema": getattr(model, "schema", None),
            "prediction_horizon": model.prediction_horizon,
            "action_horizon": model.action_horizon,
            "observation_noise": model.observation_noise,
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


def load_policy(path: str, device: str) -> tuple[ChunkBC, dict[str, Any]]:
    """Load a supported chunk-policy checkpoint; reject legacy model schemas."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = state["model"]
    policy_config = state.get("policy_config", {})
    if policy_config.get("type") != "chunk_pose":
        raise ValueError("Only ChunkBC pose checkpoints are supported")
    if (
        policy_config.get("observation_mode") != "geometry"
        or policy_config.get("arm_command_mode") != "pose"
    ):
        raise ValueError("Chunk BC requires geometry observations and pose requests")
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
                                    policy_observations(obs),
                                    hidden,
                                )
                            if request.get("trace", False):
                                trace.append(
                                    {
                                        "observations": policy_observations(obs).cpu(),
                                        "actions": action.cpu(),
                                        "qpos": obs["robot"]["qpos"].cpu(),
                                        "cube_pose": obs["cube_pose"].cpu(),
                                        "tcp_pose": obs["tcp_pose"].cpu(),
                                    }
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
        ),
        default="online",
    )
    p.add_argument(
        "--policy",
        choices=("chunk-pose",),
        default="chunk-pose",
    )
    p.add_argument("--pose-ik", choices=("dls",), default="dls")
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
        help="Showcase LeRobot dataset directory",
    )
    p.add_argument("--offline-storage", choices=("lerobot",), default="lerobot")
    p.add_argument("--offline-memory-limit-mib", type=int, default=512)
    p.add_argument("--offline-episodes", type=int, default=None)
    p.add_argument("--offline-sample-budget", type=int, default=None)
    p.add_argument("--offline-sample-hard-limit", type=int, default=None)
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
        help="Optional independent CPU sampling RNG; omitted uses the global stream",
    )
    p.add_argument("--producer-envs", type=int, default=None)
    p.add_argument("--dev-seed-base", "--dev-seed-start", type=int, default=20000)
    p.add_argument("--test-seed-base", type=int, default=None)
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
    p.add_argument(
        "--deferred-evaluation-schedule",
        choices=("updates", "updates_and_generations"),
        default="updates_and_generations",
    )

    p.add_argument("--resume", type=Path)
    p.add_argument("--viser-port", type=int, default=8080)
    p.add_argument("--tensorboard-port", type=int, default=6006)
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--qualification-only", action="store_true")
    args = p.parse_args()
    if args.offline_memory_limit_mib < 1:
        p.error("--offline-memory-limit-mib must be positive")
    if args.auto_config_input is not None:
        from examples.data_pipeline._ods_pick_place_configuration import (
            apply_configuration,
        )

        try:
            apply_configuration(args)
        except (TypeError, ValueError, OSError) as error:
            p.error(str(error))
    if args.batch_size is None:
        args.batch_size = 256
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
    cold_offline = (
        args.mode in {"offline-full", "offline-32"} and args.offline_data is None
    )
    if args.offline_episodes is not None:
        if (
            not cold_offline
            or args.offline_episodes < args.buffer_size
            or args.offline_episodes % args.buffer_size
        ):
            p.error(
                "Offline collection count must be a cache multiple in a cold offline mode"
            )
        if args.mode == "offline-32" and args.offline_episodes != args.buffer_size:
            p.error("offline-32 collects exactly the initial cache")
        if args.max_generated_episodes not in {None, args.offline_episodes}:
            p.error("Offline collection count conflicts with max-generated-episodes")
        args.max_generated_episodes = args.offline_episodes
    if (
        args.offline_sample_budget is not None
        or args.offline_sample_hard_limit is not None
    ):
        if args.resume is not None:
            p.error(
                "Protected offline resume requires saved quota state and is not supported"
            )
        if (
            args.offline_sample_budget is None
            or args.offline_sample_hard_limit is None
            or not 0 < args.offline_sample_budget <= args.offline_sample_hard_limit
            or not (cold_offline or args.offline_data is not None)
        ):
            p.error(
                "Offline quotas require positive budget <= hard limit and a resident offline corpus"
            )
    if args.max_generated_episodes is not None and (
        args.max_generated_episodes < args.buffer_size
        or args.max_generated_episodes % args.buffer_size
    ):
        p.error("Accepted episode budget must be a positive multiple of cache capacity")
    if args.defer_evaluation and args.stage != "showcase":
        p.error("Deferred evaluation is only supported for the showcase stage")
    dev = set(range(args.dev_seed_base, args.dev_seed_base + args.dev_scenes))
    test = set(range(args.test_seed_base, args.test_seed_base + args.final_test_scenes))
    if dev & test:
        p.error("Development and final-test seeds must be disjoint")
    return args


def main() -> None:
    from examples.data_pipeline._ods_pick_place_training import run

    run(_parse_args())


if __name__ == "__main__":
    main()
