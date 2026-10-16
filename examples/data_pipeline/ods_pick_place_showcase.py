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

"""Train a recurrent PickPlace policy from live ODS expert demonstrations.

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

from embodichain.data_pipeline.engine.data import OnlineDataEngine, OnlineDataEngineCfg

TASK_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "extensions/embodichain_tasks/configs/tasks/manipulation/pick_place/task.franka.yaml"
)


def policy_observations(obs: Any) -> torch.Tensor:
    """Pack current measured state only; never read demonstration annotations."""
    return torch.cat(
        (
            obs["robot"]["qpos"],
            obs["robot"]["qvel"],
            obs["cube_pose"],
            obs["goal_position"],
        ),
        dim=-1,
    ).float()


class RecurrentBC(nn.Module):
    """Small recurrent policy with frozen observation and action normalization."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
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
        features, hidden = self.gru((obs - self.mean) / self.std, hidden)
        return self.head(features), hidden

    def action(
        self, obs: torch.Tensor, hidden: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized, hidden = self(obs[:, None], hidden)
        return self.low + (normalized[:, 0] + 1) * 0.5 * (self.high - self.low), hidden

    def loss(
        self, obs: torch.Tensor, actions: torch.Tensor, burn_in: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


def load_task_config(device: str) -> dict[str, Any]:
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
    return config


def save_checkpoint(
    path: Path,
    model: RecurrentBC,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    run_id: int,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {path}")
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": vars(args),
        "run_id": run_id,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_policy(path: str, device: str) -> tuple[RecurrentBC, dict[str, Any]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = state["model"]
    model = RecurrentBC(
        weights["mean"], weights["std"], weights["low"], weights["high"]
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
                model_step = str(checkpoint_state["step"])
            if viewer:
                raw.sim.stop_visualization()
                raw.sim.sim_config.visualization.viser_server.label = (
                    f'ODS {request["name"]} / step {model_step}'
                )
                raw.sim.start_visualization()
            records = []
            for ordinal, seed in enumerate(request["seeds"]):
                obs, _ = env.reset(seed=seed, options={"save_data": False})
                if kind == "qualify":
                    result = execute_demo_episode(env)
                    success = bool(result.completed and result.all_success)
                    info = raw.get_info()
                    length = result.length
                else:
                    hidden = None
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
                                    policy_observations(obs), hidden
                                )
                            obs, _, terminated, truncated, info = env.step(action)
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
                    }
                )
                print(
                    f'{request["name"]}: {ordinal + 1}/{len(request["seeds"])} seed={seed} success={success}',
                    flush=True,
                )
            report = {
                "name": request["name"],
                "records": records,
                "success_rate": sum(r["success"] for r in records) / len(records),
                "lift_rate": sum(r["lifted"] for r in records) / len(records),
                "goal_error": sum(r["goal_error"] for r in records) / len(records),
                "episode_length": sum(r["length"] for r in records) / len(records),
            }
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


def _sample(engine: OnlineDataEngine, batch: int, length: int) -> tuple[Any, float]:
    started = time.monotonic()
    while True:
        try:
            return engine.sample_batch(batch, length), time.monotonic() - started
        except RuntimeError as error:
            if "No unlocked valid chunk" not in str(error):
                raise
            if time.monotonic() - started >= 30:
                raise TimeoutError("No eligible ODS window for 30 seconds") from error
            time.sleep(0.02)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument("--train-device", default="cuda:0")
    p.add_argument("--output", type=Path, default=Path("/tmp/ods-pick-place-showcase"))
    p.add_argument("--mode", choices=("online", "frozen"), default="online")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--updates-per-round", type=int, default=1000)
    p.add_argument("--buffer-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path)
    p.add_argument("--viser-port", type=int, default=8080)
    p.add_argument("--tensorboard-port", type=int, default=6006)
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--qualification-only", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.rounds, args.updates_per_round, args.buffer_size, args.batch_size = (
            2,
            5,
            2,
            4,
        )
    if min(args.rounds, args.updates_per_round, args.buffer_size, args.batch_size) < 1:
        p.error("Training counts must be positive")
    return args


def main() -> None:
    args = _parse_args()
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

        writer = SummaryWriter(str(args.output / "events"))
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
        mean, std = values.mean(0), values.std(0).clamp_min(1e-3)
        model = RecurrentBC(
            mean, std, torch.tensor(ready["low"]), torch.tensor(ready["high"])
        ).to(args.train_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        step = 0
        if args.resume:
            model, state = load_policy(str(args.resume), args.train_device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            optimizer.load_state_dict(state["optimizer"])
            step = state["step"]
            torch.set_rng_state(state["torch_rng"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            if state["cuda_rng"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["cuda_rng"])
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
        best_rate, best_path = -1.0, baseline
        consumed: dict[int, int] = {}
        for round_index in range(args.rounds):
            generation = 1
            if round_index and args.mode == "online":
                generation = engine.request_refill()
            model.train()
            for update_index in range(args.updates_per_round):
                update_started = time.monotonic()
                if update_index == args.updates_per_round - 1:
                    engine.wait_for_generation(generation, timeout=1800)
                batch, waited = _sample(engine, args.batch_size, 32)
                generations, counts = batch["ods_generation"].unique(return_counts=True)
                for version, count in zip(generations.tolist(), counts.tolist()):
                    consumed[version] = consumed.get(version, 0) + count
                    writer.add_scalar(
                        f"data/consumed_generation_{version}", consumed[version], step
                    )
                obs = policy_observations(batch["obs"]).to(args.train_device)
                actions = batch["actions"].to(args.train_device)
                arm, hand = model.loss(obs, actions)
                optimizer.zero_grad(set_to_none=True)
                (arm + 5 * hand).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
                writer.add_scalar("train/arm_loss", arm.item(), step)
                writer.add_scalar("train/gripper_loss", hand.item(), step)
                writer.add_scalar("data/wait_seconds", waited, step)
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
            for name in ("success_rate", "lift_rate", "goal_error", "episode_length"):
                writer.add_scalar(f"eval/{name}", report[name], step)
            for name, value in engine.get_stats().items():
                if name != "run_id":
                    writer.add_scalar(f"ods/{name}", value, step)
            writer.flush()
            if report["success_rate"] > best_rate:
                best_rate, best_path = report["success_rate"], checkpoint
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
            "consumed_generations": consumed,
            "best_checkpoint": str(best_path),
            "final_success_rate": final["success_rate"],
            "initial_success_rate": initial_test_rate,
            "learning_target_met": not args.smoke
            and final["success_rate"] >= 0.8
            and final["success_rate"] > initial_test_rate,
            "stats": engine.get_stats(),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
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
