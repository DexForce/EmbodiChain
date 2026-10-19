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

"""Independent physical evaluation of immutable BC checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from pathlib import Path
from typing import Any
import torch

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[3]
    sys.path[:0] = [
        str(repository),
        str(repository / "extensions"),
        str(repository / "extensions/embodichain_tasks"),
    ]

from examples.ods.simple_bc_demo.checkpoint import load_policy
from examples.ods.simple_bc_demo.config import load_task_config
from examples.ods.simple_bc_demo.observations import policy_observations


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


def main() -> None:
    """Evaluate a saved policy in a fresh simulator, without creating a learner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=810000)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.timeout <= 0 or not args.timeout < float("inf"):
        parser.error("episodes and timeout must be finite and positive")
    if not args.checkpoint.is_file():
        parser.error("checkpoint must be an existing file")
    # A checkpoint can contain Python/NumPy random states. Only load trusted
    # files written by this example, just as in training resume.
    model, state = load_policy(str(args.checkpoint), "cpu")
    config = load_task_config(
        args.sim_device,
        "pose" if model.arm_command_mode == "pose" else "joint",
        state.get("config", {}).get("pose_ik", "dls"),
    )
    del model
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "task_config.json").write_text(json.dumps(config, indent=2))
    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    context = mp.get_context("spawn")
    requests, results = context.Queue(), context.Queue()
    worker = context.Process(
        target=_evaluate_worker,
        args=(
            config,
            requests,
            results,
            str(args.output),
            args.viewer,
            args.viser_port,
            args.video,
        ),
    )
    worker.start()
    forced_cleanup = False
    try:
        _receive(worker, results, args.timeout)
        requests.put(
            {
                "kind": "evaluate",
                "name": "evaluation",
                "checkpoint": str(args.checkpoint.resolve()),
                "seeds": list(range(args.seed_base, args.seed_base + args.episodes)),
                "record": args.video,
            }
        )
        report = _receive(worker, results, args.timeout)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        requests.put(None)
        worker.join(30)
        if worker.is_alive():
            forced_cleanup = True
            worker.terminate()
            worker.join(10)
        if worker.is_alive():
            worker.kill()
            worker.join(5)
        requests.close()
        requests.join_thread()
        results.close()
        results.join_thread()
        (args.output / "cleanup.json").write_text(
            json.dumps(
                {
                    "pid": worker.pid,
                    "exitcode": worker.exitcode,
                    "forced_cleanup": forced_cleanup,
                    "alive": worker.is_alive(),
                },
                indent=2,
            )
        )
    if forced_cleanup or worker.exitcode != 0:
        raise RuntimeError("Evaluation worker did not shut down normally")


if __name__ == "__main__":
    main()
