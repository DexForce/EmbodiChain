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

"""Independent physical evaluation of a saved visual BC checkpoint.

Run this entry point in a separate process after the timed training pipeline
has closed. Evaluation results are quality information outside its time budget.
The caller reserves and masks both simulation and inference devices explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import torch

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[3]
    sys.path[:0] = [
        str(repository),
        str(repository / "extensions"),
        str(repository / "extensions/embodichain_tasks"),
    ]

from examples.ods.data_stream_demo.acquisition import (
    CAMERA_PROFILE_VERSION,
    camera_uids,
    load_acquisition_config,
)
from examples.ods.data_stream_demo.visual_policy import VisualChunkBC
from examples.ods.simple_bc_demo.models import rotation_from_6d
from examples.ods.simple_bc_demo.observations import policy_observations
from embodichain.utils.math import axis_angle_from_quat, quat_from_matrix

__all__ = ["evaluate_checkpoint"]


def _load_visual_policy(
    checkpoint: Path, vision: str, device: str
) -> tuple[VisualChunkBC, dict[str, Any], int]:
    if vision not in ("single", "dual"):
        raise ValueError("Visual evaluation supports only single or dual cameras")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # Only load checkpoints written by this example: optimizer/RNG records may
    # include trusted Python objects that are outside weights_only's allowlist.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state["model"]
    config = state["config"]
    cameras = tuple(
        state.get("cameras", weights.get("_extra_state", {}).get("cameras", ()))
    )
    if cameras != camera_uids(vision) or config.get("vision", vision) != vision:
        raise ValueError(
            "Checkpoint vision profile/camera order differs from evaluation"
        )
    prediction_horizon = config.get("prediction_horizon", 32)
    action_horizon = config.get("action_horizon", min(16, prediction_horizon))
    if (
        type(prediction_horizon) is not int
        or prediction_horizon < 1
        or type(action_horizon) is not int
        or not 1 <= action_horizon <= prediction_horizon
    ):
        raise ValueError(
            "Checkpoint action horizon must fit its positive prediction horizon"
        )
    normalization = state.get("normalization", {})
    statistics = {}
    for key in ("proprio_mean", "proprio_std", "target_mean", "target_std"):
        statistics[key] = torch.as_tensor(
            normalization.get(key, weights[key]), dtype=torch.float32, device="cpu"
        )
        if not torch.equal(statistics[key], weights[key].cpu()):
            raise ValueError(
                f"Checkpoint normalization differs from model buffer {key}"
            )
    model = VisualChunkBC(
        **statistics, cameras=cameras, prediction_horizon=prediction_horizon
    )
    model.load_state_dict(weights)
    model = model.to(device).eval()
    return model, state, action_horizon


def _measured_images(
    observation: Any, cameras: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    images = {}
    for camera in cameras:
        color = observation["sensor"][camera]["color"]
        if (
            color.shape not in ((1, 256, 256, 3), (1, 256, 256, 4))
            or color.dtype != torch.uint8
        ):
            raise ValueError(
                f"Camera {camera} must provide one measured uint8 RGB(A) frame"
            )
        # Own the historical image before the next simulator step mutates its
        # camera buffers. CPU scaling matches the online/offline data adapters.
        rgb = color[..., :3].movedim(-1, -3).to(device="cpu", dtype=torch.float32)
        images[camera] = rgb.div_(255)
    return images


def _relative_request(target: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
    rotation_error = rotation_from_6d(target[:, 3:9]) @ rotation_from_6d(
        geometry[:, 21:27]
    ).transpose(-1, -2)
    return torch.cat(
        (
            (target[:, :3] - geometry[:, 18:21]) / 0.02,
            axis_angle_from_quat(quat_from_matrix(rotation_error)) / 0.1,
            target[:, 9:],
        ),
        dim=-1,
    ).clamp(-1, 1)


def evaluate_checkpoint(
    checkpoint: Path,
    vision: str,
    sim_device: str = "cuda:0",
    train_device: str = "cuda:1",
    *,
    scenes: int = 20,
    seed: int = 197001,
    output: Path | None = None,
) -> dict[str, Any]:
    """Execute only the frozen visual student in a fresh physical environment.

    Args:
        checkpoint: Trusted visual BC checkpoint produced by this demo.
        vision: ``single`` or ``dual`` camera profile, matching the checkpoint.
        sim_device: Explicit tensor/renderer device of the single environment.
        train_device: Explicit device for frozen model inference.
        scenes: Number of fixed, independently seeded initial cube states.
        seed: CPU scene-generator seed, independent of training generation.
        output: Optional JSON report path. All episode failures are retained.

    Returns:
        Reloaded checkpoint identity, every physical outcome, aggregate metrics,
        and successful environment shutdown evidence.
    """
    if type(scenes) is not int or scenes < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Evaluation scenes must be positive and seed nonnegative")
    checkpoint = Path(checkpoint)
    model, state, action_horizon = _load_visual_policy(checkpoint, vision, train_device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    positions = (
        torch.tensor([-0.42, -0.08])
        + (torch.rand(scenes, 2, generator=generator) - 0.5) * 0.08
    )
    scene_digest = hashlib.sha256(positions.numpy().tobytes()).hexdigest()
    config = load_acquisition_config(vision, sim_device, producer_envs=1, seed=seed)
    config["max_episode_steps"] = 600
    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    # Import the runtime package before manager configuration, matching native
    # worker bootstrap and ensuring its dynamic bindings are initialized.
    from embodichain.lab.sim import SimulationManager
    import gymnasium as gym
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules
    from embodichain.lab.gym.utils.registration import discover_task_packages

    discover_task_packages()
    cfg = config_to_cfg(config, manager_modules=get_manager_modules())
    cfg.init_rollout_buffer = False
    cfg.filter_dataset_saving = True
    cfg.max_episode_steps = 600
    env = None
    records = []
    started = time.monotonic()
    try:
        env = gym.make(config["id"], cfg=cfg)
        raw = env.unwrapped
        raw._replay_no_auto_reset = True
        for ordinal, xy in enumerate(positions):
            scene_seed = seed + ordinal
            observation, _ = env.reset(
                seed=scene_seed,
                options={"save_data": False, "cube_xy": [xy.tolist()]},
            )
            previous_geometry = None
            previous_images = None
            queued_targets = None
            queue_index = 0
            replans = 0
            terminal = False
            truncated = False
            # The simulator updates task-state tensors that survive this loop
            # and are reset outside it. no_grad suppresses policy autograd while
            # keeping those persistent tensors writable at the next reset.
            with torch.no_grad():
                for length in range(1, 601):
                    geometry = policy_observations(observation).to(train_device)
                    if geometry.shape != (1, 51):
                        raise ValueError("Evaluation geometry schema must be [1, 51]")
                    current_images = _measured_images(observation, model.cameras)
                    if queued_targets is None or queue_index >= action_horizon:
                        history = torch.stack(
                            (
                                (
                                    geometry[:, :27]
                                    if previous_geometry is None
                                    else previous_geometry
                                ),
                                geometry[:, :27],
                            ),
                            dim=1,
                        )
                        image_history = {
                            camera: torch.stack(
                                (
                                    (
                                        current_images[camera]
                                        if previous_images is None
                                        else previous_images[camera]
                                    ),
                                    current_images[camera],
                                ),
                                dim=1,
                            ).to(train_device)
                            for camera in model.cameras
                        }
                        queued_targets = (
                            model(history, image_history) * model.target_std
                            + model.target_mean
                        )
                        if not torch.isfinite(queued_targets).all():
                            raise RuntimeError(
                                "Student produced nonfinite absolute targets"
                            )
                        queue_index = 0
                        replans += 1
                    request = _relative_request(
                        queued_targets[:, queue_index], geometry
                    )
                    queue_index += 1
                    # Retain the immediately preceding measured observation at
                    # every step, including while executing a queued chunk.
                    previous_geometry = geometry[:, :27].clone()
                    previous_images = current_images
                    observation, _, terminated, timed_out, info = env.step(
                        request.to(raw.device)
                    )
                    terminal = bool(torch.as_tensor(terminated).any())
                    truncated = bool(torch.as_tensor(timed_out).any())
                    if terminal or truncated:
                        break
            success, failed, metrics = raw.compute_task_state()
            record = {
                "scene": ordinal,
                "seed": scene_seed,
                "cube_xy": xy.tolist(),
                "success": bool(success[0]),
                "failure": bool(failed[0]),
                "lifted": bool(metrics["lifted"][0]),
                "goal_error": float(metrics["goal_error"][0]),
                "length": length,
                "replans": replans,
                "terminated": terminal,
                "truncated": truncated,
            }
            records.append(record)
            if output is not None:
                output = Path(output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(
                        {
                            "passed": False,
                            "status": "in_progress",
                            "checkpoint": str(checkpoint.resolve()),
                            "vision": vision,
                            "seed": seed,
                            "scene_digest": scene_digest,
                            "scenes": scenes,
                            "records": records,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            print(
                f"visual evaluation {ordinal + 1}/{scenes}: success={record['success']} "
                f"lifted={record['lifted']} error={record['goal_error']:.4f}",
                flush=True,
            )
        if getattr(raw, "_expert_engine", None) is not None:
            raise RuntimeError(
                "Independent student evaluation initialized an expert engine"
            )
    finally:
        try:
            if env is not None:
                env.unwrapped.close(exit_process=False)
        finally:
            SimulationManager.flush_cleanup_queue()
    report = {
        "passed": True,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "policy_step": int(state["step"]),
        "vision": vision,
        "cameras": list(model.cameras),
        "camera_profile_version": CAMERA_PROFILE_VERSION,
        "prediction_horizon": model.prediction_horizon,
        "action_horizon": action_horizon,
        "sim_device": sim_device,
        "inference_device": train_device,
        "seed": seed,
        "scene_digest": scene_digest,
        "scenes": scenes,
        "records": records,
        "success_rate": sum(record["success"] for record in records) / scenes,
        "lift_rate": sum(record["lifted"] for record in records) / scenes,
        "goal_error": sum(record["goal_error"] for record in records) / scenes,
        "episode_length": sum(record["length"] for record in records) / scenes,
        "wall_seconds": time.monotonic() - started,
        "timing_scope": "evaluation only; excluded from training pipeline time",
        "closed": True,
        "expert_initialized": False,
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vision", choices=("single", "dual"), required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--train-device", default="cuda:1")
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=197001)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    try:
        report = evaluate_checkpoint(
            args.checkpoint,
            args.vision,
            args.sim_device,
            args.train_device,
            scenes=args.scenes,
            seed=args.seed,
            output=args.output,
        )
    except BaseException:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = json.loads(args.output.read_text()) if args.output.is_file() else {}
        args.output.write_text(
            json.dumps(
                {**partial, "passed": False, "error": traceback.format_exc()}, indent=2
            )
            + "\n"
        )
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
