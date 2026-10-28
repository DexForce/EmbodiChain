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

"""Evaluate visual ACT checkpoints in independent PushCube simulator episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from ._environment import close_env, make_env, task_identity


def _load_policy(
    path: Path, device: str, *, panel: str, seeds: list[int]
) -> tuple[Any, int]:
    from embodichain_models.policy.act import ACTCfg, ACTPolicy

    # Checkpoints contain optimizer and RNG state; only load trusted local files.
    state = torch.load(path, map_location="cpu", weights_only=False)
    signature = state["signature"]
    if (
        signature.get("recipe") != "visual-act-v1"
        or signature.get("task_identity") != task_identity()
    ):
        raise ValueError("Checkpoint task or recipe does not match this environment")
    recorded = signature.get("scene_seeds") or {}
    if panel != "overfit" and set(seeds) & set(
        recorded.get("train", []) + recorded.get("validation", [])
    ):
        raise ValueError("Evaluation seeds overlap the training or validation corpus")
    selection = path.parent / "evaluation.json"
    if panel == "test" and selection.exists():
        previous = json.loads(selection.read_text())
        if previous.get("panel") == "development" and set(seeds) & set(
            previous["seeds"]
        ):
            raise ValueError("Test seeds overlap the checkpoint selection panel")
    cfg = dict(signature["model"])
    cfg["pretrained_backbone"] = False  # The checkpoint supplies all weights.
    model = ACTPolicy(ACTCfg(**cfg))
    model.load_state_dict(state["learner"]["model"])
    return model.to(device).eval(), int(state["learner"]["num_updates"])


def _interval(successes: int, total: int) -> list[float]:
    """Wilson 95% binomial confidence interval."""
    p, z = successes / total, 1.959963984540054
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / (1 + z * z / total)
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def _rollouts(
    env: Any,
    model: Any,
    *,
    mode: str,
    seeds: list[int],
    device: str,
    video_dir: Path | None,
    shuffled: Any = None,
) -> dict[str, Any]:
    from embodichain_models.policy.act import ACTTemporalEnsembler

    raw = env.unwrapped
    raw._replay_no_auto_reset = True
    low = torch.as_tensor(env.action_space.low, device=device)
    high = torch.as_tensor(env.action_space.high, device=device)
    rows = []
    success_video = failure_video = 0
    elapsed = []
    for episode_index, seed in enumerate(seeds):
        obs, _ = env.reset(seed=seed, options={"save_data": False})
        ensemble = (
            ACTTemporalEnsembler(
                model.cfg.horizon, 1, model.cfg.action_dim, device=device
            )
            if model
            else None
        )
        initial_state = obs["robot"]["qpos"].clone()
        frames: list[Any] = []
        record_video = video_dir is not None and (
            success_video < 3 or failure_video < 3
        )
        success, distance, strict = False, float("inf"), False
        for step in range(raw.max_episode_steps):
            rgb = (
                obs["sensor"]["cam_high"]["color"][..., :3]
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            if record_video:
                frames.append(rgb[0].permute(1, 2, 0).cpu().numpy().copy())
            if shuffled is not None:
                numeric, images = shuffled
                row = episode_index % numeric.data.batch_size[0]
                ids = numeric.data["frame_ids"][row, numeric.data["valid"][row]]
                rgb = images.read(ids[min(step, len(ids) - 1)].unsqueeze(0)).to(device)
            before = time.perf_counter()
            with torch.inference_mode():
                if mode == "fixed":
                    action = torch.cat(
                        (initial_state[:, :7], initial_state.new_full((1, 1), -1.0)), -1
                    )
                else:
                    assert model is not None and ensemble is not None
                    action = ensemble.update(model.predict(rgb, obs["robot"]["qpos"]))
                action = torch.maximum(torch.minimum(action, high), low)
            if action.is_cuda:
                torch.cuda.synchronize(action.device)
            elapsed.append(time.perf_counter() - before)
            obs, _, terminated, truncated, info = env.step(action)
            success = bool(info["success"][0])
            distance = float(info["metrics"]["goal_distance"][0])
            strict = distance < 0.03 and success
            if bool(torch.as_tensor(terminated).reshape(-1)[0]) or bool(
                torch.as_tensor(truncated).reshape(-1)[0]
            ):
                break
        row = {
            "seed": seed,
            "success": success,
            "strict_success": strict,
            "goal_distance": distance,
            "steps": step + 1,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if record_video and (
            (success and success_video < 3) or (not success and failure_video < 3)
        ):
            import imageio.v2 as imageio

            assert video_dir is not None
            label = "success" if success else "failure"
            video = video_dir / f"{label}-{seed}.mp4"
            with imageio.get_writer(
                video, fps=30, codec="libx264", macro_block_size=None
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
            success_video += int(success)
            failure_video += int(not success)
    successes = sum(int(row["success"]) for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "success_rate_wilson_95": _interval(successes, len(rows)),
        "strict_success_rate": sum(int(row["strict_success"]) for row in rows)
        / len(rows),
        "mean_goal_distance": sum(row["goal_distance"] for row in rows) / len(rows),
        "mean_policy_seconds": sum(elapsed) / len(elapsed),
        "episodes_detail": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--checkpoint", type=Path)
    choice.add_argument(
        "--run", type=Path, help="Select among step_*.pt using development scenes"
    )
    choice.add_argument("--baseline", choices=("fixed", "untrained"))
    parser.add_argument(
        "--panel", choices=("development", "test", "overfit"), default="test"
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--shuffle-images-report",
        type=Path,
        help="Use RGB from separate recorded episodes as a visual ablation",
    )
    args = parser.parse_args()
    if args.run is not None and args.panel != "development":
        parser.error("Checkpoint selection is restricted to the development panel")
    if args.shuffle_images_report and (
        args.checkpoint is None or args.panel != "development"
    ):
        parser.error("Image shuffling is a development-only checkpoint diagnostic")
    count = (
        args.episodes
        if args.episodes is not None
        else (50 if args.panel == "development" else 100)
    )
    start = (
        args.seed_base
        if args.seed_base is not None
        else {"development": 300000, "test": 400000, "overfit": 110000}[args.panel]
    )
    if count < 1 or start < 0:
        parser.error("episodes must be positive and seed-base nonnegative")
    seeds = list(range(start, start + count))
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    shuffled = None
    if args.shuffle_images_report:
        from embodichain.data_pipeline.datasets import (
            LeRobotDataSource,
            LeRobotImageReader,
        )
        from .train import _report

        report = _report(args.shuffle_images_report)
        path = Path(report["dataset"])
        numeric = LeRobotDataSource(
            path,
            schema=report["schema"],
            source_id="act-shuffle",
            repo_id=path.name,
            memory_limit_bytes=512 * 1024**2,
        )
        images = LeRobotImageReader(
            path, "observation.images.cam_high", repo_id=path.name
        )
        images.validate_frame_order(numeric.data["frame_ids"][numeric.data["valid"]])
        shuffled = (numeric, images)
    env = None
    try:
        env = make_env(args.device, camera=True)
        if args.run:
            checkpoints = sorted(
                args.run.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1])
            )
            if not checkpoints:
                raise ValueError("No step checkpoints found")
        else:
            checkpoints = [args.checkpoint] if args.checkpoint else [None]
        evaluations = []
        for path in checkpoints:
            if path:
                model, update = _load_policy(
                    path, args.device, panel=args.panel, seeds=seeds
                )
                mode = "policy"
            elif args.baseline == "untrained":
                from embodichain_models.policy.act import ACTCfg, ACTPolicy

                torch.manual_seed(19001)
                model = (
                    ACTPolicy(ACTCfg(pretrained_backbone=False)).to(args.device).eval()
                )
                mode, update = "untrained", 0
            else:
                model, mode, update = None, "fixed", 0
            video_dir = None
            if args.video:
                video_dir = args.output / f"videos-{update}"
                video_dir.mkdir()
            report = _rollouts(
                env,
                model,
                mode=mode,
                seeds=seeds,
                device=args.device,
                video_dir=video_dir,
                shuffled=shuffled,
            )
            report.update(
                checkpoint=str(path.resolve()) if path else None,
                update=update,
                checkpoint_sha256=(
                    hashlib.sha256(path.read_bytes()).hexdigest() if path else None
                ),
            )
            evaluations.append(report)
            del model
        if args.run:
            best = max(
                evaluations,
                key=lambda item: (
                    item["success_rate"],
                    -item["mean_goal_distance"],
                    -item["update"],
                ),
            )
            shutil.copyfile(best["checkpoint"], args.output / "best.pt")
        else:
            best = evaluations[0]
        result = {
            "panel": args.panel,
            "task_identity": task_identity(),
            "seeds": seeds,
            "shuffled_images": (
                str(args.shuffle_images_report.resolve())
                if args.shuffle_images_report
                else None
            ),
            "selected_update": best["update"],
            "evaluations": evaluations,
        }
        (args.output / "evaluation.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    key: best[key]
                    for key in (
                        "update",
                        "successes",
                        "episodes",
                        "success_rate",
                        "mean_goal_distance",
                    )
                }
            ),
            flush=True,
        )
    finally:
        close_env(env)


if __name__ == "__main__":
    main()
