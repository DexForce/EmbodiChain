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
"""Preview every outcome from three seeded learned-policy episodes in DexSim.

Run: embodichain run-example learning/blind_assembly/preview -- RUN \
    --checkpoint best --seed 30000 --output outputs/blind_peg_preview

The native camera shares the window scene and eye/target viewpoint; its explicit
intrinsics are recorded separately. Camera reads never advance physics. A
terminal state is captured before the next manual
reset; no pose or physical state is scripted. This small visual preview is
separate from the held-out episode evaluation and includes failures as well
as successes. The default visual seeds are 30000, 30001 and 30002, separate
from the nominal evaluation's default reset seed. Use the documented assembly
dependency overlays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from typing import Any

import numpy as np
from PIL import Image
import torch

__all__: list[str] = []


def _plain(value: Any) -> Any:
    """Detach owned CPU trace values without changing the simulation."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _encode(output: Path, fps: float, count: int) -> None:
    """Encode actual native RGB frames and a small episode contact sheet."""
    source = str(output / "frames" / "%06d.png")
    common = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        source,
    ]
    subprocess.run(
        common
        + [
            "-frames:v",
            str(count),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output / "preview.mp4"),
        ],
        check=True,
    )
    subprocess.run(
        common
        + [
            "-frames:v",
            str(count),
            "-filter_complex",
            "[0:v]fps=10,scale=720:-2:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
            str(output / "preview.gif"),
        ],
        check=True,
    )


def main() -> None:
    """Load the saved native policy and show a fixed seeded episode sequence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument(
        "--gym-config",
        type=Path,
        help="Explicit task/asset transfer deployment; old checkpoint assessment does not qualify the replacement.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--renderer", choices=("hybrid", "fast-rt"), default="hybrid")
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--terminal-hold", type=float, default=0.4)
    parser.add_argument("--hold-seconds", type=float, default=2)
    parser.add_argument("--eye", type=float, nargs=3, default=(0.78, -0.45, 0.73))
    parser.add_argument("--target", type=float, nargs=3, default=(0.515, -0.113, 0.46))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.fps <= 0 or args.width < 2 or args.width % 2:
        parser.error("Positive episodes/fps and an even width are required.")
    if args.terminal_hold < 0 or args.hold_seconds < 0:
        parser.error("Hold durations must be nonnegative.")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required for the native preview recordings.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory to preserve earlier recordings.")
    os.environ.setdefault("DISPLAY", ":1")
    output = args.output.resolve()
    (output / "frames").mkdir(parents=True, exist_ok=True)

    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.lab.sim.sensors import CameraCfg
    from embodichain.learning.rl.evaluation import (
        infer_policy_action,
        convert_policy_action_for_env,
    )
    from embodichain.learning.rl.policy_evaluation.cli import (
        _build_native_runtime,
        _flush_simulator,
        _resolve_input,
        _validate_native_options,
        parse_args,
    )

    native_args = parse_args([str(args.run), "--viewer"])
    vars(native_args).update(
        checkpoint=args.checkpoint,
        renderer=args.renderer,
        seed=args.seed,
        device=args.device,
        gym_config=str(args.gym_config.resolve()) if args.gym_config else None,
    )
    trace: dict[str, Any] = {
        "status": "initializing",
        "seed_sequence": list(range(args.seed, args.seed + args.episodes)),
        "episodes": [],
        "samples": [],
        "frames": [],
        "capture_source": "DexSim native RGB camera sharing the window viewpoint; no matplotlib renderer",
        "scope": "fixed seeded visual preview, all episode outcomes included; not a success-rate qualification",
        "timing_scope": "episode_simulation_seconds is control_step multiplied by configured control_dt, not a measured native clock. Camera capture publishes/renders without requesting physics steps.",
        "action_scope": "policy_action is the deterministic network output; applied_arm_action is the manager's clipped dimensionless request after warmup masking; arm_position_target is the processed joint-position command, not measured torque.",
    }
    native = camera = None
    previous_auto_reset = None
    count = 0
    try:
        resolved = _resolve_input(native_args)
        if args.checkpoint == "best" and resolved.selected_checkpoint != "best":
            raise RuntimeError(
                "A validation-selected best checkpoint is required; latest fallback is disabled."
            )
        trace["checkpoint"] = {
            "path": str(resolved.checkpoint),
            "sha256": hashlib.sha256(resolved.checkpoint.read_bytes()).hexdigest(),
            "selected": resolved.selected_checkpoint,
        }
        discover_task_packages()
        execute_init_hooks()
        _validate_native_options(native_args)
        native = _build_native_runtime(native_args, resolved, viewer=True)
        native.runtime.policy.eval()
        env = native.runtime.env
        base = getattr(env, "unwrapped", env)
        previous_auto_reset = getattr(base, "_demo_no_auto_reset", False)
        base._demo_no_auto_reset = True
        sim = base.sim
        if not sim.open_window():
            raise RuntimeError("DexSim native window did not open.")
        window = sim.get_world().get_windows()
        window.set_look_at(
            eye=np.asarray(args.eye, dtype=np.float32),
            look_at=np.asarray(args.target, dtype=np.float32),
            up=np.asarray([0, 0, 1], dtype=np.float32),
        )
        height = max(
            2, round(args.width * sim.sim_config.height / sim.sim_config.width / 2) * 2
        )
        focal = height * 28 / 24
        camera = sim.add_sensor(
            sensor_cfg=CameraCfg(
                uid="learned_assembly_preview",
                width=args.width,
                height=height,
                near=0.01,
                far=100,
                intrinsics=(focal, focal, args.width / 2, height / 2),
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=tuple(args.eye), target=tuple(args.target), up=(0.0, 0.0, 1.0)
                ),
            )
        )
        stride = max(1, round(1 / (args.fps * base.step_dt)))
        fps = 1 / (stride * base.step_dt)
        trace.update(
            camera={
                "eye": args.eye,
                "target": args.target,
                "width": args.width,
                "height": height,
                "intrinsics": [focal, focal, args.width / 2, height / 2],
                "projection_scope": "explicit native sensor intrinsics; the window lens/projection is not locked or measured by this example",
            },
            effective_fps=fps,
            configured_physics_dt=base.physics_dt,
            configured_control_dt=base.step_dt,
            fidelity=getattr(base, "assembly_fidelity", {}),
        )

        def capture(episode: int, step: int, phase: str) -> int:
            nonlocal count
            before = int(base._elapsed_steps[0])
            sim.sync_render_state()
            for _ in range(4 if count == 0 else 1):
                camera.update()
            rgb = camera.get_data()["color"][0, :, :, :3].detach().cpu().numpy().copy()
            if int(base._elapsed_steps[0]) != before:
                raise RuntimeError("Camera capture changed the control-step counter.")
            Image.fromarray(rgb).save(output / "frames" / f"{count:06d}.png")
            trace["frames"].append(
                {
                    "frame": count,
                    "episode": episode,
                    "control_step": step,
                    "episode_simulation_seconds": step * base.step_dt,
                    "phase": phase,
                }
            )
            count += 1
            return count - 1

        contacts = []
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            first = capture(episode, 0, "reset")
            reward_sum = 0.0
            started = time.perf_counter()
            for step in range(1, base.max_episode_steps + 1):
                if not sim.get_world().is_window_initialized():
                    raise RuntimeError(
                        "Native window closed before the fixed episode sequence finished."
                    )
                with torch.no_grad():
                    action = infer_policy_action(
                        native.runtime.policy,
                        observation,
                        device=native.device,
                        num_envs=1,
                    )
                    observation, reward, terminated, truncated, info = env.step(
                        convert_policy_action_for_env(env, action)
                    )
                reward_sum += float(torch.as_tensor(reward).sum())
                done = bool((terminated | truncated).any())
                trace["samples"].append(
                    {
                        "episode": episode,
                        "control_step": step,
                        "episode_simulation_seconds": step * base.step_dt,
                        "policy_action": _plain(action[0]),
                        "applied_arm_action": _plain(
                            base.action_manager.get_term("arm_delta").raw_actions[0]
                        ),
                        "arm_position_target": _plain(
                            base.action_manager.get_term("arm_delta").processed_actions[
                                0
                            ]
                        ),
                        "qpos": _plain(base.robot.get_qpos(name="arm")[0]),
                        "metrics": _plain(info.get("metrics", {})),
                    }
                )
                if step % stride == 0 or done:
                    last = capture(episode, step, "terminal" if done else "policy")
                window.set_window_title(
                    f"Learned blind peg | episode {episode+1}/{args.episodes} | t={step*base.step_dt:.2f}s"
                )
                time.sleep(max(0, started + step * base.step_dt - time.perf_counter()))
                if done:
                    trace["episodes"].append(
                        {
                            "episode": episode,
                            "seed": args.seed + episode,
                            "steps": step,
                            "simulation_seconds": step * base.step_dt,
                            "reward": reward_sum,
                            "success": bool(torch.as_tensor(info["success"]).any()),
                            "fail": bool(
                                torch.as_tensor(info.get("fail", False)).any()
                            ),
                            "truncated": bool(torch.as_tensor(truncated).any()),
                            "terminal_metrics": _plain(info.get("metrics", {})),
                            "first_frame": first,
                            "terminal_frame": last,
                        }
                    )
                    contacts.extend([first, last])
                    for _ in range(round(args.terminal_hold * fps)):
                        capture(episode, step, "terminal_hold_no_physics")
                        time.sleep(1 / fps)
                    break
            else:
                raise RuntimeError(
                    "Task episode limit did not produce a terminal result."
                )
        time.sleep(args.hold_seconds)
        trace["status"] = "completed"
    except Exception:
        trace["status"] = "failed"
        trace["error"] = traceback.format_exc()
        raise
    finally:
        trace["native_rgb_frames"] = count
        (output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        try:
            if native is not None:
                getattr(
                    native.runtime.env, "unwrapped", native.runtime.env
                )._demo_no_auto_reset = previous_auto_reset
                native.runtime.close()
        finally:
            _flush_simulator(True)
    _encode(output, fps, count)
    tiles = [
        Image.open(output / "frames" / f"{index:06d}.png").convert("RGB")
        for index in contacts
    ]
    thumb_height = round(320 * height / args.width)
    sheet = Image.new("RGB", (640, thumb_height * args.episodes))
    for index, tile in enumerate(tiles):
        sheet.paste(
            tile.resize((320, thumb_height)),
            ((index % 2) * 320, (index // 2) * thumb_height),
        )
    sheet.save(output / "contact_sheet.png")
    print(f"Native learned-policy preview: {output / 'preview.mp4'}", flush=True)


if __name__ == "__main__":
    main()
