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
"""Run ordinary and functional stepping through public EmbodiChain/Scene APIs."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from dexsim.engine.newton_physics import ProviderNewtonSolverCfg

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    CollisionPropertiesCfg,
    NewtonPhysicsCfg,
    RenderCfg,
    RigidBodyPhysicsCfg,
)
from embodichain.lab.sim.sensors import Camera, CameraCfg
from embodichain.lab.visualization import VisualizationCfg, ViserServerCfg
from solver import SimpleSolverProvider


def _save_preview(
    positions: list[torch.Tensor], reset_position: torch.Tensor, path: Path
) -> None:
    """Plot the measured functional trajectory and selected-arena reset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = torch.stack(positions).detach().cpu().numpy()
    reset_values = reset_position.detach().cpu().numpy()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for arena_id, color in enumerate(("#2563eb", "#ea580c")):
        axis.plot(
            range(len(values)),
            values[:, arena_id],
            marker="o",
            color=color,
            label=f"arena_{arena_id}",
        )
        axis.plot(
            (len(values) - 1, len(values)),
            (values[-1, arena_id], reset_values[arena_id]),
            linestyle="--",
            marker="o",
            color=color,
        )
    axis.set_xticks(range(len(values) + 1))
    axis.set_xticklabels([*(str(step) for step in range(len(values))), "reset"])
    axis.set_xlabel("Functional transition")
    axis.set_ylabel("Joint position (m)")
    axis.set_title("Spring-slider tutorial: rollout and selected-arena reset")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _capture_rgb(sim: SimulationManager, camera: Camera) -> np.ndarray:
    """Read the two native offscreen views through EmbodiChain's camera API."""
    sim.render_camera_group([camera.group_id])
    camera.update(fetch_only=True)
    return camera.get_data()["color"][..., :3].cpu().numpy().copy()


def _save_offscreen_preview(before: np.ndarray, after: np.ndarray, path: Path) -> None:
    """Save initial and replayed RGB frames for both environments."""
    from PIL import Image, ImageDraw, ImageFont

    height, width = before.shape[1:3]
    image = Image.new("RGB", (2 * width, 2 * height))
    for row, frames in enumerate((before, after)):
        for env_id in range(2):
            image.paste(Image.fromarray(frames[env_id]), (env_id * width, row * height))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for row, phase in enumerate(("Initial", "After 30 steps")):
        for env_id in range(2):
            x, y = env_id * width, row * height
            draw.rectangle((x, y, x + width - 1, y + 32), fill="white")
            draw.text(
                (x + 10, y + 5),
                f"{phase} - environment {env_id}",
                fill="black",
                font=font,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _start_video_encoder(path: Path, width: int, height: int) -> subprocess.Popen:
    """Start an H.264 encoder for the two side-by-side offscreen views."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{2 * width}x{height}",
                "-framerate",
                "10",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("--video-path requires ffmpeg on PATH") from exc


def _write_video_frame(encoder: subprocess.Popen, rgb: np.ndarray) -> None:
    """Append one RGB frame per environment without retaining the video in RAM."""
    combined = np.ascontiguousarray(np.concatenate((rgb[0], rgb[1]), axis=1))
    encoder.stdin.write(combined.tobytes())


def _finish_video_encoder(encoder: subprocess.Popen) -> None:
    """Flush the encoder and report an unsuccessful conversion."""
    encoder.stdin.close()
    error = encoder.stderr.read().decode(errors="replace")
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg could not encode the tutorial video: {error}")


def run(
    device: str,
    preview_path: Path | None = None,
    *,
    render_path: Path | None = None,
    video_path: Path | None = None,
    viser: bool = False,
    viser_port: int = 8080,
    viser_seconds: float = 30.0,
) -> dict[str, object]:
    """Demonstrate direct provider creation, parity, gradients and partial reset."""
    if viser_seconds < 0:
        raise ValueError("viser_seconds must be non-negative")
    config = SimulationManagerCfg(
        headless=True,
        render_cfg=RenderCfg(
            renderer=(
                "fast-rt"
                if render_path is not None or video_path is not None
                else "auto"
            )
        ),
        visualization=VisualizationCfg(
            backend="viser" if viser else "none",
            env_ids=[0, 1],
            viser_server=ViserServerCfg(port=viser_port),
        ),
        num_envs=2,
        enable_entity_gizmo=False,
        startup_summary="off",
        physics_cfg=NewtonPhysicsCfg(
            device=device,
            gravity=(0.0, 0.0, 0.0),
            physics_dt=0.02,
            num_substeps=2,
            requires_grad=True,
            use_cuda_graph=False,
            collision_cfg=None,
            solver_cfg=ProviderNewtonSolverCfg(
                provider_factory=SimpleSolverProvider,
                options={"stiffness": 2.0, "damping": 0.2},
            ),
        ),
    )
    sim = SimulationManager(
        config,
        requires_native_renderer=render_path is not None or video_path is not None,
    )
    try:
        slider = sim.add_articulation(
            ArticulationCfg(
                uid="slider",
                fpath=str(Path(__file__).with_name("slider.urdf")),
                asset_physics_mode="overlay",
                attrs=RigidBodyPhysicsCfg(
                    collision_props=CollisionPropertiesCfg(collision_enabled=False),
                ),
            )
        )
        sim.prepare()
        camera = None
        if render_path is not None or video_path is not None:
            camera = sim.add_sensor(
                CameraCfg(
                    uid="slider_preview",
                    width=640,
                    height=480,
                    intrinsics=(600, 600, 320, 240),
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=(0.7, -0.8, 0.55),
                        target=(0.25, 0.0, 0.2),
                        up=(0.0, 0.0, 1.0),
                    ),
                    enable_color=True,
                )
            )
        if viser:
            manifest = sim.refresh_visualization()
            if not any(node.kind == "articulation_link" for node in manifest.nodes):
                raise RuntimeError("Viser scene has no slider visual geometry")
            sim.capture_visualization(force=True)
        scene = sim.spawn_result
        initial = scene.create_functional_state()
        params = scene.functional_parameters()
        force = torch.tensor([4.0, 6.0], device=device, requires_grad=True)

        predicted = scene.functional_step(initial, force, publish=False)
        torch.testing.assert_close(
            slider.get_qpos().flatten(), initial.physical["joint_q"]
        )
        slider.set_qf(force.detach().reshape(2, 1))
        sim.update(step=1)
        torch.testing.assert_close(
            slider.get_qpos().flatten(), predicted.physical["joint_q"]
        )

        trajectory = initial
        positions = [initial.physical["joint_q"]]
        for _ in range(6):
            trajectory = scene.functional_step(trajectory, force, publish=False)
            positions.append(trajectory.physical["joint_q"])
        saved = trajectory.physical["joint_q"].clone()
        published = scene.functional_step(trajectory, force, publish=True)
        positions.append(published.physical["joint_q"])
        torch.testing.assert_close(
            slider.get_qpos().flatten(), published.physical["joint_q"]
        )
        torch.testing.assert_close(trajectory.physical["joint_q"], saved)

        reset = scene.reset_functional_state(published, arenas=["arena_0"])
        torch.testing.assert_close(
            reset.physical["joint_q"][0], initial.physical["joint_q"][0]
        )
        torch.testing.assert_close(
            reset.physical["joint_q"][1], published.physical["joint_q"][1]
        )
        if preview_path is not None:
            _save_preview(positions, reset.physical["joint_q"], preview_path)
        loss = (
            trajectory.physical["joint_q"].square().sum()
            + reset.physical["joint_q"].square().sum()
        )
        grad_force, grad_scale = torch.autograd.grad(
            loss, (force, params["force_scale"])
        )
        assert torch.isfinite(grad_force).all() and (grad_force.abs() > 0).all()
        assert torch.isfinite(grad_scale) and grad_scale.abs() > 0
        slider.reset(env_ids=[0])
        torch.testing.assert_close(
            slider.get_qpos()[0].flatten(), initial.physical["joint_q"][:1]
        )
        result = {
            "device": device,
            "native_functional_parity": True,
            "published_position": published.physical["joint_q"].detach().tolist(),
            "partial_reset_position": reset.physical["joint_q"].detach().tolist(),
            "force_gradient": grad_force.tolist(),
            "force_scale_gradient": grad_scale.item(),
        }
        if preview_path is not None:
            result["preview_path"] = str(preview_path)
        if viser or render_path is not None or video_path is not None:
            # Replay in the live EmbodiChain scene for Viser and/or native RGB.
            # The numerical and gradient checks above remain independent.
            slider.reset()
            slider.set_qf(force.detach().reshape(2, 1))
            initial_rgb = _capture_rgb(sim, camera) if camera is not None else None
            encoder = (
                _start_video_encoder(video_path, camera.cfg.width, camera.cfg.height)
                if video_path is not None
                else None
            )
            final_rgb = None
            try:
                if encoder is not None:
                    for _ in range(5):
                        _write_video_frame(encoder, initial_rgb)
                for _ in range(30):
                    sim.update(step=1)
                    if viser:
                        sim.capture_visualization(force=True)
                        time.sleep(0.05)
                    if encoder is not None:
                        final_rgb = _capture_rgb(sim, camera)
                        _write_video_frame(encoder, final_rgb)
                if encoder is not None:
                    for _ in range(5):
                        _write_video_frame(encoder, final_rgb)
            finally:
                if encoder is not None:
                    _finish_video_encoder(encoder)
            result["visualized_position"] = slider.get_qpos().flatten().tolist()
            if camera is not None:
                if render_path is not None:
                    if final_rgb is None:
                        final_rgb = _capture_rgb(sim, camera)
                    _save_offscreen_preview(initial_rgb, final_rgb, render_path)
                    result["render_path"] = str(render_path)
                if video_path is not None:
                    result["video_path"] = str(video_path)
        if viser:
            result["viser_url"] = f"http://127.0.0.1:{viser_port}"
            result["visualized_links"] = sum(
                node.kind == "articulation_link" for node in manifest.nodes
            )
            print(f"Viser preview: {result['viser_url']}", flush=True)
            time.sleep(viser_seconds)
        return result
    finally:
        sim.destroy(exit_process=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument(
        "--preview-path", type=Path, help="save the rollout and reset plot as PNG"
    )
    parser.add_argument(
        "--render-path",
        type=Path,
        help="save EmbodiChain offscreen RGB before and after replay",
    )
    parser.add_argument(
        "--video-path", type=Path, help="record a side-by-side offscreen MP4"
    )
    parser.add_argument(
        "--viser", action="store_true", help="replay in EmbodiChain Viser"
    )
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument(
        "--viser-seconds",
        type=float,
        default=30.0,
        help="seconds to keep the browser scene open after replay",
    )
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                run(
                    args.device,
                    args.preview_path,
                    render_path=args.render_path,
                    video_path=args.video_path,
                    viser=args.viser,
                    viser_port=args.viser_port,
                    viser_seconds=args.viser_seconds,
                ),
                indent=2,
            )
        )
    finally:
        SimulationManager.flush_cleanup_queue()
