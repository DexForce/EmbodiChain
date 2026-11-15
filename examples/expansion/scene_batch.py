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

"""Compare four native environments before preparation, execution and restoration.

Run ``embodichain run-example expansion/scene_batch --output /tmp/scene-batch``.
The example uses the existing UR5 Task Program and Default physical environment.
Row 0 is an unselected reference; rows 3 and 1 execute; row 2 is rejected by a
measured initial-region check. RGB is captured by a native diagnostic camera.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.sim.sensors import Camera

_SELECTED = (3, 1, 2)
_REGION_LOWER = np.array([-0.55, -0.20])
_REGION_UPPER = np.array([-0.28, 0.12])
_GOAL = np.array([-0.40, 0.48, 0.025])
_WIDTH, _HEIGHT, _BAR = 480, 360, 32
_STAGE_NAMES = {
    "before": "BEFORE PREPARATION",
    "prepared": "PREPARED SCENES",
    "executing": "TASK PROGRAM EXECUTION",
    "executed": "MEASURED OUTCOME",
    "restored": "RESTORED INITIAL STATES",
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _color(status: str) -> str:
    if "rejected" in status or "invalid" in status:
        return "#ff9b8f"
    if "accepted" in status or "restored" in status:
        return "#8ee3bc"
    return "#cbd5e1"


class _Preview:
    """Stream diagnostic RGB while retaining bounded measured state samples."""

    def __init__(
        self, env: EmbodiedEnv, camera: Camera, output: Path, stride: int
    ) -> None:
        self.env, self.camera, self.output, self.stride = env, camera, output, stride
        self.steps = 0
        self.fps = 1.0 / (env.step_dt * stride)
        self.writer = imageio.get_writer(
            output / "preview.mp4", fps=self.fps, codec="libx264", quality=8
        )
        self.status = ["reference / shared physics", "selected", "selected", "selected"]
        self.keys: dict[str, np.ndarray] = {}
        self.phases: dict[str, dict[str, Any]] = {}
        self.samples: list[tuple[float, np.ndarray, np.ndarray]] = []

    def capture(self, stage: str, *, keyframe: bool = False) -> None:
        clock = self.env.sim.simulation_time
        with self.env.sim.render_frame(capture_camera_images=False):
            self.camera.update()
            rgb = self.camera.get_data()["color"][..., :3].detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * (255 if rgb.max() <= 1 else 1), 0, 255)
            rgb = np.array(rgb, dtype=np.uint8, copy=True)
        if self.env.sim.simulation_time != clock:
            raise RuntimeError("Diagnostic rendering advanced physics")
        positions = (
            self.env.sim.get_rigid_object("cube")
            .get_local_pose(to_matrix=True)[:, :3, 3]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        qpos = self.env.robot.get_qpos().detach().cpu().numpy().copy()
        self.samples.append((clock, positions, qpos))
        frame = Image.new("RGB", (2 * _WIDTH, 64 + 2 * (_HEIGHT + _BAR)), "#111827")
        draw = ImageDraw.Draw(frame)
        draw.text((18, 10), _STAGE_NAMES[stage], fill="white", font=_font(22))
        draw.text(
            (18, 38),
            f"native RGB | simulation time {clock:.3f} s | one shared World clock",
            fill="#cbd5e1",
            font=_font(14),
        )
        for row in range(4):
            x, y = row % 2 * _WIDTH, 64 + row // 2 * (_HEIGHT + _BAR)
            draw.text(
                (x + 10, y + 7),
                f"ENV {row}  |  {self.status[row]}",
                fill=_color(self.status[row]),
                font=_font(16),
            )
            frame.paste(Image.fromarray(rgb[row]), (x, y + _BAR))
        copies = max(1, round(self.fps)) if keyframe else 1
        for _ in range(copies):
            self.writer.append_data(np.asarray(frame))
        if keyframe:
            self.keys[stage] = rgb
            frame.save(self.output / f"{stage}.png")
            self.phases[stage] = {
                "simulation_time_s": clock,
                "cube_xyz_m": positions.tolist(),
                "status": list(self.status),
            }
            print(f"[scene-batch] {stage}: {self.status}", flush=True)

    def observe(self) -> None:
        self.steps += 1
        if self.steps % self.stride == 0:
            self.capture("executing")

    def close(self) -> None:
        self.writer.close()

    def comparison(self) -> None:
        # Show one accepted row and the rejected row at the same native view.
        stages = ("prepared", "executed", "restored")
        rows = (1, 2)
        canvas = Image.new("RGB", (3 * _WIDTH, 70 + 2 * (_HEIGHT + 52)), "#111827")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (18, 12), "MEASURED BEFORE / AFTER / RESTORED", fill="white", font=_font(26)
        )
        draw.text(
            (18, 46),
            "Native camera images; the rejected row receives no task actions.",
            fill="#cbd5e1",
            font=_font(16),
        )
        for y_index, row in enumerate(rows):
            for x_index, stage in enumerate(stages):
                x, y = x_index * _WIDTH, 70 + y_index * (_HEIGHT + 52)
                status = self.phases[stage]["status"][row]
                draw.text(
                    (x + 10, y + 5),
                    f"ENV {row} | {stage.upper()}",
                    fill="white",
                    font=_font(17),
                )
                draw.text((x + 10, y + 29), status, fill=_color(status), font=_font(15))
                canvas.paste(Image.fromarray(self.keys[stage][row]), (x, y + 52))
        canvas.save(self.output / "comparison.png")
        np.savez_compressed(
            self.output / "measured_states.npz",
            simulation_time_s=np.array([sample[0] for sample in self.samples]),
            cube_xyz_m=np.stack([sample[1] for sample in self.samples]),
            qpos=np.stack([sample[2] for sample in self.samples]),
        )
        (self.output / "index.html").write_text(
            """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>四环境真实仿真对照</title>
<style>body{margin:32px auto;max-width:1440px;padding:0 20px;background:#111827;color:#e5e7eb;font:16px/1.7 system-ui}img,video{width:100%;border-radius:8px}video{max-height:75vh}a{color:#93c5fd}p{max-width:1000px}</style>
<h1>四环境真实仿真对照</h1>
<p>环境 0 是未选参照；1、3 执行抓放；2 的初始位置超出规定区域，被拒绝。
所有画面均来自原生仿真相机。环境共享物理时钟，因此未执行任务的行仍会有动力学响应。</p>
<video controls preload="metadata" poster="executed.png" src="preview.mp4"></video>
<p>关键阶段的视频停留仅供观察，不推进物理。恢复校验针对物体、关节和控制状态，仿真时钟不回退。
这些文件是诊断预览；保存资格在恢复清空旧缓冲之前记录，未写入训练数据集。</p>
<img src="comparison.png" alt="环境1与环境2在准备后、执行后和恢复后的原生相机对比">
<p><a href="before.png">准备前 · 四环境</a> · <a href="prepared.png">准备后 · 四环境</a> ·
<a href="executed.png">执行后 · 四环境</a> · <a href="restored.png">恢复后 · 四环境</a> ·
<a href="report.json">完整实测报告</a> · <a href="measured_states.npz">实测轨迹</a></p>
</html>""",
            encoding="utf-8",
        )


def _headless_video() -> None:
    # SDL is required by the native offscreen renderer even without a window.
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if "XDG_RUNTIME_DIR" not in os.environ:
        os.environ["XDG_RUNTIME_DIR"] = tempfile.mkdtemp(prefix="embodichain-xdg-")


def _run(output: Path, *, seed: int, stride: int) -> dict[str, Any]:
    _headless_video()
    import torch

    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    from embodichain.lab.sim import SimulationManager, VisualMaterialCfg
    from embodichain.lab.sim.scene import SceneView, capture_state, compare_states
    from embodichain.lab.sim.scene.expansion import ScenePoseChange, SceneVariant
    from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
    from embodichain.lab.sim.sensors import CameraCfg
    from embodichain.lab.task_program.integrations.scene_expansion import (
        execute_scene_variants,
    )
    from embodichain.lab.task_program.integrations.simulation.scene_expansion import (
        SimulationSceneExpansionHost,
    )
    from embodichain.utils.utility import load_config

    deployment = Path(__file__).resolve().parent / "engine/task.default.yaml"
    raw = load_config(deployment)
    cfg = config_to_cfg(raw, source_path=deployment)
    cfg.num_envs = cfg.sim_cfg.num_envs = 4
    cfg.seed = seed
    cfg.sim_cfg.device = "cpu"
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.render_cfg.renderer = "hybrid"
    cfg.task_program.program.count = 1
    cfg.sensor, cfg.enable_sensor = [], False
    cfg.dataset = None
    cfg.init_rollout_buffer = cfg.record_trajectory = cfg.trajectory_auto_save = False
    env = preview = None
    report: dict[str, Any] = {"selected_env_ids": list(_SELECTED), "seed": seed}
    try:
        env = REGISTERED_ENVS[raw["id"]].make(cfg=cfg)
        env.reset(seed=seed)
        cube = env.sim.get_rigid_object("cube")
        cube.set_visual_material(
            env.sim.create_visual_material(
                VisualMaterialCfg(
                    uid="preview_cube",
                    base_color=[1.0, 0.25, 0.02, 1.0],
                    roughness=0.45,
                )
            ),
            shared=False,
            update_default=True,
        )
        camera = env.sim.add_sensor(
            CameraCfg(
                uid="diagnostic_camera",
                width=_WIDTH,
                height=_HEIGHT,
                intrinsics=(520.0, 520.0, _WIDTH / 2, _HEIGHT / 2),
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=(-1.05, -1.3, 1.1),
                    target=(-0.08, 0.18, 0.20),
                    up=(0.0, 0.0, 1.0),
                ),
            )
        )
        preview = _Preview(env, camera, output, stride)
        preview.capture("before", keyframe=True)
        reference = SceneView(env.sim, env_ids=(0,))
        before_reference = capture_state(reference)

        def initial(row: int) -> ValidationResult:
            xy = cube.get_local_pose(to_matrix=True)[row, :2, 3].detach().cpu().numpy()
            valid = bool(
                np.isfinite(xy).all()
                and np.all(xy >= _REGION_LOWER)
                and np.all(xy <= _REGION_UPPER)
            )
            return ValidationResult(
                (
                    ValidationCheck(
                        "initial.start_region",
                        "passed" if valid else "failed",
                        "Measured cube origin inside the declared initial XY region",
                        {"x_m": float(xy[0]), "y_m": float(xy[1])},
                    ),
                )
            )

        positions = {1: (-0.40, -0.05), 2: (0.50, 0.30), 3: (-0.44, -0.10)}
        variants = {}
        for ordinal, row in enumerate(_SELECTED):
            pose = cube.get_local_pose(to_matrix=True)[row].clone()
            pose[:2, 3] = torch.tensor(positions[row], device=pose.device)
            variants[row] = SceneVariant(
                "scene-batch-example",
                seed,
                ordinal,
                (ScenePoseChange("cube", pose.tolist()),),
            )
        host = SimulationSceneExpansionHost(
            env, env_ids=_SELECTED, initial_validator=initial
        )

        def prepare(rows):
            result = host(rows)
            for row, value in result.items():
                preview.status[row] = (
                    "ready"
                    if value.validation.accepted
                    else "rejected: outside initial region"
                )
            report["reference_unchanged_by_preparation"] = compare_states(
                before_reference, capture_state(reference)
            ).passed
            preview.capture("prepared", keyframe=True)
            for row, value in result.items():
                if value.validation.accepted:
                    preview.status[row] = "executing Task Program"
            return result

        def measured(row: int) -> ValidationResult:
            xyz = cube.get_local_pose(to_matrix=True)[row, :3, 3].detach().cpu().numpy()
            xy_error = float(np.linalg.norm(xyz[:2] - _GOAL[:2]))
            z_error = float(abs(xyz[2] - _GOAL[2]))
            valid = xy_error < 0.06 and z_error < 0.04
            return ValidationResult(
                (
                    ValidationCheck(
                        "placement.measured",
                        "passed" if valid else "failed",
                        metrics={"xy_error_m": xy_error, "z_error_m": z_error},
                    ),
                )
            )

        original_step = env.step

        def observe_step(action):
            result = original_step(action)
            preview.observe()
            return result

        env.step = observe_step
        try:
            results = execute_scene_variants(
                env, variants, prepare_scene=prepare, measured_validator=measured
            )
        finally:
            env.step = original_step
        for row, value in results.items():
            preview.status[row] = (
                "accepted / eligible to save"
                if value.accepted
                else "rejected / no recording credit"
            )
        eligible = env.episode_context.filter_recordable(_SELECTED).cpu().tolist()
        preview.capture("executed", keyframe=True)
        snapshots = host.initial_states
        before_restore_reference = capture_state(reference)
        clock = env.sim.simulation_time
        restored = host.restore_initial_states(snapshots)
        restore_checks = {
            row: any(
                check.status == "passed" and check.check_id == "scene.restore_state"
                for check in value.validation.checks
            )
            for row, value in restored.items()
        }
        for row in _SELECTED:
            preview.status[row] = (
                "restored" if row != 2 else "restored / initial region still invalid"
            )
        preview.capture("restored", keyframe=True)
        report.update(
            rows={str(row): value.to_metadata() for row, value in results.items()},
            recording_eligible_env_ids=eligible,
            restored_rows=restore_checks,
            reference_unchanged_by_restoration=compare_states(
                before_restore_reference, capture_state(reference)
            ).passed,
            restoration_did_not_step=env.sim.simulation_time == clock,
            control_steps=preview.steps,
            preview_fps=preview.fps,
            phases=preview.phases,
            initial_region_xy_m=[_REGION_LOWER.tolist(), _REGION_UPPER.tolist()],
        )
        expected = (
            results[1].accepted
            and results[3].accepted
            and results[2].execution is None
            and set(eligible) == {1, 3}
        )
        report["passed"] = bool(
            expected
            and all(restore_checks.values())
            and report["reference_unchanged_by_preparation"]
            and report["reference_unchanged_by_restoration"]
            and report["restoration_did_not_step"]
        )
        preview.close()
        preview.comparison()
        if not report["passed"]:
            raise RuntimeError("Measured comparison checks failed; inspect report.json")
        return report
    finally:
        if preview is not None:
            preview.close()
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        if env is not None:
            env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def main() -> None:
    """Run a bounded native comparison and save local diagnostic artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/scene_batch"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Capture one video frame per N control steps",
    )
    args = parser.parse_args()
    if args.seed < 0 or args.stride < 1:
        parser.error("seed must be nonnegative and stride must be positive")
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output must be empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    report = _run(output, seed=args.seed, stride=args.stride)
    print(json.dumps({"passed": report["passed"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
