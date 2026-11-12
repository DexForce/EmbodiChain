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

"""Render G2 with the W1 documentation's OfflineRT presentation settings."""

from __future__ import annotations

__all__ = []


def _main() -> None:
    """Render one full-body image to the caller-selected output path."""
    import os

    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    from pathlib import Path
    import argparse
    import hashlib
    from importlib.metadata import version
    import json
    import math
    from PIL import Image
    import torch
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg
    from embodichain.lab.sim.sensors import CameraCfg
    from embodichain.robot_registry import resolve_robot_cfg

    parser = argparse.ArgumentParser(
        description="Render AgiBot G2 with the W1 documentation's OfflineRT settings."
    )
    parser.add_argument("--output", type=Path, default=Path("g2-render.png"))
    args = parser.parse_args()
    out = args.output.parent
    out.mkdir(parents=True, exist_ok=True)
    render_cfg = RenderCfg(renderer="rt")
    camera_cfg = CameraCfg(
        uid="preview_camera",
        width=960,
        height=1200,
        intrinsics=(2350.0, 2350.0, 480.0, 600.0),
        near=0.03,
        far=60.0,
        enable_color=True,
        enable_depth=False,
        extrinsics=CameraCfg.ExtrinsicsCfg(
            eye=(2.4, -3.2, 2.2), target=(0.06, 0.0, 0.94), up=(0.0, 0.0, 1.0)
        ),
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            render_cfg=render_cfg,
            startup_summary="off",
            robot_ik_gizmo=None,
        )
    )
    try:
        cfg = resolve_robot_cfg("AgibotG2")
        robot = sim.add_robot(cfg)
        camera = sim.add_sensor(camera_cfg)
        sim.prepare()
        sim.enable_physics(False)
        # G2's source axes differ from W1's. These mirrored source-joint poses
        # lower the upper arms and bend both elbows forward for the portrait.
        # They affect this capture only; the catalog home pose remains neutral.
        display_pose = {
            "left_arm": [math.pi / 2, -1.2, -math.pi / 2, -1.4, 0.0, 0.0, 0.0],
            "right_arm": [-math.pi / 2, -1.2, math.pi / 2, -1.4, 0.0, 0.0, 0.0],
        }
        for part, positions in display_pose.items():
            robot.set_qpos(
                torch.tensor([positions], device=robot.device), name=part, target=False
            )
        warmup_frames = 48
        for _ in range(warmup_frames):
            sim.render_camera_group([camera.group_id])
        camera.update(fetch_only=True)
        frame = camera.get_data()["color"][0].cpu().numpy()
        Image.fromarray(frame[:, :, :3]).save(args.output)
        lock = json.loads(Path(__file__).with_name("source.lock.json").read_text())
        manifest = {
            "description": "G2 catalog capture matching the W1 documentation's RT presentation.",
            "catalog_id": "AgibotG2",
            "render_cfg": render_cfg.to_dict(),
            "camera": {
                "width": camera_cfg.width,
                "height": camera_cfg.height,
                "intrinsics": camera_cfg.intrinsics,
                "eye": camera_cfg.extrinsics.eye,
                "target": camera_cfg.extrinsics.target,
                "up": camera_cfg.extrinsics.up,
                "near": camera_cfg.near,
                "far": camera_cfg.far,
            },
            "scene": {
                "presentation": "SimulationManager defaults",
                "direct_lights": [],
                "emission_color": [0.92, 0.95, 1.0],
                "emission_intensity": 80.0,
            },
            "physics_stepped": False,
            "warmup_frames": warmup_frames,
            "joint_positions": dict(
                zip(robot.joint_names, robot.get_qpos()[0].tolist())
            ),
            "root_position": cfg.init_pos,
            "source_materials_preserved": True,
            "asset": {
                key: lock[key]
                for key in ("repository", "revision", "archive_sha256", "urdf_sha256")
            },
            "runtime": {"dexsim": version("dexsim_engine"), "torch": torch.__version__},
            "preview_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "image_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        }
        args.output.with_suffix(".render.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(args.output, flush=True)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    _main()
