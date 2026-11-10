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

"""Render a scanned background, a mesh arm and Gaussian body/link appearances.

Run from the repository root with a qualified DexSim integration build::

    python examples/sim/gaussian_visuals.py --background /assets/room.ply

Without --background a synthetic Gaussian cube is used. Physics and semantic
geometry are independent of the captured appearance. Outputs include a camera
mosaic and per-frame capture metadata. No Gaussian USD import is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    DenoisingCfg,
    GaussianRenderCfg,
    GaussianSplatCfg,
    RenderCfg,
    RigidObjectCfg,
    VisualElementCfg,
    VisualObjectCfg,
)
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.visualization.cfg import VisualizationCfg

__all__ = ["main"]


def write_cube(path: Path) -> None:
    """Write a small captured-color surface fixture in native Gaussian PLY encoding."""
    axis = np.linspace(-0.125, 0.125, 10)
    positions = []
    for fixed in range(3):
        for sign in [-1, 1]:
            for u in axis:
                for v in axis:
                    point = [0.0, 0.0, 0.0]
                    point[fixed] = sign * 0.125
                    point[(fixed + 1) % 3] = u
                    point[(fixed + 2) % 3] = v
                    positions.append(point)
    fields = "x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3".split()
    with path.open("w") as stream:
        stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(positions)}\n")
        stream.write(
            "".join(f"property float {field}\n" for field in fields) + "end_header\n"
        )
        for point in positions:
            values = [*point, 1.4, -1.1, -1.1, 6, *([math.log(0.025)] * 3), 1, 0, 0, 0]
            stream.write(" ".join(map(str, values)) + "\n")


def write_arm(path: Path) -> None:
    path.write_text("""<robot name="arm">
<link name="base"><inertial><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial><visual><geometry><box size="0.2 0.2 0.3"/></geometry></visual><collision><geometry><box size="0.2 0.2 0.3"/></geometry></collision></link>
<link name="tool"><inertial><mass value="0.2"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial><visual><geometry><box size="0.1 0.1 0.4"/></geometry></visual><collision><geometry><box size="0.1 0.1 0.4"/></geometry></collision></link>
<joint name="hinge" type="revolute"><parent link="base"/><child link="tool"/><origin xyz="0 0 0.3"/><axis xyz="0 1 0"/><limit lower="-1.5" upper="1.5" effort="20" velocity="2"/></joint>
</robot>""")


def serializable(value):
    """Convert owned capture metadata into JSON-compatible values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def main() -> None:
    """Run the four-environment appearance/lifecycle demonstration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", type=Path)
    parser.add_argument(
        "--background-rotation", default="0,0,0", help="XYZ Euler degrees"
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument(
        "--renderer", choices=["hybrid", "fast-rt", "rt"], default="fast-rt"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/gaussian_visuals"))
    args = parser.parse_args()
    if args.frames < 3 or args.num_envs < 1:
        parser.error("frames >= 3 and num-envs >= 1 are required")
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="embodichain-gaussian-") as directory:
        cube = Path(directory) / "cube.ply"
        write_cube(cube)
        urdf = Path(directory) / "arm.urdf"
        write_arm(urdf)
        sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                visualization=VisualizationCfg(backend="none"),
                num_envs=args.num_envs,
                device="cpu",
                physics_dt=1 / 60,
                enable_entity_gizmo=False,
                startup_summary="off",
                render_cfg=RenderCfg(
                    renderer=args.renderer,
                    spp=16,
                    gaussian=GaussianRenderCfg(enabled=True),
                    denoising=DenoisingCfg(window="off", offscreen="optix"),
                ),
            )
        )
        room = body = arm = camera = None
        try:
            rotation = np.eye(4, dtype=np.float32)
            rotation[:3, :3] = Rotation.from_euler(
                "xyz",
                [float(x) for x in args.background_rotation.split(",")],
                degrees=True,
            ).as_matrix()
            room = sim.add_visual_object(
                VisualObjectCfg(
                    uid="background",
                    init_local_pose=rotation,
                    visuals=[
                        VisualElementCfg(
                            representation=GaussianSplatCfg(
                                file_path=str(args.background or cube)
                            )
                        )
                    ],
                )
            )
            appearance = VisualElementCfg(
                representation=GaussianSplatCfg(file_path=str(cube))
            )
            body = sim.add_rigid_object(
                RigidObjectCfg(
                    uid="cube",
                    init_pos=(0.4, 0, 1.4),
                    shape=CubeCfg(size=(0.25,) * 3),
                    visuals=[appearance],
                )
            )
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid="table",
                    body_type="static",
                    init_pos=(0, 0, 0.45),
                    shape=CubeCfg(size=(1.7, 1, 0.1)),
                )
            )
            local = np.eye(4, dtype=np.float32)
            local[:3, 3] = (0.08, 0, 0.18)
            arm = sim.add_articulation(
                ArticulationCfg(
                    uid="arm",
                    fpath=str(urdf),
                    build_pk_chain=False,
                    init_pos=(-0.5, 0, 0.65),
                    link_visuals={
                        "tool": [
                            VisualElementCfg(
                                id="scan",
                                representation=appearance.representation,
                                local_pose=local,
                            ),
                            VisualElementCfg(
                                id="mesh",
                                representation=CubeCfg(size=(0.08, 0.08, 0.35)),
                            ),
                        ]
                    },
                )
            )
            camera = sim.add_sensor(
                CameraCfg(
                    uid="camera",
                    width=320,
                    height=240,
                    intrinsics=(220, 220, 160, 120),
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=(0, -3.5, 1.8), target=(0, 0.4, 0.9)
                    ),
                    enable_depth=True,
                )
            )
            sim.prepare()
            records = []
            for frame in range(args.frames):
                arm.set_qpos(
                    torch.full((args.num_envs, 1), 0.7 * math.sin(frame / 15)),
                    target=False,
                )
                if frame == args.frames // 3 and args.num_envs > 1:
                    body.reset([1])
                    room.set_visible(False, [1])
                    body.set_gaussian_visual_state(
                        "appearance",
                        [1],
                        color_gain=(0.5, 1.3, 0.7),
                        opacity_gain=0.75,
                        scale=0.85,
                    )
                    arm.set_gaussian_visual_state(
                        "scan", [1], link_name="tool", color_gain=(1.3, 0.5, 0.5)
                    )
                if frame == 2 * args.frames // 3 and args.num_envs > 1:
                    room.reset([1])
                    body.reset([1])
                    arm.reset([1])
                sim.update(1 / 60, 1, render_final_step=False)
                with sim.render_frame():
                    sim.render_camera_group([camera.group_id])
                    camera.update(fetch_only=True)
                record = serializable(camera.capture_metadata)
                record["body_pose"] = serializable(body.get_local_pose())
                record["arm_qpos"] = serializable(arm.get_qpos())
                records.append(record)
                if frame in {0, args.frames // 3, args.frames - 1}:
                    images = camera.get_data()["color"][..., :3].detach().cpu().numpy()
                    Image.fromarray(np.concatenate(list(images), axis=1)).save(
                        args.output / f"frame_{frame:04}.png"
                    )
            (args.output / "captures.json").write_text(
                json.dumps(records, indent=2, allow_nan=False)
            )
            print(
                f"Saved {args.frames} capture records and mosaics to {args.output.resolve()}",
                flush=True,
            )
        finally:
            room = body = arm = camera = None
            sim.destroy(exit_process=False)
            sim.flush_cleanup_queue()


if __name__ == "__main__":
    main()
