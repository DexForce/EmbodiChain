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

"""Show a curved versus line-constrained 45 cm approach to an alignment frame.

Run: python examples/sim/motion/planners/curobo_approach_showcase.py

Both plans share the same Franka, initial joints, target, collision world, native
seed resets and limits. The only planning difference is the line constraint.
The fixture is included in both collision worlds. Rendering uses native
EmbodiChain cameras with kinematic playback; no physical insertion is claimed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

START = (
    1.2554341555,
    -1.3694761992,
    -1.4561792612,
    -1.8368169069,
    0.2263653278,
    1.5393618345,
    0.8410000205,
)
GOAL_POSITION = (0.45, 0.25, 0.45)
GOAL_ROTATION = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
COLORS = {"before": (0.10, 0.45, 1.0, 1.0), "after": (1.0, 0.40, 0.04, 1.0)}
WIDTH, HEIGHT, FPS = 960, 540, 30

__all__ = ["main"]


def _scene(render: bool) -> tuple[Any, Any, list[Any]]:
    """Create the calibrated robot and a physical, collision-checked fixture."""
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg, RigidBodyPhysicsCfg, RigidObjectCfg
    from embodichain.lab.sim.material import VisualMaterialCfg
    from embodichain.lab.sim.shapes import CubeCfg
    from embodichain_benchmarks.motion_generation._atomic_fixtures import (
        _create_franka_native_hand_cfg,
    )

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device="cuda",
            width=WIDTH,
            height=HEIGHT,
            render_cfg=RenderCfg(
                renderer="hybrid" if render else "no-render",
                tone_mapping_enabled=True,
                tone_mapping_exposure=0.8,
            ),
            startup_summary="off",
        )
    )
    robot = sim.add_robot(
        cfg=_create_franka_native_hand_cfg(
            {
                "uid": "showcase_franka",
                "init_qpos": [*START, 0.04, 0.04],
            }
        )
    )
    fixture = []
    # Both paths are planned against all four pieces. The opening is wider
    # than the gripper; it does not itself force the entire free-air approach
    # to remain on a line.
    pieces = (
        ("left", (0.335, 0.30, 0.29), (0.035, 0.045, 0.58)),
        ("right", (0.565, 0.30, 0.29), (0.035, 0.045, 0.58)),
        ("top", (0.45, 0.30, 0.565), (0.195, 0.045, 0.035)),
        ("bottom", (0.45, 0.30, 0.335), (0.195, 0.045, 0.035)),
    )
    for name, position, size in pieces:
        fixture.append(
            sim.add_rigid_object(
                cfg=RigidObjectCfg(
                    uid=f"alignment_{name}",
                    body_type="static",
                    init_pos=position,
                    shape=CubeCfg(
                        size=list(size),
                        visual_material=VisualMaterialCfg(
                            uid=f"alignment_material_{name}",
                            base_color=[0.92, 0.62, 0.12, 1.0],
                            roughness=0.45,
                            metallic=0.2,
                        ),
                    ),
                    attrs=RigidBodyPhysicsCfg(),
                )
            )
        )
    sim.prepare()
    return sim, robot, fixture


def _plan(robot: Any, fixture: list[Any], output: Path) -> tuple[Any, dict, dict]:
    """Plan both real trajectories and independently score their TCP geometry."""
    import numpy as np
    import torch
    from embodichain.lab.sim.motion.motion_generator import (
        MotionGenCfg,
        MotionGenOptions,
        MotionGenerator,
    )
    from embodichain.lab.sim.motion.planners import (
        CuroboPlannerCfg,
        CuroboPlanOptions,
        CuroboWorldCfg,
        CuroboLinearConstraintCfg,
        MoveType,
        PlanState,
    )

    generator = MotionGenerator(
        MotionGenCfg(
            planner_cfg=CuroboPlannerCfg(
                robot_uid=robot.uid,
                world=CuroboWorldCfg(rigid_objects=fixture),
                preserve_plan_samples=True,
                use_cuda_graph=True,
                max_attempts=8,
            )
        )
    )
    goal = torch.eye(4, device=robot.device)[None]
    goal[0, :3, :3] = goal.new_tensor(GOAL_ROTATION)
    goal[0, :3, 3] = goal.new_tensor(GOAL_POSITION)
    start = robot.get_qpos(name="arm").clone()
    generator.planner.prepare_backend(
        control_part="arm", batch_size=1, move_type=MoveType.EEF_MOVE
    )
    plans, metrics = {}, {}
    for name in COLORS:
        if generator.planner._backend_cache:
            next(iter(generator.planner._backend_cache.values())).planner.reset_seed()
        torch.manual_seed(0)
        options = CuroboPlanOptions(
            path_constraints=(
                [
                    CuroboLinearConstraintCfg(
                        position_tolerance=0.005,
                        orientation_tolerance=math.radians(3),
                    )
                ]
                if name == "after"
                else None
            )
        )
        result = generator.generate(
            [PlanState.from_xpos(goal)],
            MotionGenOptions(
                start_qpos=start,
                control_part="arm",
                plan_opts=options,
            ),
        )
        if not bool(result.success.all()):
            generator.planner.close()
            raise RuntimeError(
                f"{name} planning failed; no replacement path is displayed."
            )
        poses = robot.compute_batch_fk(
            qpos=result.positions, name="arm", to_matrix=True
        )
        delta = poses[..., :3, 3] - goal[:, None, :3, 3]
        direction = goal[:, :3, 2]
        lateral = torch.linalg.vector_norm(
            delta
            - (delta * direction[:, None]).sum(-1, keepdim=True) * direction[:, None],
            dim=-1,
        )
        relative = goal[:, None, :3, :3].transpose(-1, -2) @ poses[..., :3, :3]
        cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        sine = torch.linalg.matrix_norm(
            relative - relative.transpose(-1, -2)
        ) / math.sqrt(8)
        angle = torch.atan2(sine, cosine)
        collision = generator.planner.validate_joint_trajectory(
            result.positions, control_part="arm"
        )
        if not bool(collision.all()):
            generator.planner.close()
            raise RuntimeError(f"{name} fails native sampled collision validation.")
        metrics[name] = {
            "max_lateral_mm": float(lateral.max() * 1000),
            "max_orientation_deg": float(torch.rad2deg(angle).max()),
            "duration_s": float(result.duration[0]),
            "sampled_collision_valid": True,
        }
        plans[name] = {
            "qpos": result.positions[0].cpu().numpy(),
            "time": result.dt[0].cumsum(0).cpu().numpy(),
            "tcp": poses[0, :, :3, 3].cpu().numpy(),
        }
        np.savez(output / f"{name}.npz", **plans[name])
        print(name, metrics[name], flush=True)
    if (
        metrics["after"]["max_lateral_mm"] > 5
        or metrics["after"]["max_orientation_deg"] > 3
    ):
        generator.planner.close()
        raise RuntimeError("The constrained plan failed the independent TCP check.")
    summary = {
        "goal_position": GOAL_POSITION,
        "goal_rotation": GOAL_ROTATION,
        "start_qpos": start.cpu().tolist(),
        "approach_distance_m": 0.45,
        "seed": 0,
        "position_tolerance_m": 0.005,
        "orientation_tolerance_deg": 3,
        "metrics": metrics,
        "scope": "planned geometry and kinematic rendering; self-collision disabled by the existing adapter",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return generator, plans, metrics


def _segments(points: Any, diameter: float) -> dict:
    """Represent a polyline by native render-only cylinders in real units."""
    import numpy as np

    delta = np.diff(points, axis=0)
    length = np.linalg.norm(delta, axis=-1)
    keep = length > 1e-8
    direction = delta[keep] / length[keep, None]
    quats = []
    for d in direction:
        if d[2] < -0.99999999:
            quats.append([1.0, 0.0, 0.0, 0.0])
        else:
            q = np.r_[np.cross([0.0, 0.0, 1.0], d), 1 + d[2]]
            quats.append(q / np.linalg.norm(q))
    return dict(
        translations=((points[:-1] + points[1:]) / 2)[keep],
        orientations_xyzw=np.asarray(quats),
        scales=np.column_stack(
            [np.full(keep.sum(), diameter), np.full(keep.sum(), diameter), length[keep]]
        ),
    )


def _render(
    sim: Any, robot: Any, plans: dict, metrics: dict, output: Path, probe: bool
) -> None:
    """Render overview and overhead cameras without advancing physics."""
    import imageio.v3 as iio
    import numpy as np
    import torch
    from embodichain.lab.sim.cfg import LightCfg
    from embodichain.lab.sim.sensors import CameraCfg
    from embodichain.lab.visualization.markers import MarkerGroupCfg, MarkerPrototypeCfg

    for name, pos, intensity in (
        ("key", (1.5, -2.0, 2.5), 40.0),
        ("fill", (-1.0, 1.5, 2.0), 18.0),
    ):
        sim.add_light(LightCfg(uid=name, init_pos=pos, intensity=intensity))
    cameras = []
    for name, eye, target, up, focal in (
        ("overview", (1.25, -1.35, 1.10), (0.32, 0.0, 0.35), (0.0, 0.0, 1.0), 1000.0),
        ("overhead", (0.46, 0.025, 1.55), (0.45, 0.025, 0.45), (0.0, 1.0, 0.0), 1000.0),
    ):
        camera = sim.add_sensor(
            CameraCfg(
                uid=name,
                width=WIDTH,
                height=HEIGHT,
                intrinsics=(focal, focal, WIDTH / 2, HEIGHT / 2),
                include_visual_overlays=True,
                extrinsics=CameraCfg.ExtrinsicsCfg(eye=eye, target=target, up=up),
            )
        )
        cameras.append(camera)
    sim.prepare()
    guide = sim.add_marker_group(
        MarkerGroupCfg(
            name="guide",
            prototypes={
                "line": MarkerPrototypeCfg(
                    shape="cylinder", color=(0.12, 0.85, 0.35, 1.0)
                ),
                "tube": MarkerPrototypeCfg(
                    shape="cylinder", color=(0.12, 0.85, 0.35, 0.12)
                ),
            },
        )
    )
    ideal = np.array([[0.45, -0.20, 0.45], [0.45, 0.25, 0.45]])
    one = _segments(ideal, 0.001)
    guide.update(
        translations=np.repeat(one["translations"], 2, axis=0),
        orientations_xyzw=np.repeat(one["orientations_xyzw"], 2, axis=0),
        scales=[[0.001, 0.001, 0.45], [0.010, 0.010, 0.45]],
        prototype_indices=[0, 1],
    )
    path = sim.add_marker_group(
        MarkerGroupCfg(
            name="path",
            prototypes={
                name: MarkerPrototypeCfg(shape="cylinder", color=color)
                for name, color in COLORS.items()
            },
        )
    )
    tip = sim.add_marker_group(
        MarkerGroupCfg(
            name="tcp",
            prototypes={
                name: MarkerPrototypeCfg(
                    shape="sphere", scale=(0.008,) * 3, color=color
                )
                for name, color in COLORS.items()
            },
        )
    )
    slowdown, hold = 6.0, 0.8
    frames = (
        1
        if probe
        else math.ceil(
            (2 * hold + slowdown * max(row["duration_s"] for row in metrics.values()))
            * FPS
        )
    )
    full_start = robot.get_qpos().clone()
    ids = robot.get_joint_ids("arm")
    step = sim.physics_step_index
    for mode_index, (name, plan) in enumerate(plans.items()):
        t = (
            np.array([0.30])
            if probe
            else np.clip(
                (np.arange(frames) / FPS - hold) / slowdown, 0, plan["time"][-1]
            )
        )
        qpos = np.stack(
            [np.interp(t, plan["time"], plan["qpos"][:, j]) for j in range(7)], axis=-1
        ).astype(np.float32)
        tcp = (
            robot.compute_batch_fk(
                qpos=torch.as_tensor(qpos[None], device=robot.device),
                name="arm",
                to_matrix=True,
            )[0, :, :3, 3]
            .cpu()
            .numpy()
        )
        displacement = tcp - np.asarray(GOAL_POSITION)
        display_error = np.linalg.norm(displacement[:, [0, 2]], axis=-1) * 1000
        metrics[name]["render_max_lateral_mm"] = float(display_error.max())
        if name == "after" and display_error.max() > 5:
            raise RuntimeError("Display interpolation exceeded the 5 mm TCP tube.")
        segments = _segments(plan["tcp"], 0.002)
        path.update(
            **segments,
            prototype_indices=np.full(len(segments["translations"]), mode_index),
        )
        tip.update(translations=tcp[:1], prototype_indices=[mode_index])
        process = (
            None
            if probe
            else subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{WIDTH}x{2*HEIGHT}",
                    "-r",
                    str(FPS),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "4",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output / f"{name}_native.mp4"),
                ],
                stdin=subprocess.PIPE,
            )
        )
        try:
            for index, q in enumerate(qpos):
                full = full_start.clone()
                full[0, ids] = torch.as_tensor(q, device=robot.device)
                robot.set_qpos(full, target=False)
                tip.update(translations=tcp[index : index + 1])
                sim.sync_render_state()
                if index == 0:
                    for _ in range(8):
                        for camera in cameras:
                            camera.update()
                pictures = []
                for camera in cameras:
                    camera.update()
                    pictures.append(
                        np.ascontiguousarray(
                            camera.get_data()["color"][0, ..., :3].cpu().numpy()
                        )
                    )
                picture = np.concatenate(pictures, axis=0)
                if probe or index == round((hold + slowdown * 0.30) * FPS):
                    iio.imwrite(output / f"{name}_preview.png", picture)
                if process is not None:
                    process.stdin.write(picture.tobytes())
                if index % 30 == 0:
                    print("render", name, index, "/", frames, flush=True)
        finally:
            if process is not None:
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError("Video encoding failed.")
    assert sim.physics_step_index == step
    summary = json.loads((output / "summary.json").read_text())
    summary.update(
        metrics=metrics,
        rendering={
            "fps": FPS,
            "frames": frames,
            "slowdown": slowdown,
            "physics_steps": 0,
            "interpolation": "linear joints for kinematic display",
        },
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not probe:
        _compose(metrics, output)


def _compose(metrics: dict, output: Path) -> None:
    """Label and combine both native recordings into one synchronized video."""
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    filters = []
    for index, (name, label, color) in enumerate(
        (
            ("before", "UNCONSTRAINED", "0x4da6ff"),
            ("after", "LINE CONSTRAINED", "0xff8c33"),
        )
    ):
        caption = f"{label}  |  sampled max {metrics[name]['max_lateral_mm']:.1f} mm"
        (output / f"{name}_caption.txt").write_text(caption)
        filters.append(
            f"[{index}:v]drawbox=x=0:y=0:w=iw:h=54:color=0x101827@0.90:t=fill,drawtext=fontfile={font}:textfile={name}_caption.txt:fontcolor={color}:fontsize=28:x=18:y=12,drawbox=x=0:y={HEIGHT}:w=iw:h=44:color=0x101827@0.85:t=fill,drawtext=fontfile={font}:text='OVERHEAD - green axis and 5 mm tolerance - 6x slow motion':fontcolor=white:fontsize=22:x=18:y={HEIGHT+10}[v{index}]"
        )
    filters.append(
        "[v0][v1]hstack=inputs=2,drawbox=x=958:y=0:w=4:h=ih:color=0x101827:t=fill[out]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str((output / "before_native.mp4").resolve()),
            "-i",
            str((output / "after_native.mp4").resolve()),
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "4",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str((output / "comparison.mp4").resolve()),
        ],
        cwd=output,
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "2.6",
            "-i",
            str(output / "comparison.mp4"),
            "-frames:v",
            "1",
            str(output / "comparison_preview.png"),
        ],
        check=True,
    )


def main() -> None:
    """Plan and render the alignment-frame example from direct arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/curobo_approach_showcase"
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Render one comparison pose instead of a full video.",
    )
    args = parser.parse_args()
    os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")
    import torch
    from embodichain.lab.sim import SimulationManager
    from embodichain.utils import logger

    logger.set_log_level("WARNING")
    torch.manual_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim, robot, fixture = _scene(not args.plan_only)
    generator = None
    try:
        generator, plans, metrics = _plan(robot, fixture, args.output_dir)
        if not args.plan_only:
            _render(sim, robot, plans, metrics, args.output_dir, args.probe)
        print("Showcase outputs:", args.output_dir, flush=True)
    finally:
        if generator is not None:
            generator.planner.close()
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
