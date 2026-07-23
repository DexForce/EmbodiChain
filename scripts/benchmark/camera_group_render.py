"""Benchmark EmbodiChain multi-environment camera-group rendering.

This scene uses only primitive assets so it can reproduce the renderer timing
without downloading task assets. By default it keeps the production topology:
one head, one right-wrist and one left-wrist camera group, each with one layer
per environment. The optional merge switches exercise the camera-group APIs.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import LightCfg, RenderCfg, RigidObjectCfg
from embodichain.lab.sim.sensors import CameraCfg, plan_camera_groups
from embodichain.lab.sim.shapes import CubeCfg


CAMERAS = (
    ("head", 960, 540, (2.3, -2.3, 1.8), (0.0, 0.0, 0.35)),
    ("right_wrist", 640, 480, (1.2, -1.2, 0.9), (0.2, 0.0, 0.3)),
    ("left_wrist", 640, 480, (-1.2, -1.2, 0.9), (-0.2, 0.0, 0.3)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer", choices=("hybrid", "fast-rt"), required=True)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=25)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--merge-camera-groups", action="store_true")
    parser.add_argument("--merge-different-resolutions", action="store_true")
    parser.add_argument("--batch-camera-group-render", action="store_true")
    parser.add_argument("--include-depth", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def make_camera_configs(include_depth: bool) -> list[CameraCfg]:
    return [
        CameraCfg(
            uid=uid,
            width=width,
            height=height,
            intrinsics=(600.0, 600.0, width / 2.0, height / 2.0),
            extrinsics=CameraCfg.ExtrinsicsCfg(eye=eye, target=target),
            enable_color=True,
            enable_depth=include_depth,
        )
        for uid, width, height, eye, target in CAMERAS
    ]


def main() -> None:
    args = parse_args()
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device="cuda",
            gpu_id=args.gpu,
            num_envs=args.envs,
            arena_space=4.0,
            render_cfg=RenderCfg(renderer=args.renderer),
            batch_camera_group_render=args.batch_camera_group_render,
        )
    )
    cameras = make_camera_configs(args.include_depth)
    plan_camera_groups(
        cameras,
        enabled=args.merge_camera_groups,
        merge_different_resolutions=args.merge_different_resolutions,
    )
    sim.add_light(LightCfg(uid="key", init_pos=(0.0, -1.0, 3.0), intensity=120.0))
    sim.add_rigid_object(
        RigidObjectCfg(
            uid="cube",
            shape=CubeCfg(size=[0.45, 0.45, 0.45]),
            body_type="static",
            init_pos=(0.0, 0.0, 0.225),
        )
    )
    sensors = [sim.add_sensor(camera) for camera in cameras]
    group_ids = list(dict.fromkeys(sensor.group_id for sensor in sensors))

    def render() -> None:
        sim.render_camera_group(group_ids)
        if not args.render_only:
            for sensor in sensors:
                sensor.update(fetch_only=True)

    try:
        for _ in range(args.warmup):
            render()
        torch.cuda.synchronize(sim.device)

        samples_ms = []
        for _ in range(args.measure):
            started = time.perf_counter()
            render()
            torch.cuda.synchronize(sim.device)
            samples_ms.append((time.perf_counter() - started) * 1000.0)

        shapes = {}
        if not args.render_only:
            shapes = {
                sensor.uid: {
                    name: tuple(data.shape) for name, data in sensor.get_data().items()
                }
                for sensor in sensors
            }
        print(
            "RESULT "
            f"renderer={args.renderer} envs={args.envs} "
            f"merge={args.merge_camera_groups} "
            f"mixed_res={args.merge_different_resolutions} "
            f"batch={args.batch_camera_group_render} "
            f"depth={args.include_depth} render_only={args.render_only} "
            f"sensor_group_ids={[sensor.group_id for sensor in sensors]} "
            f"render_group_ids={group_ids} sensor_shapes={shapes} "
            f"mean_ms={np.mean(samples_ms):.2f} "
            f"median_ms={np.median(samples_ms):.2f}",
            flush=True,
        )
        sys.stdout.flush()
    finally:
        sim.destroy()


if __name__ == "__main__":
    main()
