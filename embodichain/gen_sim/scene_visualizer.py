# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

"""
Lightweight Reality Config loader for visualization.

This module provides functionality to load objects from a Reality Config JSON file
into EmbodiChain's SimulationManager. It includes support for loading the
DexForce W1 robot and correctly handles rotation matrices extracted from pose
transformations.
"""

import argparse
import math
import numpy as np
from pathlib import Path
import json
import time
import embodichain.utils.logger as logger
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    RigidBodyAttributesCfg,
    LightCfg,
    MeshCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.robots.dexforce_w1.cfg import DexforceW1Cfg


def load_objects_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_lights(sim: SimulationManager, num_lights=8, radius=5, height=8, intensity=200):
    lights = []
    for i in range(num_lights):
        angle = 2 * math.pi * i / num_lights
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = height
        uid = f"l{i+1}"
        cfg = LightCfg(uid=uid, intensity=intensity, radius=600, init_pos=[x, y, z])
        lights.append(sim.add_light(cfg))
    return lights


def extract_rotation_from_matrix(pose_matrix):
    R = np.array(
        [
            [pose_matrix[0][0], pose_matrix[0][1], pose_matrix[0][2]],
            [pose_matrix[1][0], pose_matrix[1][1], pose_matrix[1][2]],
            [pose_matrix[2][0], pose_matrix[2][1], pose_matrix[2][2]],
        ]
    )

    if np.allclose(R, np.eye(3), atol=0.001):
        return [0, 0, 0]

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy > 1e-6:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return [math.degrees(x), math.degrees(y), math.degrees(z)]


def add_objects_from_config(
    sim: SimulationManager, objects_config: dict, config_dir: Path
):
    """只加载 rigid_object，不加载 background"""
    rigid_objects = objects_config.get("rigid_object", [])

    logger.log_info(f"Loading {len(rigid_objects)} rigid objects from config...")

    for obj in rigid_objects:
        uid = obj.get("uid", "")
        if not uid:
            logger.log_warning(f"Skipping object without uid: {obj}")
            continue

        shape_info = obj.get("shape", {})
        mesh_path = shape_info.get("fpath", "")
        if not mesh_path:
            logger.log_warning(f"Skipping object {uid} without mesh path")
            continue
        mesh_path = (config_dir / mesh_path).resolve().as_posix()

        pose_matrix = obj.get("init_local_pose")
        if pose_matrix is None:
            pose_matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        pos = [pose_matrix[0][3], pose_matrix[1][3], pose_matrix[2][3]]
        rot = extract_rotation_from_matrix(pose_matrix)

        attrs_info = obj.get("attrs", {})
        physics_attrs = RigidBodyAttributesCfg(
            mass=attrs_info.get("mass", 2.0),
            static_friction=attrs_info.get("static_friction", 0.5),
            dynamic_friction=attrs_info.get("dynamic_friction", 0.5),
            restitution=attrs_info.get("restitution", 0.1),
            linear_damping=attrs_info.get("linear_damping", 0.5),
            angular_damping=attrs_info.get("angular_damping", 0.5),
            contact_offset=attrs_info.get("contact_offset", 0.002),
            rest_offset=attrs_info.get("rest_offset", 0.001),
            max_depenetration_velocity=attrs_info.get(
                "max_depenetration_velocity", 3.0
            ),
            max_linear_velocity=attrs_info.get("max_linear_velocity", 2.0),
            max_angular_velocity=attrs_info.get("max_angular_velocity", 2.0),
        )

        body_type = obj.get("body_type", "dynamic")
        max_convex_hull_num = obj.get("max_convex_hull_num", 10)
        obj_cfg = RigidObjectCfg(
            uid=uid,
            shape=MeshCfg(fpath=mesh_path),
            body_type=body_type,
            init_pos=pos,
            init_rot=rot,
            attrs=physics_attrs,
            max_convex_hull_num=max_convex_hull_num,
        )

        try:
            sim.add_rigid_object(cfg=obj_cfg)
            logger.log_info(
                f"Loaded object: {uid}, body_type={body_type}, pos={pos}, rot={rot}"
            )
        except Exception as e:
            logger.log_error(f"Failed to load object {uid}: {e}")


def add_background_from_config(
    sim: SimulationManager, objects_config: dict, config_dir: Path
):
    background_objects = objects_config.get("background", [])
    if not background_objects:
        return

    logger.log_info(
        f"Loading {len(background_objects)} background objects from config..."
    )

    for bg_obj in background_objects:
        uid = bg_obj.get("uid", "")
        if not uid:
            continue

        shape_info = bg_obj.get("shape", {})
        mesh_path = shape_info.get("fpath", "")
        if not mesh_path:
            continue
        mesh_path = (config_dir / mesh_path).resolve().as_posix()
        pose_matrix = bg_obj.get("init_local_pose")
        if pose_matrix is None:
            pose_matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        pos = [pose_matrix[0][3], pose_matrix[1][3], pose_matrix[2][3]]
        rot = extract_rotation_from_matrix(pose_matrix)
        max_convex_hull_num = bg_obj.get("max_convex_hull_num", 20)
        bg_cfg = RigidObjectCfg(
            uid=f"bg_{uid}",
            shape=MeshCfg(fpath=mesh_path),
            body_type="static",
            init_pos=pos,
            init_rot=rot,
            attrs=RigidBodyAttributesCfg(
                mass=0.2,
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.1,
            ),
            max_convex_hull_num=max_convex_hull_num,
        )

        try:
            sim.add_rigid_object(cfg=bg_cfg)
            time.sleep(1)
            logger.log_info(f"Loaded background object: {uid}")
        except Exception as e:
            logger.log_error(f"Failed to load background object {uid}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Objects Config Scene")
    parser.add_argument(
        "--objects_config",
        type=str,
        required=True,
        help="Path to objects config JSON file",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Simulation device (cpu or cuda)"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode (no GUI)"
    )
    parser.add_argument(
        "--load_w1", action="store_true", help="Load DexForce W1 robot into scene"
    )

    args = parser.parse_args()

    objects_config_path = Path(args.objects_config).resolve()
    config = load_objects_config(objects_config_path)
    config_dir = objects_config_path.parent

    sim_cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        headless=args.headless,
        physics_dt=1.0 / 100.0,
        sim_device=args.device,
        enable_rt=False,
    )
    sim = SimulationManager(sim_cfg)
    sim.init_gpu_physics()
    sim.set_manual_update(False)

    add_lights(sim)
    add_background_from_config(sim, config, config_dir)
    add_objects_from_config(sim, config, config_dir)

    if args.load_w1:
        logger.log_info("Adding DexForce W1 robot...")
        DexforceW1Cfg_dict = {
            "uid": "dexforce_w1",
            "version": "v021",
            "arm_kind": "anthropomorphic",
            "init_pos": [-1, -0.5, 0],
            "init_rot": [0, 0, 90],
        }
        try:
            sim.add_robot_v2(cfg=DexforceW1Cfg.from_dict(DexforceW1Cfg_dict))
        except Exception as e:
            logger.log_error(f"Failed to load DexForce W1 robot: {e}")

    logger.log_info("Scene setup complete! Opening window...")
    sim.open_window()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        logger.log_info("Stopping simulation...")
        sim.destroy()


if __name__ == "__main__":
    main()
