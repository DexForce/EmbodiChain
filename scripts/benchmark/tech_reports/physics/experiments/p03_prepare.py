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

"""Freeze local Franka source dependencies and shared offline P-03 commands.

Run: python -m scripts.benchmark.tech_reports.physics.experiments.p03_prepare --output PATH
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils
from scipy.spatial import ConvexHull


def frame(pos: object, quat: object) -> np.ndarray:
    """Convert a USD local joint frame to a column-vector matrix."""
    mat = np.eye(4)
    mat[:3, 3] = list(pos)
    mat[:3, :3] = Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()
    return mat


def forward(joints: list[dict], q: np.ndarray) -> np.ndarray:
    """Evaluate source USD joint frames, with the selected TCP after the hand."""
    result = np.eye(4)
    for i, j in enumerate(joints):
        motion = np.eye(4)
        if i < 7:
            motion[:3, :3] = Rotation.from_rotvec(
                np.asarray(j["axis"]) * q[i]
            ).as_matrix()
        result = (
            result
            @ np.asarray(j["parent_frame"])
            @ motion
            @ np.linalg.inv(j["child_frame"])
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset",
        type=Path,
        default=Path(
            "/home/oem/isaacsim_assets/Assets/Isaac/5.1/Isaac/IsaacLab/Factory/franka_mimic.usd"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stage = Usd.Stage.Open(str(args.asset))
    geometry = {}
    cache = UsdGeom.XformCache()
    for name in ["panda_leftfinger", "panda_rightfinger"]:
        body = stage.GetPrimAtPath("/panda/" + name)
        prim = stage.GetPrimAtPath("/panda/" + name + "/collisions")
        vertices = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        matrix = np.asarray(cache.ComputeRelativeTransform(prim, body)[0])
        vertices = (np.c_[vertices, np.ones(len(vertices))] @ matrix)[:, :3]
        vertices = vertices[ConvexHull(vertices).vertices]
        geometry[name + "_vertices"] = vertices
        geometry[name + "_simplices"] = ConvexHull(vertices).simplices
    np.savez(args.output / "fingers.npz", **geometry)
    layers, assets, missing = UsdUtils.ComputeAllDependencies(str(args.asset))
    # OmniPBR.mdl is an engine shader module; it is not a missing robot mesh.
    assert set(missing) <= {"OmniPBR.mdl"}, missing
    paths = sorted(
        {str(Path(x.realPath).resolve()) for x in layers}
        | {str(Path(x).resolve()) for x in assets}
    )
    manifest = {
        "source": str(args.asset),
        "source_origin": "Local Isaac Sim 5.1 asset bundle / Isaac/IsaacLab/Factory/franka_mimic.usd",
        "license": "Third-party NVIDIA asset terms; redistribution not granted by the benchmark code license; no asset uploaded.",
        "runtime_shader_modules": missing,
        "files": [
            {
                "path": x,
                "bytes": Path(x).stat().st_size,
                "sha256": hashlib.sha256(Path(x).read_bytes()).hexdigest(),
            }
            for x in paths
        ],
    }
    (args.output / "asset-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    prims = {p.GetName(): p for p in stage.Traverse()}
    joint_names = [f"panda_joint{i}" for i in range(1, 8)]
    chain_names = joint_names + [
        "panda_hand_joint",
        "force_sensor_joint",
        "panda_fingertip_centered_joint",
    ]
    joints = []
    for name in chain_names:
        j = UsdPhysics.Joint(prims[name])
        axis = prims[name].GetAttribute("physics:axis").Get()
        joints.append(
            dict(
                name=name,
                parent_frame=frame(
                    j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()
                ).tolist(),
                child_frame=frame(
                    j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get()
                ).tolist(),
                axis=np.eye(3)["XYZ".index(axis)].tolist() if axis else [0, 0, 0],
            )
        )
    limits = np.radians(
        [
            [
                prims[n].GetAttribute("physics:lowerLimit").Get(),
                prims[n].GetAttribute("physics:upperLimit").Get(),
            ]
            for n in joint_names
        ]
    )
    model = {
        "joint_names": joint_names + ["panda_finger_joint1", "panda_finger_joint2"],
        "tcp_link": "panda_fingertip_centered",
        "chain": joints,
        "arm_limits_rad": limits.tolist(),
        "mimic": {
            "follower": "panda_finger_joint2",
            "leader": "panda_finger_joint1",
            "position_multiplier": 1.0,
        },
        "robot_base_m": [0.15, 0.5, 0.75],
    }
    (args.output / "kinematics.json").write_text(json.dumps(model, indent=2) + "\n")
    rng = np.random.default_rng(20261022)
    offsets = np.column_stack(
        [
            rng.uniform(-0.015, 0.015, (10, 2)),
            rng.uniform(-np.deg2rad(5), np.deg2rad(5), 10),
        ]
    )
    offsets[0] = 0
    t = np.arange(501, dtype=np.float64) * 0.02
    qt = np.empty((10, len(t), 9))
    tcp_ref = np.empty((10, len(t), 7))
    object_pose = np.empty((10, 7))
    diagnostics = []
    qseed = np.array([0, -0.569, 0, -2.81, 0, 3.037, 0.741])
    active = np.array([0, 1, 3, 4, 5, 6])
    for episode, (x, y, yaw) in enumerate(offsets):
        pos = np.array([0.55 + x, 0.5 + y, 0.775])
        base = np.asarray(model["robot_base_m"])
        quat = Rotation.from_euler("z", yaw).as_quat()
        object_pose[episode] = np.r_[pos, quat]
        rot = (
            Rotation.from_euler("z", yaw).as_matrix()
            @ Rotation.from_euler("x", np.pi).as_matrix()
        )
        way = []
        for dz in [0.1, 0, 0.1]:
            target = pos - base + np.array([0, 0, dz])

            def residual(v: np.ndarray) -> np.ndarray:
                q = np.zeros(7)
                q[active] = v
                m = forward(joints, q)
                return np.r_[
                    m[:3, 3] - target,
                    Rotation.from_matrix(rot.T @ m[:3, :3]).as_rotvec(),
                ]

            sol = least_squares(
                residual,
                qseed[active],
                bounds=(limits[active, 0] + 1e-5, limits[active, 1] - 1e-5),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=400,
            )
            assert (
                np.linalg.norm(sol.fun[:3]) < 1e-5
                and np.linalg.norm(sol.fun[3:]) < 1e-5
            ), (episode, dz, sol.fun)
            qseed = np.zeros(7)
            qseed[active] = sol.x
            way.append(qseed.copy())
        knots = [
            (0, way[0], 0.04),
            (1, way[1], 0.04),
            (2, way[1], 0.015),
            (3, way[2], 0.015),
            (7, way[2], 0.015),
            (9, way[1], 0.015),
            (10, way[1], 0.04),
        ]
        for k, time in enumerate(t):
            i = min(
                np.searchsorted([x[0] for x in knots], time, side="right") - 1,
                len(knots) - 2,
            )
            a, b = knots[i : i + 2]
            u = (time - a[0]) / (b[0] - a[0])
            w = u * u * u * (10 + u * (-15 + 6 * u))
            qt[episode, k, :7] = (1 - w) * a[1] + w * b[1]
            qt[episode, k, 7:] = (1 - w) * a[2] + w * b[2]
            m = forward(joints, qt[episode, k, :7])
            tcp_ref[episode, k] = np.r_[
                m[:3, 3] + base, Rotation.from_matrix(m[:3, :3]).as_quat()
            ]
        diagnostics.append(
            {
                "initial": episode,
                "max_command_speed_rad_s": np.max(
                    np.abs(np.diff(qt[episode, :, :7], axis=0) / 0.02), axis=0
                ).tolist(),
            }
        )
    physics_t = np.arange(2001) * 0.005
    fx = np.where(
        (physics_t[:-1] >= 5) & (physics_t[:-1] < 7),
        0.2 * 0.1 * 9.81 * np.sin(2 * np.pi * 2 * (physics_t[:-1] - 5)),
        0.0,
    )
    np.savez_compressed(
        args.output / "commands.npz",
        control_t=t,
        physics_t=physics_t,
        q_target=qt,
        tcp_reference=tcp_ref,
        object_initial_pose=object_pose,
        offsets_xy_yaw=offsets,
        force_world=np.column_stack([fx, np.zeros((len(fx), 2))]),
    )
    (args.output / "trajectory-check.json").write_text(
        json.dumps(
            {
                "seed": 20261022,
                "initial_0": "nominal; 1-9 sampled",
                "diagnostics": diagnostics,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
