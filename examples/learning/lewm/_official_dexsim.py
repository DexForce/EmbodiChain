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

"""Validate the official robot export in EmbodiChain's native DexSim runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch


def audit_kinematics(root: Path, *, device: str = "cuda:6") -> dict[str, Any]:
    """Compare native articulation link poses to the compiled MuJoCo reference.

    No dynamics are stepped: this checks the imported asset, frame conventions,
    state restoration and joint-name mapping before implementing constraints.
    """
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        ArticulationRootPropertiesCfg,
        LinkPhysicsOverrideCfg,
        MassPropertiesCfg,
        RenderCfg,
        RigidBodyPhysicsCfg,
        RobotCfg,
    )

    identity = json.loads((root / "robot/robot.json").read_text())
    path = root / "robot/robot.urdf"
    if hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]:
        raise ValueError("exported robot checksum does not match the reference")
    with np.load(root / "kinematics-reference.npz", allow_pickle=False) as file:
        names, frames = file["joint_names"].tolist(), file["frame_names"].tolist()
        qpos, poses = file["qpos"], file["poses"]
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            gpu_id=int(device.split(":")[-1]),
            device=device,
            physics_dt=0.002,
            render_cfg=RenderCfg(renderer="no-render"),
            enable_entity_gizmo=False,
            robot_ik_gizmo=None,
            startup_summary="off",
        )
    )
    try:
        from scipy.spatial.transform import Rotation

        link_attrs = {}
        for body in identity["bodies"]:
            if body["mass"] <= 0:
                continue
            rotation = Rotation.from_quat(np.roll(body["iquat_wxyz"], -1)).as_matrix()
            tensor = rotation @ np.diag(body["diaginertia"]) @ rotation.T
            link_attrs[body["name"]] = LinkPhysicsOverrideCfg(
                link_names_expr=[re.escape(body["name"])],
                attrs=RigidBodyPhysicsCfg(
                    mass_props=MassPropertiesCfg(
                        mass=body["mass"],
                        inertia=tensor,
                        com_position=np.array(body["ipos"])
                        - body["urdf_frame_in_body"],
                    )
                ),
            )
        cfg = RobotCfg(
            uid="official_cube_robot",
            fpath=str(path.resolve()),
            init_local_pose=np.array(identity["root_pose"]),
            control_parts={"arm": names[:6], "eef": names[6:]},
            solver_cfg=None,
            build_pk_chain=False,
            asset_physics_mode="overlay",
            link_attrs=link_attrs,
            root_props=ArticulationRootPropertiesCfg(
                fixed_base=True, self_collision_enabled=True
            ),
            joint_drive_props=None,
        )
        robot = sim.add_robot(cfg)
        sim.prepare()
        joint_ids = [robot.joint_names.index(name) for name in names]
        bindings = {frame["name"]: frame for frame in identity["source_frames"]}
        offsets = {
            name: torch.tensor(
                bindings[name]["transform"], device=device, dtype=torch.float32
            )
            for name in frames
        }
        actual = []
        for row in qpos:
            robot.set_state(
                qpos=torch.as_tensor(row[None], device=device, dtype=torch.float32),
                joint_ids=joint_ids,
                clear_dynamics=True,
            )
            actual.append(
                np.stack(
                    [
                        (
                            robot.get_link_pose(bindings[name]["link"], to_matrix=True)[
                                0
                            ]
                            @ offsets[name]
                        )
                        .cpu()
                        .numpy()
                        for name in frames
                    ]
                )
            )
        error = np.abs(np.array(actual) - poses)
        mass = robot.body_data.mass[0].cpu().numpy()
        inertia = robot.body_data.inertia[0].cpu().numpy()
        com = robot.body_data.com_pose[0].cpu().numpy()
        mass_error, inertia_error, com_error, inertial_rows = [], [], [], []
        for body in identity["bodies"]:
            if body["mass"] == 0:
                continue
            index = robot.link_names.index(body["name"])
            expected_mass = body["mass"]
            mass_error.append(abs(mass[index] - expected_mass) / expected_mass)
            actual_r = Rotation.from_quat(com[index, 3:]).as_matrix()
            source_r = Rotation.from_quat(np.roll(body["iquat_wxyz"], -1)).as_matrix()
            source_i = source_r @ np.diag(body["diaginertia"]) @ source_r.T
            actual_i = actual_r @ np.diag(inertia[index]) @ actual_r.T
            inertia_error.append(
                float(np.linalg.norm(actual_i - source_i) / np.linalg.norm(source_i))
            )
            source_com = np.array(body["ipos"]) - body["urdf_frame_in_body"]
            com_error.append(float(np.max(np.abs(com[index, :3] - source_com))))
            inertial_rows.append(
                {
                    "name": body["name"],
                    "source_mass": expected_mass,
                    "actual_mass": float(mass[index]),
                    "source_diaginertia": body["diaginertia"],
                    "actual_diaginertia": inertia[index].tolist(),
                    "inertia_relative_error": inertia_error[-1],
                }
            )
        tolerance = 1e-5
        report = {
            "physics": "default",
            "device": device,
            "samples": len(qpos),
            "frames": len(frames),
            "joint_names": names,
            "max_position_error_m": float(error[..., :3, 3].max()),
            "max_rotation_matrix_error": float(error[..., :3, :3].max()),
            "tolerance": tolerance,
            "kinematics_passed": bool(error.max() <= tolerance),
            "physics_steps": 0,
            "dynamics_alignment_verified": False,
            "robot_urdf_sha256": identity["sha256"],
            "inertial_source": "compiled_mujoco_explicit_prebuild_link_overlays",
            "mass_max_relative_error": float(max(mass_error)),
            "inertia_max_relative_error": float(max(inertia_error)),
            "com_max_absolute_error_m": float(max(com_error)),
            "inertial_rows": inertial_rows,
            "zero_mass_frames_not_audited": [
                body["name"] for body in identity["bodies"] if body["mass"] == 0
            ],
            "source_fixed_bodies_collapsed": identity["fixed_bodies_collapsed"],
            "total_source_mass": identity["source_total_mass"],
            "total_imported_asset_mass": identity["total_mass"],
        }
        report["inertial_properties_passed"] = bool(
            max(mass_error) <= 1e-5
            and max(inertia_error) <= 1e-5
            and max(com_error) <= 1e-6
        )
        report["passed"] = (
            report["kinematics_passed"] and report["inertial_properties_passed"]
        )
        (root / "dexsim-kinematics-report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if not report["passed"]:
            raise RuntimeError(
                f"DexSim imported kinematics differ from MuJoCo: {report}"
            )
        return report
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--device", default="cuda:6")
    args = parser.parse_args()
    print(
        json.dumps(audit_kinematics(args.reference, device=args.device), indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
