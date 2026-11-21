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

"""Private rotating-anchor equality oracle; tests drive-frame/Jacobian mapping."""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")
OUT = ROOT / "alignment/point-step-response"
DT = 0.002
ANCHORS = [np.array([0.1, 0, 0]), np.array([-0.1, 0, 0])]


def cases():
    for rotation in [False, True]:
        ra = Rotation.from_euler("xyz", [0.2, -0.1, 0.3] if rotation else [0, 0, 0])
        rb = Rotation.from_euler("xyz", [-0.1, 0.3, -0.2] if rotation else [0, 0, 0])
        for residual in [np.zeros(3), np.array([1e-4, -2e-4, 1e-4])]:
            for moving in [False, True]:
                pose = [
                    np.r_[-ra.apply(ANCHORS[0]), ra.as_quat()],
                    np.r_[residual - rb.apply(ANCHORS[1]), rb.as_quat()],
                ]
                velocity = (
                    [
                        np.array([0.03, 0.1, -0.02, 0.2, -0.1, 0.3]),
                        np.array([-0.02, -0.08, 0.03, -0.15, 0.2, 0.1]),
                    ]
                    if moving
                    else [np.zeros(6), np.zeros(6)]
                )
                yield {
                    "id": int(rotation) * 4
                    + int(bool(np.linalg.norm(residual))) * 2
                    + int(moving),
                    "pose": [x.tolist() for x in pose],
                    "velocity_world": [x.tolist() for x in velocity],
                }


def source():
    import mujoco

    xml = """<mujoco><option timestep=".002" gravity="0 0 0" integrator="implicitfast" solver="Newton" iterations="100" tolerance="1e-12"/>
    <worldbody><body name="a" pos="-.1 0 0"><freejoint/><inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/></body>
    <body name="b" pos=".1 0 0"><freejoint/><inertial mass="2" pos="0 0 0" diaginertia=".2 .2 .2"/></body></worldbody>
    <equality><connect body1="a" body2="b" anchor=".1 0 0" solref=".005 1" solimp=".95 .99 .001 .5 2"/></equality></mujoco>"""
    # connect anchor is body1-local in this MuJoCo schema, matching the scalar
    # fixture; model.eq_data supplies the exact compiled anchors for the audit.
    m = mujoco.MjModel.from_xml_string(xml)
    rows = []
    print("compiled anchors", m.eq_data[0, :6].tolist(), flush=True)
    for case in cases():
        d = mujoco.MjData(m)
        for i in range(2):
            p = np.array(case["pose"][i])
            vel = np.array(case["velocity_world"][i])
            r = Rotation.from_quat(p[3:])
            d.qpos[i * 7 : i * 7 + 7] = np.r_[p[:3], np.roll(p[3:], 1)]
            d.qvel[i * 6 : i * 6 + 6] = np.r_[vel[:3], r.inv().apply(vel[3:])]
        mujoco.mj_forward(m, d)
        K, B, imp, _ = d.efc_KBIP[0]
        R = d.efc_R[0]
        r0 = Rotation.from_quat(case["pose"][0][3:])
        world_anchor = []
        point_vel = []
        for i in range(2):
            p = np.array(case["pose"][i])
            v = np.array(case["velocity_world"][i])
            offset = Rotation.from_quat(p[3:]).apply(m.eq_data[0, i * 3 : i * 3 + 3])
            world_anchor.append(p[:3] + offset)
            point_vel.append(v[:3] + np.cross(v[3:], offset))
        target = (
            (1 - DT * B)
            / (1 - DT * DT * K * imp)
            * r0.inv().apply(point_vel[1] - point_vel[0])
        )
        mujoco.mj_step(m, d)
        after = []
        for i in range(2):
            rot = Rotation.from_quat(np.roll(d.qpos[i * 7 + 3 : i * 7 + 7], -1))
            after.append(
                np.r_[
                    d.qvel[i * 6 : i * 6 + 3], rot.apply(d.qvel[i * 6 + 3 : i * 6 + 6])
                ].tolist()
            )
        rows.append(
            {
                **case,
                "gain": [float(K * imp / R), float((1 / DT - DT * K * imp) / R)],
                "target": target.tolist(),
                "anchors": m.eq_data[0, :6].reshape(2, 3).tolist(),
                "after_velocity_world": after,
            }
        )
    (OUT / "rotating-source.json").write_text(
        json.dumps({"xml": xml, "rows": rows}, indent=2) + "\n"
    )


def native():
    import torch
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        DefaultPhysicsCfg,
        RenderCfg,
        RigidObjectCfg,
        RigidBodyPhysicsCfg,
        DefaultRigidBodyPropertiesCfg,
        MassPropertiesCfg,
        CollisionPropertiesCfg,
        RigidConstraintCfg,
    )
    from embodichain.lab.sim.shapes import CubeCfg

    ref = json.loads((OUT / "rotating-source.json").read_text())
    device = os.environ.get("LEWM_DEVICE", "cuda:6")
    sim = SimulationManager(
        SimulationManagerCfg(
            physics_cfg=DefaultPhysicsCfg(
                enable_tgs=False, cache_material=True, gravity=(0, 0, 0)
            ),
            device=device,
            gpu_id=int(device.split(":")[1]),
            physics_dt=DT,
            headless=True,
            enable_default_plane=False,
            render_cfg=RenderCfg(renderer="no-render"),
            enable_entity_gizmo=False,
            robot_ik_gizmo=None,
            startup_summary="off",
        )
    )
    groups = []
    try:
        for case in ref["rows"]:
            bodies = []
            for i in range(2):
                bodies.append(
                    sim.add_rigid_object(
                        RigidObjectCfg(
                            uid=f'r{case["id"]}_{i}',
                            shape=CubeCfg(size=[0.02] * 3),
                            attrs=RigidBodyPhysicsCfg(
                                mass_props=MassPropertiesCfg(
                                    mass=i + 1, inertia=np.eye(3) * (0.1 * (i + 1))
                                ),
                                collision_props=CollisionPropertiesCfg(
                                    collision_enabled=False
                                ),
                                rigid_props=DefaultRigidBodyPropertiesCfg(
                                    has_gravity=False,
                                    linear_damping=0,
                                    angular_damping=0,
                                    min_position_iters=100,
                                    min_velocity_iters=1,
                                    sleep_threshold=0,
                                ),
                            ),
                        )
                    )
                )
            groups.append((case, bodies))
        sim.prepare()
        print("prepared", flush=True)
        torch.cuda.synchronize(device)
        print("prepared synced", flush=True)
        for case, bodies in groups:
            frames = [np.eye(4), np.eye(4)]
            for i in range(2):
                frames[i][:3, 3] = case["anchors"][i]
            (
                None
                if os.environ.get("LEWM_NO_CONSTRAINT")
                else sim.create_rigid_constraint(
                    RigidConstraintCfg(
                        name=f'eq_r{case["id"]}',
                        rigid_object_a_uid=bodies[0].uid,
                        rigid_object_b_uid=bodies[1].uid,
                        local_frame_a=frames[0],
                        local_frame_b=frames[1],
                        constraint_type="spherical",
                        linear_drive_stiffness=case["gain"][0],
                        linear_drive_damping=case["gain"][1],
                        linear_drive_velocity=(
                            None if os.environ.get("LEWM_BASELINE") else case["target"]
                        ),
                    )
                )
            )
            for i, body in enumerate(bodies):
                body.set_local_pose(torch.tensor([case["pose"][i]], device=device))
                body.set_velocity(
                    lin_vel=torch.tensor(
                        [case["velocity_world"][i][:3]], device=device
                    ),
                    ang_vel=torch.tensor(
                        [case["velocity_world"][i][3:]], device=device
                    ),
                )
        print("state set", flush=True)
        torch.cuda.synchronize(device)
        print("state synced", flush=True)
        sim.update(DT, step=1, render_final_step=False)
        print("stepped", flush=True)
        torch.cuda.synchronize(device)
        print("step synced", flush=True)
        rows = []
        for case, bodies in groups:
            v = np.array(
                [
                    np.r_[
                        b.body_data.lin_vel[0].cpu().numpy(),
                        b.body_data.ang_vel[0].cpu().numpy(),
                    ]
                    for b in bodies
                ]
            )
            err = abs(v - np.array(case["after_velocity_world"]))
            row = {
                "id": case["id"],
                "max_velocity_error": float(err.max()),
                "velocity": v.tolist(),
            }
            rows.append(row)
            print(case["id"], float(err.max()), flush=True)
        (OUT / "rotating-native.json").write_text(
            json.dumps(
                {
                    "rows": rows,
                    "scope": "8free-body cases; actualsourcecompiledanchors; no physics task change",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=["source", "native"])
    a = p.parse_args()
    source() if a.backend == "source" else native()
