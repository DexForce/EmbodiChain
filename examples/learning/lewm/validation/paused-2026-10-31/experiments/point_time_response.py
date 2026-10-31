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

"""Private nonlinear point-constraint trajectory oracle using live native states."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")
OUT = ROOT / "alignment/point-step-response"
DT = 0.002
STEPS = 500
CASES = [(0.0, 0.0, 1.0), (0.0003, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0001, -0.1, 10.0)]


def source():
    import mujoco

    spec = json.loads((OUT / "source.json").read_text())
    model = mujoco.MjModel.from_xml_string(spec["xml"])
    rows = []
    for r, v, load in CASES:
        data = mujoco.MjData(model)
        data.qpos[:] = [r / 2, -r / 2]
        data.qvel[:] = [v / 2, -v / 2]
        mujoco.mj_forward(model, data)
        K, B, d, _ = data.efc_KBIP[1]
        A0 = data.efc_R[1] * d / (1 - d)
        trace = [{"qpos": data.qpos.tolist(), "qvel": data.qvel.tolist()}]
        for _ in range(STEPS):
            data.qfrc_applied[:] = [load, -load]
            mujoco.mj_step(model, data)
            trace.append({"qpos": data.qpos.tolist(), "qvel": data.qvel.tolist()})
        rows.append(
            {
                "initial": [r, v, load],
                "K": float(K),
                "B": float(B),
                "A0": float(A0),
                "trace": trace,
            }
        )
    (OUT / "time-source.json").write_text(
        json.dumps({"rows": rows, "mujoco_version": mujoco.__version__}, indent=2)
        + "\n"
    )
    print("source4x500steps saved", flush=True)


def native():
    import torch
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        DefaultPhysicsCfg,
        RenderCfg,
        RobotCfg,
        ArticulationRootPropertiesCfg,
        JointDrivePropertiesCfg,
        RigidBodyPhysicsCfg,
        DefaultRigidBodyPropertiesCfg,
        LinkPhysicsOverrideCfg,
        MassPropertiesCfg,
        RigidConstraintCfg,
    )

    ref = json.loads((OUT / "time-source.json").read_text())
    device = "cuda:3"
    sim = SimulationManager(
        SimulationManagerCfg(
            physics_cfg=DefaultPhysicsCfg(
                enable_tgs=False, cache_material=True, gravity=(0, 0, 0)
            ),
            device=device,
            gpu_id=3,
            physics_dt=DT,
            headless=True,
            enable_default_plane=False,
            render_cfg=RenderCfg(renderer="no-render"),
            enable_entity_gizmo=False,
            robot_ik_gizmo=None,
            startup_summary="off",
        )
    )
    robots = []
    traces = []
    try:
        for i, case in enumerate(ref["rows"]):
            attrs = {
                n: LinkPhysicsOverrideCfg(
                    link_names_expr=[n],
                    attrs=RigidBodyPhysicsCfg(
                        mass_props=MassPropertiesCfg(
                            mass=m, inertia=np.eye(3) * 0.1, com_position=(0, 0, 0)
                        )
                    ),
                )
                for n, m in [("root", 1.0), ("a", 1.0), ("b", 2.0)]
            }
            robots.append(
                sim.add_robot(
                    RobotCfg(
                        uid=f"time_{i}",
                        fpath=str(ROOT / "reference/point-load.urdf"),
                        control_parts={"arm": ["a_slide", "b_slide"]},
                        init_qpos=[0.0, 0.0],
                        solver_cfg=None,
                        build_pk_chain=False,
                        asset_physics_mode="overlay",
                        link_attrs=attrs,
                        attrs=RigidBodyPhysicsCfg(
                            rigid_props=DefaultRigidBodyPropertiesCfg(
                                has_gravity=False, linear_damping=0, angular_damping=0
                            )
                        ),
                        root_props=ArticulationRootPropertiesCfg(
                            fixed_base=True,
                            min_position_iters=100,
                            min_velocity_iters=1,
                            sleep_threshold=0,
                        ),
                        joint_drive_props=JointDrivePropertiesCfg(
                            target_mode="effort",
                            stiffness=0,
                            damping=0,
                            max_effort=1e6,
                            friction=0,
                            armature=0.005,
                        ),
                    )
                )
            )
        sim.prepare()
        for i, (robot, case) in enumerate(zip(robots, ref["rows"])):
            sim.create_rigid_constraint(
                RigidConstraintCfg(
                    name=f"eq_{i}",
                    rigid_object_a_uid=f"time_{i}",
                    rigid_object_b_uid=f"time_{i}",
                    articulation_link_a="a",
                    articulation_link_b="b",
                    local_frame_a=np.eye(4),
                    local_frame_b=np.eye(4),
                    constraint_type="spherical",
                    linear_drive_stiffness=100,
                    linear_drive_damping=20,
                )
            )
            r, v, _ = case["initial"]
            robot.set_state(
                qpos=torch.tensor([[r / 2, -r / 2]], device=device),
                qvel=torch.tensor([[v / 2, -v / 2]], device=device),
                clear_dynamics=True,
                clamp_qpos=False,
            )
            traces.append(
                [
                    {
                        "qpos": robot.get_qpos()[0].cpu().tolist(),
                        "qvel": robot.get_qvel()[0].cpu().tolist(),
                    }
                ]
            )
        for step in range(STEPS):
            for i, (robot, case) in enumerate(zip(robots, ref["rows"])):
                q = robot.get_qpos()[0].cpu().numpy().astype(float)
                v = robot.get_qvel()[0].cpu().numpy().astype(float)
                x = min(abs(q[0] - q[1]) / 0.001, 1.0)
                y = 2 * x * x if x <= 0.5 else 1 - 2 * (1 - x) ** 2
                d = 0.95 + (0.99 - 0.95) * y
                R = (1 - d) / d * case["A0"]
                Kd = case["K"] * d
                k = Kd / R
                b = (1 / DT - DT * Kd) / R
                gain = (1 - DT * case["B"]) / (1 - DT * DT * Kd)
                sim.update_rigid_constraint_linear_drive(
                    f"eq_{i}", k, b, [0, gain * (v[1] - v[0]), 0]
                )
                load = case["initial"][2]
                robot.set_qf(torch.tensor([[load, -load]], device=device))
            sim.update(DT, step=1, render_final_step=False)
            for i, robot in enumerate(robots):
                traces[i].append(
                    {
                        "qpos": robot.get_qpos()[0].cpu().tolist(),
                        "qvel": robot.get_qvel()[0].cpu().tolist(),
                    }
                )
        rows = []
        for case, trace in zip(ref["rows"], traces):
            q = np.array([x["qpos"] for x in trace])
            v = np.array([x["qvel"] for x in trace])
            sq = np.array([x["qpos"] for x in case["trace"]])
            sv = np.array([x["qvel"] for x in case["trace"]])
            row = {
                "initial": case["initial"],
                "max_qpos_error": float(abs(q - sq).max()),
                "max_qvel_error": float(abs(v - sv).max()),
                "trace": trace,
            }
            rows.append(row)
            print({k: v for k, v in row.items() if k != "trace"}, flush=True)
        (OUT / "time-native.json").write_text(
            json.dumps(
                {
                    "scope": "4x500step live-state nonlinear source impulse mapping; no MuJoCo runtime in native loop",
                    "rows": rows,
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
    args = p.parse_args()
    source() if args.backend == "source" else native()
