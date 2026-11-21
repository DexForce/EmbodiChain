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

"""Private one-step equality oracle: independent position, velocity and load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")
OUT = ROOT / "alignment/point-step-response"
DT = 0.002


def source():
    import mujoco

    xml = """<mujoco><option timestep=".002" gravity="0 0 0" integrator="implicitfast"
        solver="Newton" iterations="100" tolerance="1e-12"/>
      <worldbody><body name="a"><joint name="a_slide" type="slide" axis="0 1 0" armature=".005"/>
        <inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/></body>
      <body name="b"><joint name="b_slide" type="slide" axis="0 1 0" armature=".005"/>
        <inertial mass="2" pos="0 0 0" diaginertia=".1 .1 .1"/></body></worldbody>
      <equality><connect body1="a" body2="b" anchor="0 0 0" solref=".005 1"
        solimp=".95 .99 .001 .5 2"/></equality></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    rows = []
    invmass = np.array([1 / 1.005, 1 / 2.005])
    J = np.array([1.0, -1.0])
    W = float((J * invmass) @ J)
    for position in [0.0, 1e-5, 3e-4]:
        for velocity in [0.0, 0.1, -0.1]:
            for load in [0.0, 1.0]:
                d = mujoco.MjData(m)
                q = np.array([position / 2, -position / 2])
                v = np.array([velocity / 2, -velocity / 2])
                d.qpos[:] = q
                d.qvel[:] = v
                d.qfrc_applied[:] = [load, -load]
                mujoco.mj_forward(m, d)
                K, B, imp, impp = d.efc_KBIP[1]
                R = d.efc_R[1]
                free = v + DT * invmass * np.array([load, -load])
                impulse = (
                    (1 - DT * B) * (J @ v) - DT * K * imp * (J @ q) - J @ free
                ) / (W + R)
                predicted = free + invmass * J * impulse
                mujoco.mj_step(m, d)
                np.testing.assert_allclose(d.qvel, predicted, atol=1e-12, rtol=0)
                rows.append(
                    {
                        "id": len(rows),
                        "position": position,
                        "velocity": velocity,
                        "load": load,
                        "q_before": q.tolist(),
                        "v_before": v.tolist(),
                        "K": float(K),
                        "B": float(B),
                        "impedance": float(imp),
                        "R": float(R),
                        "source_force": d.efc_force.tolist(),
                        "source_impulse": float(impulse),
                        "q_after": d.qpos.tolist(),
                        "v_after": d.qvel.tolist(),
                        "base_gain": [float(K * imp / R), float(B / R)],
                        "discrete_gain": [
                            float(K * imp / R),
                            float((1 / DT - DT * K * imp) / R),
                        ],
                        "source_relative_target": float(
                            (1 - DT * B) / (1 - DT * DT * K * imp) * velocity
                        ),
                    }
                )
    report = {"mujoco_version": mujoco.__version__, "xml": xml, "dt": DT, "rows": rows}
    (OUT / "source.json").write_text(json.dumps(report, indent=2) + "\n")
    print("source18 rows agree with independent impulse equation", flush=True)


def native(velocity_iterations):
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

    ref = json.loads((OUT / "source.json").read_text())
    device = "cuda:6"
    sim = SimulationManager(
        SimulationManagerCfg(
            physics_cfg=DefaultPhysicsCfg(
                enable_tgs=False, cache_material=True, gravity=(0, 0, 0)
            ),
            device=device,
            gpu_id=6,
            physics_dt=DT,
            headless=True,
            enable_default_plane=False,
            render_cfg=RenderCfg(renderer="no-render"),
            enable_entity_gizmo=False,
            robot_ik_gizmo=None,
            startup_summary="off",
        )
    )
    rows = []
    try:
        for mode in ["base_gain", "discrete_gain"]:
            for case in ref["rows"]:
                name = f'{mode}_{case["id"]}'
                attrs = {
                    n: LinkPhysicsOverrideCfg(
                        link_names_expr=[n],
                        attrs=RigidBodyPhysicsCfg(
                            mass_props=MassPropertiesCfg(
                                mass=mass,
                                inertia=np.eye(3) * 0.1,
                                com_position=(0, 0, 0),
                            )
                        ),
                    )
                    for n, mass in [("root", 1.0), ("a", 1.0), ("b", 2.0)]
                }
                robot = sim.add_robot(
                    RobotCfg(
                        uid=name,
                        fpath=str(ROOT / "reference/point-load.urdf"),
                        control_parts={"arm": ["a_slide", "b_slide"]},
                        init_qpos=case["q_before"],
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
                            min_velocity_iters=velocity_iterations,
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
                rows.append({"name": name, "mode": mode, "case": case, "robot": robot})
        sim.prepare()
        actual = json.loads(sim.physics.get_scene().dump_scene_config())["scene_desc"][
            "PxSceneDesc"
        ]
        assert actual["solverType"] == "PGS"
        for row in rows:
            robot = row["robot"]
            case = row["case"]
            k, b = case[row["mode"]]
            np.testing.assert_allclose(
                robot.body_data.mass.cpu(), [[1.0, 1.0, 2.0]], atol=1e-6
            )
            sim.create_rigid_constraint(
                RigidConstraintCfg(
                    name=row["name"] + "_eq",
                    rigid_object_a_uid=row["name"],
                    rigid_object_b_uid=row["name"],
                    articulation_link_a="a",
                    articulation_link_b="b",
                    local_frame_a=np.eye(4),
                    local_frame_b=np.eye(4),
                    constraint_type="spherical",
                    linear_drive_stiffness=k,
                    linear_drive_damping=b,
                    linear_drive_acceleration=False,
                )
            )
            robot.set_state(
                qpos=torch.tensor([case["q_before"]], device=device),
                qvel=torch.tensor([case["v_before"]], device=device),
                clear_dynamics=True,
                clamp_qpos=False,
            )
            robot.set_qf(torch.tensor([[case["load"], -case["load"]]], device=device))
        sim.update(DT, step=1, render_final_step=False)
        results = []
        for row in rows:
            q = row["robot"].get_qpos()[0].cpu().numpy()
            v = row["robot"].get_qvel()[0].cpu().numpy()
            case = row["case"]
            results.append(
                {
                    "mode": row["mode"],
                    "id": case["id"],
                    "q_after": q.tolist(),
                    "v_after": v.tolist(),
                    "q_error": float(np.max(abs(q - case["q_after"]))),
                    "v_error": float(np.max(abs(v - case["v_after"]))),
                    "motion_final_difference": float(
                        np.max(abs((q - case["q_before"]) / DT - v))
                    ),
                }
            )
        report = {
            "scope": "single step, no collision/arm drives; explicit source row state; native target velocity still zero",
            "velocity_iterations": velocity_iterations,
            "results": results,
        }
        path = OUT / f"native-v{velocity_iterations}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        for mode in ["base_gain", "discrete_gain"]:
            group = [r for r in results if r["mode"] == mode]
            zero = [r for r in group if ref["rows"][r["id"]]["velocity"] == 0]
            print(
                mode,
                "allvmax",
                max(r["v_error"] for r in group),
                "zerovmax",
                max(r["v_error"] for r in zero),
                flush=True,
            )
        print(path, flush=True)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=["source", "native"])
    p.add_argument("--velocity-iterations", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source() if args.backend == "source" else native(args.velocity_iterations)
