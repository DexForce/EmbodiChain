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

"""Private phase-resolved real-robot equality oracle, separate from the task.

Contacts, limits and gravity are removed only in this characterization. Native
processes consume exported forces and expected states, without importing MuJoCo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")
REF = ROOT / "alignment-point-response-pgs"
OUT = ROOT / "alignment/real14-coupled-impulse"
ASSET = json.loads((REF / "robot/robot.json").read_text())
CONTRACT = json.loads((REF / "physical-contract.json").read_text())
DT = 0.002
MODES = ("none", "point", "mimic", "both")
TIERS = ("undamped", "passive", "controlled")


def make_inputs():
    import h5py

    expert = np.load(ROOT / "alignment/expert-probe-reference.npz")
    with h5py.File(ROOT / "alignment/official-queries.h5") as f:
        q, v = f["qpos"][:], f["qvel"][:]
    specs = [
        ("initial_zero", expert["qpos"][0], np.zeros(20)),
        ("closure_static", expert["qpos"][17], np.zeros(20)),
        ("closure_moving", expert["qpos"][17], expert["qvel"][17]),
    ]
    specs += [(f"query_{i}", q[i], v[i]) for i in (5, 7, 12)]
    rows = []
    for i, (name, q0, v0) in enumerate(specs):
        q0, v0 = q0.copy(), v0.copy()
        if name == "closure_static":
            q0[6] += 0.001
        # Exactly representable shared inputs separate state-rounding errors
        # from a solver-stage difference. Cube coordinates do not couple to M14.
        q0[:14] = q0[:14].astype(np.float32)
        v0[:14] = v0[:14].astype(np.float32)
        load = np.zeros(20)
        if i >= 2:
            load[:6] = [0.01, 0.02, -0.01, 0.005, -0.002, 0.001]
            load[6], load[10] = 0.002, -0.002
        control = np.r_[
            q0[:6] + [0.005, -0.003, 0.004, -0.002, 0.001, -0.001],
            255 * np.clip(q0[6] / 0.8, 0, 1),
        ]
        rows.append(
            {
                "name": name,
                "qpos": q0.tolist(),
                "qvel": v0.tolist(),
                "load": load.tolist(),
                "control": control.tolist(),
            }
        )
    return rows


def export_source():
    import mujoco
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

    env = CubeEnv(
        env_type="single",
        ob_type="states",
        multiview=False,
        visualize_info=False,
        terminate_at_goal=True,
    )
    try:
        env.reset(seed=42)
        m, d = env.model, env.data
        m.opt.gravity[:] = 0
        original_flags = int(m.opt.disableflags)
        flags = ["CONTACT", "LIMIT", "FRICTIONLOSS", "WARMSTART"]
        for key in flags:
            original_flags |= int(getattr(mujoco.mjtDisableBit, f"mjDSBL_{key}"))
        pattern = np.zeros((m.nv, m.nv), bool)
        for i in range(m.nv):
            start, count = m.M_rowadr[i], m.M_rownnz[i]
            pattern[i, m.M_colind[start : start + count]] = True
        pattern |= pattern.T
        rows = []
        for tier in TIERS:
            m.opt.disableflags = original_flags
            if tier == "undamped":
                m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SPRING) | int(
                    mujoco.mjtDisableBit.mjDSBL_DAMPER
                )
            if tier != "controlled":
                m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            for mode in MODES:
                for case in make_inputs():
                    env.set_state(np.array(case["qpos"]), np.array(case["qvel"]))
                    d.eq_active[:] = [
                        mode in ("point", "both"),
                        mode in ("point", "both"),
                        mode in ("mimic", "both"),
                    ]
                    d.qfrc_applied[:] = case["load"]
                    d.ctrl[:] = case["control"]
                    mujoco.mj_forward(m, d)
                    mass = np.empty((m.nv, m.nv))
                    mujoco.mj_fullM(m, d, mass)
                    mass = mass[:14, :14]
                    J = d.efc_J.reshape(d.nefc, m.nv)[:, :14].copy()
                    smooth = d.qacc_smooth[:14].copy()
                    force = (
                        d.qfrc_actuator[:14] + d.qfrc_passive[:14] + d.qfrc_applied[:14]
                    )
                    a = d.qacc[:14].copy()
                    eqforce = (
                        np.linalg.solve(
                            J @ np.linalg.solve(mass, J.T) + np.diag(d.efc_R),
                            d.efc_aref - J @ smooth,
                        )
                        if len(J)
                        else np.zeros(0)
                    )
                    predicted = smooth + np.linalg.solve(mass, J.T @ eqforce)
                    row = {
                        **case,
                        "tier": tier,
                        "mode": mode,
                        "mass": mass.tolist(),
                        "applied_smooth_force": force.tolist(),
                        "smooth_acceleration": smooth.tolist(),
                        "qacc_pre_implicit": a.tolist(),
                        "J": J.tolist(),
                        "equality_ids": d.efc_id.tolist(),
                        "position": d.efc_pos.tolist(),
                        "velocity": d.efc_vel.tolist(),
                        "KBIP": d.efc_KBIP.tolist(),
                        "R": d.efc_R.tolist(),
                        "aref": d.efc_aref.tolist(),
                        "constraint_force": d.efc_force.tolist(),
                        "dense_acceleration_error": float(abs(predicted - a).max()),
                        "solver_iterations": d.solver_niter.tolist(),
                    }
                    assert np.all(d.efc_type == 0), d.efc_type
                    assert row["dense_acceleration_error"] < 1e-6, row
                    q0, v0 = d.qpos[:14].copy(), d.qvel[:14].copy()
                    row["v_pre_implicit"] = (v0 + DT * a).tolist()
                    mujoco.mj_implicit(m, d)
                    derivative = np.zeros((m.nv, m.nv))
                    for j in range(m.nv):
                        start, count = m.D_rowadr[j], m.D_rownnz[j]
                        derivative[j, m.D_colind[start : start + count]] = d.qDeriv[
                            start : start + count
                        ]
                    derivative = -derivative[:14, :14] * pattern[:14, :14]
                    np.testing.assert_allclose(
                        derivative, derivative.T, atol=1e-12, rtol=0
                    )
                    vp = v0 + DT * np.linalg.solve(mass + DT * derivative, mass @ a)
                    qp = q0 + DT * vp
                    row.update(
                        {
                            "velocity_derivative": derivative.tolist(),
                            "qpos_after": d.qpos[:14].tolist(),
                            "qvel_after": d.qvel[:14].tolist(),
                            "implicit_velocity_error": float(
                                abs(vp - d.qvel[:14]).max()
                            ),
                            "implicit_position_error": float(
                                abs(qp - d.qpos[:14]).max()
                            ),
                        }
                    )
                    assert row["implicit_velocity_error"] < 1e-10
                    assert row["implicit_position_error"] < 1e-10
                    rows.append(row)
        report = {
            "scope": "isolated real14, same canonical topology/mass/armature/equality; contact/limits/frictionloss/gravity removed; quantized shared q/v",
            "mujoco": mujoco.__version__,
            "dt": DT,
            "rows": rows,
            "iterations": int(m.opt.iterations),
            "tolerance": float(m.opt.tolerance),
            "maximum_dense_acceleration_error": max(
                r["dense_acceleration_error"] for r in rows
            ),
            "maximum_implicit_velocity_error": max(
                r["implicit_velocity_error"] for r in rows
            ),
        }
        OUT.mkdir(exist_ok=True, parents=True)
        (OUT / "source.json").write_text(json.dumps(report, indent=2) + "\n")
        print({k: v for k, v in report.items() if k != "rows"}, flush=True)
    finally:
        env.close()


def make_urdf(mode):
    path = ROOT / "reference/real14-coupled-impulse"
    path.mkdir(parents=True, exist_ok=True)
    destination = path / (
        "with-mimic.urdf" if mode in ("mimic", "both") else "without-mimic.urdf"
    )
    xml = ET.parse(REF / "robot/robot.urdf")
    for mesh in xml.getroot().iter("mesh"):
        mesh.attrib["filename"] = str(
            (REF / "robot" / mesh.attrib["filename"]).resolve()
        )
    for joint in xml.getroot().findall("joint"):
        for dynamics in joint.findall("dynamics"):
            dynamics.attrib.update({"damping": "0", "friction": "0"})
        for limit in joint.findall("limit"):
            limit.attrib.update(
                {"lower": "-1000", "upper": "1000", "effort": "1000000"}
            )
        if mode not in ("mimic", "both"):
            for mimic in joint.findall("mimic"):
                joint.remove(mimic)
    xml.write(destination)
    return destination


def run_native(mode, gains, velocity_iterations, mimic_gains):
    import torch
    from real14_link_bias import LinkBiasKinematics
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        DefaultPhysicsCfg,
        RenderCfg,
        RobotCfg,
        ArticulationRootPropertiesCfg,
        JointDrivePropertiesCfg,
        RigidBodyPhysicsCfg,
        DefaultRigidBodyPropertiesCfg,
        CollisionPropertiesCfg,
        LinkPhysicsOverrideCfg,
        MassPropertiesCfg,
        RigidConstraintCfg,
    )
    from embodichain_tasks.manipulation.ogbench_cube import (
        _base_point_compliance,
        _base_mimic_compliance,
    )

    device = "cuda:6"
    source = json.loads((OUT / "source.json").read_text())
    selected = [r for r in source["rows"] if r["mode"] == mode]
    model = LinkBiasKinematics()
    sim = SimulationManager(
        SimulationManagerCfg(
            device=device,
            gpu_id=6,
            physics_cfg=DefaultPhysicsCfg(
                enable_tgs=False, cache_material=True, gravity=(0, 0, 0)
            ),
            physics_dt=DT,
            headless=True,
            enable_default_plane=False,
            render_cfg=RenderCfg(renderer="no-render"),
            enable_entity_gizmo=False,
            robot_ik_gizmo=None,
            startup_summary="off",
        )
    )
    try:
        from embodichain.lab.sim.cfg import RigidObjectCfg
        from embodichain.lab.sim.shapes import CubeCfg

        sim.add_rigid_object(
            RigidObjectCfg(
                uid="distant_cube",
                init_pos=(20.0, 0.0, 20.0),
                shape=CubeCfg(size=[0.02] * 3),
                attrs=RigidBodyPhysicsCfg(
                    rigid_props=DefaultRigidBodyPropertiesCfg(
                        has_gravity=False,
                        min_position_iters=100,
                        min_velocity_iters=1,
                        sleep_threshold=0,
                    )
                ),
            )
        )
        links = {
            b["name"]: LinkPhysicsOverrideCfg(
                link_names_expr=[b["name"]],
                attrs=RigidBodyPhysicsCfg(
                    mass_props=MassPropertiesCfg(
                        mass=b["mass"],
                        inertia=np.array(b["inertia_matrix"]),
                        com_position=b["ipos"],
                    )
                ),
            )
            for b in ASSET["bodies"]
        }
        robot = sim.add_robot(
            RobotCfg(
                uid="oracle",
                fpath=str(make_urdf(mode)),
                init_local_pose=np.array(ASSET["root_pose"]),
                control_parts={"arm": ASSET["joint_names"]},
                init_qpos=selected[0]["qpos"][:14],
                solver_cfg=None,
                build_pk_chain=False,
                asset_physics_mode="overlay",
                link_attrs=links,
                attrs=RigidBodyPhysicsCfg(
                    rigid_props=DefaultRigidBodyPropertiesCfg(
                        has_gravity=False, linear_damping=0, angular_damping=0
                    ),
                    collision_props=CollisionPropertiesCfg(collision_enabled=True),
                ),
                root_props=ArticulationRootPropertiesCfg(
                    fixed_base=True,
                    self_collision_enabled=False,
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
                    armature=dict(
                        zip(ASSET["joint_names"], CONTRACT["dof_armature"][:14])
                    ),
                ),
            )
        )
        sim.prepare()
        ids = [robot.joint_names.index(n) for n in ASSET["joint_names"]]
        scene = json.loads(sim.physics.get_scene().dump_scene_config())
        assert scene["scene_desc"]["PxSceneDesc"]["solverType"] == "PGS"
        mass = robot.body_data.mass.cpu().numpy()[0]
        expected = {b["name"]: b["mass"] for b in ASSET["bodies"]}
        np.testing.assert_allclose(
            mass, [expected[n] for n in robot.link_names], atol=1e-6, rtol=1e-6
        )
        if mode in ("mimic", "both"):
            (mimic,) = ASSET["mimics"]
            robot.set_mimic_compliance(
                *_base_mimic_compliance(mimic["solref"], mimic["solimp"])
            )
        if mode in ("point", "both"):
            eq = CONTRACT["equality"]
            for index, endpoints in model.points:
                frame = []
                for link, local, rotation in endpoints:
                    value = np.eye(4)
                    value[:3, :3] = rotation
                    value[:3, 3] = local
                    frame.append(value)
                A0 = sum(
                    CONTRACT["body_invweight0"][eq[k][index]][0]
                    for k in ("obj1id", "obj2id")
                )
                k, b = _base_point_compliance(
                    eq["solref"][index], eq["solimp"][index], A0
                )
                sim.create_rigid_constraint(
                    RigidConstraintCfg(
                        name=f"eq_{index}",
                        rigid_object_a_uid="oracle",
                        rigid_object_b_uid="oracle",
                        articulation_link_a=endpoints[0][0],
                        articulation_link_b=endpoints[1][0],
                        local_frame_a=frame[0],
                        local_frame_b=frame[1],
                        constraint_type="spherical",
                        linear_drive_stiffness=k,
                        linear_drive_damping=b,
                        linear_drive_point_anchors=(gains == "point_exact"),
                    )
                )
        results = []
        for row in selected:
            q, v = np.array(row["qpos"][:14]), np.array(row["qvel"][:14])
            robot.set_state(
                qpos=torch.tensor(q[None], device=device, dtype=torch.float32),
                qvel=torch.tensor(v[None], device=device, dtype=torch.float32),
                joint_ids=ids,
                clear_dynamics=True,
                clamp_qpos=False,
            )
            actual_q = robot.get_qpos()[0, ids].cpu().numpy().astype(float)
            actual_v = robot.get_qvel()[0, ids].cpu().numpy().astype(float)
            np.testing.assert_array_equal(actual_q, q)
            np.testing.assert_array_equal(actual_v, v)
            if gains in ("discrete", "discrete_bias", "point_exact") and mode in (
                "point",
                "both",
            ):
                for point in model.evaluate(q, v):
                    index = point["index"]
                    columns = np.flatnonzero(np.array(row["equality_ids"]) == index)
                    K, B, d = np.array(row["KBIP"])[columns[0], :3]
                    R = row["R"][columns[0]]
                    jv = (
                        point["velocity"]
                        if gains == "point_exact"
                        else point["velocity"]
                        - np.cross(point["omega_a"], point["residual"])
                    )
                    target = (
                        point["rotation_a"].T
                        @ (jv - DT * B * point["velocity"] - DT * point["Jdotv"])
                        / (1 - DT * DT * K * d)
                    )
                    if gains in ("discrete_bias", "point_exact"):
                        extra = model.point_solver_bias(
                            q, v, common_point=(gains != "point_exact")
                        )[index]
                        target += (
                            DT * point["rotation_a"].T @ extra / (1 - DT * DT * K * d)
                        )
                    sim.update_rigid_constraint_linear_drive(
                        f"eq_{index}", K * d / R, (1 / DT - DT * K * d) / R, target
                    )
            if mimic_gains in ("discrete", "velocity") and mode in ("mimic", "both"):
                columns = np.flatnonzero(np.array(row["equality_ids"]) == 2)
                K, B, d = np.array(row["KBIP"])[columns[0], :3]
                R = row["R"][columns[0]]
                jac = np.array(row["J"])[columns[0]]
                response = jac @ np.linalg.solve(np.array(row["mass"]), jac)
                omega = np.sqrt(K * d * response / R)
                zeta = (1 / DT - DT * K * d) * response / (2 * omega * R)
                robot.set_mimic_compliance(float(omega), float(zeta))
            constraint_velocity_force = np.zeros(14)
            if mimic_gains == "velocity" and mode in ("mimic", "both"):
                # Splitting the missing target term into an applied joint force
                # gives total impulse lambda_total=lambda_native+h*sigma:
                # R*lambda_total+J*vnew = native_rhs+h*R*sigma.
                # This is a constraint term, excluded from smooth qDeriv.
                sigma = (1 - DT * B) * (jac @ v) / (DT * R)
                constraint_velocity_force = jac * sigma
            applied = np.array(row["applied_smooth_force"]) + constraint_velocity_force
            robot.set_qf(
                torch.tensor(applied[None], device=device, dtype=torch.float32),
                joint_ids=ids,
            )
            sim.update(DT, step=1, render_final_step=False)
            qp = robot.get_qpos()[0, ids].cpu().numpy().astype(float)
            vp = robot.get_qvel()[0, ids].cpu().numpy().astype(float)
            motion = (qp - q) / DT
            M, D = np.array(row["mass"]), np.array(row["velocity_derivative"])
            final_corrected = v + np.linalg.solve(M + DT * D, M @ (vp - v))
            motion_corrected = v + np.linalg.solve(M + DT * D, M @ (motion - v))
            expected_v = np.array(row["qvel_after"])
            result = {
                "name": row["name"],
                "tier": row["tier"],
                "q_after": qp.tolist(),
                "v_final": vp.tolist(),
                "v_motion": motion.tolist(),
                "constraint_velocity_force": constraint_velocity_force.tolist(),
                "v_final_corrected": final_corrected.tolist(),
                "v_motion_corrected": motion_corrected.tolist(),
                "velocity_error": float(abs(vp - expected_v).max()),
                "pre_implicit_final_error": float(
                    abs(vp - row["v_pre_implicit"]).max()
                ),
                "pre_implicit_motion_error": float(
                    abs(motion - row["v_pre_implicit"]).max()
                ),
                "final_corrected_error": float(abs(final_corrected - expected_v).max()),
                "motion_corrected_error": float(
                    abs(motion_corrected - expected_v).max()
                ),
                "motion_final_difference": float(abs(motion - vp).max()),
                "position_error": float(abs(qp - row["qpos_after"]).max()),
                "position_quantum_velocity_bound": float(
                    np.max(np.abs(np.spacing(q.astype(np.float32)))) / DT
                ),
            }
            results.append(result)
            print(
                {
                    k: result[k]
                    for k in (
                        "name",
                        "tier",
                        "velocity_error",
                        "pre_implicit_final_error",
                        "final_corrected_error",
                        "motion_final_difference",
                    )
                },
                flush=True,
            )
        report = {
            "scope": source["scope"],
            "mode": mode,
            "point_gains": gains,
            "velocity_iterations": velocity_iterations,
            "mimic_gains": mimic_gains,
            "native_scene": scene,
            "native_body_mass": mass.tolist(),
            "native_joint_order": robot.joint_names,
            "joint_ids": ids,
            "results": results,
        }
        suffix = (
            "-timemimic"
            if mimic_gains == "discrete"
            else ("-velocitymimic" if mimic_gains == "velocity" else "")
        )
        path = (
            OUT / f"native-{mode}-{gains}{suffix}-farbody-v{velocity_iterations}.json"
        )
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(path, flush=True)
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("source", "native"))
    parser.add_argument("--mode", choices=MODES, default="none")
    parser.add_argument(
        "--point-gains",
        choices=("base", "discrete", "discrete_bias", "point_exact"),
        default="base",
    )
    parser.add_argument("--velocity-iterations", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--mimic-gains", choices=("base", "discrete", "velocity"), default="base"
    )
    args = parser.parse_args()
    if args.backend == "source":
        export_source()
    else:
        run_native(
            args.mode, args.point_gains, args.velocity_iterations, args.mimic_gains
        )
