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

"""Private isolated-hinge oracle for official arm gains and force saturation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")
DT, STEPS, INERTIA, ARMATURE = 0.002, 100, 0.1, 0.1


def cases():
    contract = json.loads(
        (ROOT / "alignment-point-response-pgs/physical-contract.json").read_text()
    )
    for index in [0, 3, 5]:
        actuator = contract["actuators"][index]
        k, d = actuator["gainprm"][0], -actuator["biasprm"][2]
        limit = actuator["forcerange"][1]
        for regime in ["small_target", "initial_velocity", "near_limit", "saturated"]:
            target = np.zeros(STEPS)
            q, v = 0.0, 0.0
            if regime == "small_target":
                target[:] = 0.001 * np.sin(np.arange(STEPS) * 0.05)
            elif regime == "initial_velocity":
                q, v = 0.001, 0.05
            elif regime == "near_limit":
                target[:25] = limit / k * 1.00001
            else:
                target[:25] = 0.1
            yield {
                "name": actuator["name"].replace("/", "_") + "_" + regime,
                "regime": regime,
                "kp": k,
                "kd": d,
                "force_limit": limit,
                "q0": q,
                "v0": v,
                "target": target.tolist(),
            }


def source(rows):
    import mujoco

    result = []
    for case in rows:
        k, d, limit = case["kp"], case["kd"], case["force_limit"]
        xml = f"""<mujoco><option timestep="{DT}" gravity="0 0 0" integrator="implicitfast"
          iterations="100" tolerance="1e-8"/>
          <worldbody><body name="hinged"><joint name="hinge" type="hinge" axis="0 0 1" armature="{ARMATURE}"/>
          <inertial mass="1" pos="0 0 0" diaginertia="{INERTIA} {INERTIA} {INERTIA}"/></body></worldbody>
          <actuator><general joint="hinge" gainprm="{k}" biastype="affine" biasprm="0 {-k} {-d}"
          forcelimited="true" forcerange="{-limit} {limit}"/></actuator></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qpos[0], data.qvel[0] = case["q0"], case["v0"]
        trace = [{"qpos": float(data.qpos[0]), "qvel": float(data.qvel[0])}]
        for target in case["target"]:
            data.ctrl[0] = target
            mujoco.mj_step(model, data)
            trace.append(
                {
                    "qpos": float(data.qpos[0]),
                    "qvel": float(data.qvel[0]),
                    "actuator_force": float(data.actuator_force[0]),
                    "smooth_force_velocity_derivative": float(data.qDeriv[0]),
                }
            )
        result.append({"case": case, "xml": xml, "trace": trace})
    return {"backend": "mujoco", "version": mujoco.__version__, "results": result}


def native(rows):
    import torch
    import dexsim

    all_cases = [
        (case, mapping)
        for mapping in ["source_coefficients", "subtract_k_dt"]
        for case in rows
    ]
    dexsim.set_physics_config(
        enable_tgs=False,
        gravity=np.zeros(3, np.float32),
        cache_material=True,
        enable_friction_every_iteration=True,
    )
    config = dexsim.WorldConfig()
    config.open_windows = False
    config.renderer = dexsim.types.Renderer.NORENDER
    config.enable_gpu_sim = True
    config.direct_gpu_api = True
    world = dexsim.World(config)
    world.set_manual_update(True)
    world.enable_physics(False)
    builder = dexsim.spawn.SceneBuilder(world)
    identity = np.eye(4, dtype=np.float32)
    body = dexsim.spawn.RigidBodyPhysicsDesc.dynamic(
        mass=1.0,
        inertia=np.eye(3) * INERTIA,
        dexsim=dexsim.spawn.DexsimPhysicsDesc(
            has_gravity=False, linear_damping=0, angular_damping=0
        ),
    )
    for i, (case, mapping) in enumerate(all_cases):
        damping = case["kd"] - (DT * case["kp"] if mapping == "subtract_k_dt" else 0)
        builder.add_articulation(
            dexsim.spawn.ArticulationDesc(
                name=f"hinge_{i}",
                per_env=False,
                fixed_base=True,
                links=[
                    dexsim.spawn.LinkDesc("root", "", identity, rigid_body=body),
                    dexsim.spawn.LinkDesc("hinged", "root", identity, rigid_body=body),
                ],
                joints=[
                    dexsim.spawn.JointDesc(
                        "hinge",
                        "root",
                        "hinged",
                        dexsim.engine.JointType.REVOLUTE,
                        axis=np.array([0, 0, 1], np.float32),
                        lower_limit=-3.0,
                        upper_limit=3.0,
                        armature=ARMATURE,
                        dexsim=dexsim.spawn.DexsimJointDesc(
                            stiffness=case["kp"],
                            damping=damping,
                            max_force=case["force_limit"],
                            joint_friction=0.0,
                            drive_mode=dexsim.types.DriveType.FORCE,
                        ),
                    )
                ],
            )
        )
    scene = builder.finalize()
    handles = [scene.get_articulation(f"hinge_{i}") for i in range(len(all_cases))]
    for handle in handles:
        handle.set_articulation_flag(
            dexsim.types.ArticulationFlag.DRIVE_LIMITS_ARE_FORCES, True
        )
    world.init_gpu_physics()
    batch = scene.create_articulation_batch(handles)
    q = torch.tensor(
        [[c["q0"]] for c, _ in all_cases], dtype=torch.float32, device="cuda"
    )
    v = torch.tensor(
        [[c["v0"]] for c, _ in all_cases], dtype=torch.float32, device="cuda"
    )
    batch.apply_joint_position(q)
    batch.apply_joint_velocity(v)
    batch.apply_joint_target_velocity(torch.zeros_like(q))
    armature = torch.empty_like(q)
    batch.fetch_joint_properties(armature=armature)
    np.testing.assert_allclose(armature.cpu(), ARMATURE, atol=1e-7)
    traces = [[{"qpos": float(c["q0"]), "qvel": float(c["v0"])}] for c, _ in all_cases]
    for step in range(STEPS):
        target = torch.tensor(
            [[c["target"][step]] for c, _ in all_cases],
            dtype=torch.float32,
            device="cuda",
        )
        batch.apply_joint_target_position(target)
        world.update(DT)
        batch.fetch_joint_position(q)
        batch.fetch_joint_velocity(v)
        qa, va = q.cpu().numpy().ravel(), v.cpu().numpy().ravel()
        for i in range(len(all_cases)):
            torque = (INERTIA + ARMATURE) * (float(va[i]) - traces[i][-1]["qvel"]) / DT
            assert abs(torque) <= all_cases[i][0]["force_limit"] * 1.0001, torque
            traces[i].append(
                {
                    "qpos": float(qa[i]),
                    "qvel": float(va[i]),
                    "torque_from_momentum_Nm": torque,
                }
            )
    report = {
        "backend": "dexsim",
        "drive_limits_are_forces": True,
        "solver_iterations": "backend default; not an iteration sweep",
        "results": [
            {"case": c, "mapping": m, "trace": t}
            for (c, m), t in zip(all_cases, traces)
        ],
        "scene": json.loads(world.get_physics_scene().dump_scene_config()),
    }
    del batch, handles
    scene.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["source", "native"])
    args = parser.parse_args()
    report = (
        source(list(cases())) if args.backend == "source" else native(list(cases()))
    )
    report.update(
        {
            "scope": "isolated collisionless hinge; official gains/caps/dt/armature; no production task change",
            "inertia_kg_m2": INERTIA,
            "armature_kg_m2": ARMATURE,
            "dt_s": DT,
        }
    )
    path = ROOT / "alignment" / f"implicit-pd-{args.backend}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path, flush=True)
