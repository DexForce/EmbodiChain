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

"""Private normal-contact oracle; identical geometry/load before solver mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")
DT = 0.002
MASS = 0.07936
RADIUS = float(np.float32(0.02))
STEPS = 300


def fixtures():
    contract = json.loads(
        (ROOT / "alignment-point-response-pgs/physical-contract.json").read_text()
    )
    geoms = {g["name"]: g for g in contract["geoms"]}
    for support in ["floor", "pad"]:
        material = geoms["floor" if support == "floor" else "ur5e/robotiq/right_pad1"]
        for shape in ["sphere", "box"]:
            for motion in ["rest", "press", "impact"]:
                yield {
                    "name": f"{support}_{shape}_{motion}",
                    "support": support,
                    "shape": shape,
                    "motion": motion,
                    "load_N": 20.0 if motion == "press" else 0.0,
                    "initial_bottom_m": 0.0 if motion == "impact" else 1e-4,
                    "initial_vz_m_s": -0.5 if motion == "impact" else 0.0,
                    "support_half_size": (
                        [1.0, 1.0, 0.01]
                        if support == "floor"
                        else [0.011, 0.009375, 0.004]
                    ),
                    "support_material": material,
                    "moving_material": geoms["object_0"],
                }


def numbers(a):
    return " ".join(str(float(x)) for x in a)


def material_xml(g):
    return (
        f'condim="{g["condim"]}" friction="{numbers(g["friction"])}" '
        f'solref="{numbers(g["solref"])}" solimp="{numbers(g["solimp"])}" '
        f'priority="{g["priority"]}"'
    )


def summarize(case, rows):
    forces = np.array([r["normal_force_N"] for r in rows])
    return {
        "name": case["name"],
        "peak_normal_force_N": float(forces.max()),
        "peak_force_step": int(forces.argmax() + 1),
        "steady_z_m": float(np.mean([r["z_after_m"] for r in rows[-20:]])),
        "steady_penetration_m": float(
            RADIUS - np.mean([r["z_after_m"] for r in rows[-20:]])
        ),
        "steady_normal_force_N": float(np.mean(forces[-20:])),
        "expected_static_force_N": MASS * 9.81 + case["load_N"],
        "max_momentum_force_error_N": float(
            max(abs(r["normal_force_N"] - r["momentum_force_N"]) for r in rows)
        ),
    }


def run_source(cases):
    import mujoco

    results = []
    for case in cases:
        a, b = case["support_material"], case["moving_material"]
        size = case["support_half_size"]
        inertia = MASS * RADIUS**2 * (2 / 5 if case["shape"] == "sphere" else 2 / 3)
        moving_size = [RADIUS] if case["shape"] == "sphere" else [RADIUS] * 3
        xml = f"""<mujoco><option timestep=".002" gravity="0 0 -9.81" integrator="implicitfast"
          cone="elliptic" solver="Newton" iterations="100" tolerance="1e-8" impratio="10"/>
          <worldbody><geom name="support" type="box" size="{numbers(size)}"
            pos="0 0 {-size[2]}" {material_xml(a)}/>
          <body name="moving" pos="0 0 {RADIUS + case['initial_bottom_m']}"><freejoint/>
            <inertial mass="{MASS}" pos="0 0 0" diaginertia="{numbers([inertia]*3)}"/>
            <geom name="moving" type="{case['shape']}" size="{numbers(moving_size)}" {material_xml(b)}/>
          </body></worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qvel[2] = case["initial_vz_m_s"]
        rows, pair = [], None
        for step in range(STEPS):
            before_z, before_v = float(data.qpos[2]), float(data.qvel[2])
            data.xfrc_applied[1, 2] = -case["load_N"]
            mujoco.mj_step(model, data)
            contacts = []
            normal = 0.0
            for i in range(data.ncon):
                contact = data.contact[i]
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                normal += abs(contact.frame[2] * force[0])
                adr = contact.efc_address
                contacts.append(
                    {
                        "distance_m": float(contact.dist),
                        "force_N": force.tolist(),
                        "position": contact.pos.tolist(),
                        "efc_address": int(adr),
                    }
                )
                if pair is None and adr >= 0:
                    pair = {
                        "solref": contact.solref.tolist(),
                        "solimp": contact.solimp.tolist(),
                        "friction": contact.friction.tolist(),
                        "dim": int(contact.dim),
                        "R": data.efc_R[adr : adr + 3].tolist(),
                        "KBIP": data.efc_KBIP[adr : adr + 3].tolist(),
                        "body_invweight0": model.body_invweight0.tolist(),
                    }
            rows.append(
                {
                    "physics_step": step + 1,
                    "z_before_m": before_z,
                    "vz_before_m_s": before_v,
                    "z_after_m": float(data.qpos[2]),
                    "vz_after_m_s": float(data.qvel[2]),
                    "normal_force_N": normal,
                    "momentum_force_N": MASS * (data.qvel[2] - before_v) / DT
                    + MASS * 9.81
                    + case["load_N"],
                    "contacts": contacts,
                }
            )
        result = {
            "case": case,
            "compiled_pair": pair,
            "summary": summarize(case, rows),
            "rows": rows,
            "xml": xml,
        }
        results.append(result)
        print(result["summary"], flush=True)
    return {"backend": "mujoco", "version": mujoco.__version__, "results": results}


def run_native(cases, recipe):
    import torch
    import dexsim

    dexsim.set_physics_config(
        gravity=np.array([0, 0, -9.81], np.float32),
        enable_tgs=False,
        enable_friction_every_iteration=True,
        cache_material=True,
    )
    cfg = dexsim.WorldConfig()
    cfg.open_windows = False
    cfg.renderer = dexsim.types.Renderer.NORENDER
    cfg.enable_gpu_sim = True
    cfg.direct_gpu_api = True
    world = dexsim.World(cfg)
    world.set_manual_update(True)
    world.enable_physics(False)
    builder = dexsim.spawn.SceneBuilder(world)
    for index, case in enumerate(cases):
        size = case["support_half_size"]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [3.0 * index, 0, -size[2]]
        material = case["support_material"]
        support_geom = dexsim.spawn.GeometryDesc.cube(np.array(size) * 2)
        support_contact = dexsim.spawn.CollisionDesc.from_geometry(support_geom)
        support_contact.dexsim = dexsim.spawn.DexsimCollisionDesc(
            static_friction=material["friction"][0],
            dynamic_friction=material["friction"][0],
            restitution=0.0,
            contact_offset=1e-6,
            rest_offset=0.0,
            friction_combine_mode=1,
        )
        builder.add_cube(
            case["name"] + "_support",
            tuple(np.array(size) * 2),
            pose=pose,
            collisions=[support_contact],
            physics=dexsim.spawn.RigidBodyPhysicsDesc.static(),
            per_env=False,
        )
        pose = pose.copy()
        pose[2, 3] = RADIUS + case["initial_bottom_m"]
        moving_geom = (
            dexsim.spawn.GeometryDesc.sphere(RADIUS)
            if case["shape"] == "sphere"
            else dexsim.spawn.GeometryDesc.cube([RADIUS * 2] * 3)
        )
        moving_contact = dexsim.spawn.CollisionDesc.from_geometry(moving_geom)
        moving_contact.dexsim = dexsim.spawn.DexsimCollisionDesc(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            contact_offset=1e-6,
            rest_offset=0.0,
            friction_combine_mode=1,
        )
        if recipe == "base_force":
            pair = next(
                row["compiled_pair"]
                for row in json.loads(
                    (ROOT / "alignment/normal-contact-source-baseline.json").read_text()
                )["results"]
                if row["case"]["name"] == case["name"]
            )
            tau, ratio = pair["solref"]
            d0, dmax = pair["solimp"][:2]
            regularizer = (1 - d0) / d0 / MASS
            moving_contact.dexsim.compliant_contact_stiffness = (
                d0 / (dmax * tau * ratio) ** 2 / regularizer
            )
            moving_contact.dexsim.compliant_contact_damping = (
                2 / (dmax * tau) / regularizer
            )
            case["native_compliant_coefficients"] = {
                "stiffness_N_m": moving_contact.dexsim.compliant_contact_stiffness,
                "damping_N_s_m": moving_contact.dexsim.compliant_contact_damping,
                "source_base_R": regularizer,
                "model": "implicit force spring; source base force units, not nonlinear or discrete-time equality",
            }
        inertia = MASS * RADIUS**2 * (2 / 5 if case["shape"] == "sphere" else 2 / 3)
        physics = dexsim.spawn.RigidBodyPhysicsDesc.dynamic(
            mass=MASS,
            inertia=np.eye(3) * inertia,
            dexsim=dexsim.spawn.DexsimPhysicsDesc(
                linear_damping=0,
                angular_damping=0,
                min_position_iters=100,
                min_velocity_iters=1,
                sleep_threshold=0,
            ),
        )
        builder.add_object(
            dexsim.spawn.ObjectDesc(
                name=case["name"],
                pose=pose,
                renders=[dexsim.spawn.RenderDesc.from_geometry(moving_geom)],
                collisions=[moving_contact],
                physics=physics,
                per_env=False,
            )
        )
    scene = builder.finalize()
    handles = [scene.get_actor(c["name"]) for c in cases]
    batch = scene.create_rigid_body_batch(handles)
    world.init_gpu_physics()
    device = "cuda:0"
    queries = [
        scene.create_contact_query(
            [h], match="any", capacity=16, device=device, frame="world"
        )
        for h in handles
    ]
    initial_pose = torch.zeros((len(cases), 7), device=device)
    initial_pose[:, 3] = 1
    initial_pose[:, 4] = torch.arange(len(cases), device=device) * 3.0
    initial_pose[:, 6] = torch.tensor(
        [RADIUS + c["initial_bottom_m"] for c in cases], device=device
    )
    assert batch.apply_pose(initial_pose) == len(cases)
    velocity = torch.zeros((len(cases), 3), device=device)
    velocity[:, 2] = torch.tensor([c["initial_vz_m_s"] for c in cases], device=device)
    assert batch.apply_linear_velocity(velocity) == len(cases)
    assert batch.apply_angular_velocity(torch.zeros_like(velocity)) == len(cases)
    pose_buffer = torch.empty((len(cases), 7), device=device)
    mass_buffer = torch.empty(len(cases), device=device)
    batch.fetch_mass(mass_buffer)
    np.testing.assert_allclose(mass_buffer.cpu(), MASS, rtol=1e-6)
    batch.fetch_pose(pose_buffer)
    np.testing.assert_allclose(
        pose_buffer.cpu(), initial_pose.cpu(), atol=1e-8, rtol=1e-6
    )
    force = torch.zeros_like(velocity)
    force[:, 2] = -torch.tensor([c["load_N"] for c in cases], device=device)
    trajectories = [[] for c in cases]
    for step in range(STEPS):
        batch.fetch_pose(pose_buffer)
        batch.fetch_linear_velocity(velocity)
        before_z = pose_buffer[:, 6].cpu().numpy().copy()
        before_v = velocity[:, 2].cpu().numpy().copy()
        assert batch.apply_force(force) == len(cases)
        world.update(DT)
        batch.fetch_pose(pose_buffer)
        batch.fetch_linear_velocity(velocity)
        after_z = pose_buffer[:, 6].cpu().numpy()
        after_v = velocity[:, 2].cpu().numpy()
        if step == 0:
            print(
                "raw contacts",
                world.get_physics_scene().get_gpu_contact_num(),
                flush=True,
            )
            print(
                "first query actors",
                [(a.actor_id, a.path) for a in queries[0].actors],
                queries[0].selected_actor_ids,
                flush=True,
            )
        for index, (case, query) in enumerate(zip(cases, queries)):
            data = query.fetch()
            assert data.dropped_count == 0
            contacts = data.data[: data.count].cpu().numpy()
            normal = float((abs(contacts[:, 5]) * contacts[:, 9]).sum() / DT)
            trajectories[index].append(
                {
                    "physics_step": step + 1,
                    "z_before_m": float(before_z[index]),
                    "vz_before_m_s": float(before_v[index]),
                    "z_after_m": float(after_z[index]),
                    "vz_after_m_s": float(after_v[index]),
                    "normal_force_N": normal,
                    "momentum_force_N": float(
                        MASS * (after_v[index] - before_v[index]) / DT
                        + MASS * 9.81
                        + case["load_N"]
                    ),
                    "contacts": contacts.tolist(),
                }
            )
    results = [
        {"case": c, "summary": summarize(c, rows), "rows": rows}
        for c, rows in zip(cases, trajectories)
    ]
    for result in results:
        print(result["summary"], flush=True)
    report = {
        "backend": "dexsim",
        "recipe": recipe,
        "actual_scene": json.loads(world.get_physics_scene().dump_scene_config()),
        "results": results,
    }
    library = Path(dexsim.__file__).parent / "lib/libPhysXGpu_64.so"
    report["gpu_library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    queries.clear()
    del batch, handles
    scene.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["source", "native"])
    parser.add_argument("--recipe", choices=["hard", "base_force"], default="hard")
    args = parser.parse_args()
    cases = list(fixtures())
    report = (
        run_source(cases)
        if args.backend == "source"
        else run_native(cases, args.recipe)
    )
    suffix = "baseline" if args.recipe == "hard" else args.recipe
    destination = ROOT / "alignment" / f"normal-contact-{args.backend}-{suffix}.json"
    report.update(
        {
            "dt_s": DT,
            "mass_kg": MASS,
            "steps": STEPS,
            "scope": "independent normal contact; sphere/box with floor and flattened static pad1; no robot/servo/loop",
        }
    )
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(destination, flush=True)
