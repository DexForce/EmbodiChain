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
"""Reproduce task1186 against a procedural rigid scene and an actual dual-arm profile."""

from __future__ import annotations
import argparse, json
from copy import deepcopy
from pathlib import Path
import yaml
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, Sdf, Gf, UsdShade
import trimesh
from examples.gensim.goal_authoring.stack_pilot import digest, write
from examples.gensim.goal_authoring.hybrid_pilot import inspect_1186
from embodichain.gen_sim.task_engine.hybrid import publish_hybrid, verify_witness
from embodichain.gen_sim.task_spec import TaskSpec
from embodichain.gen_sim.services import GenSimServices, GenSimServicesCfg, ChatResponse
from embodichain.gen_sim.workflow import (
    load_task_plan,
    qualify_task_scene,
    prepare_task_deployment,
)
import numpy as np
from itertools import product


def run(root, image, execute):
    root.mkdir(parents=True, exist_ok=False)
    f = json.loads(Path(__file__).with_name("task1186_fixture.json").read_text())
    write(root / "ground.json", f["packet"])

    class Reply:
        def __init__(self, v):
            self.v = v

        def complete(self, request):
            return ChatResponse(text=json.dumps(self.v))

    with GenSimServices(
        GenSimServicesCfg(),
        injected=dict(
            task_image_grounding=Reply(dict(operations=f["operations"])),
            task_interpretation=Reply(dict(issues=[])),
        ),
    ) as svc:
        result = publish_hybrid(
            f["instruction"],
            "task1186",
            image,
            root / "plan",
            svc,
            scene_grounding=root / "ground.json",
        )
    plan = load_task_plan(root / "plan")
    graph = result["semantic_task_graph"]
    checked = inspect_1186(graph)
    if not all(checked.values()):
        raise ValueError(f"Invalid task1186 graph: {checked}")
    spec = TaskSpec.from_dict(result["task_spec"])
    for wrong in ("arm", "tray", "order"):
        bad = deepcopy(graph)
        if wrong == "arm":
            bad["nodes"][1]["call"]["resources"] = {
                "source": "left",
                "destination": "right",
            }
        if wrong == "tray":
            bad["nodes"][2]["call"]["inside"] = "service_pad"
        if wrong == "order":
            bad["nodes"][2]["depends_on"] = []
        try:
            rejected = (
                verify_witness(
                    spec, bad, support_targets=result["pending_support_targets"]
                )["status"]
                != "covered"
            )
        except ValueError:
            rejected = True
        if not rejected:
            raise ValueError(f"Negative control accepted: {wrong}")
        checked["reject_" + wrong] = rejected
    scene = root / "scene"
    scene.mkdir()
    stage = Usd.Stage.CreateNew(str(scene / "scene.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Scene").GetPrim())
    mat = UsdShade.Material.Define(stage, "/Scene/material")
    phys = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys.CreateStaticFrictionAttr(1.0)
    phys.CreateDynamicFrictionAttr(0.95)
    phys.CreateRestitutionAttr(0.0)
    entries = []
    for uid, sz, pos in [
        ("table", [1.0, 1.0, 0.05], [0, 0, 0.525]),
        ("apple", [0.045] * 3, [0, 0.18, 0.574]),
        ("orange", [0.045] * 3, [-0.07, 0.18, 0.574]),
        ("tray", [0.22, 0.19, 0.03], [0.12, -0.22, 0.566]),
        ("service_pad", [0.31, 0.40, 0.008], [0.12, 0.11, 0.554]),
    ]:
        obj = UsdGeom.Xform.Define(stage, "/Scene/" + uid)
        obj.AddTranslateOp().Set(Gf.Vec3d(*pos))
        moving = uid in {"apple", "orange", "tray"}
        if not moving:
            UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim()).CreateKinematicEnabledAttr(
                True
            )
        if moving:
            UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
            UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr(0.08)
        box = UsdGeom.Cube.Define(stage, "/Scene/" + uid + "/body")
        box.CreateSizeAttr(1.0)
        box.AddScaleOp().Set(Gf.Vec3f(*sz))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(
            mat, materialPurpose="physics"
        )
        mesh = scene / (uid + ".glb")
        parts = [trimesh.creation.box([sz[0], sz[2], sz[1]])]
        if uid == "tray":
            for side in (-1, 1):
                dims = [0.045, 0.050, 0.06]
                offset = [0.0, side * 0.139, 0.047]
                handle = UsdGeom.Cube.Define(stage, "/Scene/tray/grip" + str(side + 1))
                handle.CreateSizeAttr(1.0)
                handle.AddTranslateOp().Set(Gf.Vec3d(*offset))
                handle.AddScaleOp().Set(Gf.Vec3f(*dims))
                UsdPhysics.CollisionAPI.Apply(handle.GetPrim())
                UsdShade.MaterialBindingAPI.Apply(handle.GetPrim()).Bind(
                    mat, materialPurpose="physics"
                )
                mesh_handle = trimesh.creation.box([dims[0], dims[2], dims[1]])
                mesh_handle.apply_translation([offset[0], offset[2], -offset[1]])
                parts.append(mesh_handle)
                bracket = UsdGeom.Cube.Define(stage, "/Scene/tray/link" + str(side + 1))
                bracket.CreateSizeAttr(1.0)
                bracket.AddTranslateOp().Set(Gf.Vec3d(0.0, side * 0.115, 0.014))
                bracket.AddScaleOp().Set(Gf.Vec3f(0.018, 0.055, 0.032))
                UsdPhysics.CollisionAPI.Apply(bracket.GetPrim())
                mount = trimesh.creation.box([0.018, 0.032, 0.055])
                mount.apply_translation([0.0, 0.014, -side * 0.115])
                parts.append(mount)

        trimesh.util.concatenate(parts).export(mesh)
        entries.append(
            dict(
                uid=uid,
                name=uid,
                description=uid,
                category=uid,
                kind="rigid",
                source_prim_path="/Scene/" + uid,
                mesh_path=mesh.name,
                mesh_sha256=digest(mesh),
                pose=dict(
                    position=pos, rotation_euler_xyz_degrees=[0] * 3, scale=[1] * 3
                ),
                physics=dict(body_type="dynamic" if moving else "static"),
            )
        )
    stage.GetRootLayer().Save()
    pkg = scene / "asset.usdz"
    if not UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(scene / "scene.usda")), str(pkg)
    ):
        raise RuntimeError("USD packaging failed")
    write(
        scene / "generated_scene.json",
        dict(
            schema_version="embodichain.generated-scene/v1",
            scene_id="task1186",
            requirements_sha256=digest(root / "plan/scene_requirements.json"),
            source=dict(image_sha256=plan["image_sha256"]),
            coordinates=dict(editable_frame="Y-up", runtime_frame="Z-up", units="m"),
            runtime_asset=dict(path=pkg.name, sha256=digest(pkg)),
            objects=entries,
            graph=dict(
                nodes=[dict(object_id=e["uid"], parent_id=None) for e in entries]
            ),
            artifacts={p.name: digest(p) for p in scene.iterdir()},
        ),
    )
    components = (
        Path(__file__).resolve().parents[3]
        / "extensions/embodichain_tasks/configs/components"
    )
    runtime = root / "runtime.yaml"
    runtime.write_text(
        yaml.safe_dump(
            dict(
                physics="default",
                seed=7,
                max_episode_steps=7500,
                enable_sensor=False,
                env=dict(sim_steps_per_control=4),
                render_cfg=dict(renderer="no-render"),
                embodiment=dict(
                    component=str(
                        components
                        / "embodiments/dual_franka_robotiq_arg2f_140_handover.yaml"
                    )
                ),
            )
        )
    )
    policy = components / "execution_policies/sequential_ik_verified.yaml"
    configured_policy = yaml.safe_load(policy.read_text())
    configured_policy["motion"].update(sample_count=440, velocity_limit=1.8)
    policy = root / "policy.yaml"
    policy.write_text(yaml.safe_dump(configured_policy))
    q = qualify_task_scene(root / "plan", scene, runtime, policy, root / "qualified")
    report = json.loads((q / "scene_qualification.json").read_text())
    if report["status"] != "accepted":
        raise ValueError(report)
    installation = (
        components.parent
        / "tasks/manipulation/repeated_pick_place/task_program/integration.robotiq140.yaml"
    )
    options = yaml.safe_load(installation.read_text())["profile"]["action_options"]
    options["simulation.coordinated_transport"] = dict(
        kind="coordinated_pickment",
        release=True,
        left_to_right_arm_direction=[0.0, 1.0, 0.0],
        hand_interp_steps=55,
        hold_steps=45,
        release_steps=55,
        lift_height=0.07,
        object_motion_keyframes=32,
        grasp_seed=2026,
    )
    pair = []
    for y in (-0.139, 0.139):
        transform = np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        transform[:3, 3] = [0.0, y, 0.057]
        pair.append(transform.flatten().tolist())
    options["simulation.coordinated_transport"]["paired_object_to_eef"] = pair
    options["place"].update(
        preserve_current_object_orientation=False, release_settle_steps=80
    )
    options["hand_over"] = dict(
        kind="hand_over",
        hand_interp_steps=40,
        hold_steps=20,
        pre_grasp_distance=0.08,
        retreat_distance=0.12,
    )
    deployment = prepare_task_deployment(
        root / "plan",
        scene,
        q,
        runtime,
        policy,
        root / "prepared",
        skill_options=options,
        handover_pose=dict(
            final_position=[0.0, 0.0, 0.76], final_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0]
        ),
    )
    program = json.loads((deployment.parent / "program.json").read_text())
    final = program["program"]["items"][-1]["validators"]
    if {(v["object"], v["reference"]) for v in final} != {
        ("apple", "tray"),
        ("orange", "tray"),
        ("tray", "service_pad"),
    }:
        raise ValueError("Incomplete final acceptance")
    write(
        root / "summary.json",
        dict(
            static_checks=checked,
            qualified=report["status"],
            placements=report["placements"],
            coverage=result["coverage"],
            deployment=digest(deployment),
        ),
    )
    if execute:
        from embodichain.lab.gym.utils.registration import build_env
        from embodichain.lab.gym.utils.gym_utils import config_to_cfg
        from embodichain.lab.gym.envs.demo import execute_demo_episode
        from embodichain.utils.utility import load_config

        data = load_config(deployment)
        cfg = config_to_cfg(data, source_path=deployment)
        cfg.sim_cfg.headless = True
        env = build_env(data["id"], cfg)
        samples = []
        plate_heights = []

        def capture(*args):
            plate_heights.append(
                float(
                    env.unwrapped.sim.get_rigid_object("tray").get_local_pose(
                        to_matrix=True
                    )[0, 2, 3]
                )
            )
            samples.append(
                [
                    env.unwrapped.robot.get_qpos(name=k)[0].cpu().tolist()
                    for k in ("left_eef", "right_eef")
                ]
            )

        try:
            env.reset(seed=7, options={"save_data": False})
            episode = execute_demo_episode(env, step_observer=capture)
            write(root / "qpos.json", samples)
            write(root / "tray_height.json", plate_heights)
            poses = {
                k: env.unwrapped.sim.get_rigid_object(k)
                .get_local_pose(to_matrix=True)[0]
                .cpu()
                .tolist()
                for k in ("apple", "orange", "tray", "service_pad")
            }
            write(root / "terminal_poses.json", poses)
            relations = {}
            sizes = {
                "apple": [0.045] * 3,
                "orange": [0.045] * 3,
                "tray": [0.22, 0.19, 0.03],
                "service_pad": [0.31, 0.40, 0.008],
            }
            for goal in plan["task_document"]["objectives"]["goal"]:
                a, b = goal["subject"], goal["reference"]
                relative = np.linalg.solve(poses[b], poses[a])
                corners = (
                    np.array(list(product([-0.5, 0.5], repeat=3))) * sizes[a]
                ) @ relative[:3, :3].T + relative[:3, 3]
                margin = float(
                    np.min(np.array(sizes[b][:2]) * 0.5 - np.abs(corners[:, :2]))
                )
                gap = float(corners[:, 2].min() - sizes[b][2] * 0.5)
                relations[goal["id"]] = dict(
                    subject=a,
                    reference=b,
                    footprint_margin_m=margin,
                    vertical_gap_m=gap,
                    satisfied=bool(
                        margin >= 0 and abs(gap) < 0.006 and poses[b][2][2] > 0.98
                    ),
                )
            write(root / "terminal_relations.json", relations)
            execution = episode.to_metadata()
            execution["status"] = (
                "succeeded" if episode.completed and episode.all_success else "failed"
            )
        finally:
            env.close(exit_process=False)
        write(root / "execution.json", execution)
        segments = execution["segments"]
        checkpoints = []
        for segment in segments:
            state = segment["metadata"]["runtime"]["task_state"]["held_objects"]
            checkpoints.append(
                dict(
                    node=segment["name"],
                    start=segment["start_step"],
                    end=segment["end_step"],
                    success=segment["success"],
                    holders={x["resource_id"]: x["object_id"] for x in state},
                )
            )
        order = all(
            a["end"] <= b["start"] for a, b in zip(checkpoints, checkpoints[1:])
        )
        transfers = len(checkpoints) == 7 and all(
            checkpoints[i]["holders"] == expected
            for i, expected in enumerate(
                [
                    {"right": "apple"},
                    {"left": "apple"},
                    {},
                    {"right": "orange"},
                    {"left": "orange"},
                    {},
                    {},
                ]
            )
        )
        lifted = 0.0
        if len(checkpoints) == 7:
            start = checkpoints[-1]["start"]
            lifted = max(plate_heights[start:]) - plate_heights[start]
        write(
            root / "review.json",
            dict(
                spec_digest=plan["spec_digest"],
                graph_digest=plan["semantic_graph_sha256"],
                scene_digest=digest(scene / "generated_scene.json"),
                deployment_digest=digest(root / "prepared/deployment_manifest.json"),
                program_digest=digest(deployment.parent / "program.json"),
                static_checks=checked,
                qualified=report["status"],
                milestone_producers=result["coverage"]["producers"],
                execution=execution["status"],
                segments=checkpoints,
                ordered=order,
                verified_transfers=transfers,
                tray_lift_m=lifted,
                terminal=relations,
            ),
        )
        if not order or not transfers or lifted <= 0.02:
            raise RuntimeError("Milestone evidence incomplete")
        if execution["status"] != "succeeded" or not all(
            v["satisfied"] for v in relations.values()
        ):
            raise RuntimeError("Native program failed; see execution.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    run(args.output.resolve(), args.image, args.execute)
