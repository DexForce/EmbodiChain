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
"""Replay a grounded hybrid placement on two procedural USD blocks."""

from __future__ import annotations
import argparse
import hashlib
import json
from uuid import uuid4
from itertools import product
from pathlib import Path
import numpy as np
import yaml
from PIL import Image, ImageDraw
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils, UsdShade
import trimesh
from embodichain.gen_sim.task_engine.cli import main as plan_cli
from embodichain.gen_sim.services import ChatResponse, GenSimServices, GenSimServicesCfg
from embodichain.gen_sim.workflow import (
    load_task_plan,
    qualify_task_scene,
    prepare_task_deployment,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import build_env
from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.utils.utility import load_config


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, v):
    p.write_text(json.dumps(v, indent=2, ensure_ascii=False))


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    # Fixed overhead schematic, in the same XY frame as the generated scene.
    cubes = {
        "block_2": ([-0.42, -0.08, 0.02], 0.04),
        "block_1": ([-0.42, 0.30, 0.04], 0.08),
    }
    image = Image.new("RGB", (500, 500), "white")
    draw = ImageDraw.Draw(image)
    objects = []
    for uid, (xyz, size) in cubes.items():
        x, y = (xyz[0] + 0.67) * 1000, (0.36 - xyz[1]) * 1000
        bbox = [
            round(x - size * 500),
            round(y - size * 500),
            round(x + size * 500),
            round(y + size * 500),
        ]
        draw.rectangle(bbox, fill="orange" if uid == "block_2" else "blue")
        objects.append(
            dict(
                id=uid,
                description=uid,
                bbox=[v * 2 for v in bbox],
                provenance="observed",
            )
        )
    image.save(root / "source.png")
    write(
        root / "grounding.json",
        dict(
            image_sha256=digest(root / "source.png"),
            grounding=dict(objects=objects, bindings={k: [k] for k in cubes}),
        ),
    )
    text = "Place the small block on the larger block."

    class Reply:
        def __init__(self, value):
            self.value = value

        def complete(self, request):
            return ChatResponse(text=json.dumps(self.value))

    with GenSimServices(
        GenSimServicesCfg(),
        injected=dict(
            task_image_grounding=Reply(
                dict(
                    operations=[
                        dict(
                            template="E1",
                            object="block_2",
                            reference="block_1",
                            relation="on",
                            quote=text,
                        )
                    ]
                )
            ),
            task_interpretation=Reply(dict(issues=[])),
        ),
    ) as services:
        plan_cli(
            [
                "--route",
                "hybrid",
                "--instruction",
                text,
                "--image",
                str(root / "source.png"),
                "--scene-grounding",
                str(root / "grounding.json"),
                "--output",
                str(root / "plan"),
            ],
            services=services,
        )
    plan = load_task_plan(root / "plan")
    scene = root / "scene"
    scene.mkdir()
    stage = Usd.Stage.CreateNew(str(scene / "scene.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Scene").GetPrim())
    material = UsdShade.Material.Define(stage, "/Scene/BlockMaterial")
    surface = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    surface.CreateStaticFrictionAttr(0.99)
    surface.CreateDynamicFrictionAttr(0.97)
    surface.CreateRestitutionAttr(0.0)
    entries = []
    for uid, (xyz, size) in cubes.items():
        prim = UsdGeom.Xform.Define(stage, "/Scene/" + uid)
        prim.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        body = UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
        UsdPhysics.MassAPI.Apply(prim.GetPrim()).CreateMassAttr(
            0.05 if uid == "block_2" else 0.4
        )
        box = UsdGeom.Cube.Define(stage, "/Scene/" + uid + "/shape")
        box.CreateSizeAttr(size)
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(
            material, materialPurpose="physics"
        )
        mesh = scene / (uid + ".glb")
        trimesh.creation.box([size] * 3).export(mesh)
        entries.append(
            dict(
                uid=uid,
                name=uid,
                description=uid,
                kind="rigid",
                category="block",
                source_prim_path="/Scene/" + uid,
                mesh_path=mesh.name,
                mesh_sha256=digest(mesh),
                pose=dict(
                    position=xyz, rotation_euler_xyz_degrees=[0] * 3, scale=[1] * 3
                ),
                physics=dict(body_type="dynamic"),
                affordances=["graspable", "placeable", "support_surface"],
            )
        )
    stage.GetRootLayer().Save()
    package = scene / "asset.usdz"
    if not UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(scene / "scene.usda")), str(package)
    ):
        raise RuntimeError("Cannot package runtime USD scene")
    write(
        scene / "generated_scene.json",
        dict(
            schema_version="embodichain.generated-scene/v1",
            scene_id="two_blocks",
            requirements_sha256=digest(root / "plan/scene_requirements.json"),
            source=dict(image_sha256=plan["image_sha256"]),
            coordinates=dict(editable_frame="Y-up", runtime_frame="Z-up", units="m"),
            runtime_asset=dict(path=package.name, sha256=digest(package)),
            objects=entries,
            graph=dict(
                nodes=[dict(object_id=k, parent_id=None) for k in cubes], relations=[]
            ),
            artifacts={p.name: digest(p) for p in scene.iterdir()},
        ),
    )
    configs = (
        Path(__file__).resolve().parents[3]
        / "extensions/embodichain_tasks/configs/components"
    )
    embodiment = yaml.safe_load(
        (configs / "embodiments/ur5_dh_pgi_140_80.yaml").read_text()
    )
    embodiment["skill_profile"]["runtime_services"]["control_part_evidence"] = dict(
        kind="joint_position_constraint",
        control_parts=["eef"],
        open_qpos=[0.0],
        minimum_displacement=0.013,
    )
    (root / "embodiment.yaml").write_text(yaml.safe_dump(embodiment))
    runtime = dict(
        physics="default",
        seed=7,
        max_episode_steps=2400,
        enable_sensor=False,
        render_cfg=dict(renderer="no-render"),
        env=dict(sim_steps_per_control=4),
        embodiment=dict(
            component=str(root / "embodiment.yaml"),
            overrides=dict(init_qpos=[0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0]),
        ),
    )
    (root / "runtime.yaml").write_text(yaml.safe_dump(runtime))
    policy = yaml.safe_load(
        (configs / "execution_policies/sequential_ik_verified.yaml").read_text()
    )
    policy["requires"]["embodiment_contract"] = "single_arm_parallel_gripper"
    (root / "policy.yaml").write_text(yaml.safe_dump(policy))
    q = qualify_task_scene(
        root / "plan",
        scene,
        root / "runtime.yaml",
        root / "policy.yaml",
        root / "qualification",
    )
    verdict = json.loads((q / "scene_qualification.json").read_text())
    if verdict["status"] != "accepted":
        raise ValueError(verdict)
    path = prepare_task_deployment(
        root / "plan",
        scene,
        q,
        root / "runtime.yaml",
        root / "policy.yaml",
        root / "prepared",
        skill_options={
            "pick": dict(
                kind="pick_up",
                fixed_object_to_eef=np.diag([1.0, -1.0, -1.0, 1.0]).flatten().tolist(),
                hand_interp_steps=40,
                grasp_settle_steps=30,
            ),
            "place": dict(
                kind="place",
                preserve_current_object_orientation=True,
                hand_interp_steps=40,
                release_settle_steps=50,
            ),
        },
    )
    cfg = config_to_cfg(load_config(path), source_path=path)
    cfg.sim_cfg.headless = True
    env = build_env(load_config(path)["id"], cfg)
    try:
        env.reset(seed=7, options={"save_data": False})
        samples = []

        def capture(*_):
            samples.append(float(env.unwrapped.robot.get_qpos(name="eef")[0, 0]))

        result = execute_demo_episode(env, step_observer=capture)
        write(root / "gripper-measurements.json", samples)
        poses = {
            k: env.unwrapped.sim.get_rigid_object(k)
            .get_local_pose(to_matrix=True)[0]
            .cpu()
            .numpy()
            for k in cubes
        }
        # Compare all small-block corners in the support block's actual local frame.
        transform = np.linalg.inv(poses["block_1"]) @ poses["block_2"]
        corners = (
            np.array(list(product([-0.02, 0.02], repeat=3))) @ transform[:3, :3].T
            + transform[:3, 3]
        )
        edge = float(np.abs(corners[:, :2]).max())
        gap = float(corners[:, 2].min() - 0.04)
        from embodichain.gen_sim.task_spec import TaskSpec, evaluate_task_spec
        from embodichain.gen_sim.task_spec.publication import (
            evaluate_attempt_observations,
        )

        doc = plan["task_document"]
        spec = TaskSpec.from_dict(
            json.loads((root / "plan/task_spec.json").read_text())
        )
        (goal,) = spec.goal
        (milestone,) = spec.milestones
        (achieved,) = milestone.achieve
        if any(
            tuple(p.arguments.get(k) for k in ("subject", "reference", "relation"))
            != ("block_2", "block_1", "on")
            for p in (goal, achieved)
        ):
            raise ValueError(
                "This measurement adapter only observes the declared block stack"
            )
        supported = bool(
            edge <= 0.04 and abs(gap) < 0.004 and poses["block_1"][2, 2] > 0.99
        )
        observations = {
            goal.predicate_id: supported,
            "trajectory": [{achieved.predicate_id: supported}],
        }
        evaluated = evaluate_task_spec(spec, observations).to_dict()
        identity = dict(
            run_id=str(uuid4()),
            attempt_id=str(uuid4()),
            spec_digest=doc["spec_digest"],
            program_digest=digest(path.parent / "program.json"),
            scene_digest=digest(scene / "generated_scene.json"),
            bundle_digest=digest(root / "prepared/deployment_manifest.json"),
        )
        capture = dict(identity=identity, step=result.length, observations=observations)
        write(root / "capture.json", capture)
        write(root / "task_evaluation.json", evaluated)
        write(
            root / "certificate_evaluation.json",
            evaluate_attempt_observations(doc, identity, capture),
        )
        ok = bool(
            result.completed
            and result.all_success
            and evaluated["status"] == "satisfied"
        )
        write(root / "execution.json", result.to_metadata())
        write(
            root / "terminal.json",
            dict(
                identity=identity,
                capture_step=result.length,
                spec_digest=plan["spec_digest"],
                graph_digest=plan["semantic_graph_sha256"],
                deployment_manifest=digest(root / "prepared/deployment_manifest.json"),
                program_digest=digest(path.parent / "program.json"),
                max_xy_m=edge,
                bottom_gap_m=gap,
                poses={k: v.tolist() for k, v in poses.items()},
                passed=ok,
            ),
        )
        if not ok:
            raise ValueError(
                "Stack not accepted; inspect execution and terminal reports"
            )
    finally:
        env.close(exit_process=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output.resolve())
