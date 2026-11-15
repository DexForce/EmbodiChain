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
"""Qualify a newly authored plan against an existing pilot installation and execute it."""

from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from itertools import product
import numpy as np
import yaml, trimesh
from .stack_pilot import digest, write
from embodichain.gen_sim.workflow import (
    load_task_plan,
    load_generated_scene,
    qualify_task_scene,
    prepare_task_deployment,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import build_env
from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.utils.utility import load_config


def run(root, installation):
    plan = load_task_plan(root / "plan")
    scene = root / "scene"
    previous = load_generated_scene(installation / "scene")
    if previous["source"]["image_sha256"] != plan["image_sha256"]:
        raise ValueError("Source mismatch")
    shutil.copytree(installation / "scene", scene)
    del previous["root"]
    previous["requirements_sha256"] = digest(root / "plan/scene_requirements.json")
    for obj in previous["objects"]:
        extent = trimesh.load(scene / obj["mesh_path"], force="mesh").extents
        obj["description"] += " mesh dimensions (metres): " + str(
            extent.round(3).tolist()
        )
    write(scene / "generated_scene.json", previous)
    runtime, policy = installation / "runtime.yaml", installation / "policy.yaml"
    q = qualify_task_scene(root / "plan", scene, runtime, policy, root / "qualified")
    report = json.loads((q / "scene_qualification.json").read_text())
    if report["status"] != "accepted":
        raise ValueError(report["issues"])
    ids = report["bindings"]["objects"]
    integ = yaml.safe_load(
        (installation / "prepared/deployment/integration.yaml").read_text()
    )
    exchange = integ["runtime_services"].get("handover_pose_providers")
    poses = exchange[0] if exchange else None
    path = prepare_task_deployment(
        root / "plan",
        scene,
        q,
        runtime,
        policy,
        root / "prepared",
        skill_options=integ["profile"]["action_options"],
        handover_pose=poses,
    )
    expected = (
        {"apple": "tray", "orange": "tray", "tray": "service_pad"}
        if exchange
        else {"block_2": "block_1"}
    )
    goals = plan["task_document"]["objectives"]["goal"]
    if {ids[g["subject"]]: ids[g["reference"]] for g in goals} != expected:
        raise ValueError("Wrong native object or support")
    sizes = {
        "apple": [0.045] * 3,
        "orange": [0.045] * 3,
        "tray": [0.22, 0.19, 0.03],
        "service_pad": [0.31, 0.40, 0.008],
        "block_2": [0.04] * 3,
        "block_1": [0.08] * 3,
    }
    from copy import deepcopy
    from embodichain.gen_sim.task_spec import TaskSpec
    from embodichain.gen_sim.task_engine.hybrid import verify_witness

    controls = {}
    if exchange:
        graph = json.loads((root / "plan/semantic_task_graph.json").read_text())
        spec = TaskSpec.from_dict(
            json.loads((root / "plan/task_spec.json").read_text())
        )
        support = {g["reference"]: g["reference"] for g in goals}
        for issue in ("wrong_arm", "wrong_plate"):
            bad = deepcopy(graph)
            if issue == "wrong_arm":
                bad["nodes"][1]["call"]["resources"] = {
                    "source": "left",
                    "destination": "right",
                }
            else:
                place = bad["nodes"][2]["call"]
                relation = next(k for k in ("inside", "on") if k in place)
                place[relation] = next(
                    role for role, uid in ids.items() if uid == "service_pad"
                )
            controls[issue] = verify_witness(spec, bad, support_targets=support)[
                "status"
            ]
            if controls[issue] == "covered":
                raise ValueError("Unsafe witness accepted")
    cfg = config_to_cfg(load_config(path), source_path=path)
    cfg.sim_cfg.headless = True
    env = build_env(load_config(path)["id"], cfg)
    try:
        env.reset(seed=7, options={"save_data": False})
        episode = execute_demo_episode(env)
        write(root / "execution.json", episode.to_metadata())

        def pose(k):
            return (
                env.unwrapped.sim.get_rigid_object(k)
                .get_local_pose(to_matrix=True)[0]
                .cpu()
                .numpy()
            )

        rows = []
        for a, b in expected.items():
            relative = np.linalg.solve(pose(b), pose(a))
            corners = (
                np.array(list(product([-0.5, 0.5], repeat=3))) * sizes[a]
            ) @ relative[:3, :3].T + relative[:3, 3]
            gap = float(corners[:, 2].min() - sizes[b][2] * 0.5)
            margin = float(
                np.min(np.array(sizes[b][:2]) * 0.5 - np.abs(corners[:, :2]))
            )
            rows.append(
                dict(
                    subject=a,
                    reference=b,
                    gap=gap,
                    margin=margin,
                    satisfied=bool(
                        margin >= 0
                        and abs(gap) < (0.006 if exchange else 0.004)
                        and pose(b)[2, 2] > (0.98 if exchange else 0.99)
                    ),
                )
            )
        stages = episode.to_metadata()["segments"]
        holders = [
            {
                h["resource_id"]: ids[h["object_id"]]
                for h in s["metadata"]["runtime"]["task_state"]["held_objects"]
            }
            for s in stages
        ]
        ordered = holders == (
            [
                {"right": "apple"},
                {"left": "apple"},
                {},
                {"right": "orange"},
                {"left": "orange"},
                {},
                {},
            ]
            if exchange
            else [{"primary_manipulator": "block_2"}, {}]
        )
        ok = bool(
            episode.completed
            and episode.all_success
            and ordered
            and all(r["satisfied"] for r in rows)
        )
        write(
            root / "acceptance.json",
            dict(
                spec_digest=plan["spec_digest"],
                graph_digest=plan["semantic_graph_sha256"],
                bindings=ids,
                negative_controls=controls,
                steps=episode.length,
                terminal=rows,
                holders=holders,
                passed=ok,
            ),
        )
        if not ok:
            raise ValueError("Physical acceptance failed")
    finally:
        env.close(exit_process=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("installation", type=Path)
    a = p.parse_args()
    run(a.root.resolve(), a.installation.resolve())
