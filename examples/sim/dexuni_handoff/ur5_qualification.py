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

"""Run the external or floor-only DexPilot UR5 qualification."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import uuid
from pathlib import Path
from dexsim.runtime import Session
from dexsim.runtime.handoff import MODULE, signature
from examples.sim.attached_world.fixture import robot_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("source", choices=("external", "dexpilot"))
    parser.add_argument("dexpilot_entry", type=Path)
    args = parser.parse_args()
    p = args.output.resolve()
    p.mkdir(mode=0o700)
    source = args.source
    pilot = args.dexpilot_entry.resolve()
    asset = Path(robot_config(True).fpath)
    k = p / "writer.key"
    k.write_bytes(secrets.token_bytes(32))
    k.chmod(0o600)
    dep = Path(__file__).resolve().parents[2] / (
        "expansion/asset_generalization/configs/cube_platform_pick_place/task.ur5.dexuni.yaml"
    )
    os.environ.update(
        DEXSIM_HANDOFF_PROVIDER="embodichain",
        DEXSIM_HANDOFF_KEY_FILE=str(k),
        EMBODICHAIN_UR5_DEPLOYMENT=str(dep),
        EMBODICHAIN_MOTION_OUTPUT=str(p / "jobs"),
    )
    s = Session.create(
        directory=p / "sessions",
        headless=True,
        backend="newton",
        solver="dexuni",
        device="cuda:0",
        dt=0.01,
        num_substeps=20,
        dexuni_options={
            "mujoco_joints": list(range(14)),
            "joint_mode": "dynamic",
            "contact_mode": "auto",
            "mujoco_options": {
                "integrator": "implicitfast",
                "iterations": 50,
                "ls_iterations": 50,
                "cone": "elliptic",
                "impratio": 10,
                "enable_multiccd": True,
                "use_mujoco_contacts": True,
            },
        },
        asset_root=str(asset.parent),
        request_timeout=180,
    )

    def native(m, **kw):
        return s.invoke("dexsim.kit.mcp.dispatch", method=m, params=kw)

    def lifecycle(op, **kw):
        a = dict(operation=op, **kw)
        return s.invoke(
            MODULE, arguments=a, signature=signature(k.read_bytes(), s.session_id, a)
        )

    run_id = "ur5-" + uuid.uuid4().hex

    def invoke_turn(turn):
        subprocess.run(
            [
                "node",
                str(Path(__file__).with_name("scene_turn.mjs")),
                str(p),
                sys.prefix,
                str(pilot),
                str(turn),
                run_id,
            ],
            check=True,
        )

    def snapshot():
        return dict(
            inventory=native("scene.list_objects")["objects"],
            model=native("articulation.get_info", object_ref="manipulator"),
            simulation=native("world.status")["simulation"],
            robot=native("articulation.get_state", object_ref="manipulator"),
            bodies={
                ref: native("simulation.get_rigid_body_state", object_ref=ref)
                for ref in ["floor", "red", "blue", "left", "right"]
            },
        )

    try:
        native(
            "scene.create_primitive",
            name="floor",
            primitive="cube",
            size=[5, 5, 0.04],
            position=[0, 0, -0.02],
            physics="static",
        )
        native("scene.finalize")
        assert [x["object_ref"] for x in native("scene.list_objects")["objects"]] == [
            "floor"
        ]
        calls = []

        def stage(method, **arguments):
            calls.append([method, arguments])
            if source == "external":
                native(method, **arguments)

        stage(
            "articulation.import_urdf",
            asset_path=asset.name,
            name="manipulator",
            fixed_base=True,
        )
        for name, kind, size, pos in [
            ("red", "dynamic", [0.04] * 3, [-0.42, -0.08, 0.02]),
            ("blue", "dynamic", [0.04] * 3, [0.35, 0.18, 0.02]),
            ("left", "static", [0.16, 0.16, 0.05], [-0.28, -0.30, 0.025]),
            ("right", "static", [0.16, 0.16, 0.05], [-0.42, 0.3, 0.025]),
        ]:
            stage(
                "scene.create_primitive",
                name=name,
                primitive="cube",
                size=size,
                position=pos,
                physics=kind,
                mass=0.05,
                friction=0.97,
            )
        stage("scene.rebuild")
        arm = [f"joint{i}" for i in range(1, 7)]
        for names, props in [
            (arm, dict(target_ke=10000, target_kd=100, effort_limit=150)),
            (
                ["gripper_finger1_joint_1"],
                dict(target_ke=1000, target_kd=100, effort_limit=10000),
            ),
        ]:
            stage(
                "articulation.set_joint_state",
                object_ref="manipulator",
                joint_names=names,
                newton=props,
            )
        stage("scene.rebuild")
        home = [0, -1.57, 1.57, -1.57, -1.57, 0]
        stage(
            "articulation.set_state",
            object_ref="manipulator",
            joint_names=arm,
            joint_position=home,
        )
        stage(
            "articulation.set_joint_state",
            object_ref="manipulator",
            joint_names=arm,
            joint_target_position=home,
        )
        stage("simulation.step", dt=0.01, steps=60)

        binding = dict(
            profile="ur5-pick-place",
            paths={"UR5": "manipulator", "cube": "red", "platform": "left"},
            destinations=["left", "right"],
        )
        (p / "scene.json").write_text(
            json.dumps(
                dict(
                    binding=binding,
                    authoring=calls if source == "dexpilot" else [],
                    session_id=s.session_id,
                    run_id=run_id,
                ),
                indent=2,
            )
        )
        invoke_turn(1)
        ready = snapshot()
        assert (
            ready["simulation"]["step_index"] == 60
            and ready["simulation"]["solver"]["solver_type"] == "dexuni"
        )
        assert set(x["object_ref"] for x in ready["inventory"]) == {
            "manipulator",
            "floor",
            "red",
            "blue",
            "left",
            "right",
        }
        (p / "ready.json").write_text(json.dumps(ready, indent=2))
        invoke_turn(2)
        invoke_turn(3)
        after = snapshot()
        receipts = [
            json.loads(f.read_text()) for f in (p / "jobs").glob("*/receipt.json")
        ]
        assert {r["command_id"]: r["status"] for r in receipts} == {
            "pick": "succeeded",
            "place": "succeeded",
        }
        placed = next(r["result"] for r in receipts if r["command_id"] == "place")
        assert (
            all(placed["checks"].values())
            and placed["passed"]
            and lifecycle("describe")["phase"] == "authoring"
        )
        (p / "result.json").write_text(
            json.dumps(
                dict(
                    run_id=run_id,
                    session_id=s.session_id,
                    readback=after,
                    receipts=receipts,
                ),
                indent=2,
            )
        )
    finally:
        st = lifecycle("describe")
        if st["phase"] == "controlled":
            lifecycle("release", generation=st["generation"])
        s.shutdown()


if __name__ == "__main__":
    main()
