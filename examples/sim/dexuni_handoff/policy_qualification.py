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

"""Run a policy on a separately authored DexUni scene, recording independent native evidence."""

from __future__ import annotations
import json
import sys
import secrets
import os
import numpy as np
from pathlib import Path
from dexsim.runtime import Session
from dexsim.runtime.handoff import MODULE, signature
from embodichain.extensions.models.policies.anymal_c.profile import (
    _JOINT_NAMES,
    _DEFAULT_POSITION,
)
import subprocess
from uuid import uuid4
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=["external", "dexpilot"], required=True)
    parser.add_argument("--profile", choices=["rp1", "anymal-c"], required=True)
    parser.add_argument("--dexpilot-entry", type=Path, required=True)
    args = parser.parse_args()
    name = args.profile
    out = args.output.resolve()
    out.mkdir(mode=0o700)
    k = out / "writer.key"
    k.write_bytes(secrets.token_bytes(32))
    k.chmod(0o600)
    catalog = args.catalog.resolve(strict=True)
    files = json.loads(catalog.read_text())
    assetroot = (args.assets / name).resolve(strict=True)
    for profile in files.values():
        profile["files"] = {
            key: str((catalog.parent / value).resolve(strict=True))
            for key, value in profile["files"].items()
        }
    os.environ.update(
        DEXSIM_HANDOFF_PROVIDER="embodichain",
        DEXSIM_HANDOFF_KEY_FILE=str(k),
        EMBODICHAIN_MOTION_OUTPUT=str(out / "jobs"),
        EMBODICHAIN_LIVE_POLICY_CATALOG=str(catalog),
    )
    if name == "rp1":
        import mujoco
        from embodichain.extensions.models.policies.partymimic import PartyMimicPolicy
        from embodichain.extensions.models.policies.rp1_carry._reference import (
            make_reference,
        )

        f = files[name]["files"]
        actor = PartyMimicPolicy.from_onnx(Path(f["policy"]))
        reference = make_reference(
            "step-up", Path(f["robot_xml"]), actor, video_step=True, rp_hands=True
        )
        model = mujoco.MjModel.from_xml_path(f["reference_mjcf"])
        extra = dict(
            mujoco_joints=list(range(model.njnt)),
            mujoco_options=dict(
                iterations=100, ls_iterations=50, nconmax=128, njmax=500
            ),
        )
    else:
        extra = {}
    s = Session.create(
        directory=out / "sessions",
        backend="newton",
        solver="dexuni",
        headless=True,
        device="cuda:0",
        dt=0.005,
        num_substeps=1,
        renderer="fast-rt",
        asset_root=str(assetroot),
        request_timeout=180,
        dexuni_options=dict(joint_mode="dynamic", contact_mode="auto", **extra),
    )

    def nat(m, **kw):
        return s.invoke("dexsim.kit.mcp.dispatch", method=m, params=kw)

    def lease(op, **kw):
        a = dict(operation=op, **kw)
        return s.invoke(
            MODULE, arguments=a, signature=signature(k.read_bytes(), s.session_id, a)
        )

    try:
        if name == "anymal-c":
            nat(
                "scene.create_primitive",
                name="floor",
                primitive="cube",
                size=[8, 8, 0.1],
                position=[0, 0, -0.05],
                physics="static",
                friction=1.0,
            )
        else:
            nat("scene.import_usd", name="floor_asset", asset_path="floor.usdc")
        nat("scene.finalize")
        print("FLOOR", nat("scene.list_objects"), flush=True)
        calls = []

        def stage(method, **params):
            calls.append([method, params])
            if args.source == "external":
                return nat(method, **params)

        if name == "anymal-c":
            stage(
                "articulation.import_urdf",
                name="walker",
                asset_path="anymal.urdf",
                fixed_base=False,
                position=[0, 0, 0.76],
                rotation_euler_deg=[0, 0, 30],
            )
            stage("scene.rebuild")
            stage(
                "articulation.set_joint_state",
                object_ref="walker",
                joint_names=list(_JOINT_NAMES),
                newton=dict(
                    target_ke=220, target_kd=10, armature=0.06, effort_limit=80
                ),
            )
            stage("scene.rebuild")
            stage(
                "articulation.set_state",
                object_ref="walker",
                joint_names=list(_JOINT_NAMES),
                joint_position=_DEFAULT_POSITION.tolist(),
            )
            stage(
                "articulation.set_joint_state",
                object_ref="walker",
                joint_names=list(_JOINT_NAMES),
                joint_target_position=_DEFAULT_POSITION.tolist(),
            )
            roles = {"robot": "walker", "floor": "floor"}
        else:
            for role in ("platform", "rp1", "crate"):
                print(
                    "import",
                    role,
                    stage(
                        "scene.import_usd",
                        name=role + "_asset",
                        asset_path=role + ".usdc",
                    ),
                    flush=True,
                )
            stage(
                "scene.create_camera", name="overview", width=1280, height=720, fov=48
            )
            stage(
                "light.create",
                name="sun",
                light_type="DIRECTION",
                direction=[-0.5, 0.4, -0.7],
                intensity=4,
            )
            stage("scene.rebuild")
            root = reference.poses[0, 0]
            stage(
                "articulation.set_state",
                object_ref="base_link",
                root_pose=[*root[3:].tolist(), *root[:3].tolist()],
                joint_names=list(actor.joint_names),
                joint_position=reference.initial_joints.tolist(),
            )
            stage(
                "articulation.set_joint_state",
                object_ref="base_link",
                joint_names=list(actor.joint_names),
                joint_target_position=reference.initial_joints.tolist(),
            )
            roles = {
                "robot": "base_link",
                "floor": "floor",
                "crate": "carry_crate",
                "platform": "platform",
                "camera": "overview",
            }

        binding = {"profile": name, "paths": roles}
        runid = "policy-" + uuid4().hex
        (out / "scene.json").write_text(
            json.dumps(
                dict(
                    session_id=s.session_id,
                    binding=binding,
                    run_id=runid,
                    authoring=calls if args.source == "dexpilot" else [],
                ),
                indent=2,
            )
        )

        def turn(n):
            subprocess.run(
                [
                    "node",
                    str(Path(__file__).with_name("scene_turn.mjs")),
                    str(out),
                    sys.prefix,
                    str(args.dexpilot_entry.resolve()),
                    str(n),
                    runid,
                ],
                check=True,
            )

        turn(1)
        info = nat("articulation.get_info", object_ref=roles["robot"])
        before = dict(
            inventory=nat("scene.list_objects")["objects"],
            model=info,
            simulation=nat("world.status")["simulation"],
            robot=nat("articulation.get_state", object_ref=roles["robot"]),
            objects={
                role: nat("simulation.get_rigid_body_state", object_ref=roles[role])
                for role in roles
                if role not in ("robot", "camera")
            },
        )
        assert before["simulation"]["solver"]["solver_type"] == "dexuni"
        (out / "ready.json").write_text(json.dumps(before, indent=2))
        turn(2)
        (receipt,) = list((out / "jobs").glob("*/receipt.json"))
        result = json.loads(receipt.read_text())
        assert (
            result["done"]
            and result["status"] == "succeeded"
            and result["result"]["passed"]
        ), result
        assert lease("describe")["phase"] == "authoring"
        after = nat("articulation.get_state", object_ref=roles["robot"])
        assert after["step_index"] - before["robot"]["step_index"] == (
            1600 if name == "rp1" else 400
        )
        (out / "result.json").write_text(
            json.dumps(
                dict(
                    run_id=runid,
                    session_id=s.session_id,
                    readback=after,
                    receipt=result,
                ),
                indent=2,
            )
        )
    finally:
        st = lease("describe")
        if st["phase"] == "controlled":
            lease("release", generation=st["generation"])
        s.shutdown()


if __name__ == "__main__":
    main()
