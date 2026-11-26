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

"""Qualify a real external handoff and a DexPilot-built, floor-only handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from typing import Any

import numpy as np
from dexsim.runtime import Session
from dexsim.runtime.handoff import MODULE, signature
from embodichain.data import get_data_path
from embodichain.lab.sim.dexsim_session.writer import WriterClient


def trial(root: Path, source: str, pilot: Path) -> dict[str, Any]:
    root.mkdir(mode=0o700)
    grant = root / "writer.key"
    with os.fdopen(
        os.open(grant, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb"
    ) as f:
        f.write(secrets.token_bytes(32))
    key = grant.read_bytes()
    asset = Path(get_data_path("UniversalRobots/UR5/UR5.urdf"))
    # Session.create inherits these operator settings when starting its owner process.
    from unittest.mock import patch

    with patch.dict(
        os.environ,
        DEXSIM_HANDOFF_PROVIDER="embodichain",
        DEXSIM_HANDOFF_KEY_FILE=str(grant),
    ):
        s = Session.create(
            directory=root,
            backend="newton",
            solver="dexuni",
            device="cuda:0",
            dt=0.01,
            num_substeps=4,
            dexuni_options={"joint_mode": "dynamic", "contact_mode": "auto"},
            headless=True,
            asset_root=str(asset.parent),
            request_timeout=120,
        )

    def native(method, **params):
        return s.invoke("dexsim.kit.mcp.dispatch", method=method, params=params)

    def handoff(op, **params):
        args = {"operation": op, **params}
        return s.invoke(
            MODULE, arguments=args, signature=signature(key, s.session_id, args)
        )

    def state():
        return {
            name: native("articulation.get_state", object_ref=name)
            for name in ("arm_a", "arm_b")
        }

    try:
        native(
            "scene.create_primitive",
            name="floor",
            primitive="cube",
            size=[6, 6, 0.04],
            position=[0, 0, -0.02],
            physics="static",
        )
        native("scene.finalize")
        floor = native("scene.list_objects")["objects"]
        assert [r["object_ref"] for r in floor] == ["floor"]
        floor_pose = native("simulation.get_rigid_body_state", object_ref="floor")[
            "world_pose"
        ]
        if source == "external":
            for i, name in enumerate(("arm_a", "arm_b")):
                native(
                    "articulation.import_urdf",
                    name=name,
                    asset_path=asset.name,
                    position=[2 * i, 0, 0],
                    fixed_base=True,
                )
            native("scene.rebuild")
            initial = state()
            prior = handoff("describe")
            lease = handoff(
                "acquire",
                generation=prior["generation"],
                topology_revision=prior["topology_revision"],
                binding=dict(
                    arena="default",
                    robots={
                        "UR5": dict(
                            robot_cfg="URRobot", variant="ur5", object_ref="arm_b"
                        )
                    },
                    bodies={},
                    physics_dt=0.01,
                    steps_per_control=4,
                ),
            )
            bound = state()
            for name in initial:
                for field in (
                    "joint_position",
                    "joint_velocity",
                    "joint_target_position",
                    "root_pose",
                    "step_index",
                ):
                    assert initial[name][field] == bound[name][field]
            writer = WriterClient(s, grant)
            st = writer.call("status", {})
            cmd = dict(
                command_id="one",
                robot_uid="UR5",
                control_part="arm",
                units=["rad"] * 6,
                target=[0.08] * 6,
                control_interval_s=0.04,
                **{k: st[k] for k in ("session_id", "topology_revision", "revision")},
            )
            first = writer.call("target_and_step", cmd)
            with Session.connect(s.session_id, directory=root) as other:
                assert WriterClient(other, grant).call("target_and_step", cmd) == first
            handoff("release", generation=lease["generation"])
        else:
            subprocess.run(
                [
                    "node",
                    str(Path(__file__).with_name("pilot.mjs")),
                    str(root),
                    sys.prefix,
                    str(pilot),
                    str(
                        Path(sys.prefix)
                        / "lib"
                        / f"python{sys.version_info.major}.{sys.version_info.minor}"
                        / "site-packages"
                    ),
                ],
                check=True,
            )
        after = state()
        diagnostics = native("world.status")["simulation"]["solver"]
        assert (
            diagnostics["solver_type"] == "dexuni"
            and diagnostics["features"]["mujoco_solve_enabled"]
        )
        assert handoff("describe")["phase"] == "authoring"
        assert diagnostics["step_index"] == 4
        np.testing.assert_allclose(
            after["arm_b"]["joint_target_position"], [0.08] * 6, atol=1e-6
        )
        np.testing.assert_allclose(
            after["arm_a"]["joint_target_position"], [0] * 6, atol=1e-6
        )
        np.testing.assert_allclose(
            native("simulation.get_rigid_body_state", object_ref="floor")["world_pose"],
            floor_pose,
        )
        native("simulation.step", steps=1, dt=0.01)
        report = dict(
            session=s.session_id,
            runtime=s.info["runtime"],
            source=source,
            floor=floor_pose,
            asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
            objects=native("scene.list_objects")["objects"],
            diagnostics=diagnostics,
            readback=after,
            journal=s.info["journal"],
        )
        (root / "report.json").write_text(json.dumps(report, indent=2))
        return report
    finally:
        desc = handoff("describe")
        if desc["phase"] == "controlled":
            handoff("release", generation=desc["generation"])
        s.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dexpilot-entry", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    a, b = [
        trial(root / x, x, args.dexpilot_entry.resolve())
        for x in ("external", "dexpilot")
    ]
    assert (
        a["session"] != b["session"]
        and a["objects"] == b["objects"]
        and a["asset_sha256"] == b["asset_sha256"]
    )
    for role in ("arm_a", "arm_b"):
        for field in ("root_pose", "joint_position", "joint_target_position"):
            np.testing.assert_allclose(
                a["readback"][role][field], b["readback"][role][field], atol=1e-5
            )
    print("Both sources matched after one four-step control interval.")


if __name__ == "__main__":
    main()
