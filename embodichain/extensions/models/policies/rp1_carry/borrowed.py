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

"""Run the official RP1 step-up scene and policy on a DexSim-owned World."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree as ET

if TYPE_CHECKING:
    from dexsim.runtime import SessionContext
    from dexsim.scene import SpawnedArticulation


def _descriptor(robot: SpawnedArticulation, destination: Path) -> None:
    """Describe the live RP1 articulation for the EmbodiChain topology check."""
    from scipy.spatial.transform import Rotation

    def numbers(values: Any) -> str:
        return " ".join(str(float(value)) for value in values)

    node = ET.Element("robot", name="rp1")
    for name in robot.get_link_names():
        ET.SubElement(node, "link", name=name)
    for item in robot.joint_dof_layout:
        desc = robot.get_joint_desc(item.name)
        joint = ET.SubElement(node, "joint", name=item.name, type="revolute")
        ET.SubElement(joint, "parent", link=desc.parent_link_name)
        ET.SubElement(joint, "child", link=desc.child_link_name)
        pose = desc.origin_pose
        ET.SubElement(
            joint,
            "origin",
            xyz=numbers(pose[:3, 3]),
            rpy=numbers(Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")),
        )
        ET.SubElement(joint, "axis", xyz=numbers(desc.axis))
        ET.SubElement(
            joint,
            "limit",
            lower=str(desc.lower_limit),
            upper=str(desc.upper_limit),
            effort=str(desc.effort_limit or 0.0),
            velocity=str(desc.velocity_limit or 0.0),
        )
    ET.ElementTree(node).write(destination, encoding="utf-8")


def _attach(
    context: SessionContext,
    usd: Path,
    output: Path,
    *,
    video: bool,
    live_paths: dict[str, str] | None = None,
) -> tuple[Any, Any]:
    from dexsim import spawn
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend

    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.cfg import RigidObjectCfg, RobotCfg

    builder = context.builder
    if live_paths is None:
        builder.add_usd(name="rp1_carry", file_path=str(usd), per_env=True)
        scene = builder.finalize()
    else:
        scene = context.scene
        diag = scene.get_solver_diagnostics() or {}
        if (
            diag.get("solver_type") != "dexuni"
            or diag.get("num_substeps") != 1
            or diag.get("dt") != 0.005
        ):
            raise ValueError("PartyMimic requires a 5ms single-substep DexUni World")

    backend = get_newton_backend(context.world)
    if live_paths is None and (
        backend.cfg.num_substeps != 1 or backend.cfg.solver_cfg.rne_mode != "disabled"
    ):
        backend.cfg.num_substeps = 1
        backend.cfg.solver_cfg.rne_mode = "disabled"
        # Released SDKs can retain a graph captured with the previous schedule.
        backend.invalidate_cuda_graph("PartyMimic uses one 5 ms Newton substep")

    if video and live_paths is None:
        scene.set_environment_lighting(mode="constant", intensity=22.0)
        for name, direction, intensity in (
            ("key", (-0.52, 0.46, -0.71), 3.0),
            ("fill", (0.38, -0.46, -0.80), 0.9),
        ):
            scene.add_light(
                spawn.LightDesc(
                    name=name,
                    light_type="DIRECTION",
                    direction=direction,
                    intensity=intensity,
                )
            )

    paths = (
        {"base_link": live_paths["robot"], "carry_crate": live_paths["crate"]}
        if live_paths
        else {"base_link": "base_link", "carry_crate": "carry_crate"}
    )
    descriptor = output / "rp1.urdf"
    _descriptor(scene.get_articulation(paths["base_link"]), descriptor)
    attachment = SimulationManager.attach(
        context,
        arena="default",
        robots=[RobotCfg(uid="base_link", fpath=str(descriptor), build_pk_chain=False)],
        rigid_objects=[RigidObjectCfg(uid="carry_crate", body_type="dynamic")],
        paths=paths,
    )
    return attachment, backend


def run(
    context: SessionContext,
    *,
    files: dict[str, Path],
    options: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Load the official sparse5 actor and execute the shared step-up rollout."""
    from embodichain.extensions.models.policies.partymimic import PartyMimicPolicy

    from .simulation import _run

    started = perf_counter()
    prepared = (
        (files["scene_mjcf"], files["scene_usd"]) if "scene_mjcf" in files else None
    )
    result = _run(
        Namespace(
            output_dir=output,
            hand="rp",
            step_reference="video",
            robot_xml=files["robot_xml"],
            converter_python=files.get("converter_python"),
            device="cuda:0",
            solver="mujoco_warp",
            no_video=not options.get("video", True),
            unloaded=False,
        ),
        "step-up",
        PartyMimicPolicy.from_onnx(files["policy"]),
        context=context,
        prepared_scene=prepared,
    )
    result["wall_duration_s"] = perf_counter() - started
    metrics = output / "metrics.json"
    metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {**result, "metrics": str(metrics)}
