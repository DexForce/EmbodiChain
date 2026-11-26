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

"""Scoped PartyMimic step-up execution using only already-bound scene assets."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import numpy as np

from embodichain.extensions.models.policies.partymimic import PartyMimicPolicy
from embodichain.extensions.models.policies.rp1_carry._reference import make_reference
from embodichain.extensions.models.policies.rp1_carry.simulation import (
    _rollout,
    _verify_payload_welds,
)
from ._motion_jobs import MotionJobs

__all__: list[str] = []


class AttachedCarry:
    def __init__(
        self,
        context,
        files: dict[str, Path],
        roles: dict[str, str],
        output: Path,
        clearance: float = 0.06,
    ) -> None:
        self.context = context
        self.clearance = clearance
        self.files = files
        self.roles = roles
        self.revision = context.scene.topology_revision
        self.actor = PartyMimicPolicy.from_onnx(files["policy"])
        scene = context.scene
        diag = scene.get_solver_diagnostics() or {}
        if (
            diag.get("solver_type") != "dexuni"
            or diag.get("num_substeps") != 1
            or diag.get("dt") != 0.005
        ):
            raise ValueError("Expected the qualified DexUni 5ms carrying schedule")
        robot = scene.get_articulation(roles["robot"])
        crate = scene.get_object(roles["crate"])
        platform = scene.get_object(roles["platform"])
        scene.get_camera(roles["camera"])
        if abs(float(crate.object_desc.physics.mass) - 1.5) > 1e-4:
            raise ValueError("RP1 requires a 1.5 kg crate")
        ref = make_reference(
            "step-up", files["robot_xml"], self.actor, video_step=True, rp_hands=True
        )
        colliders = platform.object_desc.collisions
        if len(colliders) != 1 or colliders[0].geometry_type.value != "cube":
            raise ValueError("Expected a box collider on the step platform")
        pose = platform.get_world_pose() @ colliders[0].local_transform
        height = np.abs(pose[2, :3]) @ np.asarray(colliders[0].size)
        if (
            platform.get_actor_type().name.lower() != "static"
            or not np.isclose(height, 0.48, atol=0.002)
            or not np.isclose(pose[2, 3] + height * 0.5, 0.48, atol=0.002)
        ):
            raise ValueError("Expected the 0.48 m platform")
        indices = self.actor.joint_indices([j.name for j in robot.joint_dof_layout])
        if (
            np.max(np.abs(robot.get_current_qpos()[indices] - ref.initial_joints))
            > 0.15
        ):
            raise ValueError("RP1 does not match the qualified carrying posture")
        if np.linalg.norm(robot.get_world_pose()[:3, 3] - ref.poses[0, 0, :3]) > 0.05:
            raise ValueError("RP1 root is outside the qualified start")
        _verify_payload_welds(
            context.world, self.actor.keypoint_names[2:4], roles["crate"]
        )
        self.jobs = MotionJobs(context, output, self.progress)

    def progress(self):
        return {
            "physics_step": self.context.scene.get_simulation_step(),
            "simulation_time_s": self.context.scene.get_simulation_time(),
        }

    def start(self, command_id):
        return self.jobs.start(
            command_id, "rp1-carry", dict(roles=self.roles), self._execute
        )

    def _execute(self, output):
        options = Namespace(
            lead_clearance_m=self.clearance,
            output_dir=output,
            hand="rp",
            step_reference="video",
            robot_xml=self.files["robot_xml"],
            converter_python=None,
            device="cuda:0",
            solver="dexuni",
            no_video=False,
            unloaded=False,
        )
        steps = _rollout(
            options,
            "step-up",
            self.actor,
            context=self.context,
            prepared_scene=(self.files["reference_mjcf"], None),
            live_paths=self.roles,
        )
        try:
            while True:
                if self.context.scene.topology_revision != self.revision:
                    raise ValueError("Scene changed during carrying")
                try:
                    progress = next(steps)
                except StopIteration as done:
                    return done.value
                yield progress
        finally:
            steps.close()

    def close(self):
        self.jobs.close()
