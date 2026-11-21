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

"""Native simulation observations for occurrence-bound demonstration skills."""

from __future__ import annotations

import numpy as np
import torch

from embodichain.data_pipeline.real2sim import load_demo_skill
from embodichain.lab.sim.atomic_skills import (
    DemoSkillMotionProvider,
    DemoSkillObservation,
    JointPositionTarget,
)
from embodichain.lab.sim.motion.expansion.real2sim_collision import (
    BoxCollisionBody,
    SampledRobotCollisionChecker,
)

__all__: list[str] = []


class _SimulationDemoSkillEvidence:
    def __init__(self, simulation, engine, declaration, artifact, goal, binding):
        self.sim, self.robot = simulation, engine.robot
        self.artifact, self.goal = artifact, goal
        self.arm = binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        self.revision = 0
        self.sampler = None
        self.actor_ids = {}
        if declaration.contact_sensor_uid is not None:
            from embodichain.lab.sim.sensors.contact.sampling import (
                ContactEvidenceSampler,
                native_contact_revision,
            )

            sensor = simulation.get_sensor(declaration.contact_sensor_uid)
            if sensor is None:
                raise ValueError("DemoSkill contact sensor is missing")
            self.sampler = ContactEvidenceSampler(
                sensor,
                native_revision_reader=(
                    (lambda: native_contact_revision(simulation))
                    if simulation.physics_backend == "default"
                    else None
                ),
            )
            self.actor_ids = {
                role: sensor.get_actor_ids(uid) for role, uid in goal.role_uids.items()
            }
            if declaration.tcp_contact_links:
                self.actor_ids["tcp"] = sensor.get_actor_ids(
                    self.robot.uid, declaration.tcp_contact_links
                )

    def observe(self, context):
        row = int(context.env_ids[0])
        poses = {
            role: context.scene.entities[entity]
            .pose.reshape(4, 4)
            .detach()
            .cpu()
            .numpy()
            for role, entity in self.goal.role_bindings.items()
        }
        needs_tcp = any(
            "tcp" in {item.subject, item.other}
            for item in self.artifact.execution_spec.constraints
        )
        if needs_tcp:
            tcp = self.robot.compute_batch_fk(
                context.robot.qpos[:, None, list(self.arm.joint_ids)],
                name=self.arm.control_part,
                env_ids=[row],
                to_matrix=True,
            )
            if tcp is not None:
                poses["tcp"] = tcp.reshape(4, 4).detach().cpu().numpy()
        contacts = {}
        if self.sampler is not None:
            from embodichain.lab.sim.evidence import PhysicalEvidenceRequest
            from embodichain.lab.sim.sensors.contact.sampling import (
                CONTACT_ACTOR_IDS,
                CONTACT_VALID_MASK,
            )

            request = PhysicalEvidenceRequest(
                context.robot.timestamp,
                self.revision,
                context.env_ids,
                (CONTACT_ACTOR_IDS,),
            )
            self.revision += 1
            sample = self.sampler.sample(request)
            actors, valid = sample.frame.get(CONTACT_ACTOR_IDS), sample.frame.get(
                CONTACT_VALID_MASK
            )
            if (
                sample.acquisition_error is not None
                or sample.dropped_contacts
                or not bool(actors.valid.all())
                or not bool(valid.valid.all())
            ):
                raise ValueError(
                    "DemoSkill contact sample is incomplete or unavailable"
                )
            pairs = actors.values[0][valid.values[0]]
            for constraint in self.artifact.spec.constraints:
                if constraint.kind not in {"contact", "no_contact"}:
                    continue
                key = tuple(sorted((constraint.subject, constraint.other)))
                left = self.actor_ids.get(key[0])
                right = self.actor_ids.get(key[1])
                if left is None or right is None:
                    raise ValueError(
                        "DemoSkill contact roles have no explicit native actor binding"
                    )
                forward = torch.isin(pairs[:, 0], left[row]) & torch.isin(
                    pairs[:, 1], right[row]
                )
                reverse = torch.isin(pairs[:, 1], left[row]) & torch.isin(
                    pairs[:, 0], right[row]
                )
                contacts[key] = bool((forward | reverse).any())
        return DemoSkillObservation(context.robot.timestamp, row, poses, contacts)


def _create_demo_skill_source(declaration, request, engine):
    from embodichain.lab.sim import SimulationManager

    simulation = SimulationManager.get_instance(
        engine.motion_generator.planner.cfg.sim_instance_id
    )
    if simulation.get_robot(engine.robot.uid) is not engine.robot:
        raise ValueError("DemoSkill planner belongs to a different simulation")
    artifact = load_demo_skill(declaration.asset_manifest)
    goal, binding = request.invocation.goal, request.invocation.binding
    roles = {
        role: BoxCollisionBody.from_rigid_object(
            simulation.get_rigid_object(uid),
            env_id=(
                int(engine.planning_services.env_ids[0])
                if engine.planning_services.env_ids is not None
                else 0
            ),
        )
        for role, uid in goal.role_uids.items()
    }
    row = (
        int(engine.planning_services.env_ids[0])
        if engine.planning_services.env_ids is not None
        else 0
    )
    checker = SampledRobotCollisionChecker(
        engine.robot.cfg.fpath,
        joint_names=engine.robot.joint_names,
        bodies=tuple(roles.values()),
        world_from_base=engine.robot.get_local_pose(to_matrix=True)[row]
        .detach()
        .cpu()
        .numpy(),
    )
    evidence = _SimulationDemoSkillEvidence(
        simulation, engine, declaration, artifact, goal, binding
    )
    return DemoSkillMotionProvider(
        artifact,
        role_bindings=goal.role_bindings,
        robot_geometry_sha256=checker.geometry_sha256,
        role_geometry_sha256={
            role: body.geometry_sha256 for role, body in roles.items()
        },
        evidence_provider=evidence,
        motion_generator=engine.motion_generator,
    )
