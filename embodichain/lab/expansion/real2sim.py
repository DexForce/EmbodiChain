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

"""Per-arena Real2Sim bindings sharing the canonical source qualification code."""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import torch
from tensordict import TensorDict
from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
from embodichain.lab.sim.scene.expansion import SceneVariant
from embodichain.lab.task_program.integrations.scene_expansion import (
    ScenePreparationResult,
)
from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
    _prepare_tabletop_sources,
    _task_space_kinematics_factory,
    _capture_tabletop_episode,
)
from embodichain.lab.task_program.integrations.simulation.real2sim_collision import (
    TabletopPrototypeCollisionValidator,
    TabletopCollisionGuardFactory,
)

__all__: list[str] = []


class _RowPose:
    def __init__(self, entity, row):
        self.entity, self.row = entity, row

    def get_local_pose(self, **kwargs):
        return self.entity.get_local_pose(**kwargs)[self.row : self.row + 1].clone()


class _RowKinematics:
    def __init__(self, robot, row):
        self.robot, self.row = robot, row

    def compute_batch_fk(self, **kwargs):
        return self.robot.compute_batch_fk(**kwargs, env_ids=[self.row])


class _Real2SimRow:
    def __init__(
        self,
        env,
        row,
        profile,
        *,
        output: Path,
        ordinal: int,
        attempt: int,
        initial_state_id: str,
        recipe: str = "nominal",
    ):
        self.env, self.row, self.profile = env, row, profile
        source = _prepare_tabletop_sources(env, profile, env_id=row)
        (
            adapter,
            self.declarations,
            self.call_indices,
            self.object_body,
            self.target_body,
            checker,
            self.prototypes,
            unique,
        ) = source
        self.adapter = adapter.for_env_ids((row,))
        program = self.adapter.compile(env.cfg.task_program)
        calls = tuple(
            call.call_index
            for segment in program.iter_segments()
            for call in segment.calls
        )
        if calls != self.call_indices:
            raise ValueError(
                "Real2Sim requires pinned sources for every call occurrence"
            )
        validator = TabletopPrototypeCollisionValidator(
            robot=env.robot,
            checker=checker,
            prototypes=tuple(unique.values()),
            object_uid=profile.object_entity_id,
            target_uid=profile.target_entity_id,
            ground_uid="ground",
            finger_links=profile.finger_links,
            finger_joints=profile.finger_joints,
            closing_directions=profile.closing_directions,
            gripper_contact=profile.gripper_contact,
            base_link=profile.base_link,
            contact_depth_m=profile.contact_depth_m,
            base_ground_depth_m=profile.base_ground_depth_m,
            env_id=row,
        )
        from embodichain.lab.task_program.integrations.expansion import (
            TaskProgramMappedRetimeFactory,
            TaskProgramSameGridExpansionFactory,
        )

        common = dict(
            velocity_limits=profile.velocity_limits,
            acceleration_limits=profile.acceleration_limits,
            call_indices=self.call_indices,
            diagnostic_unchecked=True,
        )
        variant = SceneVariant(profile.scene_revision, profile.seed, ordinal, ())
        delegate = None
        if recipe.startswith("time_"):
            delegate = TaskProgramMappedRetimeFactory(
                duration_scale=1.15 if recipe == "time_115" else 1.30,
                phase_allowlist=profile.retime_phase_allowlist,
                **common,
            )
        elif recipe in {"joint_7", "joint_11", "task_space"}:
            delegate = TaskProgramSameGridExpansionFactory(
                operator=(
                    "task_space_residual"
                    if recipe == "task_space"
                    else "joint_residual"
                ),
                seed=7 if recipe == "joint_7" else 11,
                motion_headroom_fraction=(
                    None if recipe == "task_space" else profile.motion_headroom_fraction
                ),
                kinematics_factory=(
                    _task_space_kinematics_factory(env.robot, row)
                    if recipe == "task_space"
                    else None
                ),
                geometry_family_id=variant.variant_id,
                initial_state_id=initial_state_id,
                **common,
            )
        elif recipe != "nominal":
            raise ValueError(f"unsupported batched Real2Sim recipe: {recipe}")
        guard = TabletopCollisionGuardFactory(
            validator,
            delegate,
            velocity_limits=profile.velocity_limits,
            acceleration_limits=profile.acceleration_limits,
            ik_smoothing_trigger_fraction=(
                profile.motion_headroom_fraction if recipe == "task_space" else None
            ),
        )
        tracer = self.adapter.create_traced_configured_sources(program, guard)
        self.bridge = self.adapter.create_bridge(
            program, nominal_source_factory=tracer, plan_transform_factory=guard
        )
        self.bridge.enable_execution_trace()
        self.state = dict(
            variant=variant,
            preparation=ScenePreparationResult(
                initial_state_id,
                ValidationResult((ValidationCheck("initial_state", "passed"),)),
                {"env_id": row},
            ),
            validator=validator,
            guard=guard,
            tracer=tracer,
            delegate=delegate,
            recipe=recipe,
            attempt_id=attempt,
            layout="prepared",
            layout_parameters={},
        )
        self.output = output

    def qualify(self, *, qpos: torch.Tensor, object_pose: torch.Tensor):
        n = len(qpos)
        values = TensorDict(
            {
                "robot": {"qpos": qpos[None]},
                "rigid_objects": {self.profile.object_uid: {"pose": object_pose[None]}},
            },
            batch_size=[1, n],
        )
        buffer = TensorDict({"states": values}, batch_size=[1, n])
        # Read-only evidence view: it cannot reset, step or mutate the shared World.
        evidence = SimpleNamespace(
            _traj_steps=torch.tensor([n - 1]),
            _traj_buffer=buffer,
            robot=_RowKinematics(self.env.robot, self.row),
            step_dt=self.env.step_dt,
            sim=SimpleNamespace(
                get_rigid_object=lambda uid: _RowPose(
                    self.env.sim.get_rigid_object(uid), self.row
                )
            ),
            _active_task_program_bridge=self.bridge,
        )
        return _capture_tabletop_episode(
            evidence,
            SimpleNamespace(length=n - 1),
            profile=self.profile,
            state=self.state,
            prototypes=self.prototypes,
            call_indices=self.call_indices,
            object_body=self.object_body,
            target_body=self.target_body,
            observations={name: ("robot", name) for name in ("qpos", "qvel", "qf")},
            recorder=SimpleNamespace(dataset_full_path=str(self.output)),
            declarations=self.declarations,
        )
