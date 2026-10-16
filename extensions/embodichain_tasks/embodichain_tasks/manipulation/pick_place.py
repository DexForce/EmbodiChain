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

"""Single-cycle physical PickPlace for online imitation-learning examples."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch

from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env

__all__ = ["PickPlaceEnv"]

START = (-0.42, -0.08, 0.025)
GOAL = (-0.42, 0.28, 0.025)


@register_env("PickPlace-Franka-v1", max_episode_steps=600)
class PickPlaceEnv(EmbodiedEnv):
    """Physically lift, transport, release and withdraw from a cube.

    Args:
        cfg: Environment configuration.
        **kwargs: Arguments forwarded to EmbodiedEnv.
    """

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        if self.robot.dof != 9 or len(self.active_joint_ids) != 8:
            raise ValueError(
                "PickPlace requires the stock 9-joint Panda with 8 active joints"
            )
        self._expert_engine = None

    def _initialize_episode(
        self, env_ids: Sequence[int] | None = None, **kwargs: Any
    ) -> None:
        super()._initialize_episode(env_ids, **kwargs)
        ids = torch.as_tensor(
            range(self.num_envs) if env_ids is None else env_ids,
            device=self.device,
            dtype=torch.long,
        )
        if not hasattr(self, "_lifted"):
            self._lifted = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._stable_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._success = self._lifted.clone()
        self._lifted[ids] = False
        self._stable_steps[ids] = 0
        self._success[ids] = False
        self.goal_pose = torch.tensor(GOAL, device=self.device).repeat(self.num_envs, 1)
        cube = self.sim.get_rigid_object("cube")
        pose = cube.get_local_pose().clone()
        pose[ids, :3] = torch.tensor(START, device=self.device)
        pose[ids, :2] += (torch.rand(len(ids), 2, device=self.device) - 0.5) * 0.08
        pose[ids, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        cube.clear_dynamics(env_ids=ids)
        cube.set_local_pose(pose[ids], env_ids=ids)

    def _extend_obs(self, obs: Any, **kwargs: Any) -> Any:
        obs = super()._extend_obs(obs, **kwargs)
        # Policy inputs use full proprioception; stored commands use active joints.
        obs["robot"] = self.robot.get_proprioception().clone()
        return obs

    def _update_sim_state(self, **kwargs: Any) -> None:
        super()._update_sim_state(**kwargs)
        if not hasattr(self, "_lifted"):
            return
        cube = self.sim.get_rigid_object("cube")
        pose = cube.get_local_pose()
        velocity = cube.body_data.vel
        self._lifted |= pose[:, 2] >= START[2] + 0.05
        qpos = self.robot.get_qpos()
        tcp = self.robot.compute_fk(
            qpos=qpos[:, self.robot.get_joint_ids("arm")], name="arm", to_matrix=True
        )[:, :3, 3]
        distance = torch.linalg.vector_norm(pose[:, :3] - self.goal_pose, dim=-1)
        opened = qpos[:, 7:].amin(dim=-1) >= 0.035
        withdrawn = torch.linalg.vector_norm(tcp - pose[:, :3], dim=-1) >= 0.10
        stable = (torch.linalg.vector_norm(velocity[:, :3], dim=-1) < 0.05) & (
            torch.linalg.vector_norm(velocity[:, 3:], dim=-1) < 0.2
        )
        valid = torch.isfinite(pose).all(-1) & torch.isfinite(velocity).all(-1)
        accepted = (
            self._lifted
            & (distance <= 0.03)
            & ((pose[:, 2] - GOAL[2]).abs() <= 0.012)
            & opened
            & withdrawn
            & stable
            & valid
        )
        self._stable_steps = torch.where(accepted, self._stable_steps + 1, 0)
        self._success = self._stable_steps * self.step_dt >= 0.3 - 1e-8

    def compute_task_state(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return cached measured outcomes without advancing the evidence clock.

        Args:
            **kwargs: Reserved environment evaluation options.

        Returns:
            Per-row success, failure and measured lift/goal-error metrics.
        """
        cube = self.sim.get_rigid_object("cube").get_local_pose()
        success = getattr(
            self,
            "_success",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        lifted = getattr(self, "_lifted", torch.zeros_like(success))
        goal = torch.tensor(GOAL, device=self.device)
        return (
            success.clone(),
            cube[:, 2] < -0.05,
            {
                "lifted": lifted.clone(),
                "goal_error": torch.linalg.vector_norm(cube[:, :3] - goal, dim=-1),
            },
        )

    def is_task_success(self, **kwargs: Any) -> torch.Tensor:
        """Return the same measured success used by student evaluation.

        Args:
            **kwargs: Reserved completion-query options.

        Returns:
            An independently owned boolean mask, one value per environment.
        """
        return self.compute_task_state()[0]

    def _engine(self) -> Any:
        if self._expert_engine is None:
            from embodichain.lab.sim.atomic_actions import (
                AtomicActionEngine,
                ControlPartCommandProfile,
            )
            from embodichain.lab.sim.motion.motion_generator import (
                MotionGenerator,
                MotionGenCfg,
            )
            from embodichain.lab.sim.motion.planners import ToppraPlannerCfg

            self._expert_engine = AtomicActionEngine(
                MotionGenerator(
                    MotionGenCfg(
                        planner_cfg=ToppraPlannerCfg(
                            robot_uid=self.robot.uid,
                            sim_instance_id=self.sim.instance_id,
                        )
                    )
                ),
                control_profiles={
                    "eef": ControlPartCommandProfile.joint_positions(
                        open=torch.tensor([0.04, 0.04]), grasp=torch.zeros(2)
                    )
                },
            )
        return self._expert_engine

    def _commands(
        self, trajectory: torch.Tensor, dwell: int = 0
    ) -> Iterable[torch.Tensor]:
        if trajectory.shape[1] == 0:
            raise RuntimeError("Expert planning returned an empty trajectory")
        for action in trajectory.unbind(1):
            yield action[:, self.active_joint_ids]
        for _ in range(dwell):
            yield trajectory[:, -1, self.active_joint_ids]

    def create_demo_segments(self, **kwargs: Any) -> Iterable[DemoSegment]:
        """Lazily plan physical grasp/lift, then placement from the measured scene.

        Args:
            **kwargs: Reserved planning options.

        Yields:
            Two segments with measured acceptance.
        """
        from embodichain.lab.sim.atomic_actions import (
            ActionInvocation,
            Affordance,
            EntityState,
            GraspGoal,
            MotionPolicy,
            ObjectSemantics,
            PickUpOptions,
            PlaceGoal,
            PlaceOptions,
            SceneSnapshot,
        )

        engine = self._engine()
        cube = self.sim.get_rigid_object("cube")
        source = cube.get_local_pose(to_matrix=True).clone()
        # Fixed top-down grasp; no teleport, grasp attachment or dynamics clearing.
        grasp = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1)
        grasp[:, :3, :3] = torch.diag(
            torch.tensor([1.0, -1.0, -1.0], device=self.device)
        )
        grasp[:, :3, 3] = source[:, :3, 3]
        # Panda TCP sits near the finger tips; grasp the upper half of the cube
        # so the open fingers do not sweep its lower edges during approach.
        grasp[:, 2, 3] += 0.02
        semantics = ObjectSemantics(
            affordance=Affordance(), geometry={}, label="cube", entity_id="cube"
        )
        endpoints = {"primary": {"motion": "arm", "grasp": "eef"}}
        context = engine.initial_context(
            scene=SceneSnapshot(
                timestamp=0.0, version=0, entities={"cube": EntityState(source)}
            ),
            control_dt=self.step_dt,
        )
        pick = engine.compile(
            (
                ActionInvocation(
                    skill_id="pick_up",
                    goal=GraspGoal(semantics, grasp_xpos=grasp),
                    binding=engine.bind_control_parts("pick_up", endpoints),
                    motion_policy=MotionPolicy(sample_count=150),
                    skill_options=PickUpOptions(
                        pre_grasp_distance=0.10, lift_height=0.15, hand_interp_steps=20
                    ),
                ),
            ),
            context,
        )
        if not bool(pick.plan_success.all()):
            raise RuntimeError("PickPlace grasp planning failed")
        positions = pick.trajectory.positions
        # Give the force-driven fingers time to establish contact before lifting.
        hand_ids = self.robot.get_joint_ids("eef")
        closed = (positions[..., hand_ids] <= 0.002).all(dim=-1).all(dim=0)
        close_steps = closed.nonzero(as_tuple=False).flatten()
        if close_steps.numel():
            index = int(close_steps[0]) + 1
            positions = torch.cat(
                (
                    positions[:, :index],
                    positions[:, index - 1 : index].repeat(1, 30, 1),
                    positions[:, index:],
                ),
                dim=1,
            )
        yield DemoSegment(
            name="pick_cube",
            instruction="Grasp and lift the cube.",
            target_uid="cube",
            actions=self._commands(positions, dwell=12),
            validator=lambda: self._lifted.clone(),
        )
        held = pick.projected_context.get_held_object("arm")
        if held is None:
            raise RuntimeError("Pick plan has no held-object transform")
        measured = cube.get_local_pose(to_matrix=True).clone()
        target_object = measured.clone()
        target_object[:, :3, 3] = self.goal_pose
        target_object[:, 2, 3] += 0.015
        # Capture the real robot and object state after the grasp segment.
        place_context = engine.initial_context(
            task=pick.projected_context.task,
            scene=SceneSnapshot(
                timestamp=self.step_dt * int(self._elapsed_steps.max()),
                version=1,
                entities={"cube": EntityState(measured)},
            ),
            control_dt=self.step_dt,
        )
        tcp = self.robot.compute_fk(
            qpos=self.robot.get_qpos()[:, self.robot.get_joint_ids("arm")],
            name="arm",
            to_matrix=True,
        )
        object_to_eef = torch.bmm(torch.linalg.inv(measured), tcp)
        place = engine.compile(
            (
                ActionInvocation(
                    skill_id="place",
                    goal=PlaceGoal(torch.bmm(target_object, object_to_eef)),
                    binding=engine.bind_control_parts("place", endpoints),
                    motion_policy=MotionPolicy(sample_count=150),
                    skill_options=PlaceOptions(lift_height=0.12, hand_interp_steps=20),
                ),
            ),
            place_context,
        )
        if not bool(place.plan_success.all()):
            raise RuntimeError("PickPlace placement planning failed")
        yield DemoSegment(
            name="place_cube",
            instruction="Place the cube, release it and withdraw.",
            target_uid="cube",
            actions=self._commands(place.trajectory.positions, dwell=30),
            validator=self.is_task_success,
        )
