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

"""Gym environment that executes Action Engine programs against live state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from embodichain.gen_sim.action_engine.config import (
    RuntimePolicyCfg,
    generation_defaults,
    resolve_agent_runtime_policy,
)
from embodichain.gen_sim.action_engine.capabilities import (
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.domain import validate_seed_graph
from embodichain.gen_sim.action_engine.protocol import ACTION_ENGINE_ENV_ID
from embodichain.gen_sim.action_engine.runtime import (
    ProgramExecutor,
    evaluate_predicate,
    load_agent_execution_program,
    load_execution_program,
)
from embodichain.gen_sim.action_engine.runtime.solver_compat import (
    install_action_engine_solver_compat,
    repair_action_engine_ur5_solver_cfg,
)
from embodichain.gen_sim.action_engine.runtime.motion_policy import (
    resolve_motion_policy,
)
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env

__all__ = ["ACTION_ENGINE_ENV_ID", "ActionEngineEnv"]

_MAX_EPISODE_STEPS = int(generation_defaults()["task"]["max_episode_steps"])


@register_env(ACTION_ENGINE_ENV_ID, max_episode_steps=_MAX_EPISODE_STEPS)
class ActionEngineEnv(EmbodiedEnv):
    """EmbodiedEnv adapter for in-memory compiled execution programs."""

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs: Any) -> None:
        agent_config = kwargs.pop("agent_config", None)
        task_name = kwargs.pop("task_name", None)
        agent_config_path = kwargs.pop("agent_config_path", None)
        runtime_backend = kwargs.pop("runtime_backend", "independent")
        runtime_policy = kwargs.pop("runtime_policy", None)
        if not isinstance(agent_config, Mapping):
            raise ValueError("ActionEngineEnv requires an agent_config mapping.")
        if not isinstance(task_name, str) or not task_name:
            raise ValueError("ActionEngineEnv requires a non-empty task_name.")
        if not isinstance(agent_config_path, str) or not agent_config_path:
            raise ValueError("ActionEngineEnv requires agent_config_path.")
        self.agent_config = dict(agent_config)
        self.agent_config_path = agent_config_path
        self.task_name = task_name
        if runtime_policy is None:
            runtime_policy = resolve_agent_runtime_policy(self.agent_config)
        if not isinstance(runtime_policy, RuntimePolicyCfg):
            raise TypeError("ActionEngineEnv runtime_policy must be RuntimePolicyCfg.")
        self.runtime_policy = runtime_policy
        if runtime_backend != "independent":
            raise ValueError(
                "ActionEngineEnv only supports its independent runtime, got "
                f"{runtime_backend!r}."
            )
        self.runtime_backend = str(runtime_backend)
        self.last_execution: Any | None = None
        self._runtime_state_ready = False
        repair_action_engine_ur5_solver_cfg(getattr(cfg, "robot", None))
        super().__init__(cfg, **kwargs)
        install_action_engine_solver_compat(self.robot)
        if bool(getattr(self, "ignore_terminations_during_agent", False)):
            # Atomic trajectories execute online through env.step(). Prevent a
            # transient task signal from resetting an environment mid-program.
            self.cfg.ignore_terminations = True
        self._capture_runtime_state()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self._runtime_state_ready = False
        observation, info = super().reset(seed=seed, options=options)
        self.last_execution = None
        self._capture_runtime_state()
        return observation, info

    def _capture_runtime_state(self) -> None:
        """Capture reset-relative robot and object state used by symbolic bindings."""
        self.init_qpos = self.robot.get_qpos().clone()
        self._agent_arm_slots = self._resolve_arm_slots()
        for side in ("left", "right"):
            self._initialize_arm(side, self._agent_arm_slots.get(side))

        default_open = getattr(self, "gripper_open_state", (0.04, 0.04))
        default_close = getattr(self, "gripper_close_state", (0.0, 0.0))
        self.open_state = torch.as_tensor(
            getattr(self, "agent_open_state", default_open),
            dtype=self.init_qpos.dtype,
            device=self.init_qpos.device,
        ).flatten()
        self.close_state = torch.as_tensor(
            getattr(self, "agent_close_state", default_close),
            dtype=self.init_qpos.dtype,
            device=self.init_qpos.device,
        ).flatten()
        self.left_arm_current_gripper_state = self._hand_qpos("left")
        self.right_arm_current_gripper_state = self._hand_qpos("right")
        self.update_obj_info()
        self.agent_initial_object_poses = {
            uid: item["pose"].clone() for uid, item in self.obj_info.items()
        }
        self.agent_initial_object_heights = {
            uid: item["height"].clone() for uid, item in self.obj_info.items()
        }
        self._runtime_state_ready = True

    def _resolve_arm_slots(self) -> dict[str, dict[str, str | None] | None]:
        configured = getattr(self, "agent_arm_slots", None)
        if isinstance(configured, Mapping):
            result: dict[str, dict[str, str | None] | None] = {
                "left": None,
                "right": None,
            }
            for side in result:
                value = configured.get(side)
                if isinstance(value, str):
                    result[side] = {"arm": value, "eef": None}
                elif isinstance(value, Mapping):
                    result[side] = {
                        "arm": value.get("arm", value.get("arm_control_part")),
                        "eef": value.get(
                            "eef",
                            value.get("hand", value.get("eef_control_part")),
                        ),
                    }
            return result
        parts = getattr(self.robot, "control_parts", {}) or {}
        if "left_arm" in parts or "right_arm" in parts:
            return {
                "left": {"arm": "left_arm", "eef": "left_eef"},
                "right": {"arm": "right_arm", "eef": "right_eef"},
            }
        if "arm" in parts:
            side = str(getattr(self, "agent_single_arm_slot", "right"))
            result = {"left": None, "right": None}
            result[side] = {"arm": "arm", "eef": "hand"}
            return result
        raise ValueError("Robot exposes no arm control part for Action Engine.")

    def _initialize_arm(
        self,
        side: str,
        slot: dict[str, str | None] | None,
    ) -> None:
        arm = None if slot is None else slot.get("arm")
        eef = None if slot is None else slot.get("eef")
        arm_ids = self._control_part_ids(arm)
        eef_ids = self._control_part_ids(eef)
        setattr(self, f"{side}_arm_joints", arm_ids)
        setattr(self, f"{side}_eef_joints", eef_ids)
        arm_qpos = self.init_qpos[:, arm_ids]
        setattr(self, f"{side}_arm_init_qpos", arm_qpos.clone())
        setattr(self, f"{side}_arm_current_qpos", arm_qpos.clone())
        if arm is None or not arm_ids:
            setattr(self, f"{side}_arm_init_xpos", None)
            setattr(self, f"{side}_arm_current_xpos", None)
            return
        xpos = self.robot.compute_fk(arm_qpos, name=arm, to_matrix=True)
        setattr(self, f"{side}_arm_init_xpos", xpos.clone())
        setattr(self, f"{side}_arm_current_xpos", xpos.clone())

    def _control_part_ids(self, name: str | None) -> list[int]:
        if name is None:
            return []
        parts = getattr(self.robot, "control_parts", {}) or {}
        if name not in parts:
            return []
        return list(self.robot.get_joint_ids(name=name))

    def _hand_qpos(self, side: str) -> torch.Tensor:
        ids = list(getattr(self, f"{side}_eef_joints", ()))
        return self.init_qpos[:, ids].clone()

    def get_agent_arm_control_part(self, is_left: bool) -> str:
        value = self._agent_arm_slots["left" if is_left else "right"]
        arm = None if value is None else value.get("arm")
        if not isinstance(arm, str) or not arm:
            raise ValueError(f"{'left' if is_left else 'right'} arm is not configured.")
        return arm

    def get_agent_eef_control_part(self, is_left: bool) -> str | None:
        value = self._agent_arm_slots["left" if is_left else "right"]
        eef = None if value is None else value.get("eef")
        return str(eef) if eef else None

    def get_current_qpos_agent(self) -> tuple[torch.Tensor, torch.Tensor]:
        qpos = self.robot.get_qpos()
        return tuple(
            qpos[:, list(getattr(self, f"{side}_arm_joints", ()))].clone()
            for side in ("left", "right")
        )

    def set_current_qpos_agent(
        self,
        arm_qpos: torch.Tensor,
        is_left: bool,
    ) -> None:
        side = "left" if is_left else "right"
        setattr(self, f"{side}_arm_current_qpos", arm_qpos)

    def get_current_xpos_agent(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        qpos = self.robot.get_qpos()
        result = []
        for side in ("left", "right"):
            slot = self._agent_arm_slots.get(side)
            arm = None if slot is None else slot.get("arm")
            arm_ids = list(getattr(self, f"{side}_arm_joints", ()))
            if not arm or not arm_ids:
                result.append(None)
                continue
            result.append(
                self.robot.compute_fk(
                    qpos[:, arm_ids],
                    name=arm,
                    to_matrix=True,
                )
            )
        return result[0], result[1]

    def set_current_xpos_agent(
        self,
        arm_xpos: torch.Tensor,
        is_left: bool,
    ) -> None:
        side = "left" if is_left else "right"
        setattr(self, f"{side}_arm_current_xpos", arm_xpos)

    def get_current_gripper_state_agent(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qpos = self.robot.get_qpos()
        return tuple(
            qpos[:, list(getattr(self, f"{side}_eef_joints", ()))].clone()
            for side in ("left", "right")
        )

    def set_current_gripper_state_agent(
        self,
        arm_gripper_state: torch.Tensor,
        is_left: bool,
    ) -> None:
        side = "left" if is_left else "right"
        setattr(self, f"{side}_arm_current_gripper_state", arm_gripper_state)

    def get_arm_fk(self, qpos: torch.Tensor, is_left: bool) -> torch.Tensor:
        return self.robot.compute_fk(
            name=self.get_agent_arm_control_part(is_left),
            qpos=torch.as_tensor(qpos, device=self.robot.device),
            to_matrix=True,
        )

    def sync_agent_state_from_qpos(self, qpos: torch.Tensor) -> None:
        """Keep arm-selection seeds synchronized with the command sent to sim."""
        qpos = torch.as_tensor(
            qpos,
            dtype=self.init_qpos.dtype,
            device=self.init_qpos.device,
        )
        for side in ("left", "right"):
            arm_ids = list(getattr(self, f"{side}_arm_joints", ()))
            hand_ids = list(getattr(self, f"{side}_eef_joints", ()))
            arm_qpos = qpos[:, arm_ids]
            setattr(self, f"{side}_arm_current_qpos", arm_qpos.clone())
            slot = self._agent_arm_slots.get(side)
            arm = None if slot is None else slot.get("arm")
            if arm and arm_ids:
                xpos = self.robot.compute_fk(arm_qpos, name=arm, to_matrix=True)
                setattr(self, f"{side}_arm_current_xpos", xpos)
            setattr(
                self,
                f"{side}_arm_current_gripper_state",
                qpos[:, hand_ids].clone(),
            )

    def get_arm_ik(
        self,
        target_xpos: torch.Tensor,
        is_left: bool,
        qpos_seed: torch.Tensor | None = None,
        env_ids: list[int] | None = None,
    ) -> tuple[bool, torch.Tensor]:
        success, qpos = self.robot.compute_ik(
            name=self.get_agent_arm_control_part(is_left),
            pose=target_xpos,
            joint_seed=qpos_seed,
            env_ids=env_ids,
        )
        success_value = (
            bool(torch.as_tensor(success).all().item())
            if isinstance(success, torch.Tensor)
            else bool(success)
        )
        return success_value, qpos

    def update_obj_info(self) -> None:
        info = getattr(self, "obj_info", {})
        for uid in self.sim.get_rigid_object_uid_list():
            entity = self.sim.get_rigid_object(uid)
            if entity is None:
                continue
            pose = entity.get_local_pose(to_matrix=True)
            info[uid] = {"pose": pose, "height": pose[:, 2, 3]}
        self.obj_info = info

    def create_demo_action_list(
        self,
        regenerate: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Compile in memory when requested, then execute the program online."""
        program = load_agent_execution_program(
            self.agent_config,
            agent_config_path=self.agent_config_path,
            regenerate=regenerate,
        )
        executor = ProgramExecutor(
            program,
            self,
            max_transitions=(
                int(self.action_engine_max_transitions)
                if hasattr(self, "action_engine_max_transitions")
                else None
            ),
            settle_steps=(
                int(self.action_engine_settle_steps)
                if hasattr(self, "action_engine_settle_steps")
                else None
            ),
            record_runtime=bool(getattr(self, "action_engine_record_runtime", True)),
            record_root=getattr(self, "action_engine_record_root", None),
            runtime_policy=self.runtime_policy,
        )
        self.last_execution = executor.run(
            run_id=kwargs.get("runtime_run_id"),
            episode_index=int(kwargs.get("episode_index", 0)),
        )
        return self.last_execution

    def execute_seed_graph(
        self,
        seed_graph: Mapping[str, Any],
        *,
        runtime_run_id: str,
        episode_index: int,
        record_root: str | None = None,
    ) -> Any:
        """Execute one already validated branch graph without rewriting config."""
        program = self.preflight_seed_graph(seed_graph)
        route = getattr(self, "action_engine_ab_route", None)
        graph_route = seed_graph.get("planner_route")
        if route in {"offline", "online"} and graph_route != route:
            raise ValueError(
                f"A/B branch route {route!r} cannot execute graph route "
                f"{graph_route!r}."
            )
        executor = ProgramExecutor(
            program,
            self,
            max_transitions=(
                int(self.action_engine_max_transitions)
                if hasattr(self, "action_engine_max_transitions")
                else None
            ),
            settle_steps=(
                int(self.action_engine_settle_steps)
                if hasattr(self, "action_engine_settle_steps")
                else None
            ),
            record_runtime=bool(getattr(self, "action_engine_record_runtime", True)),
            record_root=record_root,
            runtime_policy=self.runtime_policy,
        )
        self.last_execution = executor.run(
            run_id=runtime_run_id,
            episode_index=episode_index,
        )
        return self.last_execution

    def preflight_seed_graph(self, seed_graph: Mapping[str, Any]) -> Any:
        """Validate/compile one branch graph without stepping the simulator.

        This hook is intentionally separate from :meth:`execute_seed_graph` so
        strict A/B can preflight both branches before either executor sends a
        command to the robot.
        """
        source = self.agent_config.get("source", {})
        if not isinstance(source, Mapping):
            source = {}
        uid_map = source.get("uid_map", {})
        if not isinstance(uid_map, Mapping):
            uid_map = {}
        known_objects = {str(uid) for uid in uid_map.values() if str(uid)}
        registry = build_atomic_capability_registry()
        graph = validate_seed_graph(
            seed_graph,
            known_objects=known_objects or None,
            known_actions=registry.names(),
            executable_actions=registry.executable_names(),
            require_executable=True,
        )
        for node in graph["nodes"]:
            registry.validate_binding(node)
            resolve_motion_policy(
                str(
                    self.agent_config.get(
                        "robot_profile",
                        getattr(self, "agent_robot_profile", "dual_ur10"),
                    )
                ),
                node["motion_policy"],
            )
        return load_execution_program(
            graph,
            known_objects=known_objects or None,
            registry=registry,
        )

    def _normalize_demo_action_list(self, action_list: Any) -> Any:
        """Preserve metadata on action streams that already ran online.

        ``EmbodiedEnv`` normally rebuilds returned sequences after validating
        their action width. Rebuilding an ``ExecutionResult`` would discard its
        success masks and runtime-record location, and its commands have
        already been sent to the simulator, so no replay normalization is
        needed.
        """
        if getattr(action_list, "already_executed", False):
            return action_list
        return super()._normalize_demo_action_list(action_list)

    def is_task_success(self, **_: Any) -> torch.Tensor:
        configured = getattr(self, "agent_success", None)
        if isinstance(configured, Mapping):
            return evaluate_predicate(self, configured)
        if self.last_execution is not None:
            return torch.as_tensor(
                getattr(
                    self.last_execution,
                    "runtime_success",
                    getattr(self.last_execution, "success", False),
                ),
                dtype=torch.bool,
                device=self.device,
            )
        return torch.zeros(
            int(self.num_envs),
            dtype=torch.bool,
            device=self.device,
        )

    def compute_task_state(
        self,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        success = self.is_task_success()
        return success, torch.zeros_like(success), {}
