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

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from embodichain.gen_sim.action_agent_pipeline.contracts import (
    ACTION_AGENT_ENV_ID,
    DEFAULT_VIEWER_CAMERA_UID,
    ROBOTIQ_ARG2F_140_CLOSE_QPOS as _ROBOTIQ_ARG2F_140_CLOSE_QPOS,
    ROBOTIQ_ARG2F_140_OPEN_QPOS as _ROBOTIQ_ARG2F_140_OPEN_QPOS,
)
from embodichain.gen_sim.action_agent_pipeline.config.defaults import (
    DEFAULT_MAX_EPISODE_STEPS,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.success_evaluator import (
    evaluate_configured_success,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.pytorch_solver_compat import (
    install_action_agent_pytorch_solver_compat,
)
from embodichain.gen_sim.action_agent_pipeline.utils.timing import timing_scope
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.utils import logger

__all__ = ["AgenticGenSimEnv", "AtomicActionsAgentEnv"]

_AGENT_RESERVED_KEYS = frozenset({"task_name", "config_dir"})
_REQUIRED_AGENT_KWARGS = frozenset({"agent_config", "task_name"})
_OPTIONAL_AGENT_KWARGS = frozenset({"agent_config_path"})
_AGENT_KWARGS = _REQUIRED_AGENT_KWARGS | _OPTIONAL_AGENT_KWARGS


@register_env(ACTION_AGENT_ENV_ID, max_episode_steps=DEFAULT_MAX_EPISODE_STEPS)
class AgenticGenSimEnv(EmbodiedEnv):
    """Config-driven agent environment for atomic-action tasks."""

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs: Any) -> None:
        self._agent_runtime_state_ready = False
        env_kwargs, agent_kwargs = _split_env_and_agent_kwargs(kwargs)
        super().__init__(cfg, **env_kwargs)
        install_action_agent_pytorch_solver_compat(self.robot)
        if bool(getattr(self, "ignore_terminations_during_agent", False)):
            self.cfg.ignore_terminations = True
        self._init_agents(**agent_kwargs)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if self._agent_runtime_state_ready:
            # Preserve the completed episode result before reset invalidates runtime caches.
            self.episode_success_status |= self.is_task_success()
        self._agent_runtime_state_ready = False
        obs, info = super().reset(seed=seed, options=options)
        self._draw_arrangement_debug_markers()
        self.get_states()
        self._agent_runtime_state_ready = True
        return obs, info

    def _draw_arrangement_debug_markers(self) -> None:
        debug = getattr(self, "arrangement_debug", None)
        if not isinstance(debug, Mapping) or getattr(
            self, "_arrangement_debug_drawn", False
        ):
            return
        slots = debug.get("slots", [])
        target_poses = []
        high_poses = []
        for slot in slots:
            if not isinstance(slot, Mapping):
                continue
            target = slot.get("target")
            high = slot.get("high")
            if not (
                isinstance(target, (list, tuple))
                and len(target) == 3
                and isinstance(high, (list, tuple))
                and len(high) == 3
            ):
                continue
            target_pose = torch.eye(4)
            target_pose[:3, 3] = torch.tensor(target, dtype=target_pose.dtype)
            high_pose = torch.eye(4)
            high_pose[:3, 3] = torch.tensor(high, dtype=high_pose.dtype)
            target_poses.append(target_pose)
            high_poses.append(high_pose)
        if target_poses:
            self.sim.draw_marker(
                MarkerCfg(
                    name="arrangement_target_slots",
                    axis_xpos=torch.stack(target_poses),
                    axis_size=0.004,
                    axis_len=0.06,
                )
            )
            self.sim.draw_marker(
                MarkerCfg(
                    name="arrangement_high_points",
                    axis_xpos=torch.stack(high_poses),
                    axis_size=0.002,
                    axis_len=0.035,
                )
            )
        self._arrangement_debug_drawn = True

    def is_task_success(self, **kwargs) -> torch.Tensor:
        if not getattr(self, "_agent_runtime_state_ready", False):
            return torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        return evaluate_configured_success(self)

    def compute_task_state(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor, dict]:
        success = self.is_task_success()
        fail = torch.zeros_like(success)
        return success, fail, {}

    def _init_agents(
        self,
        agent_config: Mapping[str, Any],
        task_name: str,
        agent_config_path: str | None = None,
    ) -> None:
        sections = self._validate_agent_config(agent_config)

        from embodichain.gen_sim.action_agent_pipeline.agents.compile_agent import (
            CompileAgent,
            resolve_precomputed_task_graph_path,
        )

        configured_graph = sections["TaskAgent"].get("precomputed_task_graph")
        if configured_graph is not None and not isinstance(configured_graph, str):
            raise ValueError("TaskAgent.precomputed_task_graph must be a path string.")
        # Resolve the graph once during environment construction. Runtime graph
        # execution is deterministic and never falls back to an online LLM.
        self.precomputed_task_graph_path = resolve_precomputed_task_graph_path(
            configured_path=configured_graph,
            agent_config_path=agent_config_path,
        )
        self.compile_agent = CompileAgent(
            **sections["CompileAgent"],
            task_name=task_name,
            # CompileAgent owns file-vs-directory normalization so every caller
            # follows the same cache-location policy.
            config_dir=agent_config_path,
        )

    def _validate_agent_config(
        self, agent_config: Mapping[str, Any]
    ) -> dict[str, Mapping[str, Any]]:
        missing = {"Agent", "TaskAgent", "CompileAgent"} - set(agent_config)
        if missing:
            raise ValueError(
                "Agent config is missing required sections: "
                f"{', '.join(sorted(missing))}."
            )

        sections = {}
        for section_name in ("Agent", "TaskAgent", "CompileAgent"):
            section_config = agent_config[section_name]
            if not isinstance(section_config, Mapping):
                raise ValueError(f"{section_name} config must be a mapping.")
            self._validate_agent_config_keys(section_name, section_config)
            sections[section_name] = section_config
        return sections

    def _validate_agent_config_keys(
        self, section_name: str, section_config: Mapping[str, Any]
    ) -> None:
        reserved_keys = _AGENT_RESERVED_KEYS & set(section_config)
        if reserved_keys:
            raise ValueError(
                f"{section_name} config contains reserved keys: "
                f"{', '.join(sorted(reserved_keys))}."
            )

    def get_states(self) -> None:
        # store robot states in each env.reset; keep the leading env dimension
        self.init_qpos = self.robot.get_qpos()

        self._agent_arm_slots = self._resolve_agent_arm_slots()
        for side in ("left", "right"):
            self._initialize_agent_arm_slot(side, self._agent_arm_slots.get(side))

        self.open_state = torch.as_tensor(
            getattr(
                self,
                "agent_open_state",
                getattr(self, "gripper_open_state", _ROBOTIQ_ARG2F_140_OPEN_QPOS),
            ),
            dtype=self.init_qpos.dtype,
            device=self.init_qpos.device,
        ).flatten()
        self.close_state = torch.as_tensor(
            getattr(
                self,
                "agent_close_state",
                getattr(self, "gripper_close_state", _ROBOTIQ_ARG2F_140_CLOSE_QPOS),
            ),
            dtype=self.init_qpos.dtype,
            device=self.init_qpos.device,
        ).flatten()
        self.left_arm_current_gripper_state = self._initial_gripper_state("left")
        self.right_arm_current_gripper_state = self._initial_gripper_state("right")

        self.update_obj_info()

    def _resolve_agent_arm_slots(self) -> dict[str, dict[str, str | None] | None]:
        configured_slots = getattr(self, "agent_arm_slots", None)
        if configured_slots is not None:
            return self._normalize_agent_arm_slots(configured_slots)

        if hasattr(self, "single_arm_name") or hasattr(self, "single_eef_name"):
            slot = getattr(self, "agent_single_arm_slot", "right")
            return self._normalize_agent_arm_slots(
                {
                    slot: {
                        "arm": getattr(self, "single_arm_name", "right_arm"),
                        "eef": getattr(self, "single_eef_name", "right_eef"),
                    }
                }
            )

        control_parts = getattr(self.robot, "control_parts", {}) or {}
        if "arm" in control_parts and "hand" in control_parts:
            slot = getattr(self, "agent_single_arm_slot", "left")
            return self._normalize_agent_arm_slots(
                {slot: {"arm": "arm", "eef": "hand"}}
            )

        return self._normalize_agent_arm_slots(
            {
                "left": {"arm": "left_arm", "eef": "left_eef"},
                "right": {"arm": "right_arm", "eef": "right_eef"},
            }
        )

    def _normalize_agent_arm_slots(
        self, slots
    ) -> dict[str, dict[str, str | None] | None]:
        normalized = {"left": None, "right": None}
        for side in normalized:
            slot_cfg = slots.get(side) if isinstance(slots, dict) else None
            if slot_cfg is None:
                continue
            if isinstance(slot_cfg, str):
                normalized[side] = {"arm": slot_cfg, "eef": None}
                continue
            normalized[side] = {
                "arm": slot_cfg.get("arm", slot_cfg.get("arm_control_part")),
                "eef": slot_cfg.get(
                    "eef",
                    slot_cfg.get("hand", slot_cfg.get("eef_control_part")),
                ),
            }
        return normalized

    def _initialize_agent_arm_slot(
        self, side: str, slot_cfg: dict[str, str | None] | None
    ) -> None:
        arm_name = slot_cfg.get("arm") if slot_cfg else None
        eef_name = slot_cfg.get("eef") if slot_cfg else None
        arm_joints = self._get_control_part_joint_ids(arm_name)
        eef_joints = self._get_control_part_joint_ids(eef_name)

        setattr(self, f"{side}_arm_joints", arm_joints)
        setattr(self, f"{side}_eef_joints", eef_joints)

        num_envs = int(getattr(self, "num_envs", 1))

        if arm_name is None or not arm_joints:
            setattr(
                self,
                f"{side}_arm_init_qpos",
                self.init_qpos.new_empty(num_envs, 0),
            )
            setattr(self, f"{side}_arm_init_xpos", None)
            setattr(self, f"{side}_arm_base_pose", None)
            setattr(
                self,
                f"{side}_arm_current_qpos",
                self.init_qpos.new_empty(num_envs, 0),
            )
            setattr(self, f"{side}_arm_current_xpos", None)
            return

        init_qpos = self.init_qpos[:, arm_joints]
        init_xpos = self.robot.compute_fk(init_qpos, name=arm_name, to_matrix=True)
        base_pose = self.robot.get_control_part_base_pose(arm_name, to_matrix=True)

        setattr(self, f"{side}_arm_init_qpos", init_qpos)
        setattr(self, f"{side}_arm_init_xpos", init_xpos)
        setattr(self, f"{side}_arm_base_pose", base_pose)
        setattr(self, f"{side}_arm_current_qpos", init_qpos.clone())
        setattr(self, f"{side}_arm_current_xpos", init_xpos.clone())

    def _get_control_part_joint_ids(self, control_part: str | None) -> list[int]:
        if control_part is None:
            return []
        if control_part not in (getattr(self.robot, "control_parts", {}) or {}):
            return []
        return list(self.robot.get_joint_ids(name=control_part))

    def _initial_gripper_state(self, side: str) -> torch.Tensor:
        if len(getattr(self, f"{side}_eef_joints", []) or []) == 0:
            return self.open_state.new_empty(0)
        num_envs = int(getattr(self, "num_envs", 1))
        if num_envs <= 1:
            return self.open_state
        return self.open_state.unsqueeze(0).repeat(num_envs, 1)

    def update_obj_info(self):
        # store some useful obj information; keep the leading env dimension
        obj_info = getattr(self, "obj_info", {})
        obj_uids = self.sim.get_rigid_object_uid_list()
        for obj_name in obj_uids:
            obj = self.sim.get_rigid_object(obj_name)
            obj_pose = obj.get_local_pose(to_matrix=True)

            if obj_name not in obj_info:
                obj_height = obj_pose[:, 2, 3]  # Extract the height per env
                obj_info[obj_name] = {
                    "pose": obj_pose,  # (n_envs, 4, 4)
                    "height": obj_height,  # (n_envs,)
                }
            else:
                obj_info[obj_name]["pose"] = obj_pose
                obj_info[obj_name]["height"] = obj_pose[:, 2, 3]

        self.obj_info = obj_info

    # -------------------- Common getters / setters --------------------

    def get_obs_for_agent(self):
        obs = self.get_obs()
        camera_uid = str(
            getattr(self, "viewer_camera_uid", DEFAULT_VIEWER_CAMERA_UID)
            or DEFAULT_VIEWER_CAMERA_UID
        )
        sensor_observations = obs.get("sensor", {})
        if (
            not isinstance(sensor_observations, Mapping)
            or camera_uid not in sensor_observations
        ):
            available = (
                sorted(str(uid) for uid in sensor_observations)
                if isinstance(sensor_observations, Mapping)
                else []
            )
            raise KeyError(
                f"Viewer camera {camera_uid!r} is unavailable; available sensor "
                f"UIDs: {available}."
            )
        camera_observation = sensor_observations[camera_uid]
        if (
            not isinstance(camera_observation, Mapping)
            or "color" not in camera_observation
        ):
            raise KeyError(
                f"Viewer camera {camera_uid!r} must provide a 'color' observation."
            )
        # The generated viewer_camera_uid extension selects the same sensor used
        # for the agent's visual prompt, rather than assuming a template UID.
        rgb = camera_observation["color"]
        if rgb.ndim == 4 and rgb.shape[0] > 1:
            rgb = rgb[0]
        else:
            rgb = rgb.squeeze(0)

        # Get validation camera data; use env 0 for the agent observation
        camera_data = self.event_manager.get_functor("validation_cameras")(self, None)
        result = {"rgb": rgb}
        result.update(
            {
                k: (v[0] if v.ndim > 3 and v.shape[0] > 1 else v.squeeze(0))
                for k, v in camera_data.items()
            }
        )
        return result

    def get_current_qpos_agent(self):
        return self.left_arm_current_qpos, self.right_arm_current_qpos

    def set_current_qpos_agent(self, arm_qpos, is_left):
        if is_left:
            self.left_arm_current_qpos = arm_qpos
        else:
            self.right_arm_current_qpos = arm_qpos

    def get_current_xpos_agent(self):
        return self.left_arm_current_xpos, self.right_arm_current_xpos

    def set_current_xpos_agent(self, arm_xpos, is_left):
        if is_left:
            self.left_arm_current_xpos = arm_xpos
        else:
            self.right_arm_current_xpos = arm_xpos

    def get_current_gripper_state_agent(self):
        return self.left_arm_current_gripper_state, self.right_arm_current_gripper_state

    def set_current_gripper_state_agent(self, arm_gripper_state, is_left):
        if is_left:
            self.left_arm_current_gripper_state = arm_gripper_state
        else:
            self.right_arm_current_gripper_state = arm_gripper_state

    # -------------------- IK / FK --------------------
    def get_arm_ik(self, target_xpos, is_left, qpos_seed=None, env_ids=None):
        control_part = self.get_agent_arm_control_part(is_left)
        ret, qpos = self.robot.compute_ik(
            name=control_part,
            pose=target_xpos,
            joint_seed=qpos_seed,
            env_ids=env_ids,
        )
        if isinstance(ret, torch.Tensor):
            success = bool(torch.all(ret).item())
        else:
            success = bool(ret)
        if qpos.ndim >= 2 and qpos.shape[0] == 1:
            qpos = qpos.squeeze(0)
        return success, qpos

    def get_arm_fk(self, qpos, is_left):
        control_part = self.get_agent_arm_control_part(is_left)
        xpos = self.robot.compute_fk(
            name=control_part, qpos=torch.as_tensor(qpos), to_matrix=True
        )
        return xpos

    def get_agent_arm_control_part(self, is_left: bool) -> str:
        return self._get_agent_control_part(is_left=is_left, key="arm")

    def get_agent_eef_control_part(self, is_left: bool) -> str | None:
        return self._get_agent_control_part(is_left=is_left, key="eef", required=False)

    def _get_agent_control_part(
        self, is_left: bool, key: str, required: bool = True
    ) -> str | None:
        if not hasattr(self, "_agent_arm_slots"):
            self._agent_arm_slots = self._resolve_agent_arm_slots()
        side = "left" if is_left else "right"
        slot_cfg = getattr(self, "_agent_arm_slots", {}).get(side)
        control_part = slot_cfg.get(key) if slot_cfg else None
        if control_part is None and required:
            logger.log_error(
                f"{side}_{key} is not configured for agent control.",
                error_type=ValueError,
            )
        return control_part

    # -------------------- get compiled graph for action list --------------------
    def generate_graph_for_actions(self, regenerate=False, **kwargs):
        logger.log_info(
            "Load graph for creating action list for "
            f"{self.compile_agent.task_name}.",
            color="green",
        )

        logger.log_info(
            f"Using precomputed task graph: {self.precomputed_task_graph_path}",
            color="green",
        )
        with timing_scope(
            "action_agent.task_graph.precomputed_read",
            metadata={"regenerate": bool(regenerate)},
        ):
            task_graph = self.precomputed_task_graph_path.read_text(encoding="utf-8")
        if not getattr(self, "_task_graph_logged", False):
            # The graph is immutable for this environment instance, so print
            # the full source once without flooding logs on later episodes.
            logger.log_info(
                "Task graph input (precomputed):\n"
                f"```json\n{task_graph.rstrip()}\n```",
                color="green",
            )
            self._task_graph_logged = True

        logger.log_info("Start graph compilation.", color="blue")
        compile_agent_input = dict(
            env=self,
            regenerate=regenerate,
            task_graph=task_graph,
            **kwargs,
        )
        graph_file_path, compile_kwargs, graph_content = self.compile_agent.generate(
            **compile_agent_input
        )

        return graph_file_path, compile_kwargs, graph_content

    # -------------------- get action list --------------------
    def create_demo_action_list(self, regenerate=False, *args, **kwargs):
        """Compatibility hook that executes the precomputed task graph online.

        The shared demo runner expects this historical method name. Action
        Agent cannot return an offline list because atomic actions depend on
        live simulator state, so execution is delegated to the explicitly
        named internal method and returns an ``ExecutedActionList`` marker.
        """
        return self._execute_precomputed_task_graph(regenerate, *args, **kwargs)

    def _execute_precomputed_task_graph(self, regenerate=False, *args, **kwargs):
        """Compile and execute the configured task graph against the live env."""
        with timing_scope(
            "action_agent.generate_graph_for_actions",
            metadata={"regenerate": bool(regenerate)},
        ):
            graph_file_path, compile_kwargs, _ = self.generate_graph_for_actions(
                regenerate=regenerate
            )
        atomic_action_kwargs = {
            "allow_grasp_annotation": True,
            "force_grasp_reannotate": False,
            "grasp_convex_decomposition_method": "vhacd",
        }
        for key in atomic_action_kwargs:
            if key in kwargs:
                atomic_action_kwargs[key] = kwargs[key]
        compile_kwargs.update(atomic_action_kwargs)
        grasp_runtime_defaults = getattr(self, "agent_grasp_runtime_defaults", None)
        if isinstance(grasp_runtime_defaults, Mapping):
            for key, value in grasp_runtime_defaults.items():
                compile_kwargs.setdefault(str(key), value)
        with timing_scope(
            "action_agent.execute_compiled_graph",
            metadata={
                "allow_grasp_annotation": bool(
                    atomic_action_kwargs["allow_grasp_annotation"]
                ),
                "force_grasp_reannotate": bool(
                    atomic_action_kwargs["force_grasp_reannotate"]
                ),
                "grasp_convex_decomposition_method": atomic_action_kwargs[
                    "grasp_convex_decomposition_method"
                ],
            },
        ):
            action_list = self.compile_agent.act(graph_file_path, **compile_kwargs)
        return action_list


def _split_env_and_agent_kwargs(
    kwargs: dict,
) -> tuple[dict, dict]:
    missing = _REQUIRED_AGENT_KWARGS - set(kwargs)
    if missing:
        raise ValueError(
            "AgenticGenSimEnv requires agent kwargs: " f"{', '.join(sorted(missing))}."
        )

    env_kwargs = {
        key: value for key, value in kwargs.items() if key not in _AGENT_KWARGS
    }
    agent_kwargs = {key: kwargs[key] for key in _REQUIRED_AGENT_KWARGS}
    agent_kwargs["agent_config_path"] = kwargs.get("agent_config_path")
    return env_kwargs, agent_kwargs


AtomicActionsAgentEnv = AgenticGenSimEnv
