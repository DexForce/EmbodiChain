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


"""Native substep control for a configured, single-arena kinematic replay."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

from embodichain.data import get_data_path
from embodichain.lab.gym.utils._replay import _ReplayCursor, _ReplayCommands
from embodichain.lab.gym.utils.control_timing import resolve_control_timing
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.utils._component_composition import _identifier, _mapping
from embodichain.lab.sim.cfg import MeshCfg, SurfaceDeformableObjectCfg
from embodichain.lab.sim.material import VisualMaterialCfg

__all__: list[str] = []


@dataclass(frozen=True)
class _ReplaySurface:
    mesh: str
    mesh_sha256: str
    robot_sha256: str
    object_config: dict
    visual_material: dict

    def prepare(self, cfg):
        from warp import DeterministicMode

        robot = _verified_path(cfg.robot.fpath, self.robot_sha256)
        mesh = _verified_path(self.mesh, self.mesh_sha256)
        cfg.robot.fpath = str(robot)
        physics = cfg.sim_cfg.physics_cfg
        vbd = physics.solver_cfg["vbd_options"]
        deterministic = vbd["deterministic"]
        if isinstance(deterministic, str):
            vbd["deterministic"] = DeterministicMode[deterministic]
        with np.load(mesh, allow_pickle=False) as data:
            return SurfaceDeformableObjectCfg.from_dict(
                {
                    **self.object_config,
                    "shape": MeshCfg(
                        vertices=data["vertices"],
                        triangles=data["triangles"],
                        visual_material=VisualMaterialCfg(
                            uid=self.object_config["uid"], **self.visual_material
                        ),
                    ),
                }
            )


def _load_replay_surface(value):
    config = _mapping(
        value,
        path="replay_surface",
        required=frozenset(
            {"mesh", "mesh_sha256", "robot_sha256", "object", "visual_material"}
        ),
    )
    digests = [config[key] for key in ("mesh_sha256", "robot_sha256")]
    if any(
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in digests
    ):
        raise ValueError("Replay surface requires lowercase SHA-256 asset digests.")
    return _ReplaySurface(
        _identifier(config["mesh"], path="replay_surface.mesh"),
        *digests,
        dict(config["object"]),
        dict(config["visual_material"]),
    )


def _verified_path(reference, digest):
    path = Path(get_data_path(reference))
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Replay asset digest does not match: {path}")
    return path


class _PhysicalReplay:
    def __init__(self, cfg, source):
        physics = cfg.sim_cfg.physics_cfg
        timing = resolve_control_timing(
            cfg.sim_cfg.physics_dt,
            cfg.sim_steps_per_control,
            target_control_frequency=cfg.target_control_frequency,
        )
        if (
            cfg.expert_trajectory.joint_command_mode != "position"
            or cfg.expert_trajectory.use_action_manager
        ):
            raise ValueError(
                "Native position replay requires position-only expert targets."
            )
        if (
            cfg.num_envs != 1
            or timing.sim_steps_per_control != 1
            or timing.step_dt != 1 / source.fps
            or physics.use_cuda_graph
            or cfg.replay_surface is None
            or cfg.rl_task is not None
        ):
            raise ValueError(
                "Native replay requires one arena, its resolved recorded rate, "
                "no CUDA Graph, and a verified physical replay surface."
            )
        self.snapshot = source.load()
        self.cursor = _ReplayCursor(torch.tensor([self.snapshot.steps]))
        self.surface_cfg = cfg.replay_surface.prepare(cfg)
        self.verified_mesh_sha256 = cfg.replay_surface.mesh_sha256
        self._node_shape = None
        self.hold = False

    def prepare_scene(self, env):
        self.surface = env.sim.add_deformable_object(self.surface_cfg)
        # Native joint metadata is only available after prepare. The native
        # registration API accepts the source order explicitly at this stage.
        env.sim.register_kinematic_joint_trajectory(
            env.cfg.robot.uid,
            self.snapshot.positions[None].numpy(),
            fps=self.snapshot.source.fps,
            joint_names=self.snapshot.joint_names,
            initialize_pose=True,
        )

    def require_surface(self, simulation, *, uid, mesh_sha256):
        """Qualify the native facade created from the verified mesh artifact."""
        if (
            uid != self.surface_cfg.uid
            or mesh_sha256 != self.verified_mesh_sha256
            or simulation.get_deformable_object(uid) is not self.surface
        ):
            raise ValueError(
                "Deformable evidence does not belong to the verified replay mesh."
            )
        nodes = self.surface.data.nodal_pos_w
        if (
            nodes.ndim != 3
            or nodes.shape[0] != 1
            or nodes.shape[2] != 3
            or not nodes.is_floating_point()
        ):
            raise ValueError("Replay mesh nodes require one arena and XYZ positions.")
        if self._node_shape is None:
            self._node_shape = tuple(nodes.shape)
            self._node_dtype = nodes.dtype
        elif tuple(nodes.shape) != self._node_shape or nodes.dtype != self._node_dtype:
            raise ValueError("Replay mesh topology changed after binding.")
        return self.surface, nodes

    def bind(self, env):
        self.snapshot = self.snapshot.bind(env.robot)
        self.commands = self.snapshot.commands()
        env.add_detached_uids_for_reset(
            [env.robot.uid, *(cfg.uid for cfg in env.cfg.background)]
        )

    def reset(self, env, env_ids):
        if len(env_ids):
            from dexsim.engine.newton_physics.backend_registry import get_newton_backend

            initial = self.snapshot.positions[None, 0]
            env.robot.set_qpos(initial, target=False, env_ids=env_ids)
            env.robot.set_qpos(initial, target=True, env_ids=env_ids)
            env.robot.clear_dynamics(env_ids=env_ids)
            get_newton_backend(env.sim.spawn_result.world).set_sim_time(0.0, 0)
            self.cursor.reset()
            self.hold = False

    def consume(self, env, action):
        command = action.value if isinstance(action, ControllerAction) else action
        # Validate the command kind even for safe holds; native position replay
        # cannot execute independently supplied velocity or effort targets.
        _ReplayCommands.position_target(command)
        hold = isinstance(action, ControllerAction) and action.metadata.get(
            "bridge_action_kind"
        ) in ("runtime_safe_hold", "runtime_abort_safe_hold")
        if not hold:
            self.commands.require_position(self.frame(env) + 1, command)
        self.hold = hold
        return command

    def frame(self, env):
        frame = round(
            env.sim.spawn_result.get_simulation_time() * self.snapshot.source.fps
        )
        # Keep the display/progress cursor exhausted, while reporting the actual
        # clock so final queries revoke completion after an external world step.
        self.cursor.seek(max(0, min(frame, self.snapshot.steps)))
        return frame
