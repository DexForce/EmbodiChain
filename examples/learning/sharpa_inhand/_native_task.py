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

"""Native SimulationManager Sharpa calibration task with HORA observations.

This first profile fixes cylinder scale and physical parameters to qualify the
native collection/learning path. It does not claim full upstream DR parity.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from _native_asset import _compile_usd_hand
from _simulation import SharpaSimulation
from sharpa_rl_unilab.algos.supermimic.state import unflatten_state
from sharpa_rl_unilab.tasks.sharpa_inhand.terms.geometry import (
    torch_quat_angular_velocity_from_pair,
)
from sharpa_rl_unilab.training.supermimic_mdp import SharpaSuperMimicMDP
from unilab.base.torch_env import TorchEnvState

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.sensors import ArticulationContactFilterCfg, ContactSensorCfg

__all__: list[str] = []


class _NativeTask:
    """Own native physical state, reset, targets and terminal observations."""

    def __init__(
        self,
        cfg: Any,
        upstream: Path,
        output: Path,
        *,
        num_envs: int,
        seed: int,
        scale: float = 1.0,
        source_task_profile: bool = False,
    ) -> None:
        self.source_task_profile = source_task_profile
        self.num_envs = num_envs
        self.device = torch.device(str(cfg.training.device))
        self.step_dt = 0.05
        self.cfg = cfg
        self.output = output
        assets = upstream / "src/sharpa_rl_unilab/assets"
        source = assets / "robots/sharpa_wave/scene_scale_1.xml"
        compiled = output / "assets"
        if (compiled / "source.json").exists():
            profile = json.loads((compiled / "source.json").read_text())
        else:
            _, profile = _compile_usd_hand(source, compiled)
        profile["object_scale"] = scale
        profile["source_task_profile"] = source_task_profile
        self.physical = SharpaSimulation(
            str(compiled / "hand.urdf"),
            assets / f"caches/sharpa_grasp_linspace_{scale:g}.npy",
            output / "scene",
            num_envs=num_envs,
            device=str(self.device),
            seed=seed,
            reference_profile=profile,
        )
        self.device = self.physical.device
        self.sim = self.physical.sim
        self.robot = self.physical.robot
        self.object = self.physical.object
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.signed_angle_delta = torch.zeros(num_envs, device=self.device)
        names = ["thumb", "index", "middle", "ring", "pinky"]
        tips = [f"right_{n}_DP" for n in names]
        if source_task_profile:
            from embodichain.lab.sim.sensors.contact._geometry_contact_sensor import (
                _GeometryNetForceSensor,
            )

            self.contact = _GeometryNetForceSensor(
                self.sim,
                "hand",
                "cylinder",
                tuple(
                    (tip, f"right_{name}_elastomer")
                    for name, tip in zip(names, tips, strict=True)
                ),
            )
            self.contact_history = None
        else:
            self.contact = self.sim.add_sensor(
                ContactSensorCfg(
                    uid="fingertip_contact",
                    rigid_uid_list=["cylinder"],
                    articulation_cfg_list=[
                        ArticulationContactFilterCfg(
                            articulation_uid="hand", link_name_list=tips
                        )
                    ],
                    filter_need_both_actor=True,
                    max_contacts_per_env=64,
                )
            )
            self.sim.prepare()
            ids = self.contact.get_actor_ids("hand", tips)
            objects = self.contact.get_actor_ids("cylinder")
            self.contact_history = self.contact.create_history(
                "tactile", ids, counterpart_ids=objects
            )
        self.target = torch.zeros(num_envs, 22, device=self.device)
        self.clean_contact = torch.zeros(num_envs, 5, device=self.device)
        self.previous_contact = torch.zeros_like(self.clean_contact)
        self.last_contact = torch.zeros_like(self.clean_contact)
        self.tactile = torch.zeros_like(self.clean_contact)
        self.actor_history = torch.zeros(num_envs, 3, 49, device=self.device)
        self.critic_history = torch.zeros(num_envs, 3, 58, device=self.device)
        self.proprio_history = torch.zeros(num_envs, 30, 49, device=self.device)
        authored = profile["joints"]
        from embodichain_tasks.manipulation.dexterous.sharpa_inhand import _JOINT_NAMES

        ctrl = np.array([authored[n]["ctrl_range"] for n in _JOINT_NAMES])
        self.lower = torch.tensor(
            ctrl[:, 0] * 0.9, device=self.device, dtype=torch.float32
        )
        self.upper = torch.tensor(
            ctrl[:, 1] * 0.9, device=self.device, dtype=torch.float32
        )
        kp = torch.tensor(
            [authored[n]["kp"] for n in _JOINT_NAMES], device=self.device
        ).expand(num_envs, -1)
        kd = torch.tensor(
            [authored[n]["kd"] for n in _JOINT_NAMES], device=self.device
        ).expand(num_envs, -1)
        anchor = torch.tensor([-0.09559, -0.00517, 0.61906], device=self.device).expand(
            num_envs, -1
        )
        self.xi = torch.cat(
            (
                kp,
                kd,
                torch.ones(num_envs, 1, device=self.device),
                torch.full((num_envs, 1), 0.05, device=self.device),
                torch.zeros(num_envs, 3, device=self.device),
                torch.ones(num_envs, 1, device=self.device),
                anchor,
                torch.zeros(num_envs, 2, device=self.device),
            ),
            -1,
        )
        action = SimpleNamespace(
            ctrl_lower=ctrl[:, 0],
            ctrl_upper=ctrl[:, 1],
            _target_lower=self.lower,
            _target_upper=self.upper,
            _action_scale=1 / 24,
        )
        task = SimpleNamespace(_rotation_axis=(0, 0, 1))
        env = SimpleNamespace(
            step_dt=self.step_dt,
            cfg=cfg,
            action_manager=SimpleNamespace(get_term=lambda _: action),
            event_manager=SimpleNamespace(
                get_term_cfg=lambda _: SimpleNamespace(func=task)
            ),
        )
        self.mdp = SharpaSuperMimicMDP(env, self.device)
        self.s = torch.zeros(num_envs, 89, device=self.device)
        self.state: TorchEnvState | None = None
        self.capture = self._capture
        self.identity = dict(
            self.physical.identity,
            host="EmbodiChain SimulationManager",
            upstream_environment_constructed=False,
            tactile="native ContactSensor contact force magnitude; upstream uses elastomer/object geometry-pair net forces",
            stage="HORA teacher",
            parameter_randomization=False,
            scales=[scale],
            limitation="Fixed scale/mass/COM/friction/gravity calibration. Not full upstream task parity.",
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "native_identity.json").write_text(
            json.dumps(self.identity, indent=2) + "\n"
        )
        self.xi[:, 49] = scale
        self.reset(seed=seed)

    def _capture(self) -> dict[str, torch.Tensor]:
        return {"s": self.s.clone(), "xi": self.xi.clone()}

    def _physical_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        body = self.object.body_state.clone()
        q = self.robot.body_data.fetch_state()["qpos"][
            :, self.physical.joint_ids
        ].clone()
        quat = torch.cat((body[:, 6:7], body[:, 3:6]), -1)
        return body[:, :3], quat, q

    def _observe(self, ids: torch.Tensor, *, reset: bool) -> dict[str, torch.Tensor]:
        state = unflatten_state(self.s)
        clean = self.mdp.frame(state, noisy=False)
        noisy = self.mdp.frame(state, noisy=True)
        priv = self.mdp.priv(state, self.xi)
        critic = torch.cat((clean, priv), -1)
        if reset:
            self.actor_history[ids] = noisy[ids, None, :]
            self.critic_history[ids] = critic[ids, None, :]
            self.proprio_history[ids] = noisy[ids, None, :]
        else:
            self.actor_history = torch.cat(
                (self.actor_history[:, 1:], noisy[:, None]), 1
            )
            self.critic_history = torch.cat(
                (self.critic_history[:, 1:], critic[:, None]), 1
            )
            self.proprio_history = torch.cat(
                (self.proprio_history[:, 1:], noisy[:, None]), 1
            )
        return {
            "obs": self.actor_history.flatten(1).clone(),
            "critic": self.critic_history.flatten(1).clone(),
            "priv_info": priv.clone(),
            "proprio_hist": self.proprio_history.clone(),
        }

    @torch.no_grad()
    def reset(
        self, env_indices: torch.Tensor | None = None, *, seed: int | None = None
    ) -> tuple[dict, dict]:
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_indices is None
            else env_indices
        )
        self.physical.reset(ids, seed=seed)
        pos, quat, q = self._physical_state()
        self.target[ids] = q[ids]
        self.step_count[ids] = 0
        self.signed_angle_delta[ids] = 0
        for value in (
            self.clean_contact,
            self.previous_contact,
            self.last_contact,
            self.tactile,
        ):
            value[ids] = 0
        self.contact.reset(ids)
        self.xi[ids, -2] = pos[ids, 2] - 0.02
        self.xi[ids, -1] = pos[ids, 2] + 0.02
        zeros = torch.zeros(self.num_envs, 3, device=self.device)
        assembled = torch.cat(
            (
                pos,
                quat,
                zeros,
                zeros,
                q,
                torch.zeros_like(q),
                self.clean_contact,
                self.tactile,
                self.target,
            ),
            -1,
        )
        self.s[ids] = assembled[ids]
        observation = self._observe(ids, reset=True)
        self.state = TorchEnvState(
            observation,
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            {},
        )
        return observation, {}

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> TorchEnvState:
        previous = self.s.clone()
        self.target.add_(action.clamp(-1, 1), alpha=1 / 24).clamp_(
            self.lower, self.upper
        )
        if self.source_task_profile:
            self.robot.set_state(
                target_qpos=self.target,
                joint_ids=self.physical.joint_ids,
                clamp_target_qpos=False,
            )
        else:
            self.robot.set_qpos(
                self.target, joint_ids=self.physical.joint_ids, target=True
            )
        # Force is cleared after each native World.update. Apply the cylinder's
        # weight before every physics step; zero world gravity compensates hand.
        for _ in range(12):
            self.object.add_force_torque(self.physical.gravity_force)
            self.sim.update(step=1)
        if self.source_task_profile:
            force = self.contact.measure().clamp(0, 4)
        else:
            self.contact.begin_control_step()
            self.contact.update_physics_step(1 / 240)
            force = self.contact_history.force.norm(dim=-1).clamp(0, 4)
        self.clean_contact.copy_(force)
        smooth = 0.5 * (force + self.previous_contact)
        self.previous_contact.copy_(force)
        self.last_contact.copy_(
            torch.where(torch.rand_like(force) < 0.005, self.last_contact, smooth)
        )
        self.tactile.copy_(self.last_contact)
        pos, quat, q = self._physical_state()
        linear = (pos - previous[:, :3]) / self.step_dt
        angular = torch_quat_angular_velocity_from_pair(
            quat, previous[:, 3:7], self.step_dt
        )
        velocity = (q - previous[:, 13:35]) / self.step_dt
        self.s = torch.cat(
            (
                pos,
                quat,
                linear,
                angular,
                q,
                velocity,
                self.clean_contact,
                self.tactile,
                self.target,
            ),
            -1,
        )
        self.signed_angle_delta.copy_(angular[:, 2] * self.step_dt)
        self.step_count += 1
        terminated = (pos[:, 2] < self.xi[:, -2]) | (pos[:, 2] > self.xi[:, -1])
        truncated = (self.step_count >= 400) & ~terminated
        observation = self._observe(
            torch.arange(self.num_envs, device=self.device), reset=False
        )
        reward = self.mdp.reward(unflatten_state(self.s), self.xi)
        self.state = TorchEnvState(observation, reward, terminated, truncated, {})
        return self.state

    def close(self) -> None:
        if self.source_task_profile:
            self.contact.close()
        self.contact_history = self.contact = self.robot = self.object = None
        self.sim = None
        self.physical.close()
        SimulationManager.flush_cleanup_queue()
