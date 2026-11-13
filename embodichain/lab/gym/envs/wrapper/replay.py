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

import math
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs.expert_trajectory import (
    EXPERT_TRAJECTORY_SCHEMA_VERSION,
    build_expert_action_spec,
)
from embodichain.lab.gym.utils.trajectory_state import FluidTrajectoryReader
from embodichain.lab.gym.envs.task_program._final_acceptance import (
    _replay_final_acceptance,
)
from embodichain.lab.gym.utils._replay import _ReplayCommands, _ReplayCursor
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.utils.gym_utils import load_trajectory
from embodichain.lab.gym.utils.trajectory_state import (
    capture_trajectory_state,
    restore_trajectory_state,
)
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.types import EnvObs

__all__ = ["ReplayWrapper"]


class ReplayWrapper(gym.Wrapper):
    """Replay a recorded environment trajectory.

    In ``kinematic`` mode physics is disabled and recorded robot, rigid and
    available fluid-particle states are written directly each step, producing
    observations only (no reward / success / action). In ``dynamic`` mode the
    recorded robot actions are fed back through :meth:`env.step` so physics
    re-simulates the scene; a recorded fluid starts at its source state and
    then evolves through the solver without per-frame state replacement;
    the full ``obs/reward/terminated/truncated/info`` tuple is returned. The
    ``control`` mode uses the same kinematic behavior while exposing
    :meth:`go_to_step` for interactive scrubbing.

    Dynamic replay treats missing ``meta.action_kind`` (or ``"policy"``) as raw
    policy actions. Current expert recordings use ``action_kind="expert"``;
    ``"expert_controller"`` remains accepted for stacked-branch compatibility.
    Expert recordings carry the canonical schema so qpos/qvel targets can be
    reconstructed before stepping. Schema metadata alone cannot identify
    controller actions: older position-only raw policy recordings carry it too.

    Physical-objective dynamic replay currently supports one environment and
    full resets only. Stepping past exhaustion raises instead of extending
    objective dwell time by repeating the final command.

    Cable recordings support dynamic replay from matching full-reset states
    only. Their visible state omits contact/solver history, so state scrubbing,
    partial resets and broadcasting world-space cable recordings are rejected.
    Source arena origins, when recorded, translate the captured reset positions
    into the saved frame for validation. Older files require matching placement.

    For a deployment with final acceptance rules, ``final_acceptance`` reports
    the offline evaluation of the saved terminal capture, independently of
    physics replay and Gym termination flags.

    Args:
        env: The environment to wrap (constructed without ``record_trajectory``).
        trajectory: A ``.pt`` path or loaded dict from
            :meth:`EmbodiedEnv.save_trajectory`.
        mode: ``"kinematic"``, ``"dynamic"``, or ``"control"``.
    """

    def __init__(
        self,
        env: gym.Env,
        trajectory: str | dict,
        mode: str = "dynamic",
        *,
        fluid_reader: FluidTrajectoryReader | None = None,
    ):
        super().__init__(env)
        if mode not in ("kinematic", "dynamic", "control"):
            raise ValueError(
                f"Invalid replay mode {mode!r}; use 'kinematic', 'dynamic', or "
                "'control'."
            )
        self._mode = mode
        self._trajectory = load_trajectory(trajectory)
        meta = self._trajectory["meta"]
        self._cable_replay = "cables" in self._trajectory["states"].keys()
        self._cable_source_origins: torch.Tensor | None = None
        if (
            getattr(getattr(env, "sim", None), "_cable_objects", {})
            and not self._cable_replay
        ):
            raise ValueError(
                "Cable replay requires recorded initial states for every scene cable; include them in trajectory_uids."
            )
        if self._cable_replay:
            if mode != "dynamic":
                raise NotImplementedError(
                    "Cable recordings require dynamic replay from reset."
                )
            if int(meta["num_envs"]) != env.num_envs:
                raise ValueError(
                    "World-space cable recordings require matching environment counts."
                )
            if "arena_origins_w" in meta:
                origins = torch.as_tensor(
                    meta["arena_origins_w"], dtype=torch.float32, device=env.device
                )
                if origins.shape != (env.num_envs, 3) or not bool(
                    torch.isfinite(origins).all()
                ):
                    raise ValueError("arena_origins_w must be finite (num_envs, 3).")
                self._cable_source_origins = origins
        self._fluid_reader = fluid_reader
        if getattr(getattr(env, "cfg", None), "fluid_object", ()):
            if self._fluid_reader is None:
                if not isinstance(trajectory, str):
                    raise ValueError("Fluid replay requires a trajectory file path.")
                self._fluid_reader = FluidTrajectoryReader(
                    trajectory, self._trajectory, env.cfg
                )
        elif fluid_reader is not None or "fluid_state" in meta:
            raise ValueError("A fluid trajectory requires configured fluid objects.")
        self._controller_spec = None
        self._task_outcome_replay = mode == "dynamic" and (
            getattr(env, "termination_manager", None) is not None
        )
        self._objective_replay = mode == "dynamic" and (
            getattr(env, "physical_objective", None) is not None
            or self._task_outcome_replay
        )
        self._task_initial_state = None
        if self._task_outcome_replay:
            episodes = meta.get("demo_episodes", [])
            if len(episodes) != 1 or not episodes[0].get("task_initial_state"):
                raise ValueError(
                    "Dynamic task replay requires task_initial_state metadata."
                )
            self._task_initial_state = episodes[0]["task_initial_state"]
        if self._objective_replay and env.num_envs != 1:
            raise ValueError("Physical-objective replay requires a single environment.")
        if mode == "dynamic":
            if any(int(length) <= 0 for length in meta["lengths"]):
                raise ValueError("Dynamic replay requires positive trajectory lengths.")
            self._validate_action_schema(meta)

        # Sanity-check that the trajectory matches the replay env's robot.
        traj_robot_dof = int(meta.get("robot_dof", self.env.robot.dof))
        traj_active_joint_ids = list(meta.get("active_joint_ids", []))
        env_robot_dof = int(self.env.robot.dof)
        env_active_joint_ids = list(self.env.active_joint_ids)
        if (
            traj_robot_dof != env_robot_dof
            or traj_active_joint_ids != env_active_joint_ids
        ):
            raise ValueError(
                f"Trajectory was recorded with robot_dof={traj_robot_dof} / "
                f"active_joint_ids={traj_active_joint_ids} but replay env has "
                f"robot_dof={env_robot_dof} / active_joint_ids={env_active_joint_ids}."
            )

        self._final_acceptance = _replay_final_acceptance(
            getattr(env, "_final_acceptance_rules", ()), self._trajectory
        )
        self._expand_to_env_count()

        # Per-env lengths support async vector trajectories.
        lengths = self._trajectory["meta"]["lengths"]
        self._lengths = torch.tensor(lengths, dtype=torch.long, device=self.env.device)
        # CPU indices also work when stored states and the replay use different GPUs.
        self._recorded_lengths = self._lengths.cpu()

        # Clamp replay length to the wrapped env's horizon.
        max_steps = int(self.env.max_episode_steps)
        if bool((self._lengths > max_steps).any()):
            logger.log_warning(
                f"Trajectory lengths exceed env max_episode_steps={max_steps}; clamping."
            )
            self._lengths = self._lengths.clamp(max=max_steps)
        self._replay_cursor = _ReplayCursor(self._lengths)
        self._replay_steps = self._replay_cursor.steps
        self._commands = _ReplayCommands.from_recording(self._trajectory)

    def _validate_action_schema(self, meta: dict) -> None:
        """Validate explicit controller recordings before any simulation step."""
        kind = meta.get("action_kind", "policy")
        if kind in {"policy", "raw_policy"}:
            return
        if kind not in {"expert", "expert_controller"}:
            raise ValueError(f"Unsupported trajectory action_kind: {kind!r}.")
        version = meta.get("expert_trajectory_schema_version")
        if type(version) is not int or version != EXPERT_TRAJECTORY_SCHEMA_VERSION:
            raise ValueError("Unsupported expert_trajectory_schema_version.")
        spec = build_expert_action_spec(
            joint_names=[
                self.env.robot.joint_names[joint_id]
                for joint_id in self.env.active_joint_ids
            ],
            joint_command_mode=meta.get("joint_command_mode"),
        )
        expected = spec.metadata(step_dt=self.env.step_dt)
        for key in ("joint_names", "qpos_slice", "qvel_slice"):
            value = meta.get(key)
            if key not in meta or value != expected[key]:
                raise ValueError(f"Trajectory {key} does not match the replay layout.")
            if key.endswith("_slice") and value is not None:
                if any(type(index) is not int for index in value):
                    raise ValueError(f"Trajectory {key} requires integer indices.")
        step_dt = meta.get("step_dt")
        if (
            isinstance(step_dt, bool)
            or not isinstance(step_dt, (int, float))
            or not math.isfinite(step_dt)
            or step_dt <= 0
            or not math.isclose(step_dt, self.env.step_dt, rel_tol=1e-9, abs_tol=0.0)
        ):
            raise ValueError("Trajectory step_dt does not match the replay cadence.")
        actions = self._trajectory["actions"]
        if actions.ndim != 3 or actions.shape[-1] != spec.width:
            raise ValueError(
                "Trajectory controller action width does not match schema."
            )
        if not actions.is_floating_point() or not bool(torch.isfinite(actions).all()):
            raise ValueError("Trajectory controller actions must be finite floats.")
        self._controller_spec = spec

    def _expand_to_env_count(self) -> None:
        """Broadcast a single-env trajectory to the wrapped env's env count."""
        meta = self._trajectory["meta"]
        traj_envs = int(meta["num_envs"])
        env_envs = int(self.env.num_envs)
        if traj_envs == env_envs:
            return
        if traj_envs != 1:
            raise ValueError(
                f"Trajectory has {traj_envs} envs but wrapped env has {env_envs}; "
                "only single-env trajectories can be broadcast."
            )
        self._trajectory = dict(self._trajectory)
        for key in ("states", "actions", "terminal_states"):
            if key not in self._trajectory:
                continue
            t = self._trajectory[key]
            self._trajectory[key] = t.expand(env_envs, *t.shape[1:]).clone()
        self._trajectory["meta"] = dict(
            meta, num_envs=env_envs, lengths=list(meta["lengths"]) * env_envs
        )
        if self._final_acceptance is not None:
            self._final_acceptance *= env_envs

    @property
    def final_acceptance(self) -> tuple[dict[str, Any], ...] | None:
        """Return owned, per-row final acceptance reports from the trajectory.

        Reports are available before stepping. A ``recomputed`` report contains
        the recomputed predicates, acceptance, and the separately recorded
        acceptance. ``not_recorded`` has no recomputed acceptance. ``not_run``
        represents an episode sealed before the physical gate ran. Deployments
        without final acceptance rules return None.
        """
        if self._final_acceptance is None:
            return None
        return tuple(deepcopy(row) for row in self._final_acceptance)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[EnvObs, dict]:
        if (
            (self._objective_replay or self._cable_replay)
            and options is not None
            and "reset_ids" in options
        ):
            raise ValueError(
                "Physical-objective/cable replay does not support reset_ids."
            )
        obs, info = self.env.reset(seed=seed, options=options)
        # Disable physics during restore so set_local_pose's internal update
        # does not integrate dynamics.
        if self._cable_replay:
            self._validate_cable_reset_state()
        else:
            self.env.sim.enable_physics(False)
            restore_trajectory_state(self.env, self._trajectory["states"][:, 0])
            if self._fluid_reader is not None:
                self._fluid_reader.apply(
                    self.env, torch.zeros(self.env.num_envs, dtype=torch.long)
                )
        if self._mode == "dynamic":
            self.env.sim.enable_physics(True)
            self.env._replay_no_auto_reset = True
        if self._task_outcome_replay:
            self.env.restore_task_initial_state(self._task_initial_state)
        self._replay_cursor.reset()
        recording = getattr(
            getattr(self.env, "observation_manager", None), "has_history", False
        )
        frame = self.env._capture_observation() if recording else None
        obs = frame.policy if frame is not None else self.env.get_obs()
        if self._task_outcome_replay:
            info = self.env.get_info()

        # If the wrapped environment also records this replay, replace the
        # default-reset pending state with the state restored from the file.
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        seed_recording_state = getattr(self.env, "_seed_recording_state", None)
        if seed_recording_state is not None:
            seed_recording_state(obs, env_ids)
        if frame is not None:
            self.env._record_initial_observation(frame, env_ids)
        return obs, info

    def _validate_cable_reset_state(self) -> None:
        """Validate initial observations without approximating a checkpoint."""
        initial = self._trajectory["states"][:, 0]
        for category, assets in (
            ("cables", getattr(self.env.sim, "_cable_objects", {})),
            ("articulations", self.env.sim._articulations),
            ("rigid_objects", self.env.sim._rigid_objects),
        ):
            recorded = (
                set(initial[category].keys()) if category in initial.keys() else set()
            )
            if not recorded.issubset(assets) or (
                category == "cables" and recorded != set(assets)
            ):
                raise ValueError(
                    f"Cable replay scene does not match recorded {category}."
                )
        current = initial.clone()
        capture_trajectory_state(
            self.env, current, torch.arange(self.env.num_envs, device=self.env.device)
        )
        if self._cable_source_origins is not None:
            translation = self._cable_source_origins - self.env.sim.arena_offsets.to(
                self._cable_source_origins
            )
            # Compare the captured reset in the saved world's frame. The live
            # scene and the recording keep their original poses and history.
            for cable in current["cables"].values():
                state = cable["segment_state_w"]
                state[..., :3] += translation.to(state)[:, None, :]
        for key, expected in initial.items(include_nested=True, leaves_only=True):
            observed = current[key].to(expected)
            if observed.shape != expected.shape or not torch.allclose(
                observed, expected, rtol=1e-5, atol=1e-6
            ):
                raise ValueError(
                    f"Cable replay initial state {key!r} differs from the full reset; "
                    "a recorded observation cannot restore solver history."
                )

    def step(
        self, action: Any
    ) -> tuple[EnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        env = self.env
        if self._objective_replay and bool(self._replay_cursor.exhausted.any()):
            raise RuntimeError("Physical-objective replay is exhausted; reset first.")
        n = env.num_envs
        idx = torch.arange(n, device=env.device)
        st = self._replay_cursor.indices()  # finished envs hold last

        if self._mode in ("kinematic", "control"):
            restore_trajectory_state(self.env, self._trajectory["states"][idx, st])
            if self._fluid_reader is not None:
                self._fluid_reader.apply(env, st)
                # Newton's disabled-physics update can still enter the DexUni
                # solver. Publish the authored state without a physical step.
                with env.sim.render_frame(force_visualization=True):
                    pass
            else:
                env.sim.update(env.sim_cfg.physics_dt, env.cfg.sim_steps_per_control)
            obs = env.get_obs()
            self._replay_cursor.advance()
            trunc = self._replay_steps >= self._lengths
            return (
                obs,
                torch.zeros(n, device=env.device),
                torch.zeros(n, dtype=torch.bool, device=env.device),
                trunc,
                {},
            )

        # Raw policy recordings retain their ordinary action-manager preprocessing.
        action_t = self._commands.sample(st)
        if self._controller_spec is not None:
            spec = self._controller_spec
            commands = {"qpos": action_t[:, slice(*spec.qpos_slice)]}
            if spec.qvel_slice is not None:
                commands["qvel"] = action_t[:, slice(*spec.qvel_slice)]
            action_t = ControllerAction(TensorDict(commands, batch_size=[n]))
        obs, reward, term, trunc, info = env.step(action_t)
        self._replay_cursor.advance()
        trunc = trunc | (self._replay_steps >= self._lengths)
        return obs, reward, term, trunc, info

    def go_to_step(self, step: int) -> EnvObs:
        """Scrub to a specific recorded state (kinematic).

        State index ``t`` is the state immediately before recorded action ``t``.

        Args:
            step: Target step index.

        Returns:
            The observation at the target step.
        """
        if self._cable_replay:
            raise NotImplementedError(
                "Cable recordings do not support state scrubbing."
            )
        env = self.env
        max_step = self.control_max_step
        step = max(0, min(int(step), max_step))
        env.sim.enable_physics(False)
        terminal = self._recorded_lengths == step
        if bool(terminal.all()):
            state = self._trajectory["terminal_states"]
        else:
            state = self._trajectory["states"][:, step]
            if bool(terminal.any()):
                state = state.clone()
                state[terminal] = self._trajectory["terminal_states"][terminal]
        restore_trajectory_state(env, state)
        if self._fluid_reader is not None:
            self._fluid_reader.apply(
                env, torch.full((env.num_envs,), step, dtype=torch.long)
            )
            with env.sim.render_frame(force_visualization=True):
                pass
        else:
            env.sim.update(env.sim_cfg.physics_dt, env.cfg.sim_steps_per_control)
        self._replay_cursor.seek(step)
        return env.get_obs()

    @property
    def control_max_step(self) -> int:
        """Largest state index available to interactive control replay."""
        transition_count = int(self._lengths.min().item())
        if (
            self._fluid_reader is not None
            and "terminal_states" in self._trajectory
            and bool((self._lengths == transition_count).all())
        ):
            return transition_count
        return transition_count - 1

    def close(self) -> None:
        try:
            if self._fluid_reader is not None:
                self._fluid_reader.close()
            self.env.sim.enable_physics(True)
            self.env._replay_no_auto_reset = False
        finally:
            self.env.close()
