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

"""Bounded recording of submitted commands and measured post-step states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ._robot import RobotProfile

__all__: list[str] = []


class _SequenceTrace:
    """Write one causal T-command/T+1-state trace using at most 256 frames."""

    def __init__(self, path: Path, env: Any, profile: RobotProfile, dt: float) -> None:
        import h5py

        self.env = env
        self.dt = dt
        self.order = [env.robot.joint_names.index(n) for n in profile.joint_names]
        self.objects = {
            uid: env.sim.get_rigid_object(uid)
            for uid in env.sim.get_rigid_object_uid_list()
        }
        self.file = h5py.File(path, "x")
        self.file.attrs.update(
            schema="native-skill-sequence-trace/v1",
            completed=False,
            control_dt=dt,
            state_joint_names=json.dumps(profile.joint_names),
            command_joint_names=json.dumps(env.expert_action_spec.joint_names),
            alignment="state[t] precedes command[t]; state[t+1] follows physics",
        )
        self.commands: list[np.ndarray] = []
        self.states: list[dict[str, np.ndarray]] = []
        self.count = 0
        try:
            state = self._observe()
            for name, value in state.items():
                ds = self.file.create_dataset(
                    name,
                    shape=(1, *value.shape),
                    maxshape=(None, *value.shape),
                    chunks=(256, *value.shape),
                    dtype=value.dtype,
                )
                ds[0] = value
            width = len(env.expert_action_spec.joint_names)
            self.file.create_dataset(
                "commands/qpos",
                shape=(0, width),
                maxshape=(None, width),
                chunks=(256, width),
                dtype="float32",
            )
        except BaseException:
            self.file.close()
            raise

    def _observe(self) -> dict[str, np.ndarray]:
        def capture(value):
            return value.detach().cpu().numpy().copy()

        robot = self.env.robot
        result = {
            "states/qpos": capture(robot.get_qpos()[0, self.order]),
            "states/qvel": capture(robot.get_qvel()[0, self.order]),
            "states/root_pose": capture(robot.get_local_pose()[0]),
        }
        for uid, obj in self.objects.items():
            result[f"states/objects/{uid}/pose"] = capture(obj.get_local_pose()[0])
        return result

    def append(self, command: Any) -> None:
        state = self._observe()
        self.commands.append(command.detach().cpu().numpy().copy())
        self.states.append(state)
        if len(self.commands) == 256:
            self._flush()

    def _flush(self) -> None:
        if not self.commands:
            return
        n = len(self.commands)
        ds = self.file["commands/qpos"]
        ds.resize(self.count + n, axis=0)
        ds[self.count :] = np.stack(self.commands)
        for name in self.states[0]:
            ds = self.file[name]
            ds.resize(self.count + n + 1, axis=0)
            ds[self.count + 1 :] = np.stack([state[name] for state in self.states])
        self.count += n
        self.commands.clear()
        self.states.clear()

    def close(self, *, completed: bool) -> None:
        try:
            self._flush()
            self.file.attrs.update(completed=completed, command_count=self.count)
            self.file.flush()
        finally:
            self.file.close()
