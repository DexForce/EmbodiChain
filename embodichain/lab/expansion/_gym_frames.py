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

"""Own bounded measured frame and media buffers for one native cohort."""

from __future__ import annotations
import time
import torch
from embodichain.data_pipeline.recording.media import MediaSpool

from embodichain.lab.sim.scene import capture_state

from ._gym_context import GymHostContext

from .visual import _calibrated_frames, _calibration_batch

__all__: list[str] = []


class GymFrameCollector:
    """Post-step observer; buffers only active rows and never advances the World."""

    def __init__(self, context: GymHostContext, assignments: dict) -> None:
        self.context = context
        ctx = context
        cfg, env, initial, initial_obs = ctx.cfg, ctx.env, ctx.initial, ctx.initial_obs
        calibrations = _calibration_batch(env) if cfg.capture_media else ()
        records, spools = {}, {}
        for row, (
            identity,
            episode_id,
            commit_id,
            directory,
            record,
        ) in assignments.items():
            scene_fields = {
                (entity.uid, name): tuple(getattr(entity, name))
                for entity in initial.batches[row]
                for name in (
                    "pose",
                    "velocity",
                    "qpos",
                    "qvel",
                    "target_qpos",
                    "target_qvel",
                    "qf",
                )
                if getattr(entity, name)
            }
            records[row] = {
                "scene": {
                    key: [torch.tensor(value, dtype=torch.float32)]
                    for key, value in scene_fields.items()
                },
                "qpos": [env.robot.get_qpos()[row].detach().cpu().clone()],
                "qvel": [env.robot.get_qvel()[row].detach().cpu().clone()],
                "qf": [env.robot.get_qf()[row].detach().cpu().clone()],
                "actions": [],
                "objects": {
                    uid: [
                        env.sim.get_rigid_object(uid)
                        .body_state[row]
                        .detach()
                        .cpu()
                        .clone()
                    ]
                    for uid in env.sim.get_rigid_object_uid_list()
                },
            }
            if cfg.capture_media:
                frames = _calibrated_frames(
                    env, initial_obs, row, calibration=calibrations[row]
                )
                if not frames:
                    raise ValueError(
                        "capture_media requires configured RGB/depth sensor observations"
                    )
                spool = MediaSpool(
                    directory / "media",
                    max_buffer_bytes=max(1, cfg.media_buffer_bytes // cfg.batch_size),
                    parent_episode_id=episode_id,
                    calibration=calibrations[row],
                )
                spool.append(0.0, frames)
                spools[row] = spool
        self.records, self.spools = records, spools

    def observe(self, ids, action, observation) -> None:
        ctx = self.context
        cfg, env, view, collision, timings = (
            ctx.cfg,
            ctx.env,
            ctx.view,
            ctx.collision,
            ctx.timings,
        )
        records, spools = self.records, self.spools
        ctx.journal.checkpoint()
        stamp = time.monotonic()
        frame_state = capture_state(view)
        robot_states = [
            next(entity for entity in batch if entity.uid == env.robot.uid)
            for batch in frame_state.batches
        ]
        measured_qpos = torch.tensor(
            [state.qpos for state in robot_states], dtype=torch.float32
        )
        measured_qvel = torch.tensor(
            [state.qvel for state in robot_states], dtype=torch.float32
        )
        measured_qf = torch.tensor(
            [state.qf for state in robot_states], dtype=torch.float32
        )
        measured_objects = {
            uid: torch.tensor(
                [
                    (
                        *next(e for e in batch if e.uid == uid).pose,
                        *next(e for e in batch if e.uid == uid).velocity,
                    )
                    for batch in frame_state.batches
                ],
                dtype=torch.float32,
            )
            for uid in env.sim.get_rigid_object_uid_list()
        }
        calibrations = _calibration_batch(env) if cfg.capture_media else ()
        timings["state_capture"] = (
            timings.get("state_capture", 0.0) + time.monotonic() - stamp
        )
        stamp = time.monotonic()
        if collision is not None:
            collision.sample(ids, action, frame_state)
        timings["sampled_validation"] = (
            timings.get("sampled_validation", 0.0) + time.monotonic() - stamp
        )
        for row in ids:
            record = records[row]
            value = (
                {"qpos": action.value}
                if isinstance(action.value, torch.Tensor)
                else action.value
            )

            def active_command(name):
                data = value[name][row]
                if data.shape[-1] == env.robot.dof:
                    data = data[list(env.active_joint_ids)]
                elif data.shape[-1] != len(env.active_joint_ids):
                    raise ValueError("submitted command joint layout differs")
                return data.detach().cpu()

            command = active_command("qpos")
            if env.expert_action_spec.joint_command_mode == "position_velocity":
                command = torch.cat(
                    (
                        command,
                        active_command("qvel"),
                    )
                )
            entity_states = {entity.uid: entity for entity in frame_state.batches[row]}
            for (uid, field), values in record["scene"].items():
                values.append(
                    torch.tensor(
                        getattr(entity_states[uid], field),
                        dtype=torch.float32,
                    )
                )
            record["actions"].append(command.clone())
            record["qpos"].append(measured_qpos[row].clone())
            record["qvel"].append(measured_qvel[row].clone())
            record["qf"].append(measured_qf[row].clone())
            for uid, values in record["objects"].items():
                values.append(measured_objects[uid][row].clone())
            if row in spools:
                stamp = time.monotonic()
                spools[row].append(
                    len(record["actions"]) * env.step_dt,
                    _calibrated_frames(
                        env, observation, row, calibration=calibrations[row]
                    ),
                )
                timings["media_buffering_encoding"] = (
                    timings.get("media_buffering_encoding", 0.0)
                    + time.monotonic()
                    - stamp
                )
            size = sum(
                x.numel() * x.element_size()
                for key in ("actions", "qpos", "qvel", "qf")
                for x in record[key][-1:]
            )
            size += sum(
                x.numel() * x.element_size()
                for values in record["objects"].values()
                for x in values[-1:]
            )
            size += sum(
                values[-1].numel() * values[-1].element_size()
                for values in record["scene"].values()
            )
            record["bytes"] = record.get("bytes", 0) + size
            if record["bytes"] > cfg.episode_byte_budget:
                raise BufferError("numeric episode buffer budget exceeded")
