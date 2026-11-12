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

"""Control-aligned fluid-state artifact and replay contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.lab.gym.utils.trajectory_state import (
    FluidTrajectoryReader,
    FluidTrajectoryRecorder,
)


class _Fluid:
    def __init__(self, states: torch.Tensor) -> None:
        self.data = SimpleNamespace(particle_state_w=states)
        self.num_particles = states.shape[1]
        self.written = None

    def write_particle_state_to_sim(self, state: torch.Tensor) -> None:
        self.written = state.clone()


def _env(num_envs: int) -> SimpleNamespace:
    states = torch.zeros(num_envs, 2, 6)
    fluid = _Fluid(states)
    offsets = torch.arange(num_envs, dtype=torch.float32)[:, None] * torch.tensor(
        [[10.0, 0.0, 0.0]]
    )
    return SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        cfg=SimpleNamespace(
            fluid_object=[
                SimpleNamespace(
                    uid="water",
                    particle_spacing=0.01,
                    spawn=SimpleNamespace(sample=lambda spacing: np.zeros((2, 3))),
                )
            ]
        ),
        sim=SimpleNamespace(
            arena_offsets=offsets,
            get_fluid_object=lambda uid: fluid,
        ),
        fluid=fluid,
    )


def _source(num_envs: int, length: int) -> dict:
    return {
        "meta": {
            "artifact_purpose": "recording",
            "num_envs": num_envs,
            "lengths": [length] * num_envs,
            "dt": 0.025,
            "demo_episodes": [
                {"program_hash": "program", "completed": True, "success": True}
                for _ in range(num_envs)
            ],
            "fluid_qualification": {
                "kind": "task_program",
                "accepted": True,
                "program_hash": "program",
            },
        }
    }


def _record(tmp_path: Path, env: SimpleNamespace) -> tuple[Path, dict]:
    path = tmp_path / "trajectory.pt"
    recorder = FluidTrajectoryRecorder(tmp_path, env.num_envs, 0.025)
    env.fluid.data.particle_state_w[:, :, :3] = env.sim.arena_offsets[:, None, :]
    recorder.capture(env, list(range(env.num_envs)), reset=True)
    env.fluid.data.particle_state_w[:, :, 0] += 0.2
    env.fluid.data.particle_state_w[:, :, 3] = 0.5
    recorder.capture(env, list(range(env.num_envs)))
    source = _source(env.num_envs, 1)
    source["meta"]["fluid_state"] = recorder.finish(path, [1] * env.num_envs)
    recorder.commit()
    return path, source


def test_particle_frames_restore_position_velocity_and_arena_offsets(tmp_path: Path):
    source_env = _env(1)
    path, source = _record(tmp_path, source_env)
    target = _env(2)
    cfg = SimpleNamespace(num_envs=2, fluid_object=target.cfg.fluid_object)
    reader = FluidTrajectoryReader(path, source, cfg)
    try:
        reader.apply(target, torch.tensor([0, 1]))
        assert target.fluid.written is not None
        torch.testing.assert_close(target.fluid.written[0, :, 0], torch.zeros(2))
        torch.testing.assert_close(
            target.fluid.written[1, :, 0], torch.full((2,), 10.2)
        )
        torch.testing.assert_close(target.fluid.written[1, :, 3], torch.full((2,), 0.5))
    finally:
        reader.close()


def test_particle_frames_keep_multi_env_rows_independent(tmp_path: Path):
    source_env = _env(2)
    path, source = _record(tmp_path, source_env)
    target = _env(2)
    cfg = SimpleNamespace(num_envs=2, fluid_object=target.cfg.fluid_object)
    reader = FluidTrajectoryReader(path, source, cfg)
    try:
        reader.apply(target, torch.tensor([1, 0]))
        torch.testing.assert_close(target.fluid.written[0, :, 0], torch.full((2,), 0.2))
        torch.testing.assert_close(
            target.fluid.written[1, :, 0], torch.full((2,), 10.0)
        )
    finally:
        reader.close()


def test_particle_reader_does_not_require_task_program_metadata(tmp_path: Path):
    env = _env(1)
    path, source = _record(tmp_path, env)
    for key in ("fluid_qualification", "demo_episodes", "artifact_purpose"):
        source["meta"].pop(key)
    cfg = SimpleNamespace(num_envs=env.num_envs, fluid_object=env.cfg.fluid_object)
    reader = FluidTrajectoryReader(path, source, cfg)
    try:
        reader.apply(env, torch.tensor([1]))
        torch.testing.assert_close(env.fluid.written[0, :, 0], torch.full((2,), 0.2))
    finally:
        reader.close()


def test_particle_source_rejects_malformed_and_tampered_sidecars(tmp_path: Path):
    env = _env(1)
    path, source = _record(tmp_path, env)
    cfg = SimpleNamespace(num_envs=1, fluid_object=env.cfg.fluid_object)
    sidecar = source["meta"].pop("fluid_state")
    with pytest.raises(ValueError, match="sidecars"):
        FluidTrajectoryReader(path, source, cfg)
    source["meta"]["fluid_state"] = sidecar
    manifest = path.parent / source["meta"]["fluid_state"]["manifests"][0]
    chunk = next(manifest.parent.glob("chunk-*.npz"))
    with chunk.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="content changed"):
        FluidTrajectoryReader(path, source, cfg)


def test_particle_recorder_rejects_frame_length_mismatch(tmp_path: Path):
    env = _env(1)
    recorder = FluidTrajectoryRecorder(tmp_path, 1, 0.025)
    recorder.capture(env, [0], reset=True)
    with pytest.raises(ValueError, match="align"):
        recorder.finish(tmp_path / "trajectory.pt", [1])
    recorder.abort()
    assert not recorder.directory.exists()


def test_selected_row_reset_restarts_only_its_particle_stream(tmp_path: Path):
    env = _env(2)
    recorder = FluidTrajectoryRecorder(tmp_path, 2, 0.025)
    recorder.capture(env, [0, 1], reset=True)
    recorder.capture(env, [0, 1])
    recorder.capture(env, [0], reset=True)
    recorder.capture(env, [0])

    sidecar = recorder.finish(tmp_path / "trajectory.pt", [1, 1])

    assert len(sidecar["manifests"]) == 2
    recorder.commit()
    manifest = tmp_path / sidecar["manifests"][0]
    with pytest.raises(RuntimeError, match="finalized"):
        recorder.capture(env, [0], reset=True)
    assert manifest.is_file()
    recorder.abort()


def test_partial_batch_publishes_only_selected_particle_rows(tmp_path: Path):
    env = _env(2)
    recorder = FluidTrajectoryRecorder(tmp_path, 2, 0.025, enabled_env_ids=[0])
    recorder.capture(env, [0, 1], reset=True)
    recorder.capture(env, [0, 1])
    path = tmp_path / "trajectory.pt"
    sidecar = recorder.finish(path, [1, 1], env_ids=[0])
    recorder.commit()
    assert len(sidecar["manifests"]) == len(sidecar["parent_ids"]) == 1
    assert not (recorder.directory / "env_001").exists()
    source = _source(1, 1)
    source["meta"]["fluid_state"] = sidecar
    target = _env(2)
    reader = FluidTrajectoryReader(
        path, source, SimpleNamespace(num_envs=2, fluid_object=target.cfg.fluid_object)
    )
    try:
        reader.apply(target, torch.tensor([0, 1]))
        assert target.fluid.written is not None
    finally:
        reader.close()
