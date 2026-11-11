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

"""Bounded, control-aligned fluid states for Gym trajectory replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import torch

from embodichain.data_pipeline.recording.media import (
    MediaFrameReader,
    MediaSpool,
    verify_media_manifest,
)

__all__: list[str] = []

_MAX_BUFFER_BYTES = 32 * 1024 * 1024


class FluidTrajectoryRecorder:
    """Spool one full particle-state frame per recorded control boundary."""

    def __init__(self, directory: Path, num_envs: int, step_dt: float) -> None:
        self.directory = directory / f"fluid-state-{uuid4().hex}"
        self.num_envs = num_envs
        self.step_dt = step_dt
        self._spools: list[MediaSpool | None] = [None] * num_envs
        self._parent_ids = [str(uuid4()) for _ in range(num_envs)]
        self._counts = [0] * num_envs
        self._published = False
        self._committed = False

    def capture(
        self, env: object, env_ids: Sequence[int], *, reset: bool = False
    ) -> None:
        """Capture selected rows, including the reset frame when requested."""
        if self._published:
            raise RuntimeError(
                "A finalized fluid trajectory cannot capture more frames."
            )
        ids = tuple(int(index) for index in env_ids)
        if not ids:
            return
        if reset:
            for index in ids:
                if self._spools[index] is not None:
                    shutil.rmtree(self._spools[index].directory)
                    self._spools[index] = None
                self._parent_ids[index] = str(uuid4())
                self._counts[index] = 0
        elif any(self._spools[index] is None for index in ids):
            raise RuntimeError("Fluid trajectory has no initial control frame.")
        offsets = env.sim.arena_offsets.detach().cpu().numpy()
        frames = {}
        for uid in sorted(fluid.uid for fluid in env.cfg.fluid_object):
            state = env.sim.get_fluid_object(uid).data.particle_state_w
            if state.dtype != torch.float32 or state.ndim != 3 or state.shape[-1] != 6:
                raise ValueError("Recorded fluid state must be float32 (E, N, 6).")
            values = state.detach().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("Recorded fluid state must be finite.")
            frames[uid] = values
        for index in ids:
            if self._spools[index] is None:
                self._spools[index] = MediaSpool(
                    self.directory / f"env_{index:03d}",
                    max_buffer_bytes=_MAX_BUFFER_BYTES,
                    parent_episode_id=self._parent_ids[index],
                    calibration={"coordinate_frame": "arena_local", "channels": 6},
                )
            row = {}
            for uid, values in frames.items():
                local = values[index].copy()
                local[:, :3] -= offsets[index]
                row[uid] = local
            self._spools[index].append(self._counts[index] * self.step_dt, row)
            self._counts[index] += 1

    def finish(self, trajectory_path: Path, lengths: Sequence[int]) -> dict:
        """Publish verified sidecars before the trajectory references them."""
        if self._published:
            raise RuntimeError("Fluid trajectory was already published.")
        if len(lengths) != self.num_envs:
            raise ValueError("Fluid trajectory row count differs from source.")
        manifests = []
        for index, (spool, length) in enumerate(zip(self._spools, lengths)):
            if spool is None or self._counts[index] != int(length) + 1:
                raise ValueError("Fluid frames do not align with trajectory lengths.")
            path = spool.finish()
            verify_media_manifest(
                path,
                expected_frames=self._counts[index],
                parent_episode_id=self._parent_ids[index],
                expected_timestamps=np.arange(self._counts[index]) * self.step_dt,
            )
            manifests.append(str(path.relative_to(trajectory_path.parent)))
        self._published = True
        return {
            "schema_version": 1,
            "coordinate_frame": "arena_local",
            "manifests": manifests,
            "parent_ids": list(self._parent_ids),
        }

    def commit(self) -> None:
        """Retain sidecars after their referring trajectory is durable."""
        if not self._published:
            raise RuntimeError("Fluid sidecars must be finalized before commit.")
        self._committed = True

    def abort(self) -> None:
        """Discard uncommitted particle chunks after failure or interruption."""
        if not self._committed:
            shutil.rmtree(self.directory, ignore_errors=True)


class FluidTrajectoryReader:
    """Validate and stream selected fluid frames from a qualified trajectory."""

    def __init__(self, path: str | Path, trajectory: Mapping, cfg: object) -> None:
        self.path = Path(path).resolve()
        meta = trajectory["meta"]
        qualification = meta.get("fluid_qualification")
        if (
            meta.get("artifact_purpose") != "recording"
            or not isinstance(qualification, Mapping)
            or qualification.get("kind") != "task_program"
            or qualification.get("accepted") is not True
        ):
            raise ValueError(
                "Fluid replay requires an accepted Task Program recording."
            )
        episodes = meta.get("demo_episodes", [])
        source_envs = int(meta["num_envs"])
        if len(episodes) != source_envs or not episodes:
            raise ValueError("Fluid replay source episodes do not match recorded rows.")
        program_hash = qualification.get("program_hash")
        if (
            not isinstance(program_hash, str)
            or not program_hash
            or any(
                episode.get("program_hash") != program_hash
                or episode.get("completed") is not True
                or episode.get("success") is not True
                for episode in episodes
            )
        ):
            raise ValueError("Fluid replay program provenance differs from the source.")
        target_envs = int(cfg.num_envs)
        if source_envs not in (1, target_envs):
            raise ValueError(
                "Fluid replay source must match env count or contain one row."
            )
        sidecar = meta.get("fluid_state")
        if (
            not isinstance(sidecar, Mapping)
            or sidecar.get("schema_version") != 1
            or sidecar.get("coordinate_frame") != "arena_local"
        ):
            raise ValueError("Fluid replay requires aligned particle-state sidecars.")
        paths = sidecar.get("manifests")
        parent_ids = sidecar.get("parent_ids")
        if not isinstance(paths, list) or not isinstance(parent_ids, list):
            raise ValueError("Fluid replay sidecar manifest is malformed.")
        if len(paths) != source_envs or len(parent_ids) != source_envs:
            raise ValueError("Fluid replay sidecar rows do not match the trajectory.")
        self._uids = tuple(sorted(fluid.uid for fluid in cfg.fluid_object))
        if not self._uids:
            raise ValueError("Fluid replay requires configured fluid objects.")
        expected_counts = {
            fluid.uid: int(fluid.spawn.sample(fluid.particle_spacing).shape[0])
            for fluid in cfg.fluid_object
        }
        self._readers = []
        try:
            for index, (name, parent_id) in enumerate(zip(paths, parent_ids)):
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(parent_id, str)
                    or not parent_id
                ):
                    raise ValueError(
                        "Fluid replay manifest paths and IDs must be strings."
                    )
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        "Fluid replay manifest escapes its artifact directory."
                    )
                candidate = self.path.parent / relative
                if any(
                    part.is_symlink()
                    for part in (candidate, *candidate.parents)
                    if part.is_relative_to(self.path.parent)
                ):
                    raise ValueError("Fluid replay manifest cannot use symlinks.")
                manifest_path = candidate.resolve()
                if not manifest_path.is_relative_to(self.path.parent):
                    raise ValueError(
                        "Fluid replay manifest escapes its artifact directory."
                    )
                count = int(meta["lengths"][index]) + 1
                declared = verify_media_manifest(
                    manifest_path,
                    expected_frames=count,
                    parent_episode_id=parent_id,
                    expected_timestamps=np.arange(count) * float(meta["dt"]),
                )
                if declared.get("calibration") != {
                    "coordinate_frame": "arena_local",
                    "channels": 6,
                }:
                    raise ValueError("Fluid replay particle frame convention differs.")
                if tuple(sorted(declared["features"])) != self._uids:
                    raise ValueError(
                        "Fluid replay particle UIDs differ from the scene."
                    )
                for uid, schema in declared["features"].items():
                    if (
                        len(schema["shape"]) != 2
                        or schema["shape"][0] != expected_counts[uid]
                        or schema["shape"][1] != 6
                        or np.dtype(schema["dtype"]) != np.dtype("float32")
                    ):
                        raise ValueError("Fluid replay requires float32 (N, 6) frames.")
                self._readers.append(MediaFrameReader(manifest_path))
        except Exception:
            self.close()
            raise
        self._source_envs = source_envs
        self._target_envs = target_envs

    def apply(self, env: object, steps: torch.Tensor) -> None:
        """Write selected control-boundary frames without advancing physics."""
        if tuple(steps.shape) != (self._target_envs,):
            raise ValueError("Fluid replay step indices do not match target rows.")
        indices = [int(value) for value in steps.detach().cpu().tolist()]
        offsets = env.sim.arena_offsets
        selected: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        for row, step in enumerate(indices):
            source = 0 if self._source_envs == 1 else row
            key = source, step
            if key not in selected:
                _, selected[key] = self._readers[source].read(step)
        for uid in self._uids:
            fluid = env.sim.get_fluid_object(uid)
            values = []
            for row, step in enumerate(indices):
                source = 0 if self._source_envs == 1 else row
                state = selected[source, step][uid]
                if (
                    state.shape != (fluid.num_particles, 6)
                    or not np.isfinite(state).all()
                ):
                    raise ValueError("Fluid replay particle count or state is invalid.")
                values.append(state)
            world = torch.as_tensor(np.stack(values), device=env.device).clone()
            world[:, :, :3] += offsets[:, None, :]
            fluid.write_particle_state_to_sim(world)

    def close(self) -> None:
        for reader in getattr(self, "_readers", ()):
            reader.close()
