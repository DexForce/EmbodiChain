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

"""On-demand physical grasp labels for a configured parallel-jaw grasp service."""

from __future__ import annotations

import fcntl
import hashlib
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

from embodichain.data import get_data_path
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalGraspPoseGeneratorCfg,
    ParallelJawGraspCollisionCfg,
)
from embodichain.toolkits.graspkit.pg_grasp._ranking import (
    _closing_axis_deviation,
    _grasp_costs,
    _normalized_ranking_weights,
)
from embodichain.toolkits.graspkit.pg_grasp.fixed_geometry import (
    FixedGripperCollisionGeometry,
)

__all__ = []

_LABEL_CACHE_DIR = Path.home() / ".cache/embodichain/grasp_annotator_cache/validated"


def _cache_key(
    urdf: Path, vertices: torch.Tensor, triangles: torch.Tensor, signature: dict
) -> str:
    """Hash canonical little-endian float32 vertices, int32 indices and policy JSON."""
    with urdf.open("rb") as stream:
        hand = hashlib.file_digest(stream, "md5").hexdigest()
    mesh = hashlib.md5()
    for value, dtype in ((vertices, "<f4"), (triangles, "<i4")):
        mesh.update(value.detach().cpu().numpy().astype(dtype, copy=False).tobytes())
    mesh.update(
        json.dumps(
            signature, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )
    return f"{hand}-{mesh.hexdigest()}"


def _rank_labels(
    tcp: torch.Tensor,
    vertices: torch.Tensor,
    object_poses: torch.Tensor,
    directions: torch.Tensor,
    geometry: FixedGripperCollisionGeometry,
    algorithm: AntipodalGraspPoseGeneratorCfg,
    collision: ParallelJawGraspCollisionCfg,
    part: str = "center",
    *,
    axis: torch.Tensor | None = None,
    positive: bool | torch.Tensor = True,
    middle_gap: float | None = None,
    middle_only: bool = False,
    on_support: bool = True,
    qualify: Callable[[torch.Tensor], torch.Tensor] | None = None,
    alignment_angle: float | None = None,
    tcp_tolerance_deg: float | None = None,
) -> torch.Tensor:
    """Rank each live row by geometric cost, TCP +Z agreement, then cache index."""
    contact = tcp @ torch.linalg.inv(
        tcp.new_tensor(geometry.provenance["grasp_from_tcp"])
    )
    points = geometry.points.to(tcp)
    weights = _normalized_ranking_weights(
        algorithm.angle_weight, algorithm.center_weight
    )
    orders = []
    batch = len(object_poses)
    if part in ("top", "bottom"):
        axis = tcp.new_tensor([0.0, 0.0, 1.0])
        positive = part == "top"
    if axis is not None:
        axis = torch.nn.functional.normalize(axis.to(tcp).expand(batch, 3), dim=-1)
    signs = torch.as_tensor(positive, device=tcp.device).expand(batch)
    directions = torch.nn.functional.normalize(
        directions.to(tcp).expand(batch, 3), dim=-1
    )
    for row, (pose, approach) in enumerate(zip(object_poses, directions, strict=True)):
        world = pose @ contact
        mesh = vertices @ pose[:3, :3].T + pose[:3, 3]
        lo, hi = mesh.amin(0), mesh.amax(0)
        center = (lo + hi) / 2 if algorithm.center_mode == "bounds" else mesh.mean(0)
        centers = world[:, :3, 3]
        closing = torch.nn.functional.normalize(world[:, :3, 0], dim=-1)
        valid = (
            _closing_axis_deviation(closing, approach)
            <= algorithm.approach_deviation_angle
        )
        if axis is not None:
            projected = mesh @ axis[row]
            low, high = projected.amin(), projected.amax()
            span = high - low
            positions = centers @ axis[row]
            if middle_only:
                valid &= (positions >= low + span / 3) & (positions <= high - span / 3)
            else:
                # Signed coordinates put the requested object end on the positive side.
                sign = torch.where(signs[row], 1.0, -1.0)
                fraction = (
                    (1 + middle_gap) / 2
                    if middle_gap is not None
                    else (0.5 if algorithm.center_mode == "bounds" else 0.65)
                )
                cut = (low + high) * sign / 2 + (fraction - 0.5) * span
                valid &= positions * sign > cut
                if algorithm.center_mode == "bounds":
                    center = center + sign * axis[row] * span / 4
                else:
                    center = mesh[projected * sign > cut].mean(0)
        if collision.filter_ground_collision and on_support:
            # A label can come from a different resting orientation. Recheck
            # the open hand against the current support, in bounded tiles.
            for begin in range(0, len(world), 256):
                block = world[begin : begin + 256, 2]
                clearance = (block[:, :3] @ points.T).amin(1) + block[:, 3] - lo[2]
                valid[begin : begin + 256] &= clearance >= collision.collision_threshold
        if alignment_angle is not None:
            valid &= (world[:, :3, 2] @ approach) >= math.cos(alignment_angle)
        if qualify is not None:
            valid &= qualify(world)
        ids = torch.where(valid)[0]
        if len(ids):
            costs = _grasp_costs(world[ids], centers[ids], approach, center, weights)
            z = torch.nn.functional.normalize(tcp[ids, :3, 2], dim=-1)
            align = z @ (pose[:3, :3].T @ approach)
            tie = align.argsort(descending=True, stable=True)
            ranked = tie[costs[tie].argsort(stable=True)]
            if tcp_tolerance_deg is not None:
                ranked = ranked[
                    align[ranked] >= math.cos(math.radians(tcp_tolerance_deg))
                ]
            ids = ids[ranked]
        orders.append(ids)
    return torch.nn.utils.rnn.pad_sequence(orders, batch_first=True, padding_value=-1)


def _world_pose_rows(
    local: torch.Tensor, order: torch.Tensor, object_poses: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Express valid labels in each live object's world frame, preserving rank."""
    rows = []
    for pose, ids in zip(object_poses, order, strict=True):
        candidates = pose @ local[ids[ids >= 0]]
        rows.append(
            (
                candidates,
                torch.arange(len(candidates), device=pose.device, dtype=pose.dtype),
            )
        )
    return rows


@dataclass
class ValidatedGraspCache:
    """Resolve object-local labels for one configured parallel-jaw hand."""

    hand: dict
    algorithm: dict
    collision_cfg: ParallelJawGraspCollisionCfg
    geometry: FixedGripperCollisionGeometry
    qualify: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None

    force_refresh: bool = False
    physics: str = "default"
    _refreshed: set[str] = field(default_factory=set, init=False, repr=False)

    @classmethod
    def from_geometry(
        cls,
        hand: dict,
        algorithm: dict,
        collision: ParallelJawGraspCollisionCfg,
        geometry: FixedGripperCollisionGeometry,
        *,
        force_refresh: bool = False,
        physics: str = "default",
        qualify: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> ValidatedGraspCache:
        """Use the mounted TCP and canonical contact frame for the validator's hand."""
        source = geometry.provenance
        tcp = np.asarray(source["root_from_tcp"])
        description = dict(
            hand,
            fpath=source["urdf_path"],
            root_link=source["root_link"],
            default_root_from_tcp=tcp.tolist(),
            root_from_contact_frame=(
                tcp @ np.linalg.inv(source["grasp_from_tcp"])
            ).tolist(),
            collision_links=list(dict.fromkeys(p["link"] for p in source["parts"])),
            mesh_paths=source["mesh_paths"],
        )
        if any(
            np.float32(source["joint_positions"][name]) != np.float32(value)
            for name, value in description["joint_commands"]["open"].items()
        ):
            from .gripper_geometry import build_gripper_collision_geometry

            geometry = build_gripper_collision_geometry(
                description, point_spacing=source["point_spacing_m"]
            )
        return cls(
            description,
            algorithm,
            collision,
            geometry,
            qualify=qualify,
            physics=physics,
            force_refresh=force_refresh,
        )

    def _with_commands(self, values: torch.Tensor) -> ValidatedGraspCache:
        """Map execution-order joint commands into the native validator hand."""
        defaults = self.hand["joint_commands"]
        names = (
            self.hand.get("binding", {}).get("control_joint_names")
            or self.hand["control_parts"]["eef"]
        )
        if values.shape != (2, len(names)):
            raise ValueError(
                "Invocation command width disagrees with the native hand binding."
            )
        commands = {}
        for command, row in zip(("open", "grasp"), values, strict=True):
            native = defaults[command]
            if torch.equal(row, row.new_tensor([native[name] for name in names])):
                commands[command] = native
                continue
            # Runtime buffers round URDF endpoints to float32. Retain the native
            # value when an override reuses either calibrated endpoint.
            effective = {}
            for name, value in zip(names, row.tolist(), strict=True):
                effective[name] = next(
                    (
                        target[name]
                        for target in defaults.values()
                        if row.new_tensor(target[name]).item() == value
                    ),
                    value,
                )
            if native.keys() - effective.keys():
                # Bindings can command only the leader of a coupled pair.
                xml = ET.parse(get_data_path(self.hand["fpath"]))
                mimics = {
                    j.attrib["name"]: j.find("mimic") for j in xml.findall("joint")
                }

                def resolve(name: str) -> float:
                    if name not in effective:
                        mimic = mimics[name]
                        effective[name] = (
                            native[name]
                            if mimic is None
                            else (
                                resolve(mimic.attrib["joint"])
                                * float(mimic.get("multiplier", "1"))
                                + float(mimic.get("offset", "0"))
                            )
                        )
                    return effective[name]

                effective = {name: resolve(name) for name in native}
            commands[command] = effective
        if commands == defaults:
            return self
        # Bound views share the per-key force-refresh bookkeeping.
        bound = copy(self)
        bound.hand = dict(self.hand, joint_commands=commands)
        if commands["open"] != defaults["open"]:
            from .gripper_geometry import build_gripper_collision_geometry

            bound.geometry = build_gripper_collision_geometry(
                bound.hand, point_spacing=self.geometry.provenance["point_spacing_m"]
            )
        return bound

    def __call__(
        self,
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        attrs: dict | None,
        object_poses: torch.Tensor,
        directions: torch.Tensor,
        part: str = "center",
        *,
        hand_commands: tuple[torch.Tensor, torch.Tensor] | None = None,
        **options,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Share physical validation only across rows executing the same commands."""
        if hand_commands is None:
            return self._rank(
                vertices, triangles, attrs, object_poses, directions, part, **options
            )
        pairs = torch.stack(hand_commands, dim=1).detach().cpu()
        unique, groups = pairs.unique(dim=0, return_inverse=True)
        if len(unique) == 1:
            return self._with_commands(unique[0])._rank(
                vertices, triangles, attrs, object_poses, directions, part, **options
            )
        pools, keys = [], []
        orders = [None] * len(pairs)
        offset = 0
        for index, pair in enumerate(unique):
            rows = (groups == index).nonzero().flatten().tolist()
            row_options = options.copy()
            for name, shape in (("axis", (len(pairs), 3)), ("positive", (len(pairs),))):
                value = options.get(name)
                if isinstance(value, torch.Tensor):
                    row_options[name] = value.expand(shape)[rows]
            local, order, key = self._with_commands(pair)._rank(
                vertices,
                triangles,
                attrs,
                object_poses[rows],
                directions.expand(len(pairs), 3)[rows],
                part,
                **row_options,
            )
            pools.append(local)
            keys.append(key)
            for row, ids in zip(rows, order, strict=True):
                orders[row] = ids[ids >= 0] + offset
            offset += len(local)
        order = torch.nn.utils.rnn.pad_sequence(
            orders, batch_first=True, padding_value=-1
        )
        return torch.cat(pools), order, "+".join(keys)

    def ranked_world_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        attrs: dict | None = None,
        on_support: bool = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return validated world-frame candidates and their row-local rank costs."""
        local, order, _ = self(
            mesh_vertices,
            mesh_triangles,
            attrs,
            obj_poses,
            approach_direction,
            on_support=on_support,
        )
        return _world_pose_rows(local, order, obj_poses)

    def _load_labels(
        self,
        key: str,
        urdf: Path,
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        signature: dict,
    ) -> dict[str, torch.Tensor]:
        """Read the physical labels, running one isolated validator on a cache miss."""
        directory = _LABEL_CACHE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{key}.pt"
        if not path.exists() or (self.force_refresh and key not in self._refreshed):
            with (directory / f"{key}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not path.exists() or (
                    self.force_refresh and key not in self._refreshed
                ):
                    print(f"Validating antipodal grasps for {path.name}", flush=True)
                    with TemporaryDirectory(dir=directory) as temp:
                        work = Path(temp)
                        import trimesh

                        # Text OBJ preserves float32 coordinates in the native mesh importer.
                        trimesh.Trimesh(
                            vertices=vertices.detach().cpu().numpy(),
                            faces=triangles.detach().cpu().numpy(),
                            process=False,
                        ).export(work / "target.obj", digits=17)
                        (work / "job.json").write_text(
                            json.dumps(signature | {"urdf": str(urdf)})
                        )
                        subprocess.run(
                            [sys.executable, "-m", __name__, str(work)], check=True
                        )
                        (work / "labels.pt").replace(path)
                    if self.force_refresh:
                        self._refreshed.add(key)
        return torch.load(path, weights_only=True, map_location="cpu")

    def _rank(
        self,
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        attrs: dict | None,
        object_poses: torch.Tensor,
        directions: torch.Tensor,
        part: str = "center",
        *,
        axis: torch.Tensor | None = None,
        positive: bool | torch.Tensor = True,
        middle_gap: float | None = None,
        middle_only: bool = False,
        on_support: bool = True,
        pose_adjustment: dict | None = None,
        tcp_tolerance_deg: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Return object-frame TCPs, ordered label indices per scene and the cache key.

        Validation runs in a separate physics process on the first request for
        a hand, mesh and sampling configuration. Each call ranks the stored TCPs
        for the current object placements and approach directions. A padding
        index of -1 marks the end of a scene's eligible candidates.
        """
        from embodichain.lab.scripts.label_grasps import (
            _DEFAULT_NUM_ENVS,
            _SamplingCfg,
            _VALIDATION_REVISION,
        )

        hand = self.hand
        urdf = Path(get_data_path(hand["fpath"]))
        native_urdf = urdf
        if hand.get("extract_subtree") or hand.get("mesh_paths"):
            from embodichain.data.urdf import materialize_urdf_subtree

            native_urdf = materialize_urdf_subtree(
                urdf,
                hand["root_link"],
                mesh_paths={
                    k: get_data_path(v) for k, v in hand.get("mesh_paths", {}).items()
                },
            )
        description = {
            k: hand[k]
            for k in (
                "model_id",
                "root_link",
                "pad_links",
                "collision_links",
                "root_from_contact_frame",
                "min_opening_width",
                "max_opening_width",
                "qpos_limits",
                "joint_drive_props",
                "attrs",
                "link_attrs",
            )
            if k in hand
        }
        defaults = _SamplingCfg()
        sampling = _SamplingCfg(
            sample_count=self.algorithm.get("sample_count", defaults.sample_count),
            candidates_per_direction=self.algorithm.get(
                "max_candidates", defaults.candidates_per_direction
            ),
        )
        algorithm = AntipodalGraspPoseGeneratorCfg(
            **(
                dict(
                    approach_direction_samples=1,
                    approach_alignment_angle=math.radians(
                        sampling.approach_tolerance_deg
                    ),
                    sample_count=sampling.sample_count,
                    max_candidates=sampling.candidates_per_direction,
                )
                | self.algorithm
            )
        )
        config = {
            "physics": self.physics,
            "gripper": dict(
                description=description,
                root_from_tcp=hand["default_root_from_tcp"],
                open_qpos=hand["joint_commands"]["open"],
                close_qpos=hand["joint_commands"]["grasp"],
            ),
            "rigid": {
                "attrs": {
                    group: {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in props.items()
                    }
                    for group, props in (attrs or {}).items()
                    if props is not None
                },
                "shape": {
                    "collision": {
                        "approximation": "convex_decomposition",
                        "max_hulls": 12,
                        "acd_method": "visacd",
                    }
                },
            },
            "collision_threshold": self.collision_cfg.collision_threshold,
            "close_wait_steps": 60,
            "enable_disturbance": True,
        }
        settings = dict(
            algorithm=algorithm.to_dict(),
            collision=self.collision_cfg.to_dict(),
            point_spacing=self.geometry.provenance["point_spacing_m"],
        )
        signature = dict(
            schema=_VALIDATION_REVISION,
            target_mesh_format="obj",
            config=config,
            sampling=sampling.to_dict(),
            generator=settings,
            num_envs=_DEFAULT_NUM_ENVS,
            engine=version("dexsim-engine"),
            hand_geometry=self.geometry.provenance["points_sha256"],
            hand_files=sorted(self.geometry.provenance["files_sha256"].values()),
            mesh_shape=[list(vertices.shape), list(triangles.shape)],
        )
        if pose_adjustment:
            signature["pose_adjustment"] = pose_adjustment
        key = _cache_key(urdf, vertices, triangles, signature)
        labels = self._load_labels(key, native_urdf, vertices, triangles, signature)
        poses = labels["grasp_pose"].to(object_poses)
        check = None
        if self.qualify is not None:
            widths = labels["open_length"].to(poses)

            def check(world: torch.Tensor) -> torch.Tensor:
                return self.qualify(world, widths)

        order = _rank_labels(
            poses,
            vertices.to(poses),
            object_poses,
            directions,
            self.geometry,
            algorithm,
            self.collision_cfg,
            part,
            axis=axis,
            positive=positive,
            middle_gap=middle_gap,
            middle_only=middle_only,
            on_support=on_support,
            qualify=check,
            alignment_angle=self.algorithm.get("approach_alignment_angle"),
            tcp_tolerance_deg=tcp_tolerance_deg,
        )
        print(f"Ranked {len(poses)} validated grasps from {key}.pt", flush=True)
        return poses, order, key


def _adjust_candidates(poses: torch.Tensor, settings: dict) -> torch.Tensor:
    """Apply requested tool adjustments in the object frame before physical testing."""
    poses = poses.clone()
    if settings.get("mirrored"):
        poses[:, :3, :2] *= -1
    if settings.get("rotate_upright"):
        axis = poses.new_tensor(settings["upright_axis"])
        angle = (
            torch.where(poses[:, :3, 1] @ axis < 0, -1.0, 1.0)
            * settings["rotate_upright"]
        )
        y, z = poses[:, :3, 1].clone(), poses[:, :3, 2].clone()
        cos, sin = angle.cos()[:, None], angle.sin()[:, None]
        poses[:, :3, 1] = cos * y + sin * z
        poses[:, :3, 2] = cos * z - sin * y
    if settings.get("retract"):
        poses[:, :3, 3] -= settings["retract"] * poses[:, :3, 2]
    if "grasp_frame_to_eef" in settings:
        poses = poses @ poses.new_tensor(settings["grasp_frame_to_eef"])
    return poses


def _generate(work: Path) -> None:
    from embodichain.lab.scripts.label_grasps import _SamplingCfg, _run

    job = json.loads((work / "job.json").read_text())
    _run(
        job["config"],
        _SamplingCfg(**job["sampling"]),
        work / "target.obj",
        Path(job["urdf"]),
        work / "labels.pt",
        work,
        job["num_envs"],
        False,
        generator_settings=job["generator"],
        candidate_transform=(
            (lambda poses: _adjust_candidates(poses, job["pose_adjustment"]))
            if "pose_adjustment" in job
            else None
        ),
    )


if __name__ == "__main__":
    _generate(Path(sys.argv[1]))
