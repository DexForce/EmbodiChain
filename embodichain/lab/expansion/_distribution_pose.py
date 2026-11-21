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

"""Full orientation features and recorded-model TCP reconstruction."""

from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation

__all__: list[str] = []


def orientation(poses: np.ndarray, times: np.ndarray) -> dict:
    q = np.asarray(poses, dtype=float)[:, 3:7]
    if (
        q.shape != (len(times), 4)
        or not np.isfinite(q).all()
        or np.any(np.linalg.norm(q, axis=1) < 1e-12)
    ):
        raise ValueError("Orientation requires finite nonzero xyzw quaternions")
    rot = Rotation.from_quat(q)
    canonical = rot.as_quat(canonical=True)
    vectors = rot.as_rotvec()
    angle = (rot[:-1].inv() * rot[1:]).magnitude()
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("Increasing orientation times required")
    return {
        "initial_xyzw": canonical[0].tolist(),
        "final_xyzw": canonical[-1].tolist(),
        "initial_rotvec_rad": vectors[0].tolist(),
        "final_rotvec_rad": vectors[-1].tolist(),
        "angular_path_rad": float(angle.sum()),
        "angular_speed_p50_p95": np.quantile(angle / dt, [0.5, 0.95]).tolist(),
        "net_rotation_rad": float((rot[0].inv() * rot[-1]).magnitude()),
        "rotation_bin_initial": ",".join(
            map(str, np.floor((vectors[0] + np.pi) / (np.pi / 6)).astype(int))
        ),
        "rotation_bin_final": ",".join(
            map(str, np.floor((vectors[-1] + np.pi) / (np.pi / 6)).astype(int))
        ),
    }


class RecordedTCP:
    """Cache one recorded URDF chain; reconstruct in bounded 256-frame batches."""

    def __init__(self, deployment: dict):
        import pytorch_kinematics as pk

        robot = deployment.get("robot") or {}
        bindings = list((robot.get("gripper_bindings") or {}).values())
        self.reason = None
        if len(bindings) != 1:
            self.reason = "single_explicit_gripper_binding_required"
            return
        binding = bindings[0]
        self.link = binding.get("root_link")
        transform = binding.get("defaults", {}).get("default_root_from_tcp")
        asset = robot.get("fpath")
        if not self.link or transform is None or not asset or not Path(asset).is_file():
            self.reason = "recorded_tcp_calibration_or_asset_missing"
            return
        self.offset = np.asarray(transform, dtype=float)
        if (
            self.offset.shape != (4, 4)
            or not np.isfinite(self.offset).all()
            or not np.allclose(self.offset[3], [0, 0, 0, 1])
            or not np.allclose(
                self.offset[:3, :3].T @ self.offset[:3, :3], np.eye(3), atol=1e-6
            )
            or not np.isclose(np.linalg.det(self.offset[:3, :3]), 1)
        ):
            raise ValueError("Invalid TCP transform")
        if robot.get("body_scale") not in (None, [1.0, 1.0, 1.0]):
            self.reason = "scaled_robot_not_supported"
            return
        content = Path(asset).read_bytes()
        self.chain = pk.build_chain_from_urdf(content)
        self.names = self.chain.get_joint_parameter_names()
        self.evidence = {
            "method": "fk_from_measured_qpos_and_root",
            "root_link": self.link,
            "root_from_tcp": self.offset.tolist(),
            "urdf_sha256": hashlib.sha256(content).hexdigest(),
        }

    def poses(self, episode) -> np.ndarray | None:
        import torch

        if self.reason:
            return None
        robots = [
            s for s in episode.metadata.get("initial_state", ()) if s["kind"] == "robot"
        ]
        if len(robots) != 1:
            raise ValueError("TCP requires one recorded robot")
        robot = robots[0]
        fields = {
            s["field"]: s["key"]
            for s in episode.metadata["scene_state_fields"]
            if s["entity_uid"] == robot["uid"]
        }
        q = episode.observations[fields["qpos"]].cpu()
        roots = episode.observations[fields["pose"]].cpu().numpy()
        indices = [list(robot["joint_names"]).index(n) for n in self.names]
        out = []
        with torch.no_grad():
            for start in range(0, len(q), 256):
                local = (
                    self.chain.forward_kinematics(
                        q[start : start + 256, indices].float()
                    )[self.link]
                    .get_matrix()
                    .numpy()
                    @ self.offset
                )
                rp = roots[start : start + 256]
                world = np.broadcast_to(np.eye(4), local.shape).copy()
                world[:, :3, :3] = Rotation.from_quat(rp[:, 3:]).as_matrix()
                world[:, :3, 3] = rp[:, :3]
                tcp = world @ local
                out.append(
                    np.c_[tcp[:, :3, 3], Rotation.from_matrix(tcp[:, :3, :3]).as_quat()]
                )
        return np.concatenate(out)
