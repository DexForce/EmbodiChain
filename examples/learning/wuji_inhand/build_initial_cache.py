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
"""Build palm-up initial-contact assets from measured native screening traces.

These starts require opposing pads, low endpoint velocity and bounded
penetration. Maintaining the grasp and rotating remain policy objectives;
long stationary holding is not claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from _asset import _prepared_sha256

__all__ = ["main"]


def _screened_batches(screen: Path, prepared: Path) -> tuple[list[Path], int]:
    """Return the batches a screening run lists and its feedback step count.

    Batch files left by another run in the same directory are ignored. When the
    manifest records the prepared-asset hashes, they must match ``prepared``.

    Args:
        screen: Directory written by ``screen.py``.
        prepared: Prepared assets that the cache will copy.

    Returns:
        Listed batch files and the preparation feedback step count.

    Raises:
        ValueError: If the screening run used different prepared assets.
        FileNotFoundError: If a listed batch is missing.
    """
    manifest = json.loads((screen / "screening.json").read_text())
    recorded = manifest.get("prepared_assets_sha256")
    if recorded is not None and recorded != _prepared_sha256(prepared):
        raise ValueError("The screening run used different prepared assets")
    files = [screen / f"batch_{record['batch']}.pt" for record in manifest["batches"]]
    missing = [file.name for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"Screening batches are missing: {missing}")
    return files, int(manifest["preparation_feedback_steps"])


def _holding_targets(packet: dict[str, Any], feedback_steps: int) -> np.ndarray:
    """Return the fixed controller targets held while a batch was measured.

    Args:
        packet: One loaded ``batch_*.pt`` screening packet.
        feedback_steps: The run's preparation feedback step count.

    Returns:
        Holding targets, one row per screened arena.

    Raises:
        ValueError: If feedback changed targets that the packet does not record.
    """
    if "target" in packet:
        return packet["target"].numpy()
    if feedback_steps:
        raise ValueError(
            "This screening packet predates recorded holding targets; rerun "
            "screen.py because preparation feedback changed them"
        )
    # Without feedback the reset preload is held unchanged.
    return packet["initial"][:, 27:47].numpy()


def main() -> None:
    """Select measured contacts and freeze a self-contained training-asset cache."""
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--screen", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    batches, feedback_steps = _screened_batches(a.screen, a.prepared)
    a.output.mkdir(exist_ok=False, parents=True)
    m = mujoco.MjModel.from_xml_path(str((a.prepared / "scene.xml").resolve()))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    palm = m.body("right_palm_link").id
    if d.xmat[palm].reshape(3, 3)[2, 0] < 0.999:
        raise ValueError("Initial-contact cache requires a palm-up Wuji model")
    qa = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    oa = int(m.joint("object_joint").qposadr[0])
    obj = m.geom("object").id
    tips = [m.geom(f"right_finger{i}_link4_geom_3").id for i in range(1, 6)]
    kept = [[] for _ in range(8)]
    records = []
    scales = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    for file in batches:
        packet = torch.load(file, weights_only=False, map_location="cpu")
        states = packet["physical_pose_trace"].numpy()
        forces = packet["fingertip_contact_forces"].numpy()
        targets = _holding_targets(packet, feedback_steps)
        for row in range(states.shape[1]):
            scale = scales[row // 32]
            m.geom_size[obj, :2] = [0.02 * scale, 0.016 * scale]
            history = []
            for frame in range(min(20, len(states))):
                s = states[frame, row]
                d.qpos[qa] = s[:20]
                d.qpos[oa : oa + 7] = s[20:27]
                mujoco.mj_kinematics(m, d)
                delta = d.geom_xpos[tips] - s[20:23]
                radial = delta[:, :2] / np.maximum(
                    np.linalg.norm(delta[:, :2], axis=-1, keepdims=True), 1e-6
                )
                opposite = (radial[1:] * radial[:1]).sum(-1) < -0.15
                touch = (forces[frame, row] > 0.02) & (
                    np.abs(delta[:, 2]) < 0.8 * 0.016 * scale
                )
                valid = bool(touch[0] and np.sum(touch[1:] & opposite) >= 2)
                history.append(valid)
                if frame >= 7 and all(history[-5:]):
                    prev = states[frame - 1, row]
                    linear = np.linalg.norm(s[20:23] - prev[20:23]) / 0.05
                    speed = np.max(np.abs(s[:20] - prev[:20])) / 0.05
                    if linear > 0.05 or speed > 2.0:
                        continue
                    mujoco.mj_forward(m, d)
                    depth = max(
                        [
                            max(0.0, -float(c.dist))
                            for c in d.contact[: d.ncon]
                            if obj in c.geom
                        ]
                        or [0.0]
                    )
                    if depth > 0.003:
                        continue
                    kept[row // 32].append(np.r_[s, targets[row]])
                    records.append(
                        {
                            "batch": file.name,
                            "row": row,
                            "frame": frame,
                            "scale": scale,
                            "linear_speed": float(linear),
                            "max_joint_speed": float(speed),
                            "max_penetration_m": depth,
                        }
                    )
                    break
        print(file.name, [len(k) for k in kept], flush=True)
    for scale, rows in zip(scales, kept):
        np.save(
            a.output / f"candidates_{scale:g}.npy",
            np.asarray(rows, dtype=np.float32).reshape(-1, 47),
        )
    (a.output / "initial_contact_manifest.json").write_text(
        json.dumps(
            {
                "method": "Actual native measured starting states with >=250ms prior opposing-pad contact, low endpoint velocities and<=3mm object penetration. Static long-hold success is not claimed. Full-policy qualification remains20s.",
                "prepared": str(a.prepared.resolve()),
                "screen": str(a.screen.resolve()),
                "counts": [len(k) for k in kept],
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )

    shutil.copytree(a.prepared / "compiled", a.output / "compiled")
    shutil.copy2(a.prepared / "scene.xml", a.output / "scene.xml")
    profile_path = a.output / "compiled/source.json"
    profile = json.loads(profile_path.read_text())
    profile["native_source"] = str((a.output / "compiled/hand.usda").resolve())
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    entries = []
    for scale, rows in zip(scales, kept):
        file = a.output / f"candidates_{scale:g}.npy"
        entries.append(
            {
                "scale": scale,
                "count": len(rows),
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            }
        )
    (a.output / "cache_manifest.json").write_text(
        json.dumps(
            {
                "task_version": "wuji-palm-up-v2",
                "layout": "measured q20,object position3/quaternionWXYZ4,holding target20",
                "initial_contact_only": True,
                "static_20second_holding_claim": False,
                "source_native_screen": str(a.screen.resolve()),
                "scales": entries,
            },
            indent=2,
        )
        + "\n"
    )
    if any(len(rows) == 0 for rows in kept):
        raise RuntimeError(
            "A size has no valid initial contact state; do not train this cache"
        )


if __name__ == "__main__":
    main()
