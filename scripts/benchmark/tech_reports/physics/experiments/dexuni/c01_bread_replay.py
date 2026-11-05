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
"""Freeze and record the user-requested C01 single-bread Isaac Lab variant.

Run: imported by physics/platforms/dexuni/isaaclab/c01_bread.py.
No performance timing; accepted two-good reference recordings stay immutable.
"""

from __future__ import annotations

from collections.abc import Callable
import datetime as dt
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .c01_common import PHASES, packing_metrics
from .c01_replay import freeze_inputs, load_inputs, sha256, write_json

__all__ = ["prepare", "record", "analyze"]


def prepare(reference: Path, output: Path, worker: Path) -> dict[str, Any]:
    """Keep the original left-arm prefix, hold home, and omit milk geometry."""
    inputs = load_inputs(reference)
    freeze_inputs(inputs, output, worker)
    source = output / "source"
    shutil.copy2(Path(__file__), source / Path(__file__).name)
    end = 702  # 11.7 s, after the complete existing left-home motion.
    total = 942  # 15.7 s, including four seconds of genuine settling.
    original = inputs["joints"]
    joints = {"joint_names": original["joint_names"].copy()}
    joints["time"] = np.arange(total + 1, dtype=float) / 60
    for key in ("targets", "tcp_target", "tcp_baked"):
        joints[key] = np.concatenate(
            (
                original[key][: end + 1],
                np.repeat(original[key][end : end + 1], total - end, axis=0),
            )
        )
    right = [i for i, n in enumerate(joints["joint_names"]) if n.startswith("RIGHT_")]
    if not np.allclose(
        joints["targets"][:, right],
        np.repeat(joints["targets"][:1, right], total + 1, axis=0),
        rtol=0,
        atol=2e-6,
    ):
        raise ValueError("The single-bread variant must keep the right arm at home")
    for name in ("joint_commands", "goods_meshes"):
        (output / "inputs" / f"{name}.npz").rename(
            output / "inputs" / f"{name}_reference.npz"
        )
    mesh = {
        k: v.copy() for k, v in inputs["mesh"].items() if not k.startswith("pouch_")
    }
    np.savez_compressed(output / "inputs/joint_commands.npz", **joints)
    np.savez_compressed(output / "inputs/goods_meshes.npz", **mesh)
    phases = [row for row in PHASES if row[0] <= 11.7] + [(15.7, "settle")]
    scene = {
        **inputs["manifest"]["scene"],
        "goods": ["bread"],
        "good_arms": {"bread": "left"},
        "pouch": None,
        "phases": phases,
    }
    inputs.update(joints=joints, mesh=mesh, scene=scene)
    write_json(
        output / "inputs/variant.json",
        dict(
            variant="single_bread_no_milk",
            active_goods=["bread"],
            reference_manifest_sha256=sha256(reference / "manifest.json"),
            command_prefix_frames=end + 1,
            prefix_equal=True,
            right_arm_at_home_tolerance=2e-6,
            right_home_max_baked_difference=float(
                abs(joints["targets"][:, right] - joints["targets"][:1, right]).max()
            ),
            settling_s=4.0,
            duration_s=15.7,
            note="Reference NPZs retained as provenance; no milk particles, gas cavity or right-arm cycle are created.",
        ),
    )
    return inputs


def record(
    inputs: dict[str, Any],
    output: Path,
    step: Callable[[], None],
    readings: Callable[[], dict[str, np.ndarray]],
    capture: Callable[[], np.ndarray] | None,
    diagnostics: Callable[[], dict[str, Any]],
    audit: dict[str, Any],
    *,
    until: float | None,
) -> None:
    """Record actual clock, named joints, geometry and centered physical frames."""
    joints, mesh, scene = inputs["joints"], inputs["mesh"], inputs["scene"]
    total = 942 if until is None else min(942, round(until * 60))
    preview = output / "preview"
    preview.mkdir()
    manifest = dict(
        task="C01",
        platform="Isaac Lab / native Newton VBD",
        variant="single_bread_no_milk",
        status="recording",
        user_scene_approved=False,
        performance_measurements_started=False,
        reference_user_approved=True,
        reference_manifest_sha256=sha256(inputs["run"] / "manifest.json"),
        scene=scene,
        camera=inputs["manifest"]["camera"],
        joint_state_order=joints["joint_names"].tolist(),
        coupling="one_way",
        joint_mode="kinematic",
        initial_model=audit,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        input_sha256={
            str(p.relative_to(output / "inputs")): sha256(p)
            for p in (output / "inputs").rglob("*")
            if p.is_file()
        },
        execution_source_sha256={
            str(p.relative_to(output / "source")): sha256(p)
            for p in (output / "source").rglob("*")
            if p.is_file() and p.suffix != ".pyc"
        },
    )
    write_json(output / "manifest.json", manifest)
    states: dict[str, list[np.ndarray]] = {}
    rows = []
    writer = None
    if capture is not None:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont

        writer = imageio.get_writer(
            str(preview / "video.mp4"),
            fps=30,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        )
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    try:
        for frame in range(total + 1):
            if frame:
                step()
            sample = readings()
            t = float(sample["time"])
            if abs(t - joints["time"][frame]) > 1e-7:
                raise RuntimeError(f"Actual physics clock mismatch at {frame}: {t}")
            if not all(np.isfinite(v).all() for v in sample.values()):
                raise RuntimeError(f"Nonfinite physical state at {t}")
            qe = float(abs(sample["qpos"] - joints["targets"][frame]).max())
            te = float(
                np.linalg.norm(
                    sample["tcp"] - joints["tcp_baked"][frame], axis=-1
                ).max()
            )
            if qe > 2e-5 or te > 2e-5:
                raise RuntimeError(f"Named joint/TCP mismatch at {t}: {qe}, {te}")
            for k, v in sample.items():
                states.setdefault(k, []).append(np.array(v, copy=True))
            row = packing_metrics(
                sample["bag"],
                {"bread": sample["bread"].mean(axis=0)},
                {"mouth": mesh["bag_mouth_group"], "bottom": mesh["bag_bottom_group"]},
            )
            row.update(time=t, joint_replay_error=qe, tcp_replay_error_m=te)
            rows.append(row)
            if writer is not None and frame % 2 == 0:
                image = Image.fromarray(capture())
                name = next(label for end, label in scene["phases"] if t <= end + 1e-7)
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, image.width, 42), fill="black")
                draw.text(
                    (12, 9),
                    f"Isaac Lab | C01 single bread (no milk) | t={t:.2f}s | {name} | in bag: {int(row['bread_in_bag'])}/1",
                    font=font,
                    fill="white",
                )
                writer.append_data(np.asarray(image))
                if frame in (0, 252, 462, 582, 702, 822, total):
                    image.save(preview / f"still_{t:05.2f}s.jpg")
            if frame % 120 == 0:
                print(
                    f"C01 single bread frame={frame}/{total} actual={t:.3f} in_bag={row['bread_in_bag']}",
                    flush=True,
                )
        manifest.update(
            status="completed" if total == 942 else "completed_diagnostic_truncated",
            frames=total // 2 + 1 if writer is not None else 0,
            effective=diagnostics(),
        )
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if writer is not None:
            writer.close()
        np.savez_compressed(
            preview / "states.npz", **{k: np.asarray(v) for k, v in states.items()}
        )
        write_json(preview / "metrics.json", rows)
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)


def analyze(run: Path) -> dict[str, Any]:
    """Compute single-bread grip, signed tet volume and late bag motion."""
    with np.load(run / "preview/states.npz") as archive:
        s = dict(archive)
    with np.load(run / "inputs/goods_meshes.npz") as archive:
        m = dict(archive)

    def determinant(x: np.ndarray) -> np.ndarray:
        """Return six times the oriented tetrahedron volume."""
        return np.einsum(
            "...i,...i->...",
            x[..., 1, :] - x[..., 0, :],
            np.cross(x[..., 2, :] - x[..., 0, :], x[..., 3, :] - x[..., 0, :]),
        )

    ratios = determinant(
        s["bread"].astype(float)[:, m["bread_tetrahedra"]]
    ) / determinant(m["bread_vertices"][m["bread_tetrahedra"]])
    t = s["time"]
    hold = (t >= 4.2 - 1e-8) & (t <= 8.9 + 1e-8)
    relative = s["bread"].mean(axis=1) - s["tcp"][:, 0]
    drift = (
        np.linalg.norm(relative[hold] - relative[hold][0], axis=-1)
        if hold.any()
        else np.array([np.nan])
    )
    bag = s["bag"]
    speed = np.linalg.norm(np.diff(bag, axis=0) / np.diff(t)[:, None, None], axis=-1)
    late = t >= t[-1] - 1 - 1e-8
    import json

    row = json.loads((run / "preview/metrics.json").read_text())[-1]
    result = dict(
        states_sha256=sha256(run / "preview/states.npz"),
        finite=all(np.isfinite(v).all() for v in s.values()),
        samples=len(t),
        actual_duration_s=float(t[-1]),
        bread_centre_in_bag=row["bread_in_bag"],
        minimum_signed_tet_volume_ratio=float(ratios.min()),
        inverted_tet_pose_count=int((ratios <= 0).sum()),
        bread_hold_drift_m=float(drift.max()),
        bread_lift_centre_above_table_m=float(
            s["bread"].mean(axis=1)[:, 2].max() - 0.86
        ),
        final_bag_speed_p95_m_s=float(np.percentile(speed[late[1:]], 95)),
        final_bag_centre_span_m=np.ptp(bag[late].mean(axis=1), axis=0),
        max_joint_replay_error=max(
            r["joint_replay_error"]
            for r in json.loads((run / "preview/metrics.json").read_text())
        ),
        max_tcp_replay_error_m=max(
            r["tcp_replay_error_m"]
            for r in json.loads((run / "preview/metrics.json").read_text())
        ),
    )
    write_json(run / "analysis/current_metrics.json", result)
    np.savez_compressed(
        run / "analysis/plot_data.npz",
        time=t,
        bag_node_speed_p95=np.r_[0, np.percentile(speed, 95, axis=1)],
        bread_grip_drift=np.linalg.norm(
            relative - relative[min(252, len(relative) - 1)], axis=-1
        ),
        min_signed_tet_ratio=ratios.min(axis=1),
    )
    return result
