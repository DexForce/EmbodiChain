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
"""Replay the accepted C02 commands and record independent platform states.

Run through either platform worker; this harness performs no timing.
"""

from __future__ import annotations
from collections.abc import Callable
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "OUTPUT",
    "load",
    "freeze",
    "record",
    "sha",
    "write_json",
    "table_pieces",
    "seed_paper",
    "audit_model",
    "fixed_link_expansion",
]
OUTPUT = (
    Path(__file__).resolve().parents[3].parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/c02_bag_frame_packing"
)
PAPER_FIELDS = (
    "particle_q",
    "particle_mass",
    "particle_inv_mass",
    "particle_radius",
    "tri_indices",
    "tri_poses",
    "tri_materials",
    "edge_indices",
    "edge_rest_angle",
    "edge_bending_properties",
)


def sha(path: Path) -> str:
    """Hash one retained file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write metadata, including NumPy arrays."""
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item(),
        )
        + "\n"
    )


def load(reference: Path) -> dict[str, Any]:
    """Verify the accepted native inputs before reusing their exact bytes."""
    manifest = json.loads((reference / "manifest.json").read_text())
    for rel, h in manifest["input_sha256"].items():
        assert sha(reference / "inputs" / rel) == h, rel
    for rel, h in json.loads(
        (reference / "analysis/platform_port/accepted_native_sha256.json").read_text()
    ).items():
        assert sha(reference / rel) == h, rel
    result = dict(reference=reference, manifest=manifest)
    for key, path in [
        ("bag", "inputs/bag.npz"),
        ("materials", "inputs/bag_materials.npz"),
        ("joints", "inputs/joint_commands.npz"),
        ("initial", "analysis/current_checks/initial_model.npz"),
    ]:
        with np.load(reference / path) as a:
            result[key] = dict(a)
    with np.load(reference / "preview/states.npz") as a:
        result["tcp_targets"] = a["tcp_targets"].copy()
    assert len(result["joints"]["time"]) == 3841
    return result


def freeze(inputs: dict[str, Any], output: Path, worker: Path) -> None:
    """Freeze inputs and execution source in a new output directory."""
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(inputs["reference"] / "inputs", output / "inputs")
    shutil.copytree(
        inputs["reference"] / "source/reference", output / "source/reference"
    )
    shutil.copy2(worker, output / "source" / worker.name)
    shutil.copy2(Path(__file__), output / "source/c02_replay.py")
    shutil.copy2(
        Path(__file__).with_name("c02_controller.py"),
        output / "source/c02_controller.py",
    )
    shutil.copy2(
        inputs["reference"] / "source/c02.py", output / "source/dexsim_c02_reference.py"
    )
    shutil.copy2(
        output / "inputs/joint_commands.npz",
        output / "inputs/accepted_native_joint_commands.npz",
    )
    shutil.copy2(
        inputs["reference"] / "manifest.json", output / "source/native_manifest.json"
    )
    shutil.copy2(
        inputs["reference"] / "analysis/current_checks/initial_model.npz",
        output / "inputs/native_initial_model.npz",
    )
    (output / "analysis").mkdir()


def table_pieces() -> list[tuple]:
    """Return the original tabletop and four legs."""
    return [("table", (0.66, -0.01, 0.898), (0.60, 1.32, 0.044))] + [
        (f"leg{i}", (x, y, 0.438), (0.044, 0.044, 0.876))
        for i, (x, y) in enumerate(
            ((x, y) for x in (0.40, 0.91) for y in (-0.59, 0.57))
        )
    ]


def seed_paper(model: Any, inputs: dict[str, Any]) -> None:
    """Restore authored node masses/rest tensors once, before solver creation.

    EC's scalar config does not expose the original heterogeneous masses and
    triangle/hinge arrays. Its public RuntimeControl.initialize_model hook is
    used only at initialization. This function applies no runtime force.
    """
    initial = inputs["initial"]
    assert model.particle_count == 1381 and model.tri_count == 2712
    assert np.array_equal(model.tri_indices.numpy(), initial["tri_indices"])
    assert np.array_equal(model.edge_indices.numpy(), initial["edge_indices"])
    for key in PAPER_FIELDS:
        target = getattr(model, key)
        assert target.numpy().shape == initial[key].shape, (
            key,
            target.numpy().shape,
            initial[key].shape,
        )
        target.assign(initial[key])
    import warp as wp

    wp.synchronize_device(model.device)


def audit_model(model: Any, inputs: dict[str, Any], output: Path) -> dict[str, Any]:
    """Read back the initialized geometry, rest material, masses and rigid flags."""
    for key in PAPER_FIELDS:
        actual = getattr(model, key).numpy()
        expected = inputs["initial"][key]
        assert np.array_equal(actual, expected), (
            key,
            float(np.max(np.abs(actual - expected))),
            actual.ravel()[:5],
            expected.ravel()[:5],
        )
    assert np.all(model.particle_inv_mass.numpy() > 0)
    fields = PAPER_FIELDS + (
        "tri_areas",
        "particle_flags",
        "shape_flags",
        "shape_transform",
        "shape_scale",
        "shape_type",
        "shape_body",
        "body_mass",
        "body_inertia",
        "body_flags",
        "shape_material_ke",
        "shape_material_kd",
        "shape_material_mu",
        "shape_margin",
        "shape_gap",
    )
    np.savez_compressed(
        output / "analysis/initial_model.npz",
        **{k: getattr(model, k).numpy() for k in fields},
        body_labels=np.asarray(model.body_label),
        shape_labels=np.asarray(model.shape_label),
    )
    return dict(
        nodes=1381,
        triangles=2712,
        world_pins=0,
        authored_paper_fields_byte_equal=list(PAPER_FIELDS),
        dynamic_goods=["can", "carton"],
        external_forces_added=False,
    )


def fixed_link_expansion(
    urdf: Path, names: list[str], available: list[str]
) -> tuple[list[int], np.ndarray]:
    """Resolve merged USD fixed links from actual retained body poses."""
    import xml.etree.ElementTree as ET

    graph = {}
    for j in ET.parse(urdf).getroot().findall("joint"):
        if j.get("type") != "fixed":
            continue
        a, b = j.find("parent").get("link"), j.find("child").get("link")
        o = j.find("origin")
        m = np.eye(4)
        if o is not None:
            m[:3, 3] = np.fromstring(o.get("xyz", "0 0 0"), sep=" ")
            m[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(o.get("rpy", "0 0 0"), sep=" ")
            ).as_matrix()
        graph.setdefault(a, []).append((b, m))
        graph.setdefault(b, []).append((a, np.linalg.inv(m)))
    known = {n: (i, np.eye(4)) for i, n in enumerate(available)}
    queue = list(known)
    for a in queue:
        for b, m in graph.get(a, []):
            if b not in known:
                i, parent = known[a]
                known[b] = (i, parent @ m)
                queue.append(b)
    assert set(names) <= set(known), set(names) - set(known)
    return [known[n][0] for n in names], np.stack([known[n][1] for n in names])


def _sample(
    raw: dict[str, np.ndarray], inputs: dict[str, Any], frame: int
) -> dict[str, np.ndarray]:
    """Fit the original paper frame and evaluate actual primitive containment."""
    s = dict(raw)
    n = int(inputs["bag"]["paper_count"])
    rest = inputs["bag"]["vertices"][:n]
    cur = s["bag"][:n]
    a, b = rest.mean(0), cur.mean(0)
    u, _, vt = np.linalg.svd((rest - a).T @ (cur - b))
    c = np.eye(3)
    c[2, 2] = np.linalg.det(u @ vt)
    rot = (u @ c @ vt).T
    pos = b - rot @ a
    err = np.sqrt(np.mean(np.linalg.norm(cur - (rest @ rot.T + pos), axis=1) ** 2))
    local = (s["snack_pose"][:, 4:] - pos) @ rot
    rel = rot.T @ Rotation.from_quat(s["snack_pose"][:, :4]).as_matrix()
    extent = np.empty((2, 3))
    extent[0] = 0.0325 * np.linalg.norm(rel[0, :, :2], axis=1) + 0.059 * np.abs(
        rel[0, :, 2]
    )
    extent[1] = np.abs(rel[1]) @ np.array((0.024, 0.030, 0.059))
    loaded = (
        (np.abs(local[:, :2]) + extent[:, :2] < np.array((0.068, 0.163))).all(-1)
        & (local[:, 2] - extent[:, 2] > -0.007)
        & (local[:, 2] + extent[:, 2] < 0.25)
    )
    s.update(
        tcp_targets=s.get("tcp_targets", inputs["tcp_targets"][frame]),
        bag_position=pos,
        bag_rotation=rot,
        shape_error=np.asarray(err),
        loaded=loaded,
    )
    return s


def record(
    platform: str,
    inputs: dict[str, Any],
    output: Path,
    step: Callable,
    readings: Callable,
    capture: Callable | None,
    diagnostics: Callable,
    audit: dict[str, Any],
    until: float | None = None,
) -> None:
    """Record physical outer states and an optional full 30 fps video."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    native = inputs["manifest"]
    joints = inputs["joints"]
    total = 3840 if until is None else min(3840, round(until * 60))
    assert total > 0
    preview = output / "preview"
    preview.mkdir()
    states = {}
    writer = None
    manifest = dict(
        task="C02",
        platform=platform,
        status="recording",
        performance_measurements=False,
        reference_user_approved=True,
        reference_manifest_sha256=sha(inputs["reference"] / "manifest.json"),
        joint_mode="kinematic",
        coupling="one_way",
        controller="Original W1BagController with observed bag feedback; reference lift tail; external process provides IK only",
        joint_state_order=native["joint_state_order"],
        link_state_order=native["link_state_order"],
        table_centre=native["table_centre"],
        table_half=native["table_half"],
        initial_model=audit,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        executable=sys.executable,
    )
    trajectory = output / "inputs/controller_trajectory.json"
    if trajectory.exists():
        manifest["controller_trajectory"] = json.loads(trajectory.read_text())
        manifest["controller"] += "; explicit platform-specific robot TCP correction"
    try:
        if capture is not None:
            writer = imageio.get_writer(
                preview / "video.mp4",
                fps=30,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )
        for frame in range(total + 1):
            if frame:
                step()
            sample = _sample(readings(), inputs, frame)
            assert abs(float(sample["time"]) - frame / 60) < 1e-7, (
                frame,
                sample["time"],
            )
            assert (
                max(
                    abs(
                        sample["qpos"]
                        - sample.get("joint_command", joints["targets"][frame])
                    )
                )
                < 2e-5
            ), (
                frame,
                "joint replay",
            )
            assert all(np.isfinite(v).all() for v in sample.values()), frame
            for key, v in sample.items():
                states.setdefault(key, []).append(np.array(v, copy=True))
            # Keep the supplied demo's paper-body acceptance limits. Fail
            # early rather than recording a full task around a collapsed bag.
            time_value = float(sample["time"])
            limit = 0.04 if time_value < 11 - 1e-8 else 0.025
            assert float(sample["shape_error"]) <= limit, (
                frame,
                "reference paper shape",
                float(sample["shape_error"]),
            )
            if time_value >= 11 - 1e-8:
                assert float(sample["bag_rotation"][2, 2]) >= np.cos(np.radians(30)), (
                    frame,
                    "reference bag upright",
                    float(sample["bag_rotation"][2, 2]),
                )
            if writer is not None and frame % 2 == 0:
                img = Image.fromarray(capture().astype(np.uint8))
                draw = ImageDraw.Draw(img)
                draw.rectangle((0, 0, 1280, 36), fill="black")
                draw.text(
                    (12, 10),
                    f"C02 | {platform} | original paper bag + two rigid goods | {frame/60:.2f}s",
                    fill="white",
                )
                writer.append_data(np.asarray(img))
                if frame in (0, 600, 1200, 2100, 2820, 3000, 3360, 3480, 3600, 3840):
                    img.save(preview / f"still_{frame/60:05.2f}s.jpg")
            if frame % 30 == 0:
                print(
                    f'C02 {platform} frame={frame}/{total} loaded={sample["loaded"].tolist()} '
                    f'shape={float(sample["shape_error"]):.5f} '
                    f'tilt_deg={np.degrees(np.arccos(np.clip(sample["bag_rotation"][2, 2], -1, 1))):.2f}',
                    flush=True,
                )
        manifest.update(
            status="completed",
            actual_steps=total,
            actual_time_s=total / 60,
            reference_paper_shape_limits_passed=True,
        )
    except BaseException as exc:
        manifest.update(
            status="failed",
            error=repr(exc),
            actual_steps=len(states.get("time", [])) - 1,
        )
        raise
    finally:
        if writer is not None:
            writer.close()
        if "joint_command" in states:
            np.savez_compressed(
                output / "inputs/joint_commands.npz",
                time=np.asarray(states["time"]),
                joint_names=joints["joint_names"],
                targets=np.stack(states["joint_command"]),
            )
        if states:
            np.savez_compressed(
                preview / "states.npz", **{k: np.stack(v) for k, v in states.items()}
            )
        manifest.update(
            effective=diagnostics(),
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            input_sha256={
                str(p.relative_to(output / "inputs")): sha(p)
                for p in (output / "inputs").rglob("*")
                if p.is_file()
            },
            source_sha256={
                str(p.relative_to(output / "source")): sha(p)
                for p in (output / "source").rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            },
        )
        write_json(output / "manifest.json", manifest)
