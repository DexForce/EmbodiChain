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
"""Render recorded physical poses without stepping or prescribing any motion."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["render_trace"]


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _triangles(geometry: Any) -> np.ndarray:
    if geometry.kind == "mesh":
        vertices = np.asarray(geometry.vertices, dtype=float)
        faces = np.asarray(geometry.triangles, dtype=int)
    elif geometry.kind == "box":
        vertices = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        ) * np.asarray(geometry.size[:3])
        faces = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ]
        )
    elif geometry.kind == "cylinder":
        radius, half_height = geometry.size[:2]
        angle = np.linspace(0, 2 * np.pi, 49)[:-1]
        ring = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
        vertices = np.concatenate(
            [
                np.column_stack((ring, np.full(len(ring), -half_height))),
                np.column_stack((ring, np.full(len(ring), half_height))),
                np.array([[0, 0, -half_height], [0, 0, half_height]]),
            ]
        )
        n = len(ring)
        faces = np.asarray(
            [
                face
                for i in range(n)
                for face in (
                    [i, (i + 1) % n, (i + 1) % n + n],
                    [i, (i + 1) % n + n, i + n],
                    [2 * n, (i + 1) % n, i],
                    [2 * n + 1, i + n, (i + 1) % n + n],
                )
            ]
        )
    elif geometry.kind == "sphere":
        radius = geometry.size[0]
        azimuth = np.linspace(0, 2 * np.pi, 25)[:-1]
        polar = np.linspace(0, np.pi, 13)
        vertices = radius * np.array(
            [
                [np.sin(p) * np.cos(a), np.sin(p) * np.sin(a), np.cos(p)]
                for p in polar
                for a in azimuth
            ]
        )
        n = len(azimuth)
        faces = np.asarray(
            [
                face
                for j in range(len(polar) - 1)
                for i in range(n)
                for face in (
                    [j * n + i, j * n + (i + 1) % n, (j + 1) * n + i],
                    [
                        j * n + (i + 1) % n,
                        (j + 1) * n + (i + 1) % n,
                        (j + 1) * n + i,
                    ],
                )
            ]
        )
    else:
        raise ValueError(f"Unsupported recorded geometry {geometry.kind!r}.")
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Geometry {geometry.name!r} has no renderable triangles.")
    local = vertices @ _rotation(np.asarray(geometry.quaternion)).T
    return (local + np.asarray(geometry.position))[faces]


def _metric_text(metrics: dict[str, Any]) -> str:
    values = []
    for key, label in (("fall_m", "Fall"), ("advance_m", "Advance")):
        if metrics.get(key) is not None:
            values.append(f"{label}: {1000 * metrics[key]:.3f} mm")
    if metrics.get("insertion_depth_m") is not None:
        values.append(f"Insertion: {1000 * metrics['insertion_depth_m']:.3f} mm")
        values.append("Seated" if metrics.get("success_rate") == 1 else "Not seated")
    if metrics.get("ideal_advance_m") is not None:
        values.append(f"Ideal lead: {1000 * metrics['ideal_advance_m']:.3f} mm")
    for key, label in (
        ("lead_relative_error", "Lead error"),
        ("ratio_relative_error", "Ratio error"),
    ):
        if metrics.get(key) is not None:
            values.append(f"{label}: {100 * metrics[key]:.3f}%")
    return "   |   ".join(values)


def render_trace(
    trace_path: str | Path,
    output_path: str | Path,
    *,
    fps: int = 20,
    collision_colors: bool = False,
) -> dict[str, str | float]:
    """Render actual recorded body poses as an animation and three-frame sheet.

    Args:
        trace_path: JSON with asset_manifest, kind, backend, timestep,
            sample_steps, frames and metrics. Each frame maps body names to
            seven-value world poses with xyzw quaternions.
        output_path: Destination ending in .gif or .mp4. MP4 needs FFmpeg.
        fps: Playback frame rate. Displayed playback speed follows recorded
            physical sample times; no ideal motion is synthesized.
        collision_colors: Render every collision piece in a stable distinct
            color instead of the asset's visible geometries.

    Returns:
        Absolute animation and contact-sheet paths and actual playback speed.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import animation, pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from embodichain.gen_sim.assembly import AssemblyAsset

    trace_file = Path(trace_path).resolve()
    output = Path(output_path).resolve()
    if fps <= 0:
        raise ValueError("Playback frame rate must be positive.")
    if output.suffix.lower() not in {".gif", ".mp4"}:
        raise ValueError("Recorded animation output must end in .gif or .mp4.")
    trace = json.loads(trace_file.read_text())
    manifest = Path(trace["asset_manifest"])
    if not manifest.is_absolute():
        manifest = trace_file.parent / manifest
    asset = AssemblyAsset.load(manifest)
    frames = trace["frames"]
    steps = np.asarray(trace["sample_steps"], dtype=int)
    timestep = float(trace["timestep"])
    if not frames or len(frames) != len(steps) or timestep <= 0:
        raise ValueError(
            "Recorded frames and sample steps must be nonempty and aligned."
        )
    if len(steps) > 1 and not np.all(np.diff(steps) > 0):
        raise ValueError("Recorded sample steps must strictly increase.")
    if len(steps) > 2:
        intervals = np.diff(steps)
        if not np.all(intervals[:-1] == intervals[0]) or intervals[-1] > intervals[0]:
            raise ValueError(
                "Recorded sample steps need a constant stride with an optional shorter final interval."
            )
    poses: list[dict[str, np.ndarray]] = []
    for frame in frames:
        parsed = {}
        for body in asset.bodies:
            pose = np.asarray(frame[body.name], dtype=float)
            if (
                pose.shape != (7,)
                or not np.isfinite(pose).all()
                or np.linalg.norm(pose[3:]) < 1e-12
            ):
                raise ValueError(f"Invalid recorded pose for body {body.name!r}.")
            parsed[body.name] = pose
        poses.append(parsed)
    playback_speed = (
        fps * float(steps[-1] - steps[0]) * timestep / (len(steps) - 1)
        if len(steps) > 1
        else 0.0
    )
    meshes = []
    piece_index = 0
    for body in asset.bodies:
        body_triangles = []
        body_colors = []
        for geometry in body.geometries:
            if not (geometry.collision if collision_colors else geometry.visible):
                continue
            color = np.asarray(geometry.rgba, dtype=float)
            if collision_colors:
                color = np.array(
                    [*colorsys.hsv_to_rgb((piece_index * 0.618034) % 1, 0.6, 0.85), 1.0]
                )
            triangles = _triangles(geometry)
            body_triangles.append(triangles)
            body_colors.append(np.broadcast_to(color, (len(triangles), 4)))
            piece_index += 1
        if body_triangles:
            meshes.append(
                (
                    body.name,
                    np.concatenate(body_triangles),
                    np.concatenate(body_colors),
                )
            )
    if not meshes:
        raise ValueError("Recorded asset has no selected geometry to render.")
    witnesses = {}
    if trace["kind"] == "gear":
        for index, (name, triangles, _) in enumerate(meshes):
            vertices = triangles.reshape(-1, 3)
            radius = float(np.max(np.linalg.norm(vertices[:, :2], axis=1)))
            top = float(np.max(vertices[:, 2])) + radius * 0.025
            witnesses[name] = (
                np.array([[0, 0, top], [radius * 0.72, 0, top]]),
                "#00d4ef" if index % 2 == 0 else "#e52282",
            )
    bounds = []
    for frame in poses:
        for name, triangles, _ in meshes:
            pose = frame[name]
            world = triangles.reshape(-1, 3) @ _rotation(pose[3:]).T + pose[:3]
            bounds.extend((world.min(axis=0), world.max(axis=0)))
    lower, upper = np.min(bounds, axis=0), np.max(bounds, axis=0)
    span = np.maximum(upper - lower, max(float(np.max(upper - lower)), 1e-6) * 0.12)
    center = (lower + upper) / 2
    lower, upper = center - 0.58 * span, center + 0.58 * span
    title = str(trace.get("title", trace["kind"].replace("_", " ").title()))
    backend = str(trace["backend"])
    route = str(trace.get("route", "direct"))
    metrics = _metric_text(trace.get("metrics", {}))

    def setup_axis(axis: Any) -> None:
        axis.computed_zorder = False
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_zlim(lower[2], upper[2])
        axis.set_box_aspect(span)
        axis.view_init(elev=55 if trace["kind"] == "gear" else 25, azim=-65)
        for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
            coordinate.set_major_locator(MaxNLocator(3))
            coordinate.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value * 1000:g}")
            )
            coordinate.pane.set_alpha(0.04)
        axis.set_xlabel("x (mm)", labelpad=5)
        axis.set_ylabel("y (mm)", labelpad=5)
        axis.set_zlabel("z (mm)", labelpad=5)
        axis.tick_params(labelsize=8)
        axis.grid(True, alpha=0.15)

    def draw_geometry(axis: Any, index: int) -> list[Any]:
        world_meshes = []
        colors = []
        for name, triangles, color in meshes:
            pose = poses[index][name]
            world_meshes.append(triangles @ _rotation(pose[3:]).T + pose[:3])
            colors.append(color)
        # One collection sorts individual faces across body boundaries, avoiding
        # a whole plate incorrectly occluding the peg above it in mplot3d.
        collection = Poly3DCollection(
            np.concatenate(world_meshes),
            facecolors=np.concatenate(colors),
            edgecolors=(0.08, 0.1, 0.15, 0.11),
            linewidths=0.22,
            shade=True,
            lightsource=matplotlib.colors.LightSource(azdeg=135, altdeg=45),
            zorder=1,
        )
        axis.add_collection3d(collection)
        return [collection]

    def draw_witnesses(axis: Any, index: int) -> list[tuple[Any, str, np.ndarray]]:
        markers = []
        for name, (local, color) in witnesses.items():
            pose = poses[index][name]
            world = local @ _rotation(pose[3:]).T + pose[:3]
            for width, line_color in ((5.0, "#1a2735"), (2.6, color)):
                (line,) = axis.plot(
                    *world.T,
                    color=line_color,
                    linewidth=width,
                    marker="o",
                    markevery=[-1],
                    markersize=5,
                    markeredgewidth=0.6,
                    markeredgecolor="#1a2735",
                    zorder=2,
                )
                markers.append((line, name, local))
        return markers

    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(9.6, 6.8), dpi=100, facecolor="white")
    axis = figure.add_subplot(111, projection="3d")
    setup_axis(axis)
    collections = draw_geometry(axis, 0)
    markers = draw_witnesses(axis, 0)
    figure.subplots_adjust(top=0.85, bottom=0.24, left=0.04, right=0.92)
    figure.suptitle(title, fontsize=17, fontweight="bold", y=0.96)
    figure.text(
        0.5,
        0.9,
        f"{backend} / {route}   |   Recorded simulation poses   |   Mean playback {playback_speed:g}x",
        ha="center",
        fontsize=10,
        color="#3e4c5b",
    )
    time_text = figure.text(0.5, 0.13, "", ha="center", fontsize=12)
    figure.text(0.5, 0.077, metrics, ha="center", fontsize=10, color="#234965")
    figure.text(
        0.5,
        0.037,
        "Offline mesh rendering of physical states; visual appearance is not a precision certificate.",
        ha="center",
        fontsize=8,
        color="#667380",
    )
    marker_note = (
        "Radial markers show recorded orientation; render-only, no effect on dynamics."
        if witnesses
        else ""
    )
    figure.text(0.5, 0.012, marker_note, ha="center", fontsize=8, color="#667380")

    def update(index: int) -> list[Any]:
        world_meshes = []
        for name, triangles, _ in meshes:
            pose = poses[index][name]
            world_meshes.append(triangles @ _rotation(pose[3:]).T + pose[:3])
        collections[0].set_verts(np.concatenate(world_meshes))
        for line, name, local in markers:
            pose = poses[index][name]
            world = local @ _rotation(pose[3:]).T + pose[:3]
            line.set_data_3d(*world.T)
        time_text.set_text(f"Simulation time: {steps[index] * timestep:.3f} s")
        return [*collections, *(line for line, _, _ in markers), time_text]

    video = animation.FuncAnimation(
        figure, update, frames=len(frames), interval=1000 / fps, blit=False
    )
    writer = (
        animation.PillowWriter(fps=fps)
        if output.suffix.lower() == ".gif"
        else animation.FFMpegWriter(
            fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]
        )
    )
    video.save(output, writer=writer)
    plt.close(figure)

    sheet = output.with_name(f"{output.stem}.frames.png")
    figure = plt.figure(figsize=(13.5, 5.3), dpi=130, facecolor="white")
    for column, index in enumerate([0, len(frames) // 2, len(frames) - 1], start=1):
        axis = figure.add_subplot(1, 3, column, projection="3d")
        setup_axis(axis)
        draw_geometry(axis, index)
        draw_witnesses(axis, index)
        axis.set_title(f"t = {steps[index] * timestep:.3f} s", fontsize=11)
    figure.suptitle(
        f"{title}  |  {backend} / {route}", fontsize=15, fontweight="bold", y=0.98
    )
    figure.text(0.5, 0.08, metrics, ha="center", fontsize=10, color="#234965")
    figure.text(
        0.5,
        0.04,
        "Actual recorded physics poses; start / middle / final. Geometry colors do not alter collisions.",
        ha="center",
        fontsize=8,
        color="#667380",
    )
    figure.text(0.5, 0.012, marker_note, ha="center", fontsize=8, color="#667380")
    figure.subplots_adjust(left=0.03, right=0.93, top=0.86, bottom=0.28, wspace=0.14)
    figure.savefig(sheet)
    plt.close(figure)
    return {
        "animation": str(output),
        "contact_sheet": str(sheet),
        "playback_speed": playback_speed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--collision-colors", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            render_trace(
                args.trace,
                args.output,
                fps=args.fps,
                collision_colors=args.collision_colors,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
