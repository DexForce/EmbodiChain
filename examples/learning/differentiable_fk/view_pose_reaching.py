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

"""Preview a frozen FK-supervised Franka predictor against its initial posture.

Run: embodichain run-example learning/differentiable_fk/view_pose_reaching
Train with learn_pose_reaching first. This viewer performs CPU inference only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Any
import urllib.request

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import viser
from viser.extras import ViserUrdf
from viser.transforms import SO3

from embodichain.compute.kinematics import pose_error
from embodichain.data import get_data_path
from .learn_pose_reaching import _cpu_state, load_predictor

__all__ = ["serve"]


def _learning_curve(history: list[dict[str, float]]) -> go.Figure:
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.18)
    for row, name, label in (
        (1, "position_mean_mm", "Position / mm"),
        (2, "rotation_mean_deg", "Rotation / deg"),
    ):
        figure.add_trace(
            go.Scatter(
                x=[item["step"] for item in history],
                y=[item[name] for item in history],
                mode="lines+markers",
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(title_text=label, row=row, col=1)
    figure.update_xaxes(title_text="Training update", row=2, col=1)
    figure.update_layout(
        title="Independent validation",
        height=330,
        margin=dict(l=55, r=10, t=35, b=35),
        font=dict(size=10),
    )
    return figure


def serve(
    run_dir: Path, host: str = "127.0.0.1", port: int = 8103, smoke: bool = False
) -> None:
    """Compare held-out and edited TCP targets using a frozen CPU predictor.

    Args:
        run_dir: Training directory containing exported weights and evaluation.
        host: Local server bind address.
        port: Preferred HTTP port; Viser reports the actual selected port.
        smoke: Check scene construction, one target edit and HTTP, then stop.
    """
    torch.set_num_threads(4)
    metadata = json.loads((run_dir / "metrics.json").read_text())
    with np.load(run_dir / "evaluation.npz", allow_pickle=False) as saved:
        current_q, targets = saved["current_q"].copy(), saved["target_pose"].copy()
    model = load_predictor(run_dir)
    frozen = _cpu_state(model)
    urdf = Path(get_data_path("Franka/Panda/PandaWithHand.urdf"))
    server = viser.ViserServer(
        host=host, port=port, label="Learn reaching through differentiable FK"
    )
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/floor", width=2.5, height=3.2, cell_size=0.1, section_size=0.5
    )
    roots, robots, actual_frames, target_frames, lines, goal_labels = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for name, offset, color, title in (
        ("before", -0.65, (0.34, 0.55, 0.91), "Before training"),
        ("after", 0.65, (0.13, 0.68, 0.54), "After training"),
    ):
        root = "/" + name
        roots.append(
            server.scene.add_frame(root, show_axes=False, position=(0.0, offset, 0.0))
        )
        robots.append(
            ViserUrdf(server, urdf, root_node_name=root, mesh_color_override=color)
        )
        actual_frames.append(
            server.scene.add_frame(root + "/tcp", axes_length=0.06, axes_radius=0.002)
        )
        target_frames.append(
            server.scene.add_frame(
                root + "/target", axes_length=0.10, axes_radius=0.003
            )
        )
        lines.append(
            server.scene.add_line_segments(
                root + "/error",
                points=np.zeros((1, 2, 3), dtype=np.float32),
                colors=(244, 172, 55),
                line_width=4,
            )
        )
        goal_labels.append(server.scene.add_label(root + "/goal_label", "Goal TCP"))
        server.scene.add_label(root + "/title", title, position=(0.0, 0.0, 1.12))

    @server.on_client_connect
    def camera(client: viser.ClientHandle) -> None:
        client.camera.position = (2.4, 0.0, 1.5)
        client.camera.look_at = (0.25, 0.0, 0.45)
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.fov = float(np.deg2rad(45))

    server.gui.add_markdown(
        "## Reaching learned through FK\nA network predicts joints from **current joints + target pose**. Blue: zero-initialized output. Green: validation-selected model. Large axes mark the goal; yellow lines show position error."
    )
    case = server.gui.add_slider(
        "Test case", 1, max(2, len(current_q)), 1, 1, disabled=len(current_q) == 1
    )
    with server.gui.add_folder("Edit target", expand_by_default=True):
        translation = [
            server.gui.add_slider(f"Offset {axis} (m)", -0.05, 0.05, 0.001, 0.0)
            for axis in "XYZ"
        ]
        angles = [
            server.gui.add_slider(f"{name} (deg)", -15.0, 15.0, 1.0, 0.0)
            for name in ("Roll", "Pitch", "Yaw")
        ]
        reset = server.gui.add_button("Reset offsets")
    controls = translation + angles

    @reset.on_click
    def reset_offsets(_: viser.GuiEvent) -> None:
        for control in controls:
            control.value = 0.0

    status = server.gui.add_markdown("Loading prediction…")
    errors = server.gui.add_markdown("")
    server.gui.add_plotly(_learning_curve(metadata["history"]), aspect=1.1)
    summary = [
        f"### All {metadata['test_count']} held-out targets",
        "| Metric | Before | After |",
        "| --- | --- | --- |",
    ]
    for key, label in (
        ("position_mean_mm", "Position mean / mm"),
        ("position_p95_mm", "Position P95 / mm"),
        ("rotation_mean_deg", "Rotation mean / deg"),
        ("rotation_p95_deg", "Rotation P95 / deg"),
    ):
        summary.append(
            f"| {label} | {metadata['before'][key]:.3f} | {metadata['after'][key]:.3f} |"
        )
    summary.append(
        f"| Within 2 mm and 2 deg | {metadata['before']['success_rate']:.1%} | {metadata['after']['success_rate']:.1%} |"
    )
    server.gui.add_markdown("\n".join(summary))
    server.gui.add_markdown(
        "Weights stay frozen. Edited goals are outside the saved evaluation and may exceed the learned local region. This is a kinematic preview without collision, contact or physical execution."
    )

    def infer(snapshot: tuple[float, ...]) -> dict[str, Any]:
        index = int(snapshot[0]) - 1
        target = targets[index].copy()
        target[:3, 3] += np.asarray(snapshot[1:4], dtype=np.float32)
        rotation = SO3.from_rpy_radians(*np.deg2rad(snapshot[4:7])).as_matrix()
        target[:3, :3] = rotation @ target[:3, :3]
        q = torch.from_numpy(current_q[index : index + 1].copy())
        goal = torch.from_numpy(target[None])
        with torch.no_grad():
            predicted = model(q, goal)
            joints = torch.cat((q, predicted))
            poses = model.fk(joints)
            position, orientation = pose_error(poses, goal)
        if not torch.isfinite(joints).all():
            raise FloatingPointError("Non-finite frozen prediction")
        if ((joints < model.fk.lower - 1e-6) | (joints > model.fk.upper + 1e-6)).any():
            raise ValueError("Frozen prediction exceeds joint limits")
        return {
            "q": joints.numpy(),
            "poses": poses.numpy(),
            "target": target,
            "position_mm": position.norm(dim=-1).numpy() * 1000,
            "rotation_deg": torch.rad2deg(orientation.norm(dim=-1)).numpy(),
            "edited": any(value != 0 for value in snapshot[1:]),
            "case": index + 1,
        }

    def render(result: dict[str, Any]) -> None:
        for j, robot in enumerate(robots):
            lookup = dict(zip(model.fk.model.joint_names, result["q"][j]))
            robot.update_cfg(
                np.asarray(
                    [
                        lookup.get(name, 0.04)
                        for name in robot.get_actuated_joint_names()
                    ]
                )
            )
            actual_frames[j].position = result["poses"][j, :3, 3]
            actual_frames[j].wxyz = SO3.from_matrix(result["poses"][j, :3, :3]).wxyz
            target_frames[j].position = result["target"][:3, 3]
            target_frames[j].wxyz = SO3.from_matrix(result["target"][:3, :3]).wxyz
            lines[j].points = np.stack(
                (result["poses"][j, :3, 3], result["target"][:3, 3])
            )[None]
            goal_labels[j].position = result["target"][:3, 3] + np.array(
                [0.0, 0.0, 0.07]
            )
            roots[j].visible = True
        status.content = f"**{'Edited target · outside saved evaluation' if result['edited'] else 'Held-out test target'} {result['case']}**\n\nFrozen checkpoint: update {metadata['best_step']}."
        rows = ["| Current target | Before | After |", "| --- | --- | --- |"]
        for key, label in (
            ("position_mm", "Position / mm"),
            ("rotation_deg", "Rotation / deg"),
        ):
            rows.append(f"| {label} | {result[key][0]:.3f} | {result[key][1]:.3f} |")
        errors.content = "\n".join(rows)

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fk-preview")
    future: Future[dict[str, Any]] | None = None
    pending: tuple[int, tuple[float, ...]] | None = None
    requested: tuple[float, ...] | None = None
    revision = active_revision = 0
    previous_case = int(case.value)
    smoke_edited = False
    print(f"Preview: http://{host}:{server.get_port()}", flush=True)
    try:
        while True:
            if int(case.value) != previous_case:
                for control in controls:
                    control.value = 0.0
                previous_case = int(case.value)
            snapshot = (
                float(case.value),
                *(float(control.value) for control in controls),
            )
            if snapshot != requested:
                requested, revision = snapshot, revision + 1
                pending = (revision, snapshot)
                status.content = "Computing frozen prediction…"
                errors.content = ""
                for root in roots:
                    root.visible = False
            if future is not None and future.done():
                result = future.result()
                future = None
                if active_revision == revision:
                    render(result)
                    if smoke and not smoke_edited:
                        translation[0].value = 0.01
                        smoke_edited = True
                    elif smoke:
                        assert result["edited"]
                        assert all(robot.get_actuated_joint_names() for robot in robots)
                        assert not any(
                            parameter.requires_grad for parameter in model.parameters()
                        )
                        for name, value in _cpu_state(model).items():
                            if isinstance(value, torch.Tensor):
                                torch.testing.assert_close(
                                    value, frozen[name], atol=0, rtol=0
                                )
                            else:
                                assert value == frozen[name]
                        connect_host = "127.0.0.1" if host == "0.0.0.0" else host
                        with urllib.request.urlopen(
                            f"http://{connect_host}:{server.get_port()}", timeout=10
                        ) as response:
                            assert response.status == 200
                        print(
                            "Viewer smoke passed: robot scenes, target edit, bounded predictions, frozen weights and HTTP",
                            flush=True,
                        )
                        break
            if future is None and pending is not None:
                active_revision, work = pending
                pending = None
                future = executor.submit(infer, work)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        server.stop()


def main() -> None:
    """Open the frozen-model comparison, or run its bounded smoke check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/learn_fk_reaching"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8103)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    serve(args.run, args.host, args.port, args.smoke)


if __name__ == "__main__":
    main()
