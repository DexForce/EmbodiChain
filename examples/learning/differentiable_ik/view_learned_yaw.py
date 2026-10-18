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

"""Preview a frozen grasp-yaw network and optional online optimization reference.

Run: python -m examples.learning.differentiable_ik.view_learned_yaw
The viewer never trains. Target edits run inference on one CPU worker; a reference
search is optional and explicitly requested. Stale results cannot replace inputs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from queue import Empty, SimpleQueue
import time
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
import trimesh
import viser
from viser.extras import ViserUrdf

from embodichain.compute.kinematics import DifferentiableIKCfg
from embodichain.data import get_data_path
from embodichain.learning.layers import DifferentiableIKLayer
from .learn_grasp_yaw import (
    METHODS,
    SPLITS,
    current_heading,
    evaluate_angles,
    grid_reference,
    load_predictors,
    optimize_task_angles,
)

_NAMES = ("Keep current heading", "Learned grasp yaw", "Online reference")
_MODEL_NAMES = {
    "learned": "IK-trained network",
    "supervised": "Teacher-supervised network",
    "constant": "Shared constant yaw",
}
_COLORS = ("#568ce8", "#22ae89", "#e4a547")


def _rotation(yaw: float) -> np.ndarray:
    return np.array([0.0, np.cos(yaw / 2), np.sin(yaw / 2), 0.0])


def _curve(data: dict[str, Any], method: str) -> go.Figure:
    figure = go.Figure()
    if data["reference_ready"]:
        figure.add_trace(
            go.Scatter(
                x=np.rad2deg(data["grid_angles"]),
                y=data["grid_cost"][0],
                mode="lines",
                line={"color": "#a6aebc"},
                name="Sampled reference",
                showlegend=False,
            )
        )
    for j, name in enumerate(("heading", method, "online")):
        i = METHODS.index(name)
        if data["valid"][0, i] and (name != "online" or data["reference_ready"]):
            figure.add_trace(
                go.Scatter(
                    x=[np.rad2deg(data["angle"][0, i])],
                    y=[data["cost"][0, i]],
                    mode="markers",
                    marker={"color": _COLORS[j], "size": 10},
                    name=name,
                    showlegend=False,
                )
            )
    figure.update_layout(
        height=240,
        margin=dict(l=45, r=8, t=25, b=40),
        title="Yaw → joint displacement²",
        font={"size": 10},
    )
    figure.update_xaxes(title="Yaw / degrees", range=[-180, 180])
    figure.update_yaxes(title="rad²")
    return figure


def _learning_curve(metadata: dict[str, Any]) -> go.Figure:
    figure = go.Figure()
    for run in metadata["training"]:
        if run["seed"] == metadata["selected_viewer_seed"]:
            figure.add_trace(
                go.Scatter(
                    x=[x["step"] for x in run["history"]],
                    y=[x["validation_loss"] for x in run["history"]],
                    mode="lines+markers",
                    name=run["method"],
                )
            )
    figure.update_layout(
        height=240,
        margin=dict(l=45, r=8, t=25, b=40),
        title="Training · independent validation",
        font={"size": 10},
        legend={"orientation": "h", "y": -0.3},
    )
    figure.update_xaxes(title="Training update")
    figure.update_yaxes(title="Task loss")
    return figure


def serve(
    run_dir: Path, host: str = "127.0.0.1", port: int = 8101, smoke: bool = False
) -> None:
    """Show held-out cases and evaluate user edits using frozen checkpoints."""
    metadata = json.loads((run_dir / "metrics.json").read_text())
    with np.load(run_dir / "replay.npz", allow_pickle=False) as source:
        archive = {key: source[key] for key in source.files}
    torch.set_num_threads(4)
    k, models, cfg = load_predictors(run_dir, "cpu")
    inner = DifferentiableIKLayer(k, DifferentiableIKCfg(max_iterations=cfg.ik_steps))
    check = DifferentiableIKLayer(
        k, DifferentiableIKCfg(max_iterations=cfg.check_steps)
    )
    urdf = Path(get_data_path("Franka/Panda/PandaWithHand.urdf"))
    server = viser.ViserServer(
        host=host, port=port, label="Learned grasp yaw · differentiable IK"
    )
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/floor", width=3.5, height=4.0, cell_size=0.1, section_size=0.5
    )
    robots, objects, goals, pedestals, legs, titles = [], [], [], [], [], []
    cylinder = trimesh.creation.cylinder(radius=0.025, height=0.12, sections=48)
    cylinder.visual.face_colors = [239, 196, 81, 255]
    for j, (name, offset, color) in enumerate(
        zip(
            ("heading", "learned", "reference"),
            (-0.95, 0.0, 0.95),
            ((0.34, 0.55, 0.91), (0.13, 0.68, 0.54), (0.89, 0.65, 0.28)),
        )
    ):
        root = "/" + name
        server.scene.add_frame(root, show_axes=False, position=(0.0, offset, 0.0))
        robots.append(
            ViserUrdf(server, urdf, root_node_name=root, mesh_color_override=color)
        )
        objects.append(server.scene.add_mesh_trimesh(root + "/cylinder", cylinder))
        goals.append(
            server.scene.add_frame(
                root + "/grasp", axes_length=0.065, axes_radius=0.0025
            )
        )
        pedestals.append(
            server.scene.add_box(
                root + "/support_top",
                color=(218, 224, 233),
                dimensions=(0.2, 0.2, 0.03),
            )
        )
        legs.append(
            server.scene.add_box(
                root + "/support_leg",
                color=(198, 207, 219),
                dimensions=(0.06, 0.06, 0.1),
            )
        )
        titles.append(
            server.scene.add_label(
                root + "/title", _NAMES[j], position=(0.0, 0.0, 1.12)
            )
        )

    @server.on_client_connect
    def camera(client: viser.ClientHandle) -> None:
        client.camera.position = (2.8, 0.0, 1.55)
        client.camera.look_at = (0.25, 0.0, 0.45)
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.fov = float(np.deg2rad(45))

    commands: SimpleQueue[str] = SimpleQueue()
    server.gui.add_markdown(
        "## Learned grasp orientation\nA frozen network chooses yaw from **current joints + grasp position**. Position and downward approach stay fixed."
    )
    split_control = server.gui.add_dropdown(
        "Held-out split", SPLITS, initial_value="iid"
    )
    per_split = int((archive["split"] == "iid").sum())
    case_control = server.gui.add_slider(
        "Case", 1, max(2, per_split), 1, 1, disabled=per_split == 1
    )
    model_control = server.gui.add_dropdown(
        "Model", tuple(_MODEL_NAMES), initial_value="learned"
    )
    show_current = server.gui.add_checkbox("Show current posture", False)
    with server.gui.add_folder("Move cylinder", expand_by_default=False):
        offsets = [
            server.gui.add_slider(f"Offset {axis} (m)", -0.12, 0.12, 0.005, 0.0)
            for axis in "XYZ"
        ]
        reset = server.gui.add_button("Reset target")
    reference_button = server.gui.add_button("Compute online reference", color="orange")
    reference_button.on_click(lambda _: commands.put("reference"))
    reset.on_click(lambda _: commands.put("reset"))
    status = server.gui.add_markdown("")
    metrics = server.gui.add_markdown("")
    server.gui.add_plotly(_learning_curve(metadata), aspect=1.0)
    with server.gui.add_folder("Yaw reference curve", expand_by_default=False):
        curve = server.gui.add_plotly(go.Figure(), aspect=1.0)
    info = server.gui.add_markdown("")
    server.gui.add_markdown(
        "**Kinematic candidates.** No collision/contact/dynamics validation. Joint endpoint displacement is not energy or execution time."
    )

    def saved_case(index: int) -> dict[str, Any]:
        data = {
            key: (
                value.copy()
                if key == "grid_angles"
                else value[index : index + 1].copy()
            )
            for key, value in archive.items()
        }
        data["reference_ready"] = True
        return data

    def compute(
        q_array: np.ndarray, p_array: np.ndarray, with_reference: bool
    ) -> dict[str, Any]:
        start = time.perf_counter()
        q = torch.as_tensor(q_array, dtype=torch.float32).reshape(1, 7)
        p = torch.as_tensor(p_array, dtype=torch.float32).reshape(1, 3)
        with torch.no_grad():
            heading = current_heading(k, q)
            d = {"q": q, "p": p, "heading": heading}
            angles = [
                heading,
                *(
                    models[name](q, p, heading)
                    for name in ("constant", "learned", "supervised")
                ),
            ]
            repeated = {
                key: value.expand(4, *value.shape[1:]) for key, value in d.items()
            }
            result = evaluate_angles(check, repeated, torch.cat(angles))
        # Unknown references display the common starting posture, not stale results.
        values = {}
        for key, value in result.items():
            if key == "q":
                missing = q.expand(2, -1)
            elif key == "valid":
                missing = torch.zeros(2, dtype=torch.bool)
            else:
                missing = torch.full((2,), float("nan"), dtype=value.dtype)
            values[key] = torch.cat((value, missing)).unsqueeze(0)
        ready = False
        grid_angles = torch.linspace(-torch.pi, torch.pi, cfg.grid_points)
        grid_cost = torch.full((1, cfg.grid_points), float("nan"))
        found = torch.zeros(1, dtype=torch.bool)
        if with_reference:
            angle, found = optimize_task_angles(
                inner, d, cfg.reference_updates, cfg.chunk_size
            )
            online = evaluate_angles(check, d, angle)
            grid, grid_angles, grid_cost = grid_reference(
                check, d, cfg.grid_points, cfg.grid_chunk_size
            )
            for key in values:
                values[key][:, 4] = online[key]
                values[key][:, 5] = grid[key]
            ready = True
        result = {key: value.numpy() for key, value in values.items()}
        result.update(
            current_q=q.numpy(),
            positions=p.numpy(),
            grid_angles=grid_angles.numpy(),
            grid_cost=grid_cost.numpy(),
            reference_found=found.numpy(),
            reference_ready=ready,
            seconds=time.perf_counter() - start,
        )
        return result

    executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="grasp-yaw-inference"
    )
    future = None
    generation, submitted_generation, revision = 0, -1, 0
    selected = -1
    last_input = None
    last_view = None
    pending_prediction = False
    failed_generation = -1
    active = saved_case(0)
    try:
        print(f"Learned-yaw preview: http://{host}:{server.get_port()}", flush=True)
        while True:
            indices = np.flatnonzero(archive["split"] == split_control.value)
            selected_now = int(
                indices[min(int(case_control.value) - 1, len(indices) - 1)]
            )
            if selected_now != selected:
                selected = selected_now
                for handle in offsets:
                    handle.value = 0.0
            inputs = (
                selected,
                model_control.value,
                tuple(float(h.value) for h in offsets),
            )
            if inputs != last_input:
                generation += 1
                failed_generation = -1
                revision += 1
                last_input = inputs
                if inputs[2] == (0.0, 0.0, 0.0):
                    active = saved_case(selected)
                    pending_prediction = False
                    status.content = "Saved held-out task. Model weights are frozen."
                else:
                    pending_prediction = True
                    status.content = "New target: running frozen-network inference…"
            position = archive["positions"][selected] + np.asarray(
                inputs[2], dtype=np.float32
            )
            with_reference = False
            while True:
                try:
                    command = commands.get_nowait()
                except Empty:
                    break
                if command == "reset":
                    for handle in offsets:
                        handle.value = 0.0
                elif command == "reference" and future is None:
                    with_reference = True
                    failed_generation = -1
            if future is not None and future.done():
                computed = None
                try:
                    computed = future.result()
                except Exception as exc:
                    if generation == submitted_generation:
                        failed_generation = generation
                        status.content = (
                            f"Inference failed: {type(exc).__name__}: {exc}"
                        )
                if generation == submitted_generation and computed is not None:
                    active = computed
                    revision += 1
                    pending_prediction = False
                    status.content = f"{'Prediction + online reference' if active['reference_ready'] else 'Frozen prediction'}: {active['seconds']:.2f}s. No training."
                    np.savez_compressed(
                        run_dir / "last_interactive.npz",
                        **{
                            key: value
                            for key, value in active.items()
                            if isinstance(value, np.ndarray)
                        },
                    )
                elif generation != submitted_generation:
                    status.content = "Inputs changed; stale computation discarded."
                future = None
            if future is None and (
                (pending_prediction and failed_generation != generation)
                or with_reference
            ):
                submitted_generation = generation
                future = executor.submit(
                    compute,
                    archive["current_q"][selected].copy(),
                    position.copy(),
                    with_reference,
                )
                status.content = (
                    "Computing numerical reference…"
                    if with_reference
                    else "Predicting with frozen networks…"
                )
            reference_button.disabled = future is not None
            key = (inputs, bool(show_current.value), pending_prediction, revision)
            if key != last_view:
                method = model_control.value
                indices_to_show = (0, METHODS.index(method), 4)
                with server.atomic():
                    titles[1].text = _MODEL_NAMES[method]
                    for j, method_index in enumerate(indices_to_show):
                        use_current = (
                            pending_prediction
                            or bool(show_current.value)
                            or (j == 2 and not active["reference_ready"])
                        )
                        q = (
                            archive["current_q"][selected]
                            if use_current
                            else active["q"][0, method_index]
                        )
                        mapping = dict(zip(metadata["joint_names"], q))
                        robots[j].update_cfg(
                            np.array(
                                [
                                    mapping.get(name, 0.03 if "finger" in name else 0.0)
                                    for name in robots[j].get_actuated_joint_names()
                                ]
                            )
                        )
                        objects[j].position = position + np.array([0.0, 0.0, -0.035])
                        goals[j].position = position
                        angle = active["angle"][0, method_index]
                        goals[j].visible = bool(
                            not pending_prediction and np.isfinite(angle)
                        )
                        if np.isfinite(angle):
                            goals[j].wxyz = _rotation(float(angle))
                        pedestals[j].position = position + np.array([0.0, 0.0, -0.11])
                        height = max(0.01, float(position[2] - 0.125))
                        legs[j].dimensions = (0.06, 0.06, height)
                        legs[j].position = (
                            float(position[0]),
                            float(position[1]),
                            height / 2,
                        )
                    if pending_prediction:
                        metrics.content = "Updating target; showing current joints."
                        curve.visible = False
                        info.content = ""
                    else:
                        lines = [
                            "| Method | Cost (rad²) | Valid |",
                            "| --- | ---: | :---: |",
                        ]
                        for label, i in zip(
                            ("Keep heading", method, "Online reference"),
                            indices_to_show,
                        ):
                            known = i != 4 or active["reference_ready"]
                            cost = f"{active['cost'][0,i]:.3f}" if known else "pending"
                            valid = (
                                ("✓" if active["valid"][0, i] else "FAIL")
                                if known
                                else "—"
                            )
                            lines.append(f"| {label} | {cost} | {valid} |")
                        metrics.content = "\n".join(lines)
                        curve.visible = True
                        curve.figure = _curve(active, method)
                        i = METHODS.index(method)
                        valid_pair = (
                            active["valid"][0, 0]
                            and active["valid"][0, i]
                            and active["cost"][0, 0] > 1e-9
                        )
                        reduction = (
                            100 * (1 - active["cost"][0, i] / active["cost"][0, 0])
                            if valid_pair
                            else None
                        )
                        quality = (
                            f"{reduction:+.1f}% lower cost than heading."
                            if reduction is not None
                            else "No paired-valid improvement statistic."
                        )
                        info.content = f"**{quality}**\n\nYaw: {np.rad2deg(active['angle'][0,i]):.1f}°. IK residual: {active['position_mm'][0,i]:.3g} mm / {np.rad2deg(active['rotation_rad'][0,i]):.3g}°.\n\nFirst seed {metadata['selected_viewer_seed']}; checkpoint selected on validation. Edited targets are outside the saved evaluation."
                        if (
                            active["reference_ready"]
                            and not active["reference_found"][0]
                        ):
                            info.content += "\n\nNo feasible inner-budget reference candidate; fallback is flagged."
                last_view = key
            if smoke:
                before = {
                    name: {
                        key: value.clone() for key, value in model.state_dict().items()
                    }
                    for name, model in models.items()
                }
                trial = compute(
                    archive["current_q"][0],
                    archive["positions"][0]
                    + np.array([0.01, 0.0, 0.0], dtype=np.float32),
                    False,
                )
                assert np.isfinite(trial["q"][:, :4]).all()
                for name, model in models.items():
                    assert not any(p.requires_grad for p in model.parameters())
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(
                            value, before[name][key], atol=0, rtol=0
                        )
                import urllib.request

                with urllib.request.urlopen(
                    f"http://{host}:{server.get_port()}"
                ) as response:
                    assert response.status == 200
                print(
                    "Viewer smoke passed: meshes, frozen predictions, unchanged weights and HTTP",
                    flush=True,
                )
                break
            time.sleep(0.025)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/learn_grasp_yaw"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    serve(args.run, args.host, args.port, args.smoke)


if __name__ == "__main__":
    main()
