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

"""Compare a stopped duration objective with autograd path optimization in Viser.

Run: embodichain run-example learning/differentiable_toppra/view_path_optimization
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
from threading import Lock
import time
from typing import Any
import urllib.request

import numpy as np
import plotly.graph_objects as go
import torch
import viser
from viser.extras import ViserUrdf

from .optimize_path import (
    ShowcaseCfg,
    ShowcaseRobot,
    Snapshot,
    check_gradient,
    load_run,
    optimize,
    save_run,
)

_COLORS = ("#568ce8", "#22ae89")


def _figure(title: str) -> go.Figure:
    result = go.Figure()
    result.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={"text": title, "font": {"size": 12}},
        font={"size": 10},
        margin={"l": 42, "r": 12, "t": 30, "b": 35},
        height=205,
    )
    return result


def _history(state: Snapshot) -> go.Figure:
    fig = _figure("Actual duration · accepted updates")
    fig.add_trace(
        go.Scatter(
            x=[x["iteration"] for x in state.history],
            y=[x["duration"] for x in state.history],
            mode="lines+markers",
            line={"color": _COLORS[1]},
            marker={"size": 4},
            name="Current duration",
        )
    )
    if state.before["duration"] is not None:
        fig.add_hline(
            y=state.before["duration"], line_dash="dot", line_color=_COLORS[0]
        )
    fig.update_xaxes(title="Accepted update")
    fig.update_yaxes(title="seconds")
    return fig


def _gradient(state: Snapshot) -> go.Figure:
    fig = _figure("Current free-node ∂T/∂q · s/rad")
    if state.gradient is None:
        fig.add_annotation(
            text=(
                "No autograd objective"
                if state.cfg.detach_duration
                else "Forward timing only"
            ),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    else:
        bound = max(float(np.abs(state.gradient).max()), 1e-12)
        fig.add_trace(
            go.Heatmap(
                z=state.gradient,
                x=[f"J{i+1}" for i in range(7)],
                y=[f"Node {i+1}" for i in np.flatnonzero(~state.fixed)],
                colorscale="RdBu_r",
                zmin=-bound,
                zmax=bound,
                colorbar={"thickness": 10, "title": "s/rad"},
            )
        )
    return fig


def _limits(state: Snapshot) -> go.Figure:
    fig = _figure("Max joint limit utilization · native samples")
    for name, row, color in (
        ("Initial", state.before, _COLORS[0]),
        ("Current", state.after, _COLORS[1]),
    ):
        times = row["dt"].cumsum()
        if not len(times):
            continue
        for key, limit, dash in (
            ("velocities", "velocity", "solid"),
            ("accelerations", "acceleration", "dash"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=(np.abs(row[key]) / row[limit]).max(-1),
                    name=name + " " + key,
                    line={"color": color, "dash": dash},
                )
            )
    fig.add_hline(y=1, line_color="#b1bac7", line_dash="dot")
    fig.update_layout(legend={"orientation": "h", "font": {"size": 8}})
    fig.update_xaxes(title="Physical time / s")
    fig.update_yaxes(title="utilization", range=[0, 1.08])
    return fig


def sample_index(row: dict, physical_time: float) -> int:
    """Hold the latest native sample, without interpolating a new trajectory."""
    if row["length"] and physical_time >= row["duration"]:
        return row["length"] - 1
    return int(
        np.clip(
            np.searchsorted(row["dt"].cumsum(), physical_time, side="right") - 1,
            0,
            max(0, row["length"] - 1),
        )
    )


def serve(
    host: str = "127.0.0.1",
    port: int = 8105,
    urdf: Path | None = None,
    run_dir: Path | None = None,
    output: Path | None = None,
    smoke: bool = False,
) -> None:
    """Open a CPU-only interactive gradient demonstration and optional replay.

    All numerical work and export run on one worker. Snapshots own their arrays;
    UI edits invalidate earlier generations without mutating an outstanding graph.
    """
    torch.set_num_threads(1)
    robot = ShowcaseRobot(urdf)
    cfg = (
        ShowcaseCfg(**json.loads((run_dir / "run.json").read_text())["config"])
        if run_dir
        else ShowcaseCfg()
    )
    export_root = output or Path("outputs/retiming_showcase") / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    server = viser.ViserServer(host=host, port=port, label="Retiming Lab")
    server.gui.configure_theme(
        control_layout="fixed",
        control_width="large",
        dark_mode=True,
        show_logo=False,
        show_share_button=False,
        brand_color=(34, 174, 137),
    )
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/floor",
        width=2.4,
        height=3.0,
        cell_size=0.1,
        section_size=0.5,
        shadow_opacity=0.15,
    )
    robots = []
    titles = []
    duration_labels = []
    paths = []
    nodes = []
    pins = []
    cursors = []
    for name, offset, color, title in (
        ("before", -0.65, (0.34, 0.55, 0.91), "INITIAL · STOPPED GRADIENT"),
        ("after", 0.65, (0.13, 0.68, 0.54), "AUTOGRAD · OPTIMIZED"),
    ):
        root = "/" + name
        server.scene.add_frame(root, show_axes=False, position=(0.0, offset, 0.0))
        robots.append(
            ViserUrdf(
                server, robot.urdf, root_node_name=root, mesh_color_override=color
            )
        )
        titles.append(
            server.scene.add_label(
                root + "/title",
                title,
                position=(0.0, 0.0, 1.20),
                anchor="center",
                font_screen_scale=1.1,
            )
        )
        duration_labels.append(
            server.scene.add_label(
                root + "/duration",
                "Preparing…",
                position=(0.0, 0.0, 1.10),
                anchor="center",
            )
        )
        rgb = (86, 140, 232) if name == "before" else (34, 174, 137)
        paths.append(
            server.scene.add_line_segments(
                root + "/path", np.zeros((1, 2, 3)), rgb, line_width=3
            )
        )
        nodes.append(
            server.scene.add_point_cloud(
                root + "/nodes",
                np.zeros((6, 3)),
                rgb,
                point_size=0.012,
                point_shape="circle",
                precision="float32",
            )
        )
        pins.append(
            server.scene.add_point_cloud(
                root + "/fixed",
                np.zeros((2, 3)),
                (242, 183, 77),
                point_size=0.021,
                point_shape="diamond",
                precision="float32",
            )
        )
        cursors.append(
            server.scene.add_frame(
                root + "/tcp", show_axes=False, origin_radius=0.013, origin_color=rgb
            )
        )

    @server.on_client_connect
    def camera(client: viser.ClientHandle) -> None:
        client.camera.position = (2.6, 0.0, 1.8)
        client.camera.look_at = (0.25, 0.0, 0.55)
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.fov = float(np.deg2rad(48))

    server.gui.add_markdown(
        "## Retiming Lab\n**Differentiate duration. Reshape the path.**\n\n`Q → TOPPRA → T → ∂T/∂Q` · fixed joint targets"
    )
    case = server.gui.add_dropdown(
        "Path case", ("detour", "via_joint", "stationary"), initial_value=cfg.case
    )
    mode = server.gui.add_dropdown(
        "Mode", ("optimize", "retime"), initial_value=cfg.mode
    )
    with server.gui.add_folder("Motion budgets", expand_by_default=False):
        vscale = server.gui.add_slider(
            "Velocity multiplier", 0.25, 1.0, 0.05, cfg.velocity_scale
        )
        ascale = server.gui.add_slider(
            "Acceleration multiplier", 0.25, 1.0, 0.05, cfg.acceleration_scale
        )
        joint = server.gui.add_dropdown(
            "Bottleneck",
            ("None", *(f"J{i+1}" for i in range(7))),
            initial_value="None" if cfg.bottleneck < 0 else f"J{cfg.bottleneck+1}",
        )
        jv = server.gui.add_slider(
            "Joint velocity factor", 0.25, 1.0, 0.05, cfg.bottleneck_velocity
        )
        ja = server.gui.add_slider(
            "Joint acceleration factor", 0.25, 1.0, 0.05, cfg.bottleneck_acceleration
        )
        server.gui.add_markdown(
            "Reference v ≤ 1 rad/s and URDF cap. Reference a = **2 rad/s² planning budget**."
        )
    detached = server.gui.add_checkbox(
        "Detach duration",
        cfg.detach_duration,
        hint="Disconnect the duration autograd objective; no path update",
    )
    buttons = server.gui.add_button_group("Update", ("Reset", "Step", "+30", "Cancel"))
    status = server.gui.add_markdown("Preparing a deterministic path…")
    metrics = server.gui.add_markdown("")
    play = server.gui.add_checkbox(
        "Play",
        False,
        hint="Shared physical seconds; the faster path holds at its endpoint",
    )
    clock = server.gui.add_slider("Time / seconds", 0.0, 1.0, 0.005, 0.0)
    speed = server.gui.add_dropdown(
        "Playback speed", ("0.25×", "0.5×", "1×", "2×"), initial_value="1×"
    )
    history_plot = server.gui.add_plotly(
        _figure("Duration · accepted updates"), aspect=0.55
    )
    tabs = server.gui.add_tab_group()
    with tabs.add_tab("Gradient"):
        gradient_plot = server.gui.add_plotly(_figure("∂T/∂q · free nodes"), aspect=0.6)
        check = server.gui.add_button("Check local derivative")
        derivative = server.gui.add_markdown(
            "Autograd versus centered finite differences; optimizer uses autograd only."
        )
    with tabs.add_tab("Limits"):
        limits_plot = server.gui.add_plotly(
            _figure("Native velocity / acceleration samples"), aspect=0.7
        )
    with tabs.add_tab("Export"):
        destination = server.gui.add_text("Output root", str(export_root))
        export = server.gui.add_button("Export current result")
        exported = server.gui.add_markdown(
            "JSON + NPZ + PNG. Existing exports are preserved."
        )
    server.gui.add_markdown(
        "Kinematic preview · sampled checks · local first-order descent. Fixed middle nodes need not appear on the uniform time grid."
    )

    lock = Lock()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="retiming-showcase")
    revision = 0
    pending: tuple | None = None
    progress: deque = deque(maxlen=1)
    future: Future | None = None
    current: Snapshot | None = None
    active_id = -1
    active_action = ""
    export_count = 0
    busy = False
    last_render = 0.0
    smoke_phase = 0
    discarded = 0
    started = time.monotonic()

    def controls(steps: int = 30) -> ShowcaseCfg:
        return ShowcaseCfg(
            case=case.value,
            mode=mode.value,
            detach_duration=detached.value,
            steps=steps,
            velocity_scale=float(vscale.value),
            acceleration_scale=float(ascale.value),
            bottleneck=-1 if joint.value == "None" else int(joint.value[1:]) - 1,
            bottleneck_velocity=float(jv.value),
            bottleneck_acceleration=float(ja.value),
        )

    def request(action: str = "optimize", steps: int = 30) -> None:
        nonlocal revision, pending, current, export_count
        play.value = False
        with lock:
            revision += 1
            progress.clear()
            if action == "cancel":
                pending = None
                if current is not None:
                    current = replace(
                        current,
                        status="cancelled",
                        reason="Cancelled at last displayed accepted state",
                    )
                status.content = "**Cancelled** · last displayed valid path retained"
                return
            cfg_now = controls(steps)
            resume = None
            if action == "continue" and current is not None:
                previous = asdict(current.cfg)
                requested = asdict(cfg_now)
                previous.pop("steps")
                requested.pop("steps")
                if previous == requested:
                    resume = current
                    cfg_now = controls(min(100, current.iteration + steps))
            target = None
            if action == "export":
                export_count += 1
                target = (
                    Path(destination.value)
                    / f"result_{export_count:03d}_{time.time_ns()}"
                )
            pending = (revision, action, cfg_now, resume, current, target)
            status.content = (
                "Updating… older results remain paused until this request finishes."
            )

    for control in (case, mode, vscale, ascale, joint, jv, ja, detached):
        control.on_update(lambda _: request())

    @buttons.on_click
    def update(event: viser.GuiEvent) -> None:
        action = event.target.value
        if action == "Cancel":
            request("cancel")
        elif action == "Reset":
            request(steps=0)
        elif action == "Step":
            request("continue", 1)
        else:
            request("continue", 30)

    @check.on_click
    def derivative_check(_: viser.GuiEvent) -> None:
        request("check")

    @export.on_click
    def export_result(_: viser.GuiEvent) -> None:
        request("export")

    def worker(job: tuple) -> Any:
        job_id, action, settings, resume, saved, target = job

        def cancelled() -> bool:
            with lock:
                return job_id != revision

        def publish(state: Snapshot) -> None:
            with lock:
                if job_id == revision:
                    progress.append((job_id, state))

        if action == "load":
            return load_run(robot, run_dir)
        if action == "check":
            return (
                {"status": "pending", "passed": None}
                if saved is None
                else check_gradient(robot, saved)
            )
        if action == "export":
            if saved is None:
                raise ValueError("No completed result to export")
            save_run(robot, saved, target)
            return target
        return optimize(
            robot, settings, resume=resume, cancelled=cancelled, publish=publish
        )

    def render(state: Snapshot, reset_clock: bool = False) -> None:
        nonlocal current, last_render
        current = state
        last_render = time.monotonic()
        before, after = state.before, state.after
        t0, t1 = before["duration"], after["duration"]
        gain = "—" if t0 is None or t1 is None or t0 == 0 else f"{100*(1-t1/t0):+.2f}%"
        disconnected = state.cfg.detach_duration
        titles[0].text = (
            "REFERENCE BUDGET"
            if state.cfg.mode == "retime"
            else "INITIAL · STOPPED GRADIENT"
        )
        titles[1].text = (
            "CURRENT BUDGET"
            if state.cfg.mode == "retime"
            else "DURATION DETACHED" if disconnected else "AUTOGRAD · OPTIMIZED"
        )
        for i, row in enumerate((before, after)):
            duration_labels[i].text = (
                "Solver failed"
                if row["duration"] is None
                else f"{row['duration']:.3f} seconds"
            )
            xyz = row["tcp"].astype(np.float32)
            paths[i].points = (
                np.stack((xyz[:-1], xyz[1:]), 1)
                if len(xyz) > 1
                else np.zeros((1, 2, 3), np.float32)
            )
            nodes[i].points = row["nodes_tcp"].astype(np.float32)
            pins[i].points = row["nodes_tcp"][state.fixed].astype(np.float32)
        fixed_error = float(
            np.abs(state.current[state.fixed] - state.initial[state.fixed]).max()
        )
        marker = "PASS" if after["valid"] else "FAIL"
        extra = ""
        if state.cfg.case == "via_joint" and after["length"]:
            closest = np.abs(after["positions"] - state.initial[3]).max(-1).min()
            extra = f"\nMiddle node: geometry fixed; closest output sample **{closest:.5f} rad**."
        metrics.content = (
            (
                f"| Initial | Current | Change |\n|---:|---:|---:|\n| {t0:.3f} s | {t1:.3f} s | **{gain}** |"
                if t0 is not None and t1 is not None
                else "Solver result unavailable"
            )
            + f"\n\n**{state.iteration} accepted** · {state.seconds:.2f} s compute · sampled checks **{marker}**\n\nFixed-node error: {fixed_error:.1e} rad."
            + extra
        )
        status.content = f"**{state.status.upper()}** · {state.reason}"
        history_plot.figure = _history(state)
        gradient_plot.figure = _gradient(state)
        limits_plot.figure = _limits(state)
        check.disabled = state.gradient is None
        if disconnected:
            derivative.content = "`duration.requires_grad = False` · no autograd update. This is not a claim that the mathematical derivative is zero."
        elif state.cfg.mode == "retime":
            derivative.content = (
                "Forward timing only · no autograd objective in this mode."
            )
        else:
            derivative.content = "Derivative not checked for the displayed path. Compare autograd with centered finite differences."
        maximum = max(t0 or 0.0, t1 or 0.0)
        clock.max = maximum if maximum > 0 else 1.0
        clock.disabled = maximum == 0
        play.disabled = maximum == 0 or not before["success"] or not after["success"]
        if reset_clock:
            clock.value = 0.0
        elif clock.value > maximum:
            clock.value = maximum

    def animate() -> None:
        state = current
        if state is None:
            return
        for i, row in enumerate((state.before, state.after)):
            if not row["length"]:
                continue
            index = sample_index(row, float(clock.value))
            lookup = dict(zip(robot.names, row["positions"][index]))
            robots[i].update_cfg(
                np.asarray(
                    [lookup.get(n, 0.04) for n in robots[i].get_actuated_joint_names()]
                )
            )
            cursors[i].position = row["tcp"][index]

    request("load" if run_dir else "optimize")
    previous_time = time.monotonic()
    print(f"Retiming Lab: http://{host}:{server.get_port()}", flush=True)
    try:
        while True:
            now = time.monotonic()
            if smoke and now - started > 60:
                raise TimeoutError("Viewer smoke exceeded 60 seconds")
            if future is not None and future.done():
                try:
                    result = future.result()
                    with lock, server.atomic():
                        if active_id == revision:
                            if isinstance(result, Snapshot):
                                render(result, True)
                            elif active_action == "check":
                                passed = result.get("passed")
                                badge = (
                                    "SKIPPED"
                                    if passed is None
                                    else "PASS" if passed else "NOT VERIFIED"
                                )
                                status.content = (
                                    f"**CHECKED** · local derivative {badge}"
                                )
                                ag, fd = result.get("autograd"), result.get(
                                    "finite_difference"
                                )
                                ag_text = "not evaluated" if ag is None else f"{ag:.6f}"
                                fd_text = "not evaluated" if fd is None else f"{fd:.6f}"
                                rel = result.get("relative_error")
                                detail = result.get(
                                    "reason", "Current local branch only"
                                )
                                if rel is not None:
                                    detail += f" · relative error {rel:.2e}"
                                derivative.content = (
                                    f"**Local derivative: {badge}**\n\n"
                                    f"| Method | s/rad |\n|---|---:|\n"
                                    f"| Autograd gᵀd | {ag_text} |\n"
                                    f"| Centered difference | {fd_text} |\n\n{detail}."
                                )
                            else:
                                status.content = "**EXPORTED** · current result saved"
                                exported.content = f"Saved **{result}**"
                        else:
                            discarded += 1
                except Exception as exc:
                    if active_id == revision:
                        status.content = f"**FAILED** · {exc}"
                    if smoke:
                        raise
                future = None
                busy = False
                with lock:
                    progress.clear()
            if now - last_render >= 0.2:
                with lock:
                    update = progress.pop() if progress else None
                with lock, server.atomic():
                    if update is not None and update[0] == revision:
                        render(update[1])
            if future is None:
                with lock:
                    job = pending
                    pending = None
                if job is not None:
                    active_id, active_action = job[:2]
                    future = executor.submit(worker, job)
                    busy = True
            buttons.visible = mode.value != "retime"
            detached.disabled = mode.value == "retime"
            check.disabled = busy or current is None or current.gradient is None
            export.disabled = busy or current is None
            if play.value and current is not None and not busy:
                maximum = max(
                    current.before["duration"] or 0.0, current.after["duration"] or 0.0
                )
                clock.value = min(
                    maximum,
                    float(clock.value)
                    + (now - previous_time) * float(speed.value[:-1]),
                )
                if clock.value >= maximum:
                    play.value = False
            previous_time = now
            animate()
            if smoke and smoke_phase == 1 and busy:
                detached.value = True
                smoke_phase = 2
            if smoke and not busy and pending is None and current is not None:
                if smoke_phase == 0:
                    assert (
                        current.after["valid"]
                        and current.after["duration"]
                        < 0.95 * current.before["duration"]
                    )
                    initial, _ = robot.initial("detour")
                    row = robot.row(
                        initial, robot.velocity * 0.25, robot.acceleration * 0.5
                    )
                    assert sample_index(row, row["duration"]) == row["length"] - 1
                    request("continue", 100)
                    smoke_phase = 1
                elif smoke_phase == 2 and current.cfg.detach_duration:
                    assert current.gradient is None and current.iteration == 0
                    np.testing.assert_array_equal(current.initial, current.current)
                    mode.value = "retime"
                    smoke_phase = 3
                elif smoke_phase == 3 and current.cfg.mode == "retime":
                    assert not buttons.visible
                    np.testing.assert_array_equal(current.current, current.initial)
                    mode.value = "optimize"
                    case.value = "stationary"
                    detached.value = False
                    smoke_phase = 4
                elif (
                    smoke_phase == 4
                    and current.cfg.case == "stationary"
                    and current.cfg.mode == "optimize"
                    and not current.cfg.detach_duration
                ):
                    assert (
                        current.after["length"] == 2 and current.after["duration"] == 0
                    )
                    assert clock.disabled
                    assert discarded >= 1
                    assert all(r.get_actuated_joint_names() for r in robots)
                    connect_host = "127.0.0.1" if host == "0.0.0.0" else host
                    with urllib.request.urlopen(
                        f"http://{connect_host}:{server.get_port()}", timeout=10
                    ) as response:
                        assert response.status == 200
                    save_run(robot, current, export_root / "viewer_smoke")
                    print(
                        "Viewer smoke passed: meshes, editing, detached/static branches, revision isolation, native playback, HTTP and export",
                        flush=True,
                    )
                    break
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        with lock:
            revision += 1
        executor.shutdown(wait=True, cancel_futures=True)
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8105)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    serve(args.host, args.port, args.urdf, args.run, args.output, args.smoke)


if __name__ == "__main__":
    main()
