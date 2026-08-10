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

"""Gradio layout and event bindings for the engine workspace.

The workflow layer supplies all callbacks; this module only owns presentation
and wires components to those callbacks.
"""

from __future__ import annotations

import gradio as gr

from app_asset_engine import build_asset_engine_panel, cleanup_asset_engine_session
from app_config import (
    DEBUG_ENGINE_ACTION,
    DEBUG_ENGINE_ASSET,
    DEBUG_ENGINE_SCENE,
    DEBUG_ENGINES,
    DEFAULT_ROBOT_PROFILE,
    DEXFORCE_LOGO,
    LANGUAGE_EN,
    ROBOT_PROFILES,
    UI_TEXT,
)
from app_processes import get_request_session_id
from app_workflows import (
    cleanup_workflow_session,
    format_status,
    preview_saved_scene,
    refresh_saved_scenes,
    reset_scene_engine,
    run_action_engine_from_current,
    run_scene_engine,
    stop_action_engine,
    ui_snapshot,
)

__all__ = ["build_app"]


def select_engine(selected_engine: str):
    """Show the selected engine panel without starting a pipeline."""
    button_updates = tuple(
        gr.update(variant="primary" if engine == selected_engine else "secondary")
        for engine, _ in DEBUG_ENGINES
    )
    return (
        *button_updates,
        gr.update(visible=selected_engine == DEBUG_ENGINE_ASSET),
        gr.update(visible=selected_engine == DEBUG_ENGINE_SCENE),
        gr.update(visible=selected_engine == DEBUG_ENGINE_ACTION),
    )


def action_engine_snapshot(request: gr.Request) -> tuple[object, ...]:
    """Adapt this session's runtime snapshot to the Action status widgets."""
    session_id = get_request_session_id(request)
    video, task, progress, status, _initial, _edited, _objects = ui_snapshot(session_id)
    return video, task, progress, status


def run_action_engine_panel(
    task_text: str,
    robot_profile: str | None,
    request: gr.Request,
) -> tuple[object, ...]:
    """Run the Action engine and return its latest UI snapshot."""
    video, task, progress, status, _initial, _edited, _objects = (
        run_action_engine_from_current(task_text, robot_profile, request)
    )
    return video, task, progress, status


def cleanup_app_session(request: gr.Request) -> None:
    """Stop every engine process owned by a disconnected Gradio session."""
    cleanup_workflow_session(request)
    cleanup_asset_engine_session(request)


def build_app() -> gr.Blocks:
    """Build the engine-only Gradio application."""
    with gr.Blocks(title="EmbodiChain Gradio") as app:
        if DEXFORCE_LOGO.is_file():
            with gr.Row(equal_height=True):
                gr.Image(
                    value=str(DEXFORCE_LOGO),
                    show_label=False,
                    container=False,
                    height=58,
                    width=183,
                )

        with gr.Row():
            asset_engine_button = gr.Button("Asset_engine", variant="primary")
            scene_engine_button = gr.Button("Scene_engine", variant="secondary")
            action_engine_button = gr.Button("Action_engine", variant="secondary")

        asset_engine = build_asset_engine_panel()
        with gr.Column(visible=False) as scene_engine_panel:
            gr.Markdown(
                "## Scene engine\n"
                "Upload one image to generate a Scene Engine export. "
                "The resulting Viser page is shown below."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    scene_image = gr.Image(
                        label=UI_TEXT[LANGUAGE_EN]["input_image"],
                        sources=["upload", "webcam"],
                        type="filepath",
                        format="png",
                        height=300,
                    )
                    with gr.Row():
                        scene_run = gr.Button("Generate scene", variant="primary")
                        scene_reset = gr.Button("Reset Scene Engine", variant="stop")
                with gr.Column(scale=2):
                    scene_progress = gr.Slider(
                        0,
                        100,
                        value=0,
                        step=1,
                        label=UI_TEXT[LANGUAGE_EN]["progress"],
                        interactive=False,
                    )
                    scene_status = gr.Markdown(format_status("Idle."))
                    scene_output = gr.Textbox(
                        label="Scene output directory (hash-named)",
                        interactive=False,
                    )
                    scene_preview = gr.HTML(
                        "<div style='padding: 1rem; color: #6b7280;'>"
                        "The Viser preview will appear here after generation."
                        "</div>"
                    )

        with gr.Column(visible=False) as action_engine_panel:
            gr.Markdown(
                "## Action engine\n"
                "Select a generated Scene Engine export to inspect it in Viser. "
                "Scene selection is currently independent from DexSim execution."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    action_scene_list = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Generated scenes",
                        info="Complete scenes stored under .gen_sim/scenes.",
                    )
                    action_scene_refresh = gr.Button("Refresh scenes")
                    action_scene_status = gr.Markdown(
                        "**Scene list:** open Action engine or refresh to load scenes."
                    )
                    action_task = gr.Textbox(
                        label="Task description",
                        placeholder="e.g. Put the bottle on the table",
                    )
                    action_robot = gr.Radio(
                        choices=ROBOT_PROFILES,
                        value=DEFAULT_ROBOT_PROFILE,
                        label=UI_TEXT[LANGUAGE_EN]["robot"],
                    )
                    with gr.Row():
                        action_run = gr.Button("Run DexSim", variant="primary")
                        action_stop = gr.Button(
                            "Stop Action Engine",
                            variant="stop",
                        )
                with gr.Column(scale=2):
                    action_scene = gr.HTML(
                        "<div style='padding: 1rem; color: #6b7280;'>"
                        "Select a generated scene to preview it."
                        "</div>"
                    )
                    action_video = gr.Video(
                        label=UI_TEXT[LANGUAGE_EN]["single_video_preview"],
                        height=320,
                        autoplay=True,
                        loop=True,
                    )
                    action_current_task = gr.Textbox(
                        label=UI_TEXT[LANGUAGE_EN]["current_task"],
                        interactive=False,
                    )
                    action_progress = gr.Slider(
                        0,
                        100,
                        value=0,
                        step=1,
                        label=UI_TEXT[LANGUAGE_EN]["progress"],
                        interactive=False,
                    )
                    action_status = gr.Markdown(
                        format_status("Load or generate a scene first.")
                    )
                    action_refresh_timer = gr.Timer(2.0)

        for engine, button in zip(
            (engine for engine, _label in DEBUG_ENGINES),
            (asset_engine_button, scene_engine_button, action_engine_button),
        ):
            button.click(
                select_engine,
                inputs=[gr.State(engine)],
                outputs=[
                    asset_engine_button,
                    scene_engine_button,
                    action_engine_button,
                    asset_engine["panel"],
                    scene_engine_panel,
                    action_engine_panel,
                ],
                queue=False,
            )

        action_engine_button.click(
            refresh_saved_scenes,
            inputs=[action_scene_list],
            outputs=[action_scene_list, action_scene_status],
            queue=False,
        )

        scene_run.click(
            run_scene_engine,
            inputs=[scene_image],
            outputs=[scene_progress, scene_status, scene_output, scene_preview],
        )
        scene_reset.click(
            reset_scene_engine,
            outputs=[
                scene_image,
                scene_progress,
                scene_status,
                scene_output,
                scene_preview,
            ],
            queue=False,
        )
        action_scene_refresh.click(
            refresh_saved_scenes,
            inputs=[action_scene_list],
            outputs=[action_scene_list, action_scene_status],
            queue=False,
        )
        action_scene_list.change(
            preview_saved_scene,
            inputs=[action_scene_list],
            outputs=[action_scene, action_scene_status],
        )
        action_run.click(
            run_action_engine_panel,
            inputs=[action_task, action_robot],
            outputs=[
                action_video,
                action_current_task,
                action_progress,
                action_status,
            ],
        )
        action_stop.click(
            stop_action_engine,
            outputs=[
                action_scene,
                action_scene_status,
                action_video,
                action_current_task,
                action_progress,
                action_status,
            ],
            queue=False,
        )
        action_refresh_timer.tick(
            action_engine_snapshot,
            outputs=[
                action_video,
                action_current_task,
                action_progress,
                action_status,
            ],
            queue=False,
        )

        app.unload(cleanup_app_session)

    return app
