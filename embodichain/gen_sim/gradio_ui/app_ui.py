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

"""Gradio layout and event bindings.

The workflow layer supplies all callbacks; this module only owns presentation
and wires components to those callbacks.
"""

from __future__ import annotations

from app_workflows import *  # noqa: F401,F403 - callbacks/constants form the UI contract.
from app_asset_engine import build_asset_engine_panel


def select_application_mode(selected_mode: str | None):
    """Switch between the full product UI and the focused engine UI."""
    is_debug = selected_mode == APP_MODE_DEBUG
    debug_css = (
        "<style>.demo-only { display: none !important; }</style>" if is_debug else ""
    )
    return (
        gr.update(variant="secondary" if is_debug else "primary"),
        gr.update(variant="primary" if is_debug else "secondary"),
        gr.update(visible=is_debug),
        gr.update(value=debug_css),
        APP_MODE_DEBUG if is_debug else APP_MODE_DEMO,
        gr.update(visible=is_debug),
    )


def select_debug_engine(selected_engine: str):
    """Expose an explicit active state without starting any pipeline."""
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


def action_engine_snapshot():
    """Adapt the shared runtime snapshot to the five Action-engine widgets."""
    video, task, progress, status, initial, edited, _objects = ui_snapshot()
    return video, task, progress, status, initial or edited


def run_action_engine_panel(task_text: str, robot_profile: str | None):
    run_action_engine_from_current(task_text, robot_profile)
    return action_engine_snapshot()


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="EmbodiChain Gradio") as demo:
        app_mode = gr.State(APP_MODE_DEMO)
        run_mode = gr.State(TOP_MODE_INTERACT)
        action_mode = gr.State(None)
        language = gr.State(LANGUAGE_EN)
        last_seen_input_revision = gr.State(0)
        interact_prebuilt_scene_dir = gr.State(None)
        mode_style = gr.HTML(value="", visible=True)
        with gr.Row():
            demo_mode_button = gr.Button("Demo", variant="primary")
            debug_mode_button = gr.Button("Debug", variant="secondary")
        with gr.Row(visible=False) as debug_controls:
            asset_engine_button = gr.Button("Asset_engine", variant="primary")
            scene_engine_button = gr.Button("Scene_engine", variant="secondary")
            action_engine_button = gr.Button("Action_engine", variant="secondary")
        with gr.Column(visible=False) as debug_engine_area:
            asset_engine = build_asset_engine_panel()
            with gr.Column(visible=False) as scene_engine_panel:
                gr.Markdown(
                    "## Scene engine\n"
                    "Upload one image to generate a Scene Engine export. "
                    "The resulting Viser page is shown below."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        debug_scene_image = gr.Image(
                            label=UI_TEXT[LANGUAGE_EN]["input_image"],
                            sources=["upload", "webcam"],
                            type="filepath",
                            format="png",
                            height=300,
                        )
                        debug_scene_run = gr.Button("Generate scene", variant="primary")
                    with gr.Column(scale=2):
                        debug_scene_progress = gr.Slider(
                            0,
                            100,
                            value=0,
                            step=1,
                            label=UI_TEXT[LANGUAGE_EN]["progress"],
                            interactive=False,
                        )
                        debug_scene_status = gr.Markdown(format_status("Idle."))
                        debug_scene_output = gr.Textbox(
                            label="Scene output directory (hash-named)",
                            interactive=False,
                        )
                        debug_scene_preview = gr.HTML(
                            "<div style='padding: 1rem; color: #6b7280;'>"
                            "The Viser preview will appear here after generation."
                            "</div>"
                        )
            with gr.Column(visible=False) as action_engine_panel:
                gr.Markdown(
                    "## Action engine\nUses the Gym scene produced by Scene engine (not merely a rendered GLB), then generates the action config and launches DexSim. This retains collisions, poses and physics metadata required by simulation."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        debug_action_task = gr.Textbox(
                            label="Task description",
                            placeholder="e.g. Put the bottle on the table",
                        )
                        debug_action_robot = gr.Radio(
                            choices=ROBOT_PROFILES,
                            value=DEFAULT_ROBOT_PROFILE,
                            label=UI_TEXT[LANGUAGE_EN]["robot"],
                        )
                        debug_action_load = gr.Button("Load current scene")
                        debug_action_run = gr.Button("Run DexSim", variant="primary")
                    with gr.Column(scale=2):
                        debug_action_scene = gr.Model3D(
                            label="Input Gym scene preview",
                            height=420,
                            clear_color=(0.94, 0.94, 0.94, 1.0),
                        )
                        debug_action_video = gr.Video(
                            label=UI_TEXT[LANGUAGE_EN]["single_video_preview"],
                            height=320,
                            autoplay=True,
                            loop=True,
                        )
                        debug_action_current_task = gr.Textbox(
                            label=UI_TEXT[LANGUAGE_EN]["current_task"],
                            interactive=False,
                        )
                        debug_action_progress = gr.Slider(
                            0,
                            100,
                            value=0,
                            step=1,
                            label=UI_TEXT[LANGUAGE_EN]["progress"],
                            interactive=False,
                        )
                        debug_action_status = gr.Markdown(
                            format_status("Load or generate a scene first.")
                        )
                        debug_action_refresh_timer = gr.Timer(2.0)
        with gr.Row(equal_height=True, elem_classes="demo-only"):
            if DEXFORCE_LOGO.is_file():
                gr.Image(
                    value=str(DEXFORCE_LOGO),
                    show_label=False,
                    container=False,
                    height=58,
                    width=183,
                )
            heading = gr.Markdown(UI_TEXT[LANGUAGE_EN]["heading"])
        with gr.Row(elem_classes="demo-only"):
            auto_button = gr.Button("Auto", variant="secondary")
            interact_button = gr.Button("Interact", variant="primary")
            parallel_env_button = gr.Button("Parallel Simulation", variant="secondary")
            language_button = gr.Button("中文", variant="secondary")
        with gr.Row(elem_classes="demo-only"):
            with gr.Column(scale=4):
                instruction = gr.HTML(
                    "<div style='font-size: 20px; font-weight: 700; "
                    "line-height: 1.35; min-height: 86px; display: flex; "
                    "align-items: center;'>"
                    "Upload one image, enter one task, then EmbodiChain "
                    " will generate what you want."
                    "</div>"
                )
            with gr.Column(scale=1):
                robot_profile = gr.Radio(
                    choices=ROBOT_PROFILES,
                    value=DEFAULT_ROBOT_PROFILE,
                    label=UI_TEXT[LANGUAGE_EN]["robot"],
                )

        with gr.Row(elem_classes="demo-only"):
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label=UI_TEXT[LANGUAGE_EN]["input_image"],
                    sources=["upload", "webcam"],
                    type="filepath",
                    format="png",
                    height=320,
                )
                with gr.Row():
                    with gr.Column():
                        task_input = gr.Textbox(
                            label=UI_TEXT[LANGUAGE_EN]["task_description"],
                            placeholder=UI_TEXT[LANGUAGE_EN]["task_placeholder"],
                            lines=1,
                        )
                        random_task_input_button = gr.Button("Random Task")
                    with gr.Column():
                        env_input = gr.Textbox(
                            label=UI_TEXT[LANGUAGE_EN]["scene_description"],
                            placeholder=UI_TEXT[LANGUAGE_EN]["scene_placeholder"],
                            lines=1,
                        )
                        random_scene_input_button = gr.Button("Random Scene")
                scene_mode = gr.Radio(
                    choices=scene_mode_choices(LANGUAGE_EN),
                    value=SCENE_MODE_INITIAL,
                    label=UI_TEXT[LANGUAGE_EN]["scene_mode"],
                )
                with gr.Row():
                    generate_button = gr.Button("Generate", variant="primary")
                    rerun_simulation_button = gr.Button("Run Task", variant="secondary")
                    reset_button = gr.Button("Reset", variant="stop")
            with gr.Column(scale=2):
                current_image = gr.Video(
                    label=UI_TEXT[LANGUAGE_EN]["single_video_preview"],
                    height=420,
                    elem_id="embodichain-video-preview",
                    autoplay=True,
                    loop=True,
                )
                current_task = gr.Textbox(
                    label=UI_TEXT[LANGUAGE_EN]["current_task"],
                    interactive=False,
                    lines=2,
                )

        progress = gr.Slider(
            minimum=0,
            maximum=100,
            value=0,
            step=1,
            label=UI_TEXT[LANGUAGE_EN]["progress"],
            interactive=False,
            elem_classes="demo-only",
        )
        status = gr.Markdown(format_status("Idle."), elem_classes="demo-only")
        with gr.Row(elem_classes="demo-only"):
            model = gr.Model3D(
                label=UI_TEXT[LANGUAGE_EN]["initial_preview"],
                height=520,
                clear_color=(0.94, 0.94, 0.94, 1.0),
            )
            edited_model = gr.Model3D(
                label=UI_TEXT[LANGUAGE_EN]["edited_preview"],
                height=520,
                clear_color=(0.94, 0.94, 0.94, 1.0),
            )
        object_model = gr.Model3D(
            label=UI_TEXT[LANGUAGE_EN]["object_preview"],
            height=360,
            clear_color=(0.94, 0.94, 0.94, 1.0),
            elem_classes="demo-only",
        )

        refresh_timer = gr.Timer(2.0)
        top_mode_outputs = [
            auto_button,
            interact_button,
            parallel_env_button,
            generate_button,
            rerun_simulation_button,
            random_task_input_button,
            random_scene_input_button,
            reset_button,
            current_image,
            run_mode,
            action_mode,
        ]
        demo_mode_button.click(
            select_application_mode,
            inputs=[gr.State(APP_MODE_DEMO)],
            outputs=[
                demo_mode_button,
                debug_mode_button,
                debug_controls,
                mode_style,
                app_mode,
                debug_engine_area,
            ],
            queue=False,
        )
        debug_mode_button.click(
            select_application_mode,
            inputs=[gr.State(APP_MODE_DEBUG)],
            outputs=[
                demo_mode_button,
                debug_mode_button,
                debug_controls,
                mode_style,
                app_mode,
                debug_engine_area,
            ],
            queue=False,
        )
        for engine, button in zip(
            (engine for engine, _ in DEBUG_ENGINES),
            (
                asset_engine_button,
                scene_engine_button,
                action_engine_button,
            ),
        ):
            button.click(
                select_debug_engine,
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
        debug_scene_run.click(
            run_scene_engine,
            inputs=[debug_scene_image],
            outputs=[
                debug_scene_progress,
                debug_scene_status,
                debug_scene_output,
                debug_scene_preview,
            ],
        )
        debug_action_load.click(
            action_engine_snapshot,
            outputs=[
                debug_action_video,
                debug_action_current_task,
                debug_action_progress,
                debug_action_status,
                debug_action_scene,
            ],
            queue=False,
        )
        debug_action_run.click(
            run_action_engine_panel,
            inputs=[debug_action_task, debug_action_robot],
            outputs=[
                debug_action_video,
                debug_action_current_task,
                debug_action_progress,
                debug_action_status,
                debug_action_scene,
            ],
        )
        debug_action_refresh_timer.tick(
            action_engine_snapshot,
            outputs=[
                debug_action_video,
                debug_action_current_task,
                debug_action_progress,
                debug_action_status,
                debug_action_scene,
            ],
            queue=False,
        )
        auto_button.click(
            select_top_mode,
            inputs=[
                gr.State(TOP_MODE_AUTO),
                gr.State(None),
                run_mode,
                action_mode,
                language,
            ],
            outputs=top_mode_outputs,
            queue=False,
        )
        interact_button.click(
            select_top_mode,
            inputs=[
                gr.State(TOP_MODE_INTERACT),
                gr.State(None),
                run_mode,
                action_mode,
                language,
            ],
            outputs=top_mode_outputs,
            queue=False,
        )
        parallel_env_button.click(
            select_top_mode,
            inputs=[
                gr.State(None),
                gr.State(TOP_MODE_PARALLEL_ENV),
                run_mode,
                action_mode,
                language,
            ],
            outputs=top_mode_outputs,
            queue=False,
        )
        language_button.click(
            toggle_language,
            inputs=[language, run_mode, action_mode],
            outputs=[
                auto_button,
                interact_button,
                parallel_env_button,
                generate_button,
                rerun_simulation_button,
                random_task_input_button,
                random_scene_input_button,
                reset_button,
                language_button,
                heading,
                instruction,
                robot_profile,
                image_input,
                task_input,
                env_input,
                scene_mode,
                current_image,
                current_task,
                progress,
                model,
                edited_model,
                object_model,
                language,
            ],
            queue=False,
        )
        generate_button.click(
            run_generate_for_top_mode,
            inputs=[
                run_mode,
                action_mode,
                scene_mode,
                robot_profile,
                image_input,
                task_input,
                env_input,
                interact_prebuilt_scene_dir,
                language,
            ],
            outputs=[
                image_input,
                task_input,
                env_input,
                current_image,
                current_task,
                progress,
                status,
                model,
                edited_model,
                object_model,
            ],
        )
        random_task_input_button.click(
            randomize_interact_task_input,
            inputs=[run_mode, language],
            outputs=[
                image_input,
                task_input,
                env_input,
                scene_mode,
                interact_prebuilt_scene_dir,
                model,
                edited_model,
                object_model,
            ],
            queue=False,
        )
        random_scene_input_button.click(
            randomize_interact_scene_input,
            inputs=[run_mode, language],
            outputs=[env_input],
            queue=False,
        )
        rerun_simulation_button.click(
            rerun_current_simulation,
            inputs=[
                run_mode,
                action_mode,
                robot_profile,
            ],
            outputs=[
                image_input,
                task_input,
                env_input,
                current_image,
                current_task,
                progress,
                status,
                model,
                edited_model,
                object_model,
            ],
            queue=False,
        )
        image_input.upload(
            clear_interact_prebuilt_scene,
            outputs=[interact_prebuilt_scene_dir],
            queue=False,
        )
        scene_mode.change(
            scene_mode_input_updates,
            inputs=[scene_mode],
            outputs=[task_input, env_input],
            queue=False,
        )
        reset_button.click(
            clear_interact_prebuilt_scene,
            outputs=[interact_prebuilt_scene_dir],
            queue=False,
        )
        reset_button.click(
            run_reset_or_stop,
            inputs=[run_mode],
            outputs=[
                image_input,
                task_input,
                env_input,
                current_image,
                current_task,
                progress,
                status,
                model,
                edited_model,
                object_model,
            ],
            queue=False,
        )
        refresh_timer.tick(
            synced_ui_snapshot,
            inputs=[run_mode, action_mode, last_seen_input_revision],
            outputs=[
                image_input,
                task_input,
                env_input,
                current_image,
                current_task,
                progress,
                status,
                model,
                edited_model,
                object_model,
                rerun_simulation_button,
                last_seen_input_revision,
                scene_mode,
                robot_profile,
                parallel_env_button,
                action_mode,
            ],
            queue=False,
        )
    return demo
