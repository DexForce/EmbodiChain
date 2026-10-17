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

"""Explain workspace decisions using saved evidence, without a simulation handle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__: list[str] = []

_INK = (224, 232, 244)
_MUTED = (151, 172, 198)
_PASSED = (40, 220, 100)
_FAILED = (245, 65, 65)


def _checks(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "scene" in value:
        checks = value["scene"]["initial_validation"]
    elif "initial_validation" in value:
        checks = value["initial_validation"]
    else:
        checks = value
    if isinstance(checks, dict):
        checks = checks["checks"]
    return {check.get("name", check.get("check_id")): check for check in checks}


class _WorkspacePresentation:
    """Show the native red/green cloud and real object placement together."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        self.rows = report["candidates"]
        self.reachability = report["workspace"]["reachability"]
        self.fonts: dict[int, Any] = {}

    def _text(
        self,
        draw: Any,
        position: tuple[int, int],
        text: str,
        size: int = 20,
        color: tuple[int, int, int] = _INK,
    ) -> None:
        if size not in self.fonts:
            try:
                self.fonts[size] = ImageFont.truetype("DejaVuSans.ttf", size)
            except OSError:
                self.fonts[size] = ImageFont.load_default()
        draw.text(position, text, font=self.fonts[size], fill=color)

    def paint(
        self,
        rgb: np.ndarray | None,
        stage: str,
        index: int | None = None,
        *,
        paused: bool,
    ) -> np.ndarray:
        image = Image.new("RGB", (1280, 720), (10, 20, 35))
        if rgb is not None:
            image.paste(Image.fromarray(rgb).resize((896, 672)), (16, 24))
        draw = ImageDraw.Draw(image)
        title, color = "Red/green robot workspace at object initialization height", _INK
        if stage == "negative":
            title, color = "RED START: fixed-grasp IK failed; no expert action", _FAILED
        elif stage in ("initial", "execution") and index is not None:
            title, color = f"GREEN START #{index+1}: IK passed", _PASSED
            if stage == "execution":
                title += " - execute pick & place"
        elif stage == "result" and index is not None:
            accepted = self.rows[index]["scene"]["accepted"]
            title = f"GREEN START #{index+1}: task {'PASSED' if accepted else 'FAILED'}"
            color = _PASSED if accepted else _FAILED
        self._text(draw, (16, 0), title, 20, color)
        self._text(draw, (936, 40), "FIXED-GRASP WORKSPACE", 19)
        for y, legend, fill in (
            (90, "IK / limits / FK pass", _PASSED),
            (130, "IK / limits / FK fail", _FAILED),
        ):
            draw.ellipse((938, y, 953, y + 15), fill=fill)
            self._text(draw, (966, y - 4), legend, 18, fill)
        self._text(draw, (936, 188), "Upright object origins", 20)
        self._text(
            draw,
            (936, 220),
            f"Z = {self.reachability['object_center_height_m']:.3f} m",
            20,
        )
        self._text(draw, (936, 252), "Same downward grasp", 18, _MUTED)
        if stage == "negative":
            checks = _checks(self.report["negative"])
            support = checks["scene.support"]["status"]
            self._text(draw, (936, 315), f"Table: {support.upper()}", 22)
            self._text(draw, (936, 357), "IK target: FAILED", 23, _FAILED)
            if self.report["mode"] == "task":
                self._text(
                    draw,
                    (936, 405),
                    f"Expert actions: {self.report['negative_executed_steps']}",
                    24,
                )
            self._text(draw, (936, 448), "Rejected before execution", 18, _FAILED)
        elif index is not None:
            row = self.rows[index]
            check = _checks(row)["workspace.pose_ik"]
            self._text(
                draw, (936, 315), f"IK target: {check['status'].upper()}", 22, _PASSED
            )
            if stage == "result":
                self._text(
                    draw, (936, 357), f"Lift: {100*row['measured_lift_m']:.1f} cm", 23
                )
                self._text(
                    draw, (936, 399), f"Stable: {row['stable_control_steps']} steps", 23
                )
                self._text(
                    draw, (936, 441), f"Actions: {row['executed_steps']}", 20, _MUTED
                )
            else:
                self._text(draw, (936, 357), f"Cache row: {row['cache_index']}", 20)
                self._text(draw, (936, 401), "Read the same fixed-grasp", 18, _MUTED)
                self._text(draw, (936, 428), "IK/FK checks used by task", 18, _MUTED)
        else:
            self._text(draw, (936, 315), "Initialize in RED first:", 20, _FAILED)
            self._text(draw, (936, 352), "the IK target is rejected.", 18)
            self._text(draw, (936, 405), "Then initialize in GREEN:", 20, _PASSED)
            self._text(draw, (936, 442), "solve and pick & place.", 18)
        self._text(
            draw,
            (936, 513),
            f"Green points: {self.reachability['reachable_count']}",
            19,
            _PASSED,
        )
        self._text(
            draw,
            (936, 548),
            f"Red points: {self.reachability['rejected_count']}",
            19,
            _FAILED,
        )
        self._text(
            draw, (936, 605), "INSPECTION PAUSE" if paused else "RECORDED EXECUTION", 18
        )
        self._text(
            draw,
            (936, 639),
            "No physics steps added" if paused else "Original control cadence",
            16,
            _MUTED,
        )
        self._text(
            draw,
            (16, 700),
            "Every colored point was checked with the same fixed grasp: pose IK, joint limits and full-pose FK.",
            14,
            _MUTED,
        )
        return np.asarray(image)


def _write_walkthrough(report: dict[str, Any], output: Path) -> dict[str, Any]:
    """Keep every source frame; insert explicitly labelled inspection cards."""
    import imageio.v2 as imageio

    painter = _WorkspacePresentation(report)
    events = []
    seen = set()
    for event in report["video_events"]:
        tokens = event["stage"].split("_")
        if event["stage"] == "negative_rejected":
            stage, index = "negative", None
        elif len(tokens) == 3 and tokens[0] == "candidate":
            index = int(tokens[1])
            if index >= len(report["candidates"]):
                continue
            stage = "result" if tokens[2] == "result" else "initial"
            wanted = (
                ("initial", "result")
                if report["mode"] == "task"
                else ("settled", "rejected")
            )
            if tokens[2] not in wanted:
                continue
        else:
            continue
        if (stage, index) not in seen:
            events.append((event, stage, index))
            seen.add((stage, index))
    source_frames = inspection_frames = event_index = 0
    current_stage, current_index = events[0][1:] if events else ("overview", None)
    video = output / "workspace_explained.mp4"
    with imageio.get_reader(str(output / "preview.mp4")) as reader:
        fps = float(reader.get_meta_data()["fps"])
        with imageio.get_writer(str(video), fps=fps) as writer:
            first_rgb = imageio.imread(output / events[0][0]["keyframe"])
            overview = painter.paint(first_rgb, "overview", paused=True)
            imageio.imwrite(output / "workspace_overview.png", overview)
            for _ in range(round(3 * fps)):
                writer.append_data(overview)
                inspection_frames += 1

            def insert_cards(cursor: int) -> None:
                nonlocal event_index, current_stage, current_index, inspection_frames
                while (
                    event_index < len(events)
                    and events[event_index][0]["after_frame"] == cursor
                ):
                    event, current_stage, current_index = events[event_index]
                    rgb = imageio.imread(output / event["keyframe"])
                    card = painter.paint(rgb, current_stage, current_index, paused=True)
                    if current_stage == "negative":
                        imageio.imwrite(output / "workspace_rejection.png", card)
                    elif current_stage == "initial" and current_index == 0:
                        imageio.imwrite(output / "workspace_green_start.png", card)
                    duration = 4 if current_stage == "negative" else 1.5
                    for _ in range(round(duration * fps)):
                        writer.append_data(card)
                        inspection_frames += 1
                    event_index += 1

            for rgb in reader:
                insert_cards(source_frames)
                writer.append_data(
                    painter.paint(
                        rgb,
                        "negative" if current_stage == "negative" else "execution",
                        current_index if report["mode"] == "task" else None,
                        paused=False,
                    )
                )
                source_frames += 1
            insert_cards(source_frames)
            if event_index != len(events):
                raise ValueError("Inspection events do not match recorded video frames")
    return {
        "video": video.name,
        "overview": "workspace_overview.png",
        "rejection": "workspace_rejection.png",
        "green_start": "workspace_green_start.png",
        "source_video": "preview.mp4",
        "source_frames": source_frames,
        "inspection_frames": inspection_frames,
        "fps": fps,
        "physics_steps_added": 0,
    }
