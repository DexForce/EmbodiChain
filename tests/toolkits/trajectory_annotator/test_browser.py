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

"""Real-browser contracts for editing, asynchronous frames and persistence.

Install optional test tooling with ``pip install playwright`` and
``python -m playwright install chromium firefox``. System Chrome is also used
when available. Missing optional browsers are explicitly skipped.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.no_sim


@pytest.fixture(params=["chromium", "firefox"])
def page(request: pytest.FixtureRequest) -> Iterator[Page]:
    """Launch a clean browser context without touching user browser profiles."""
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        browser_type = getattr(runtime, request.param)
        options: dict[str, Any] = {"headless": True}
        if request.param == "chromium":
            executable = os.environ.get(
                "ANNOTATION_CHROMIUM_EXECUTABLE"
            ) or shutil.which("google-chrome")
            if executable:
                options["executable_path"] = executable
        if (
            "executable_path" not in options
            and not Path(browser_type.executable_path).is_file()
        ):
            pytest.skip(
                f"Install the optional browser: python -m playwright install {request.param}"
            )
        browser = browser_type.launch(**options)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1100}, accept_downloads=True
        )
        tab = context.new_page()
        errors: list[str] = []
        tab.on("pageerror", lambda error: errors.append(str(error)))
        try:
            yield tab
            assert not errors, f"Browser JavaScript errors: {errors}"
        finally:
            context.close()
            browser.close()


def open_project(page: Page, url: str) -> None:
    """Wait for a real decoded frame, not just a loaded HTML document."""
    from playwright.sync_api import expect

    page.goto(url)
    expect(page.locator("#workspace")).to_be_visible()
    expect(page.locator("#frame-current")).to_have_text("0")


def go(page: Page, index: int) -> None:
    """Use the user-facing frame jump control."""
    from playwright.sync_api import expect

    page.locator("#goto-frame").fill(str(index))
    page.locator("#goto-frame").press("Enter")
    expect(page.locator("#frame-current")).to_have_text(str(index))


def create_skill_interval(page: Page) -> None:
    """Create the known inclusive interval [2, 5], containing four frames."""
    from playwright.sync_api import expect

    go(page, 2)
    page.keyboard.press("i")
    page.locator("#label").fill("翻转包裹")
    page.locator("#label").press("Enter")
    go(page, 5)
    page.keyboard.press("o")
    page.keyboard.press("Enter")
    expect(page.locator("#skill-interval-count")).to_have_text("1")


def test_web_fitting_comparison_download_rejection_and_invalidation(
    page: Page,
    annotation_server: tuple[str, Path],
    fitting_oracle: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Exercise the new production web workflow without coupling browser tests to IK."""
    import numpy as np
    from playwright.sync_api import expect

    fitting_oracle["accepted"] = False
    open_project(page, annotation_server[0])
    expect(page.locator("#fit-run")).to_be_disabled()
    create_skill_interval(page)
    page.locator("#fit-joints").fill("known.npz")
    page.locator("#fit-profile").fill("robot.json")
    page.locator("#fit-semantics").select_option("measured_state")
    page.locator("#fit-offset").fill("0")
    page.locator("#fit-run").click()
    expect(page.locator("#fit-state")).to_have_text("未通过校验", timeout=15000)
    expect(page.locator("#fit-result")).to_be_visible()
    expect(page.locator("#fit-proposed-legend")).to_be_visible()
    expect(page.locator("#fit-message")).to_contain_text(
        "patched path exceeds joint limits"
    )
    page.locator("#fit-joint").select_option("1")
    expect(page.locator("#fit-cursor")).to_contain_text("hand_joint")
    page.locator("#fit-exaggeration").select_option("20")
    chart = page.locator("#fit-joint-chart").bounding_box()
    # The plot spans 0..2.99 s. This point is video t=.2 s, where the VFR
    # recording's displayed frame is 5 (its nonzero first PTS is normalized).
    x = chart["x"] + 62 + 0.2 / 2.99 * (chart["width"] - 76)
    page.mouse.click(x, chart["y"] + 65)
    expect(page.locator("#frame-current")).to_have_text("5")
    with page.expect_download() as download:
        page.locator("#fit-download").click()
    destination = tmp_path / "fit.npz"
    download.value.save_as(destination)
    with np.load(destination, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["patched_qpos"], data["original_qpos"])
    page.locator("#edit").click()
    page.locator("#end-frame").fill("6")
    page.locator("#commit").click()
    expect(page.locator("#fit-state")).to_have_text("标注已变化")
    expect(page.locator("#fit-download")).to_have_attribute("aria-disabled", "true")
    # An unobservable segment must surface the error and release the job guard,
    # allowing a corrected configuration/selection to be fitted again.
    fitting_oracle["error"] = True
    page.locator("#fit-run").click()
    expect(page.locator("#fit-state")).to_have_text("拟合失败")
    expect(page.locator("#fit-message")).to_contain_text("No observable closing event")
    expect(page.locator("#fit-run")).to_be_enabled()
    fitting_oracle["error"], fitting_oracle["accepted"] = False, True
    page.locator("#fit-run").click()
    expect(page.locator("#fit-state")).to_have_text("通过校验")
    expect(page.locator("#fit-proposed-legend")).to_be_hidden()
    expect(page.locator("#fit-parameters")).to_contain_text("lift_height")


def test_keyboard_edit_delete_undo_and_resume(
    page: Page, annotation_server: tuple[str, Path]
) -> None:
    from playwright.sync_api import expect

    url, destination = annotation_server
    open_project(page, url)
    video = page.locator(".video-card").bounding_box()
    timeline = page.locator(".timeline-card").bounding_box()
    annotations = page.locator(".annotations-card").bounding_box()
    editor = page.locator(".skill-editor-card").bounding_box()
    assert timeline["x"] == pytest.approx(video["x"], abs=1)
    assert timeline["width"] == pytest.approx(video["width"], abs=1)
    assert timeline["y"] >= video["y"] + video["height"]
    assert editor["x"] == pytest.approx(annotations["x"], abs=1)
    assert editor["width"] == pytest.approx(annotations["width"], abs=1)
    assert editor["y"] >= annotations["y"] + annotations["height"]
    expect(page.locator("#editor-title")).to_have_text("新建原子技能区间")
    expect(page.locator(".header-shortcuts")).to_contain_text("←前一帧")
    expect(page.locator(".header-shortcuts")).to_contain_text("→后一帧")
    for hint in ("空格播放 / 暂停", "I标注起始帧", "O标注结束帧"):
        expect(page.locator(".header-shortcuts")).to_contain_text(hint)
    page.keyboard.press("ArrowLeft")
    expect(page.locator("#frame-current")).to_have_text("0")
    page.keyboard.press("Shift+ArrowRight")
    expect(page.locator("#frame-current")).to_have_text("10")
    create_skill_interval(page)
    expect(page.locator("#skill-interval-list tr")).to_have_count(1)
    expect(page.locator("#skill-interval-list td").nth(3)).to_have_text("4")
    expect(page.locator(".timeline-skill-interval")).to_have_count(1)
    page.locator("#label").focus()
    page.locator("#label").dispatch_event(
        "keydown", {"key": "Enter", "isComposing": True}
    )
    expect(page.locator("#label")).to_be_focused()
    page.keyboard.press("i")
    page.keyboard.press("Backspace")
    expect(page.locator("#start-frame")).to_have_value("")
    expect(page.locator("#skill-interval-count")).to_have_text("1")
    expect(page.locator("#frame-current")).to_have_text("5")
    page.locator("#video-stage").focus()
    page.keyboard.press("e")
    expect(page.locator("#editor-title")).to_have_text("编辑原子技能区间")
    page.locator("#start-frame").fill("3")
    page.locator("#start-frame").press("Enter")
    go(page, 6)
    page.keyboard.press("o")
    page.locator("#skill-interval-list tr").focus()
    page.keyboard.press("Enter")
    expect(page.locator("#skill-interval-list td").nth(1)).to_have_text("3")
    expect(page.locator("#skill-interval-list td").nth(2)).to_have_text("6")
    page.keyboard.press("Delete")
    expect(page.locator("#skill-interval-count")).to_have_text("0")
    expect(page.locator(".timeline-skill-interval")).to_have_count(0)
    page.keyboard.press("Control+z")
    expect(page.locator("#skill-interval-count")).to_have_text("1")
    page.keyboard.press("Control+Shift+z")
    expect(page.locator("#skill-interval-count")).to_have_text("0")
    page.keyboard.press("Control+z")
    expect(page.locator("#skill-interval-count")).to_have_text("1")
    page.keyboard.press("Control+s")
    expect(page.locator("#save-status")).to_have_text("已保存")
    saved = json.loads(destination.read_text())
    assert [
        (
            skill_interval["label"],
            skill_interval["start_frame"],
            skill_interval["end_frame"],
        )
        for skill_interval in saved["segments"]
    ] == [("翻转包裹", 3, 6)]
    page.reload()
    expect(page.locator("#skill-interval-count")).to_have_text("1")
    expect(page.locator("#skill-interval-list td").nth(2)).to_have_text("6")
    expect(page.locator("#frame-current")).to_have_text("0")
    screenshot = os.environ.get("ANNOTATION_SCREENSHOT")
    if screenshot:
        page.screenshot(
            path=screenshot.replace(
                ".png", f"-{page.context.browser.browser_type.name}.png"
            ),
            full_page=True,
        )


def test_timeline_frame_geometry_and_free_drag_inside_skill_intervals(
    page: Page, annotation_server: tuple[str, Path]
) -> None:
    from playwright.sync_api import expect

    url, _ = annotation_server
    open_project(page, url)
    project = page.request.get(f"{url}/api/project").json()
    # The fixture has 90 frames. Include one-frame intervals, long labels, and
    # both source edges; none of these may widen a bar past its frame cells.
    intervals = [
        dict(id="first", label="首帧技能", start_frame=0, end_frame=0),
        dict(id="single", label="很长的单帧原子技能名称", start_frame=12, end_frame=12),
        dict(id="middle", label="翻转包裹", start_frame=20, end_frame=39),
        dict(id="last", label="末帧技能", start_frame=89, end_frame=89),
    ]
    response = page.request.put(
        f"{url}/api/annotations",
        headers={"X-Annotation-Token": project["session_token"]},
        data={
            "segments": intervals,
            "expected_revision": project["document"]["revision"],
        },
    )
    assert response.ok
    page.reload()
    expect(page.locator("#skill-interval-count")).to_have_text("4")
    expect(page.locator("#frame-current")).to_have_text("0")

    def check_geometry() -> dict[str, float]:
        bounds = page.locator("#timeline").bounding_box()
        cell = bounds["width"] / 90
        for interval in intervals:
            bar = page.locator(f'[data-skill-id="{interval["id"]}"]').bounding_box()
            assert bar["x"] == pytest.approx(
                bounds["x"] + interval["start_frame"] * cell, abs=0.1
            )
            assert bar["width"] == pytest.approx(
                (interval["end_frame"] - interval["start_frame"] + 1) * cell,
                abs=0.1,
            )
        return bounds

    bounds = check_geometry()
    y = bounds["y"] + 46  # Inside the colored skill bars, rather than empty track.

    def x_at(frame: int) -> float:
        return bounds["x"] + (frame + 0.5) * bounds["width"] / 90

    page.mouse.move(x_at(30), y)
    page.mouse.down()
    try:
        # Starting inside [20, 39] must seek frame 30, never snap to frame 20.
        expect(page.locator("#frame-current")).to_have_text("30")
        for frame in (35, 45, 25, 12, 0, 89):
            page.mouse.move(x_at(frame), y, steps=4)
            expect(page.locator("#frame-current")).to_have_text(str(frame))
    finally:
        page.mouse.up()
    expect(page.locator("#frame-current")).to_have_text("89")
    cursor = page.locator("#playhead").bounding_box()
    assert cursor["x"] == pytest.approx(x_at(89), abs=0.1)

    page.locator("#zoom-in").click()
    bounds = check_geometry()
    page.locator("#timeline-scroll").evaluate(
        "element => { element.scrollLeft = element.clientWidth / 2; }"
    )
    bounds = check_geometry()
    # The same coordinate rule must hold after zooming and horizontal scrolling.
    page.mouse.click(x_at(30), bounds["y"] + 46)
    expect(page.locator("#frame-current")).to_have_text("30")


def test_pause_marks_visible_frame_and_latest_seek_wins(
    page: Page, annotation_server: tuple[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from playwright.sync_api import expect
    from embodichain.toolkits.trajectory_annotator._frames import FrameReader

    original = FrameReader.preview

    def delayed(self: FrameReader, index: int) -> Any:
        if index in {1, 40}:
            time.sleep(0.35)
        return original(self, index)

    monkeypatch.setattr(FrameReader, "preview", delayed)
    open_project(page, annotation_server[0])
    with page.expect_response(lambda response: "/api/frames/1?" in response.url):
        page.keyboard.press("Space")
        page.keyboard.press("i")
    page.evaluate(
        "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
    )
    expect(page.locator("#frame-current")).to_have_text("0")
    expect(page.locator("#start-frame")).to_have_value("0")
    expect(page.locator("#playback-state")).to_have_text("已暂停")
    page.keyboard.press("Escape")
    page.locator("#goto-frame").fill("40")
    page.locator("#goto-frame").press("Enter")
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    expect(page.locator("#frame-current")).to_have_text("43")
    expect(page.locator("#video-frame")).to_have_attribute("alt", "视频第 43 帧")
    page.keyboard.press("End")
    expect(page.locator("#frame-current")).to_have_text("89")
    page.keyboard.press("ArrowRight")
    expect(page.locator("#frame-current")).to_have_text("89")
    page.keyboard.press("Space")
    expect(page.locator("#frame-current")).not_to_have_text("89")
    page.keyboard.press("Space")


def test_failed_save_keeps_disk_and_exports_current_edits(
    page: Page, annotation_server: tuple[str, Path]
) -> None:
    from playwright.sync_api import expect

    url, destination = annotation_server
    open_project(page, url)
    create_skill_interval(page)
    expect(page.locator("#save-status")).to_have_text("已保存")
    original = destination.read_bytes()
    page.route(
        "**/api/annotations",
        lambda route: route.fulfill(status=507, json={"detail": "模拟磁盘写入失败"}),
    )
    page.keyboard.press("Delete")
    expect(page.locator("#save-status")).to_have_text("保存失败")
    assert destination.read_bytes() == original
    with page.expect_download() as download:
        page.locator("#export").click()
    assert json.loads(Path(download.value.path()).read_text())["segments"] == []
    page.unroute("**/api/annotations")
    page.locator("#save").click()
    expect(page.locator("#save-status")).to_have_text("已保存")
    assert json.loads(destination.read_text())["segments"] == []


def test_edit_during_save_does_not_clear_newer_dirty_state(
    page: Page, annotation_server: tuple[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from playwright.sync_api import expect
    from embodichain.toolkits.trajectory_annotator._storage import AnnotationStore

    entered, release = threading.Event(), threading.Event()
    original = AnnotationStore.save

    def delayed(
        self: AnnotationStore, skill_intervals: object, expected_revision: object
    ) -> dict[str, Any]:
        if expected_revision == 0:
            entered.set()
            assert release.wait(timeout=10)
        return original(self, skill_intervals, expected_revision)

    monkeypatch.setattr(AnnotationStore, "save", delayed)
    open_project(page, annotation_server[0])
    create_skill_interval(page)
    try:
        assert entered.wait(timeout=3)
        page.keyboard.press("Delete")
        expect(page.locator("#skill-interval-count")).to_have_text("0")
    finally:
        release.set()
    expect(page.locator("#save-status")).to_have_text("已保存")
    saved = json.loads(annotation_server[1].read_text())
    assert saved["revision"] == 2
    assert saved["segments"] == []
