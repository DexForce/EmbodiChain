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
"""Build a self-contained, simulator-free HTML view of retained DexUni results."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import quote, unquote

__all__ = ["main"]

_REPO = Path(__file__).resolve().parents[5]
_DEFAULT_ROOT = _REPO / "outputs/benchmarks/tech_reports/dexuni"
_ASSETS = _REPO / "scripts/benchmark/tech_reports/assets/physics/dexuni"
# Representative moments from the retained recordings, not measurement frames.
_PREVIEW_SECONDS = {
    "S01-C": 5,
    "S01-T": 5,
    "S02": 8,
    "S03": 6,
    "S04": 8,
    "M01": 8,
    "M02": 13,
    "M03": 16,
    "M04": 16,
    "M05": 14,
    "M06": 12,
    "C01": 23,
    "C02": 58,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_url(path: Path, output: Path) -> str:
    return quote(os.path.relpath(path, output.parent), safe="/")


def _poster(video: Path, second: int) -> str | None:
    if not video.is_file():
        return None
    import imageio_ffmpeg

    with tempfile.TemporaryDirectory(prefix="dexuni-poster-") as directory:
        frame = Path(directory) / "frame.jpg"
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(second),
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                "scale=1120:-2",
                "-q:v",
                "4",
                str(frame),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        if not frame.is_file():
            raise ValueError(f"No video frame at {second}s: {video}")
        return "data:image/jpeg;base64," + base64.b64encode(frame.read_bytes()).decode()


def _refresh_media(
    directory: Path, output: Path, second: int, include_posters: bool
) -> dict | None:
    """Load separate preview evidence without importing new measurement values."""
    asset_path = directory / "robot_asset_override.json"
    if not asset_path.is_file():
        return None
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = directory / "preview/manifest.json"
    if not manifest_path.is_file():
        return {"completed": False}
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("performance_measurements_started", False) or manifest.get(
        "performance_measurements", False
    ):
        raise ValueError(f"Expected an untimed media recording: {manifest_path}")
    video = manifest_path.with_name("video.mp4")
    if not video.is_file():
        video = directory / "preview/video.mp4"
    completed = manifest.get("status") == "completed" and video.is_file()
    effective = manifest.get("effective", {})
    asset = json.loads(asset_path.read_text())
    asset_root = Path(asset["selected_robot_urdf"]).parent
    public_asset = {
        "model": "W1-030",
        "link_count": asset["link_count"],
        "joint_count": asset["joint_count"],
        "sha256": {
            str(Path(name).relative_to(asset_root)): digest
            for name, digest in asset["sha256"].items()
        },
    }
    return {
        "completed": completed,
        "steps": effective.get("step_index", manifest.get("frames")),
        "time_s": effective.get("sim_time", manifest.get("actual_simulation_time_s")),
        "poster_second": second,
        "poster": _poster(video, second) if completed and include_posters else None,
        "video_url": _local_url(video, output) if completed else None,
        "manifest_sha256": _sha(manifest_path),
        "robot_asset": public_asset,
        "asset_record_sha256": _sha(asset_path),
    }


def _payload(
    root: Path, output: Path, include_posters: bool, preview_root: Path | None = None
) -> dict:
    comparison = root / "comparison.json"
    data = json.loads(comparison.read_text())
    publication = json.loads((_ASSETS / "publication.json").read_text())
    remote = (
        f"https://huggingface.co/datasets/{publication['repo_id']}/blob/"
        f"{publication['revision']}/reports/{publication['report_id']}"
    )
    verified = []
    commentary = json.loads(
        Path(__file__).with_name("task_commentary.json").read_text()
    )
    if set(commentary) != {task["code"] for task in data["tasks"]}:
        raise ValueError("Editorial task coverage must match the retained comparison")
    for task in data["tasks"]:
        task["editorial"] = commentary[task["code"]]
        source = root / task["source"]
        actual = _sha(source)
        if actual != task["source_sha256"]:
            raise ValueError(f"Frozen evidence hash mismatch: {source}")
        verified.append({"path": task["source"], "sha256": actual})
        task["group"] = task["code"][0]
        stats = task["stats"]
        task["kind"] = (
            "formal"
            if all(
                stats.get(p, {}).get("processes") == 5
                for p in ("embodichain", "isaaclab")
            )
            else "pilot" if task["code"] == "C02" else "single" if stats else "demo"
        )
        task["report_url"] = remote + "/browse/" + task["report"]
        task["source_url"] = remote + "/browse/" + task["source"]
        video = root / task["video"]
        task["video_url"] = _local_url(video, output) if video.is_file() else None
        task["poster_second"] = _PREVIEW_SECONDS[task["code"]]
        task["poster"] = (
            _poster(video, task["poster_second"]) if include_posters else None
        )
        if preview_root is not None:
            refresh = _refresh_media(
                preview_root / task["code"].lower(),
                output,
                task["poster_second"],
                include_posters,
            )
            if refresh is not None:
                task["refresh"] = refresh
        task_root = root / Path(task["video"]).parts[0]
        code = task["code"]
        unsupported = code in {"M05", "C01"}
        task["capability"] = {
            "status": (
                "unsupported"
                if unsupported
                else "limited" if code == "C02" else "supported"
            ),
            "label": (
                "完整场景未支持"
                if unsupported
                else "可运行 · 保形受限" if code == "C02" else "已实现对照"
            ),
            "reason": (
                "本次 Isaac Lab / Newton 后端未实现充气包气腔、压载与恢复的完整场景。"
                if code == "M05"
                else (
                    "本次 Isaac Lab / Newton 后端未实现包含充气奶包的双商品装袋；单面包替代场景不计为完整支持。"
                    if code == "C01"
                    else (
                        "24×12 倒袋失败；96×24 完成提袋录像，但严格保形检查未全过。"
                        if code == "C02"
                        else "两侧已实现本任务；具体效果差异见检查结果。"
                    )
                )
            ),
        }
        recordings = []
        refresh = task.get("refresh", {})
        if refresh.get("completed"):
            recordings.append(
                dict(
                    platform="embodichain",
                    label="EmbodiChain · 最新 W1 资产",
                    url=refresh["video_url"],
                    poster=refresh["poster"],
                    note="使用更新后的 W1 模型重新录制。",
                )
            )
        franka_patterns = {
            "S01-C": "media/s01c/raw_data/{platform}/*/rgb_oblique.mp4",
            "S01-T": "media/s01t/raw_data/{platform}/*/rgb_oblique.mp4",
            "S02": "media/s02/raw_data/{platform}/*/rgb_oblique.mp4",
            "S03": "preview/raw_data/{platform}/*/rgb_oblique.mp4",
            "M06": "preview/{platform}/{platform}.mp4",
        }
        if code in franka_patterns:
            for platform, label in (
                ("embodichain", "EmbodiChain · Franka"),
                ("isaaclab", "Isaac Lab"),
            ):
                matches = sorted(
                    task_root.glob(franka_patterns[code].format(platform=platform))
                )
                if len(matches) > 1:
                    raise ValueError(
                        f"Ambiguous preview source for {code} {platform}: {matches}"
                    )
                if matches:
                    path = matches[0]
                    recordings.append(
                        dict(
                            platform=platform,
                            label=label,
                            url=_local_url(path, output),
                            poster=(
                                _poster(path, task["poster_second"])
                                if include_posters
                                else None
                            ),
                            note="本场景使用 Franka，机器人资产无变更。",
                        )
                    )
        alternatives = []
        if code in {"S04", "M01", "M02", "M03", "M04"}:
            alternatives = [
                ("Isaac Lab", "preview/isaaclab/video.mp4", "Isaac Lab 对应场景。")
            ]
        elif code == "C02":
            alternatives = [
                (
                    "Isaac Lab · 96×24",
                    "platforms/isaaclab/preview/video.mp4",
                    "完成提袋录像，严格保形检查未全部通过。",
                ),
                (
                    "Isaac Lab · 24×12 · 失败",
                    "budget_reduction/i24_s12_video/preview/video.mp4",
                    "倒袋失败，未完成装袋提袋。",
                ),
            ]
        for label, relative, note in alternatives:
            path = task_root / relative
            if path.is_file():
                recordings.append(
                    dict(
                        platform="isaaclab",
                        label=label,
                        url=_local_url(path, output),
                        poster=(
                            _poster(path, task["poster_second"])
                            if include_posters
                            else None
                        ),
                        note=note,
                    )
                )
        task["recordings"] = recordings
        ec = next((r for r in recordings if r["platform"] == "embodichain"), None)
        # Presentation uses only the selected current EC asset, never the old composite.
        task["video_url"] = ec["url"] if ec else None
        task["poster"] = ec["poster"] if ec else None
    c02_path = root / "c02_bag_frame_packing/performance_comparison/results.json"
    c02 = json.loads(c02_path.read_text())
    data["c02"] = [
        {
            "label": c["label"].replace("x", "×"),
            "short": c["common_initial_12s"]["physics_s"],
            "full": c.get("complete_64s", {}).get("physics_s"),
            "samples_short": c["common_initial_12s"]["samples"],
            "samples_full": c.get("complete_64s", {}).get("samples"),
        }
        for c in c02["configurations"]
    ]
    data["c02_figures"] = []
    if include_posters:
        refreshed_c02 = next(t for t in data["tasks"] if t["code"] == "C02").get(
            "refresh", {}
        )
        if refreshed_c02.get("poster"):
            data["c02_figures"].append(
                {"label": "EC 12×6 · 新资产录像", "poster": refreshed_c02["poster"]}
            )
        for label, relative in (
            (
                "Isaac 24×12 · 倒袋失败",
                "budget_reduction/i24_s12_video/preview/video.mp4",
            ),
            ("Isaac 96×24 · 保形受限", "platforms/isaaclab/preview/video.mp4"),
        ):
            poster = _poster(root / "c02_bag_frame_packing" / relative, 58)
            if poster:
                data["c02_figures"].append({"label": label, "poster": poster})
    e2e_path = root.parent / "maintenance/e2e-sdk-2026-10-27/RESULTS.json"
    data["e2e"] = None
    if e2e_path.is_file():
        e2e = json.loads(e2e_path.read_text())
        data["e2e"] = {
            "sdk_commit": e2e["sdk"]["source_head"],
            "cases": [
                {
                    k: case[k]
                    for k in (
                        "platform",
                        "task",
                        "steps",
                        "exit_code",
                        "cold_automatic_download",
                        "finite_states",
                        "observations",
                    )
                }
                for case in e2e["cases"]
            ],
        }
        verified.append(
            {"path": str(e2e_path.relative_to(root.parent)), "sha256": _sha(e2e_path)}
        )
    data["publication"] = {"revision": publication["revision"], "url": remote}
    data["provenance"] = {"comparison_sha256": _sha(comparison), "sources": verified}
    return data


def _bundle_media(data: dict, output: Path) -> None:
    """Copy the selected platform videos into a portable, hash-indexed folder."""
    media = output.parent / "media"
    media.mkdir(parents=True, exist_ok=True)
    manifest = []
    for task in data["tasks"]:
        for index, recording in enumerate(task["recordings"]):
            source = (output.parent / unquote(recording["url"])).resolve(strict=True)
            name = f"{task['code'].lower()}_{recording['platform']}_{index}.mp4"
            destination = media / name
            if source != destination.resolve():
                shutil.copy2(source, destination)
            recording["url"] = "media/" + name
            manifest.append(
                dict(
                    task=task["code"],
                    label=recording["label"],
                    file=recording["url"],
                    bytes=destination.stat().st_size,
                    sha256=_sha(destination),
                )
            )
        ec = next(
            (r for r in task["recordings"] if r["platform"] == "embodichain"), None
        )
        task["video_url"] = ec["url"] if ec else None
        if task.get("refresh", {}).get("completed"):
            task["refresh"]["video_url"] = ec["url"] if ec else None
    (media / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )


def main() -> None:
    """Render embedded data, screenshots and UI into one offline HTML report.

    Existing local comparison and source evidence are required and hash-checked.
    Recordings are optional: screenshots are embedded when present, while video
    playback uses relative links to the retained local files. No data is fetched
    or uploaded and no simulator is imported.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--bundle-media",
        action="store_true",
        help="Copy selected MP4 files beside --output for a portable report. No upload.",
    )
    parser.add_argument(
        "--preview-root",
        type=Path,
        help="Optional isolated new-asset recordings, with lowercase task directories.",
    )
    parser.add_argument(
        "--no-posters", action="store_true", help="Skip video frame extraction."
    )
    args = parser.parse_args()
    if args.bundle_media and args.output is None:
        parser.error("--bundle-media requires an explicit --output release path")
    root = args.root.expanduser().resolve()
    output = (args.output or root / "report.html").expanduser().resolve()
    if output.suffix.lower() != ".html":
        parser.error("--output must be an .html file")
    preview_root = (
        args.preview_root.expanduser().resolve() if args.preview_root else None
    )
    data = _payload(root, output, not args.no_posters, preview_root)
    from performance_figures import build_performance_figures

    data["figures"] = build_performance_figures(
        data["tasks"], output.parent / "figures"
    )
    for figure in data["figures"].values():
        figure["svg_url"] = "figures/" + figure["svg_name"]
        figure["png_url"] = "figures/" + figure["png_name"]
    if args.bundle_media:
        _bundle_media(data, output)
    template = Path(__file__).with_name("report_template.html").read_text()
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace(
        "<", "\\u003c"
    )
    assert template.count("__REPORT_DATA__") == 1
    rendered = template.replace("__REPORT_DATA__", encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(
        f"Built {output} ({output.stat().st_size:,} bytes; {len(data['tasks'])} tasks)"
    )
    print("All retained task source hashes verified; original evidence unchanged.")


if __name__ == "__main__":
    main()
