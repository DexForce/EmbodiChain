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

"""Publish one consistent delivery without judging task success or changing trials."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
from uuid import uuid4

import imageio.v2 as imageio
import imageio_ffmpeg
import psutil

from .catalog import write_json
from .usage import UsageMeter, collect_usage, refresh_usage

__all__ = ["finalize_run"]


def _read(path: Path, issues: list[str]) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        json.dumps(value, allow_nan=False)
        return value
    except (ValueError, OSError) as exc:
        issues.append(f"{path}: {exc}")
        return {}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _live(process: dict) -> bool:
    if "pid" not in process or "created_at" not in process:
        return False
    try:
        actual = psutil.Process(process["pid"])
        return (
            actual.create_time() == process["created_at"]
            and actual.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


def _execution(root: Path, issues: list[str]) -> dict:
    records = [
        path
        for path in (
            root / "launch.json",
            root / "search.json",
            root / "delivery_state.json",
            root / "pause.json",
        )
        if path.exists()
    ]
    states = [
        _read(path, issues)
        for path in sorted(records, key=lambda path: path.stat().st_mtime)
    ]
    heartbeat = (
        _read(root / "heartbeat.json", issues)
        if any(state.get("status") == "running" for state in states)
        else {}
    )
    for state in states:
        if state.get("status") == "running" and (
            _live(state)
            or (
                state.get("output")
                and heartbeat.get("launch") == state["output"]
                and time.time() - heartbeat.get("time", 0) < 10
            )
        ):
            raise RuntimeError(
                "Experiment host is still running; stop it before finalizing"
            )
    record = states[-1] if states else {}
    if record.get("status") == "running":
        record = {**record, "status": "interrupted", "recovered": True}
    if not record:
        latest = _read(root / "latest_attempt.json", issues)
        record = {"status": "unknown", "process": latest.get("process", {})}
        if latest:
            process = record["process"]
            record["status"] = (
                "timed_out"
                if process.get("timed_out")
                else (
                    "interrupted"
                    if process.get("interrupted")
                    else "completed" if process.get("returncode") == 0 else "failed"
                )
            )
    if record.get("status") == "exited":
        record = {
            **record,
            "status": (
                "completed"
                if record.get("process", {}).get("returncode") == 0
                else "failed"
            ),
        }
    stop = _read(root / "stop.json", issues)
    if record.get("status") == "paused_by_user" or (
        record.get("status") == "interrupted"
        and record.get("output")
        and stop.get("launch") == record["output"]
    ):
        record = {**record, "recorded_status": record["status"], "status": "stopped"}
    settings = sorted((root / "codex").glob("*/agent_settings.json"))
    defaults = _read(settings[-1], issues) if settings else {}
    record = {
        **record,
        "model": record.get("model", defaults.get("model")),
        "reasoning_effort": record.get(
            "reasoning_effort", defaults.get("reasoning_effort")
        ),
    }
    return record


def _attempt(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / (path if len(path.parts) > 1 else Path("attempts") / path)
    path = path.resolve()
    if path.parent != root / "attempts" or not path.is_dir():
        raise ValueError(f"Selected attempt is outside this run or missing: {value}")
    return path


def _probe(video: Path) -> dict:
    reader = imageio.get_reader(str(video))
    try:
        meta = reader.get_meta_data()
        fps = float(meta["fps"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid video frame rate")
        frames = sum(1 for _ in reader)
        if frames == 0:
            raise ValueError("Video has no decodable frames")
        decoded = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-xerror",
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=300,
        )
        if decoded.returncode:
            raise ValueError(
                f"Video decode failed: {decoded.stderr.decode(errors='replace')[-1000:]}"
            )
        return {
            "frames": frames,
            "fps": fps,
            "duration_seconds": frames / fps,
            "width": meta["size"][0],
            "height": meta["size"][1],
            "codec": meta.get("codec"),
            "pixel_format": meta.get("pix_fmt"),
            "sha256": _hash(video),
        }
    finally:
        reader.close()


def _reproduction(root: Path, attempt: Path, manifest: dict, issues: list[str]) -> dict:
    process = _read(attempt / "process.json", issues)
    command = process.get("command", [])
    script = (
        Path(command[command.index("--script") + 1]) if "--script" in command else None
    )
    config = attempt / "lab.json"
    code = attempt / "code"
    hashes = {
        str(path.relative_to(attempt)): _hash(path)
        for path in sorted(code.rglob("*"))
        if path.is_file() and path.resolve().is_relative_to(attempt)
    }
    if config.exists():
        hashes["lab.json"] = _hash(config)
    inputs = _read(attempt / "inputs.json", issues)
    recorded = inputs.get("files_sha256")
    integrity = (
        "not_recorded"
        if recorded is None
        else (
            "matched"
            if all(hashes.get(path) == value for path, value in recorded.items())
            else "mismatch"
        )
    )
    if integrity == "mismatch":
        issues.append(f"Frozen code/config changed after execution: {attempt.name}")
    replay = []
    if integrity != "mismatch" and (
        script is not None
        and script.resolve().is_relative_to(code)
        and script.is_file()
    ):
        replay = [
            manifest.get("python", "python"),
            "-B",
            "-m",
            "embodichain.gen_sim.agent_lab",
            "run",
            "--run-dir",
            str(root),
            "--script",
            str(script),
        ]
        if config.exists():
            replay += ["--config", str(config)]
    return {
        "command": shlex.join(replay) if replay else None,
        "cwd": manifest.get("repo"),
        "files_sha256": hashes,
        "recorded_files_sha256": recorded,
        "file_integrity": integrity,
        "config": _read(config, issues),
        "environment": _read(attempt / "environment.json", issues),
        "source_fingerprint": _read(attempt / "source_fingerprint.json", issues),
        "runtime_hashes": _read(attempt / "runtime_hashes.json", issues),
    }


def _usage_lines(resources: dict) -> list[str]:
    timing, tokens = resources["timing"], resources["tokens"]
    labels = {
        "reported": "已报告",
        "reconstructed": "按已有记录重建",
        "partial": "部分记录",
        "unavailable": "未知",
    }

    def duration(value: float | None) -> str:
        if value is None:
            return "未知"
        seconds = round(value)
        return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}（{value:.3f} 秒）"

    def count(key: str) -> str:
        value = tokens.get(key)
        return f"{value:,}" if value is not None else "未知"

    total = count("total_tokens")
    if tokens["status"] == "partial" and tokens["total_tokens"] is not None:
        total = "至少 " + total
    lines = [
        "## 累计耗时与用量",
        f"- **累计全程耗时：{'至少 ' if timing['status'] == 'partial' and timing['total_wall_seconds'] is not None else ''}{duration(timing['total_wall_seconds'])}**；{labels[timing['status']]}。",
        f"- Codex 会话累计：{duration(timing['codex_wall_seconds'])}，包含工具和仿真等待，不是纯思考时间。",
        f"- 仿真进程累计：{duration(timing['simulation_wall_seconds'])}，不是仿真时钟或视频时长。子项可能重叠，不与全程耗时重复相加。",
        f"- **Token 总量：{total}**；{labels[tokens['status']]}。",
        f"- 输入：{count('input_tokens')}；其中缓存命中：{count('cached_input_tokens')}；输出：{count('output_tokens')}。",
        f"- 推理输出子项：{count('reasoning_output_tokens')}。总量只按输入加输出计算，不再加缓存或推理子项。",
        f"- 覆盖：{tokens['invocations']} 次 Codex 启动，{tokens['reported_turns']} 个完整用量记录，{tokens['partial_turns']} 个部分记录，{tokens['missing_turns']} 个无用量数据的回合。",
        f"- 数据截至 Unix 时间：`{resources['as_of']}`；仅当前 run，包含失败重试，排除启动间停机间隔及其他对话。",
        "- 缺失用量不记为零、不按文字长度猜测，不将 token 数解释为实际费用或账户额度消耗。",
        "- 历史记录重建可能遗漏未记录的宿主初始化/整理开销；完整来源与统计边界见 result.json.resources。",
        "",
    ]
    configurations = resources.get("configurations", [])
    if configurations:
        lines += [
            "### 启动配置分段",
            "以下耗时和 token 按启动记录；请求配置不等于已确认的实际配置。",
            "| 启动 | 请求模型 / 强度 | 会话记录中的模型 / 强度 | 耗时（秒） | Token / 完整性 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for item in configurations:
            observed = (
                "; ".join(
                    f"{s['model']} / {s['reasoning_effort']}"
                    for s in item["observed_settings"]
                )
                or "未取得"
            )
            lines.append(
                f"| {item['invocation']} | {item['requested_model']} / {item['requested_reasoning_effort']} | {observed} | {item['wall_seconds']} | {item['total_tokens']} / {item['token_status']} |"
            )
        lines += ["", "中途换模型或强度时，全程用量不能全部归到最后一次配置。", ""]
    return lines


def _report(result: dict) -> str:
    run = result["run"]
    video = result["video"]
    lines = [
        f"# 实验报告：{run.get('task_id') or run['run_id']}",
        "",
        *_usage_lines(result["resources"]),
        "## 结论",
        f"- 运行状态：`{result['execution'].get('status', 'unknown')}`",
        f"- 验证范围：`{result['assessment']['scope']}`",
        "- 任务结论：尚未独立验收。退出码、可播放录像及 Codex 声明均不自动证明任务成功。",
        f"- Codex 声明状态：`{result['assessment']['agent_claim'].get('status', '未提交状态')}`；声明完整任务完成：`{result['assessment']['agent_claim'].get('task_completed', '未声明')}`",
        f"- 本轮声明模式：`{run.get('mode')}`；最近启动请求的模型：`{result['execution'].get('model')}`；思考强度：`{result['execution'].get('reasoning_effort')}`。各次启动见分段记录。",
        "",
        "## 任务",
        str(run.get("instruction") or "未记录任务描述"),
        "",
        f"验收要求：{run.get('acceptance') or '未记录'}",
        "",
        "## 视频",
    ]
    if video:
        lines += [
            f"[打开视频]({Path(run['directory']) / 'final/video.mp4'})",
            f"- 来源尝试：`{result['selection']['attempt_id']}`",
            f"- 选择依据：{result['selection']['reason']}",
            f"- 性质：`{video['kind']}`，不代表完整任务成功。",
            f"- 已解码：{video['frames']} 帧，{video['fps']} fps，{video['duration_seconds']:.3f} 秒。",
            f"- SHA-256：`{video['sha256']}`",
        ]
    else:
        lines.append(f"没有发布视频：{result['video_missing_reason']}")
        lines.append(
            f"选择的尝试：`{result['selection']['attempt_id']}`；{result['selection']['reason']}"
        )
    lines += [
        "",
        "## 完成与未完成",
        "以下内容来自 Codex 声明，不是整理器的独立验收结论；完整原文保留在 result.json。",
    ]
    claim = result["assessment"]["agent_claim"]
    for key in ("summary", "rationale", "reason", "usage_notes"):
        if claim.get(key):
            lines += ["", str(claim[key])]
    stages = claim.get("required_task_stages", {})
    if isinstance(stages, dict):
        lines += [f"- `{key}`：{value}" for key, value in stages.items()]
    evidence = claim.get("evidence", [])
    if isinstance(evidence, list):
        for item in evidence:
            description = (
                item.get(
                    "finding",
                    item.get("observation", json.dumps(item, ensure_ascii=False)),
                )
                if isinstance(item, dict)
                else str(item)
            )
            lines.append(f"- {description}")
    if not claim:
        lines.append("未提交交接声明；不能据此补写已完成的任务阶段。")
    elif "solver_result" in claim:
        lines += [
            "```json",
            json.dumps(claim["solver_result"], ensure_ascii=False, indent=2),
            "```",
        ]
    lines += [
        "",
        "## 尝试与证据",
        "| 尝试 | 执行完成 | 错误 |",
        "| --- | --- | --- |",
    ]
    for item in result["attempts"]:
        error = (
            str(item["worker"].get("error") or "").replace("|", "/").replace("\n", " ")
        )
        lines.append(
            f"| [{item['attempt_id']}]({Path(run['directory']) / item['path'] / 'result.json'}) | {item['worker'].get('execution_completed')} | {error} |"
        )
    review = result["assessment"]["review_record"]
    if review:
        lines += [
            "",
            f"既有评审记录：`{review.get('status', '未记录状态')}`，评审者：`{review.get('reviewer', '未记录')}`。",
            f"[查看原评审记录](<{Path(run['directory']) / 'acceptance.json'}>)。整理器未重新验证该评审，完整记录也保留在 result.json。",
        ]
    lines += [
        "",
        "## 问题与限制",
        "- 保留原始尝试；没有跨尝试拼接、补帧或伪造缺失录像。",
        "- 视频时长按编码帧率计算；物理仿真时间以原始 metrics/telemetry 为准。",
        "- 同一 JSON 生成本报告。无视频、无交接声明或未验证任务，均不会补成成功。",
    ]
    lines += [f"- {issue}" for issue in result["issues"]]
    if isinstance(claim.get("limitations"), list):
        lines += [f"- Codex 记录的限制：{item}" for item in claim["limitations"]]
    lines += [
        "",
        "## 复现",
        f"工作目录：`{result['reproduction'].get('cwd')}`",
        "```sh",
        result["reproduction"].get("command") or "# 缺少可复现的冻结脚本入口",
        "```",
        "命令创建新的尝试，不覆盖原录像。运行前核对 result.json 中的环境、代码和资产指纹；它不是完整环境快照。",
        "",
    ]
    return "\n".join(lines)


def finalize_run(
    root: Path,
    *,
    attempt: Path | str | None = None,
    execution: dict | None = None,
    usage_meter: UsageMeter | None = None,
) -> dict:
    """Publish final/result.json, report.md and an optional validated MP4.

    Selection is explicit CLI input, handoff.selected_attempt, legacy replay,
    then the latest decodable diagnostic attempt (never inferred to be best).
    Each immutable publication is exposed through an atomically replaced final
    symlink. Repeating unchanged finalization reuses the existing publication.

    Args:
        root: An existing Agent Lab run directory.
        attempt: Optional attempt path, relative to the run or absolute.
        execution: Terminal host state supplied by a lifecycle owner.
        usage_meter: Optional active host meter, closed at the report snapshot
            after video validation/copy. Caller must also close it on errors.

    Returns:
        The versioned result manifest. Task success remains unverified.

    Raises:
        RuntimeError: A host or simulation worker is still running.
        ValueError: The run manifest is missing or final is a real directory.
    """
    root = root.resolve()
    issues: list[str] = []
    manifest = _read(root / "run.json", issues)
    if not manifest:
        raise ValueError(f"Missing or invalid run.json: {root}")
    observed_execution = _execution(root, issues)
    execution = (
        {**observed_execution, **execution}
        if execution is not None
        else observed_execution
    )
    if execution.get("status") == "running":
        raise RuntimeError("Cannot publish a final delivery while still running")
    handoff = _read(root / "workspace/handoff.json", issues)
    claim = handoff or _read(root / "workspace/candidate.json", issues)
    summary = _read(root / "summary.json", issues)
    attempts = []
    for directory in sorted((root / "attempts").glob("*")):
        if not directory.is_dir() or directory.is_symlink():
            continue
        process = _read(directory / "process.json", issues)
        if _live(process):
            raise RuntimeError(f"Simulation worker is still running: {directory}")
        worker = _read(directory / "result.json", issues)
        attempts.append(
            {
                "attempt_id": directory.name,
                "path": str(directory.relative_to(root)),
                "worker": {
                    key: worker.get(key)
                    for key in ("execution_completed", "stage", "error")
                },
                "process": process,
                "metrics": _read(directory / "metrics.json", issues),
                "experiment": _read(directory / "experiment.json", issues),
            }
        )
    previous = _read(root / "final/result.json", [])
    requested = attempt if attempt is not None else handoff.get("selected_attempt")
    selection_source = "cli" if attempt is not None else "handoff"
    if requested is None and previous.get("selection", {}).get("source") == "cli":
        requested = previous["selection"]["attempt_id"]
        selection_source = "cli"
    reason = "显式选择本轮尝试" if requested else "旧入口的新进程重跑记录"
    if requested is None:
        replay = summary.get("replay")
        requested = replay.get("attempt") if isinstance(replay, dict) else None
        selection_source = "replay"
    candidates = (
        [requested]
        if requested is not None
        else [item["path"] for item in reversed(attempts)]
    )
    if requested is None:
        selection_source = "diagnostic"
        reason = "未指定最终尝试，回退到最近可解码的诊断录像；不代表最好或成功的一次"
        issues.append(reason)
    selected = None
    video = None
    for candidate in candidates:
        try:
            directory = _attempt(root, candidate)
            candidate_record = next(
                item for item in attempts if item["attempt_id"] == directory.name
            )
            if selected is None:
                selected = candidate_record
            source = directory / "video.mp4"
            if source.resolve().parent != directory:
                raise ValueError("Video resolves outside the selected attempt")
            info = _probe(source)
            selected = candidate_record
            video = {
                **info,
                "path": "video.mp4",
                "source": str(source.relative_to(root)),
                "source_sha256": info["sha256"],
                "kind": (
                    "complete_attempt"
                    if selected["worker"]["execution_completed"]
                    and selected["process"].get("returncode") == 0
                    and not selected["process"].get("timed_out")
                    and not selected["process"].get("interrupted")
                    else "partial_attempt"
                ),
            }
            expected = selected["metrics"].get("video_frames")
            if expected is not None and expected != info["frames"]:
                video["kind"] = "partial_attempt"
                issues.append(
                    f"Recorded frame count {expected} differs from decoded count {info['frames']}"
                )
            break
        except (
            ValueError,
            TypeError,
            OSError,
            RuntimeError,
            KeyError,
            StopIteration,
            subprocess.SubprocessError,
        ) as exc:
            issues.append(f"Video candidate {candidate}: {str(exc)[:1200]}")
    if not claim and selected:
        worker_claim = _read(root / selected["path"] / "result.json", issues).get(
            "solver_result"
        )
        if worker_claim is not None:
            claim = {"solver_result": worker_claim}
    result = {
        "schema_version": "agent-lab-delivery/v1",
        "run": {
            "run_id": root.name,
            "directory": str(root),
            "task_id": manifest.get("task", {}).get("task_id"),
            "instruction": manifest.get("task", {}).get("instruction"),
            "acceptance": manifest.get("task", {}).get("acceptance"),
            "mode": manifest.get("mode"),
            "approved_asset_changes": manifest.get("approved_asset_changes", []),
            "repo_revision": manifest.get("repo_revision"),
        },
        "execution": execution,
        "resources": collect_usage(root),
        "assessment": {
            "scope": manifest.get("objective", "solve"),
            "verification_status": "not_run",
            "task_success": None,
            "agent_claim": claim,
            "review_record": _read(root / "acceptance.json", issues),
        },
        "selection": {
            "attempt_id": selected["attempt_id"] if selected else None,
            "source": selection_source,
            "reason": reason,
        },
        "video": video,
        "video_missing_reason": (
            None
            if video
            else "本轮没有可发布的选定录像；可能未进入仿真、选择无效或视频损坏。详见 issues。"
        ),
        "attempts": attempts,
        "reproduction": (
            _reproduction(root, root / selected["path"], manifest, issues)
            if selected
            else {}
        ),
        "issues": issues,
    }
    final = root / "final"
    if final.exists() and not final.is_symlink():
        raise ValueError(f"Refusing to replace an existing real directory: {final}")
    input_hash = hashlib.sha256(
        json.dumps(result, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if (
        usage_meter is None
        and previous.get("publication", {}).get("input_sha256") == input_hash
    ):
        published_video = previous.get("video")
        if (
            (final / "report.md").is_file()
            and (final / "report.md").read_text() == _report(previous)
            and (
                (
                    video is None
                    and published_video is None
                    and not (final / "video.mp4").exists()
                )
                or (
                    published_video is not None
                    and (final / "video.mp4").is_file()
                    and _hash(final / "video.mp4") == previous["video"]["sha256"]
                )
            )
        ):
            return previous
    publication = root / ".deliveries" / uuid4().hex
    publication.mkdir(parents=True)
    input_video = dict(video) if video else None
    if video:
        try:
            _copy_video(root / video["source"], publication / "video.mp4", video)
        except (ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            result["video"] = None
            result["video_missing_reason"] = (
                "选定录像无法转换为标准 MP4/H.264；原始录像仍保留。"
            )
            issues.append(f"Video encoding failed: {str(exc)[:1200]}")
    if usage_meter is not None:
        usage_meter.close()
    result["resources"] = refresh_usage(root)
    input_hash = hashlib.sha256(
        json.dumps(
            {**result, "video": input_video}, sort_keys=True, ensure_ascii=False
        ).encode()
    ).hexdigest()
    result["publication"] = {
        "id": publication.name,
        "directory": str(publication.relative_to(root)),
        "input_sha256": input_hash,
    }
    write_json(publication / "result.json", result)
    (publication / "report.md").write_text(_report(result), encoding="utf-8")
    pointer = root / f".final-{publication.name}"
    pointer.symlink_to(publication.relative_to(root), target_is_directory=True)
    os.replace(pointer, final)
    return result


def _copy_video(source: Path, destination: Path, video: dict) -> None:
    with tempfile.TemporaryDirectory(
        prefix="encode-", dir=destination.parent.parent
    ) as temporary:
        encoded = Path(temporary) / "video.mp4"
        if video["codec"] == "h264" and str(video["pixel_format"]).startswith(
            "yuv420p"
        ):
            shutil.copy2(source, encoded)
            if _hash(encoded) != video["source_sha256"]:
                raise ValueError("Source video changed during publication")
        else:
            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-v",
                    "error",
                    "-xerror",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-movflags",
                    "+faststart",
                    str(encoded),
                ],
                check=True,
                capture_output=True,
                timeout=300,
            )
            normalized = _probe(encoded)
            if normalized["frames"] != video["frames"]:
                raise ValueError("Encoding changed the decoded frame count")
            video.update(normalized)
        shutil.copy2(encoded, destination)
