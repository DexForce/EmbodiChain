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

"""Strict, post-timing integrity checks for every saved LeRobot video frame.

The verifier software-decodes AV1 videos sequentially, hashes raw YUV420p
frames, and reconciles actual counts with official LeRobot episode metadata.
It does not establish lossless RGB fidelity or every SDK random-seek position.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import traceback
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

__all__ = ["verify_dataset"]

_AUDIT_REVISION = "full-video-integrity/v1"


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _inventory(
    root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    import pyarrow.parquet as pq

    _require(not (root / "INCOMPLETE").exists(), "Dataset is marked INCOMPLETE")
    info_file = root / "meta/info.json"
    info = json.loads(info_file.read_text())
    _require(
        info["total_episodes"] > 0 and info["total_frames"] > 0,
        "Official corpus must contain accepted episodes and frames",
    )
    cameras = {
        key: value
        for key, value in info["features"].items()
        if value["dtype"] == "video"
    }
    _require(bool(cameras), "Official metadata contains no recorded video cameras")
    fps = float(info["fps"])
    _require(fps > 0 and math.isfinite(fps), "Invalid official FPS")
    columns = ["episode_index", "length", "dataset_from_index", "dataset_to_index"]
    for camera in cameras:
        columns.extend(
            f"videos/{camera}/{field}"
            for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
        )
    metadata_files = [info_file, *sorted((root / "meta/episodes").rglob("*.parquet"))]
    _require(len(metadata_files) > 1, "Official episode metadata is missing")
    hashes = {str(path.relative_to(root)): _digest(path) for path in metadata_files}
    files: dict[str, dict[str, Any]] = {}
    count = frames = 0
    for meta in metadata_files[1:]:
        for batch in pq.ParquetFile(meta).iter_batches(
            batch_size=64, columns=columns, use_threads=False
        ):
            for episode in batch.to_pylist():
                length = int(episode["length"])
                _require(
                    episode["episode_index"] == count
                    and episode["dataset_from_index"] == frames
                    and episode["dataset_to_index"] == frames + length,
                    "Official episode order/length offsets differ",
                )
                _require(length > 0, "Official episode has no frames")
                for camera, feature in cameras.items():
                    relative = info["video_path"].format(
                        video_key=camera,
                        chunk_index=int(episode[f"videos/{camera}/chunk_index"]),
                        file_index=int(episode[f"videos/{camera}/file_index"]),
                    )
                    path = (root / relative).resolve()
                    _require(
                        root.resolve() in path.parents,
                        "Video metadata escapes the corpus",
                    )
                    _require(path.is_file(), f"Missing registered video: {relative}")
                    start = float(episode[f"videos/{camera}/from_timestamp"])
                    stop = float(episode[f"videos/{camera}/to_timestamp"])
                    _require(
                        math.isclose(
                            stop - start, length / fps, rel_tol=0, abs_tol=1e-4
                        ),
                        "Episode video duration differs from its official frame count",
                    )
                    group = files.setdefault(
                        relative,
                        {
                            "camera": camera,
                            "expected_frames": 0,
                            "episodes": [],
                            "spans": [],
                            "shape": list(feature["shape"]),
                            "fps": fps,
                        },
                    )
                    _require(
                        group["camera"] == camera,
                        "Multiple cameras reference one video file",
                    )
                    group["expected_frames"] += length
                    group["episodes"].append(count)
                    group["spans"].append([start, stop])
                count += 1
                frames += length
    _require(
        count == info["total_episodes"] and frames == info["total_frames"],
        "Official episode and info frame/episode totals disagree",
    )
    actual = {str(path.relative_to(root)) for path in (root / "videos").rglob("*.mp4")}
    _require(
        actual == set(files),
        f"Unregistered or missing MP4 files: actual={sorted(actual)}, registered={sorted(files)}",
    )
    for relative, group in files.items():
        end = 0.0
        for start, stop in sorted(group["spans"]):
            _require(
                math.isclose(start, end, rel_tol=0, abs_tol=1e-4),
                f"Video spans overlap or contain gaps: {relative}",
            )
            end = stop
        _require(
            math.isclose(end, group["expected_frames"] / fps, rel_tol=0, abs_tol=1e-4),
            f"Video span total differs: {relative}",
        )
    _require(
        all(
            sum(
                group["expected_frames"]
                for group in files.values()
                if group["camera"] == camera
            )
            == frames
            for camera in cameras
        ),
        "A camera is missing official episode frames",
    )
    return info, files, hashes


def _decode_file(
    root: Path,
    relative: str,
    expected: dict[str, Any],
    artifacts: Path,
    index: int,
    *,
    ffmpeg: str,
    ffprobe: str,
    threads: int,
    timeout: float,
    sink: list[dict[str, Any]],
) -> dict[str, Any]:
    path = root / relative
    before = _digest(path)
    record: dict[str, Any] = {
        "file": relative,
        "camera": expected["camera"],
        "expected_frames": expected["expected_frames"],
        "episodes": expected["episodes"],
        "compressed_sha256": before,
        "decoded_pixel_format": "yuv420p",
        "frame_hash_scope": "SHA256 of FFmpeg decoded rawvideo YUV420p planes; not RGB or BC-normalized tensors",
    }
    sink.append(record)
    probe_command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v",
        "-show_entries",
        "stream=index,codec_name,codec_type,width,height,pix_fmt,r_frame_rate,nb_frames,time_base,duration",
        "-of",
        "json",
        str(path),
    ]
    probe = subprocess.run(
        probe_command, capture_output=True, text=True, timeout=timeout, check=False
    )
    record.update(probe_command=probe_command, probe_exitcode=probe.returncode)
    (artifacts / f"{index:04d}.probe.stderr").write_text(probe.stderr)
    _require(
        probe.returncode == 0, f"ffprobe failed for {relative}: {probe.stderr[-2000:]}"
    )
    streams = json.loads(probe.stdout)["streams"]
    _require(len(streams) == 1, f"Expected exactly one video stream in {relative}")
    stream = streams[0]
    _require(stream["codec_name"] == "av1", f"Unexpected recorded codec in {relative}")
    _require(
        [stream["height"], stream["width"], 3] == expected["shape"],
        f"Camera dimensions differ from official metadata in {relative}",
    )
    _require(
        math.isclose(
            float(Fraction(stream["r_frame_rate"])),
            expected["fps"],
            rel_tol=0,
            abs_tol=1e-8,
        ),
        f"Actual video FPS differs from official metadata: {relative}",
    )
    _require(
        stream.get("pix_fmt") == "yuv420p",
        f"Unexpected AV1 stored pixel format in {relative}",
    )
    record["probe"] = stream
    framehash = artifacts / f"{index:04d}.decoded-frame-sha256.txt"
    error_log = artifacts / f"{index:04d}.decode.stderr"
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-err_detect",
        "explode",
        "-hwaccel",
        "none",
        "-threads:v",
        str(threads),
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-fps_mode",
        "passthrough",
        "-c:v",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-threads:v",
        str(threads),
        "-filter_threads",
        "1",
        "-filter_complex_threads",
        "1",
        "-f",
        "framehash",
        "-hash",
        "sha256",
        "-y",
        str(framehash),
    ]
    started = time.monotonic()
    with error_log.open("wb") as stderr:
        process = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            timeout=timeout,
            check=False,
        )
    record.update(
        decode_command=command,
        decode_exitcode=process.returncode,
        decode_seconds=time.monotonic() - started,
        decoded_frame_hashes=str(framehash),
        decode_error_log=str(error_log),
    )
    errors = error_log.read_text(errors="replace")
    _require(
        process.returncode == 0 and not errors.strip(),
        f"Strict full decode failed for {relative} (exit={process.returncode}): {errors[-2000:]}",
    )
    count = 0
    previous_pts = None
    with framehash.open() as values:
        for line in values:
            if line.startswith("#") or not line.strip():
                continue
            fields = [value.strip() for value in line.split(",")]
            _require(
                len(fields) == 6 and fields[0] == "0" and len(fields[-1]) == 64,
                "Malformed decoded frame-hash record",
            )
            pts = int(fields[2])
            _require(
                previous_pts is None or pts > previous_pts,
                "Decoded frames have duplicate or reversed presentation timestamps",
            )
            previous_pts = pts
            count += 1
    record["decoded_frames"] = count
    _require(
        count == expected["expected_frames"],
        f'Actual decoded frame count differs for {relative}: decoded={count}, official={expected["expected_frames"]}',
    )
    if str(stream.get("nb_frames", "")).isdigit():
        _require(
            int(stream["nb_frames"]) == count,
            f"Container declared frame count differs from full decode: {relative}",
        )
    _require(
        _digest(path) == before,
        f"Recorded video changed while being audited: {relative}",
    )
    record["decoded_frame_hash_file_sha256"] = _digest(framehash)
    record["passed"] = True
    return record


def verify_dataset(
    root: Path | str,
    output: Path | str,
    *,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
    threads: int = 1,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Software-decode all saved AV1 frames and retain an independent receipt.

    Args:
        root: Finalized LeRobot corpus containing official ``meta/info.json``,
            episode Parquet metadata and AV1/YUV420p camera files.
        output: New receipt JSON path outside the immutable corpus. The sibling
            ``<output-stem>-artifacts`` directory must also be new. Existing
            successful or failed evidence is never overwritten.
        ffmpeg: Software decoder executable, resolved from PATH when omitted.
        ffprobe: Stream metadata executable, resolved from PATH when omitted.
        threads: FFmpeg input/output codec threads, from one to eight. Files
            are decoded sequentially. The caller controls process CPU affinity;
            the CLI caps its inherited CPU budget at eight cores.
        timeout: Positive finite timeout in seconds for each probe/decode call.

    Returns:
        A persisted receipt with ``passed`` and ``all_saved_video_frames_decoded``
        flags, official/actual counts, decoder commands/versions, source hashes
        and artifact paths. Decode or metadata failures return a failed receipt.
        Frame hashes cover decoded raw YUV420p planes, rather than RGB tensors;
        success proves software decode/count integrity, not lossless camera
        fidelity, learned policy quality or all SDK random-seek positions.

    Raises:
        FileExistsError: Receipt or artifact directory already exists.
        ValueError: Output would modify the immutable input corpus.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    _require(
        root not in output.parents and output != root,
        "Audit output must be outside the immutable corpus",
    )
    artifacts = output.parent / (output.stem + "-artifacts")
    if output.exists() or artifacts.exists():
        raise FileExistsError(f"Audit evidence already exists: {output} or {artifacts}")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(exist_ok=False)
    result: dict[str, Any] = {
        "revision": _AUDIT_REVISION,
        "passed": False,
        "all_saved_video_frames_decoded": False,
        "dataset": str(root),
        "scope": "post-timing sequential software decode/count integrity; no lossless RGB, policy-quality or all-SDK-seek claim",
        "decoded_pixel_format": "yuv420p",
        "frame_hash_scope": "SHA256 of decoded rawvideo YUV420p planes; not RGB or BC-normalized tensors",
        "files": [],
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "threads_per_file": threads,
        "files_in_parallel": 1,
        "verifier_sha256": _digest(Path(__file__)),
    }
    started = time.monotonic()
    try:
        _require(
            type(threads) is int and 1 <= threads <= 8,
            "Software decoder threads must be in [1,8]",
        )
        _require(
            math.isfinite(timeout) and timeout > 0,
            "Probe/decode timeout must be finite and positive",
        )
        ffmpeg = ffmpeg or shutil.which("ffmpeg")
        ffprobe = ffprobe or shutil.which("ffprobe")
        _require(
            ffmpeg is not None and ffprobe is not None,
            "FFmpeg/ffprobe executables are required",
        )
        for name, executable in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
            version = subprocess.run(
                [executable, "-version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=min(timeout, 30.0),
            )
            result[name + "_version"] = version.stdout.splitlines()[0]
        info, files, metadata_hashes = _inventory(root)
        result.update(
            expected_episodes=info["total_episodes"],
            expected_frames_per_camera=info["total_frames"],
            metadata_sha256=metadata_hashes,
        )
        for index, (relative, expected) in enumerate(sorted(files.items())):
            _decode_file(
                root,
                relative,
                expected,
                artifacts,
                index,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                threads=threads,
                timeout=timeout,
                sink=result["files"],
            )
        counts: dict[str, int] = defaultdict(int)
        for record in result["files"]:
            counts[record["camera"]] += record["decoded_frames"]
        _require(
            all(value == info["total_frames"] for value in counts.values()),
            "Actual per-camera decoded total differs from official SDK total",
        )
        for relative, value in metadata_hashes.items():
            _require(
                _digest(root / relative) == value,
                f"Official metadata changed during audit: {relative}",
            )
        result.update(
            passed=True,
            decoded_frames_per_camera=dict(counts),
            total_decoded_camera_frames=sum(counts.values()),
            all_saved_video_frames_decoded=True,
        )
    except Exception as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
    result["seconds_outside_training_pipeline"] = time.monotonic() - started
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-count", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    args = parser.parse_args()
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        ARROW_NUM_THREADS="1",
    )
    _require(
        args.ffmpeg is not None and args.ffprobe is not None,
        "FFmpeg/ffprobe executables are required",
    )
    _require(
        1 <= args.cpu_count <= 8 and 1 <= args.threads <= args.cpu_count,
        "CPU budget must be 1..8 and cover decoder threads",
    )
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available[-args.cpu_count :])
    try:
        result = verify_dataset(
            args.dataset,
            args.output,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            threads=args.threads,
            timeout=args.timeout,
        )
    except (FileExistsError, ValueError) as error:
        print(
            json.dumps(
                {"passed": False, "error": repr(error), "receipt_written": False}
            ),
            flush=True,
        )
        raise SystemExit(2) from error
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "passed",
                    "error",
                    "expected_frames_per_camera",
                    "decoded_frames_per_camera",
                    "seconds_outside_training_pipeline",
                )
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    _main()
