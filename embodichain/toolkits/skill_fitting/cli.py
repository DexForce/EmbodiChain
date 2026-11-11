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

"""Fit a video-selected joint segment with the native atomic skill engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
from typing import Sequence

import numpy as np

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> None:
    """Run offline reconstruction and persist accepted or rejected candidates.

    Args:
        argv: Explicit command arguments, or the process command line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", type=Path)
    inputs.add_argument(
        "--hdf5", type=Path, help="HDF5 image sequence used by the annotator."
    )
    parser.add_argument("--image-key")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--video-stream", type=int)
    parser.add_argument("--joints", type=Path, required=True)
    parser.add_argument("--robot-profile", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--segment-id")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--skill",
        required=True,
        choices=("pickup", "pick_up", "slide", "twist", "handover", "hand_over"),
    )
    parser.add_argument(
        "--time-offset",
        type=float,
        required=True,
        help="Joint timestamp corresponding to video time zero, in seconds.",
    )
    parser.add_argument("--qpos-key", default="qpos")
    parser.add_argument("--timestamps-key", default="timestamps")
    parser.add_argument("--joint-dt", type=float)
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument(
        "--qpos-semantics",
        required=True,
        choices=("measured_state", "actual_submission", "design_reference"),
    )
    parser.add_argument(
        "--phase-hints",
        type=Path,
        help="JSON complete half-open phase partition, relative to segment.",
    )
    parser.add_argument(
        "--fit-config", type=Path, help="Closed JSON SkillFitCfg overrides."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from embodichain.data_pipeline.skill_fitting import file_digest, save_skill_fit
    from embodichain.toolkits.trajectory_annotator._storage import AnnotationStore
    from ._pipeline import FitInputs, fit_recording

    try:
        if args.hdf5:
            from embodichain.toolkits.trajectory_annotator.hdf5 import Hdf5FrameSource

            source = Hdf5FrameSource(args.hdf5, image_key=args.image_key, fps=args.fps)
        else:
            from embodichain.toolkits.trajectory_annotator.video import VideoFrameSource

            source = VideoFrameSource(args.video, stream_index=args.video_stream)
        annotation_digest = (
            None if args.annotations is None else file_digest(args.annotations)
        )
        # Never allow result export to replace any input.
        input_paths = [
            args.video,
            args.hdf5,
            args.joints,
            args.robot_profile,
            args.annotations,
            args.phase_hints,
            args.fit_config,
        ]
        if args.output.resolve() in {p.resolve() for p in input_paths if p is not None}:
            raise ValueError("Output must differ from every input file.")
        try:
            info = source.prepare(threading.Event(), lambda **_: None)
            revision, selected_id = None, None
            if args.annotations:
                store = AnnotationStore(args.annotations.resolve(), 8 * 1024**2)
                store.initialize(info)
                document = store.snapshot()["document"]
                matches = [
                    s for s in document["segments"] if s["id"] == args.segment_id
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "--segment-id must select exactly one existing annotation."
                    )
                selected = matches[0]
                first, last = selected["start_frame"], selected["end_frame"]
                revision, selected_id = document["revision"], selected["id"]
            else:
                first = 0 if args.start_frame is None else args.start_frame
                last = (
                    info.frame_count - 1 if args.end_frame is None else args.end_frame
                )
            if info.kind == "video":
                base = info.details["time_base"]
                clock = np.asarray(source.pts, dtype=np.float64)
                clock = (clock - clock[0]) * base["numerator"] / base["denominator"]
                durations = (
                    np.r_[np.diff(clock), np.diff(clock)[-1]]
                    if len(clock) > 1
                    else np.array([1 / info.fps])
                )
            else:
                clock = np.arange(info.frame_count, dtype=np.float64) / info.fps
                durations = np.full(info.frame_count, 1 / info.fps)
            source.check_unchanged()
        finally:
            source.close()
        inputs = FitInputs(
            joints=str(args.joints),
            robot_profile=str(args.robot_profile),
            qpos_semantics=args.qpos_semantics,
            time_offset=args.time_offset,
            qpos_key=args.qpos_key,
            timestamps_key=args.timestamps_key,
            joint_dt=args.joint_dt,
            env_index=args.env_index,
            device=args.device,
            fit_config=(
                {}
                if args.fit_config is None
                else json.loads(args.fit_config.read_text())
            ),
            phase_hints=(
                None
                if args.phase_hints is None
                else json.loads(args.phase_hints.read_text())
            ),
        )
        if not 0 <= first <= last < info.frame_count:
            raise ValueError("Invalid inclusive video interval.")
        skill = {"pickup": "pick_up", "handover": "hand_over"}.get(
            args.skill, args.skill
        )
        arrays, report = fit_recording(
            inputs,
            skill_id=skill,
            frame_times=clock[first : last + 1],
            frame_durations=durations[first : last + 1],
            start_frame=first,
            end_frame=last,
            provenance=dict(
                source_identity=info.identity,
                source_timing=info.timing,
                annotation_revision=revision,
                annotation_id=selected_id,
                annotation_sha256=annotation_digest,
            ),
        )
        source.check_unchanged()
        if (
            args.annotations is not None
            and file_digest(args.annotations) != annotation_digest
        ):
            raise ValueError("Annotation revision changed during fitting.")
        save_skill_fit(args.output, arrays, report)
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "accepted": report["accepted"],
                    "parameters": report["parameters"],
                    "native_joint_rmse": report["native_joint_rmse"],
                    "tcp_errors": report["tcp_errors"],
                    "failures": report["failures"],
                },
                indent=2,
            )
        )
        if not report["accepted"]:
            raise SystemExit(2)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Skill fitting failed: {exc}\n")


if __name__ == "__main__":
    main()
