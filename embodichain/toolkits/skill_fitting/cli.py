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
import sys
import threading
from typing import Sequence

import numpy as np

__all__ = ["main"]


def _bundle_main(argv: Sequence[str]) -> None:
    """Fit sealed portable inputs using their saved frame clock."""
    parser = argparse.ArgumentParser(
        description="Fit a portable reconstructed-scene bundle."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-task-output", type=Path)
    parser.add_argument("--package-output", type=Path)
    parser.add_argument("--package-video", type=Path)
    parser.add_argument("--video-stream", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.output is None and args.package_output is None:
        parser.error("Choose --output for fitting or --package-output for repackaging.")
    if args.run_task_output and args.output is None:
        parser.error("--run-task-output requires --output.")
    if args.package_video and args.package_output is None:
        parser.error("--package-video requires --package-output.")
    if args.video_stream is not None and args.package_video is None:
        parser.error("--video-stream requires --package-video.")
    from ._bundle import load_inputs, pack_inputs, _video_source
    from ._episode import fit_episode
    from ._export import export_run_task
    from embodichain.data_pipeline.skill_fitting import save_skill_fit

    inputs, document, clock, durations = load_inputs(args.bundle)
    inputs.device = args.device
    if args.package_output is not None:
        root = args.bundle.expanduser().resolve()
        visual = _video_source(root, json.loads((root / "fitting.json").read_text()))
        video = args.package_video or (None if visual is None else visual[0])
        stream = args.video_stream
        if args.package_video is None and visual is not None:
            stream = visual[1]
        package = pack_inputs(
            args.package_output,
            inputs,
            document,
            clock,
            durations,
            video=video,
            video_stream=stream,
        )
        print(json.dumps(dict(package=str(package)), indent=2))
    if args.output is None:
        return
    if args.output.resolve().is_relative_to(args.bundle.resolve()):
        if args.output.resolve().exists():
            raise FileExistsError(
                "Choose a new fit output; bundle inputs must stay unchanged."
            )
    arrays, report = fit_episode(
        inputs, document, lambda a, b: (clock[a : b + 1], durations[a : b + 1])
    )
    save_skill_fit(args.output, arrays, report)
    archive = None
    if args.run_task_output and report["accepted"]:
        archive = export_run_task(args.run_task_output, arrays, report, inputs)
    print(
        json.dumps(
            dict(
                accepted=report["accepted"],
                output=str(args.output.resolve()),
                run_task=None if archive is None else str(archive),
                failures=report["failures"],
            ),
            indent=2,
        )
    )
    if not report["accepted"]:
        raise SystemExit(2)


def main(argv: Sequence[str] | None = None) -> None:
    """Run offline reconstruction and persist accepted or rejected candidates.

    Args:
        argv: Explicit command arguments, or the process command line.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(value == "--bundle" or value.startswith("--bundle=") for value in argv):
        _bundle_main(argv)
        return
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
    parser.add_argument(
        "--scene",
        type=Path,
        help="Reconstructed scene YAML or physical_export directory; auto-discovered beside joints when present.",
    )
    parser.add_argument(
        "--object-id",
        default="",
        help="Scene grasp target; default selects the object nearest the observed grasp.",
    )
    parser.add_argument(
        "--participant",
        default="auto",
        help="auto, configured, or one configured motion/hand control-part name.",
    )
    parser.add_argument("--annotations", type=Path)
    parser.add_argument(
        "--package-output",
        type=Path,
        help="New portable input folder; requires --all-segments.",
    )
    parser.add_argument(
        "--package-video",
        type=Path,
        help="Include the selected source video for --bundle web startup.",
    )
    parser.add_argument("--segment-id")
    parser.add_argument(
        "--all-segments",
        action="store_true",
        help="Fit all saved annotations and preserve uncovered samples as DemoSkill sources.",
    )
    parser.add_argument(
        "--run-task-output",
        type=Path,
        help="New directory for native run-task skills and demo-only tensors; requires --all-segments.",
    )
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--skill",
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
    if args.package_output and not args.all_segments:
        parser.error("--package-output requires --all-segments.")
    if args.package_video and (
        not args.package_output
        or args.hdf5
        or not args.package_video.is_file()
        or not args.video.is_file()
        or not args.package_video.samefile(args.video)
    ):
        parser.error(
            "--package-video must name the --video source and requires --package-output."
        )
    if args.all_segments:
        if args.annotations is None or any(
            v is not None
            for v in (args.segment_id, args.skill, args.start_frame, args.end_frame)
        ):
            parser.error(
                "--all-segments requires --annotations and cannot be combined with --segment-id, --skill or frame bounds."
            )
    elif args.skill is None or args.run_task_output is not None:
        parser.error(
            "Single-segment fitting requires --skill; --run-task-output requires --all-segments."
        )
    from embodichain.data_pipeline.skill_fitting import file_digest, save_skill_fit
    from embodichain.toolkits.trajectory_annotator._storage import AnnotationStore
    from embodichain.toolkits.trajectory_annotator._skills import _canonical_skill_id
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
            args.scene,
            args.annotations,
            args.phase_hints,
            args.fit_config,
        ]
        if args.output.resolve() in {p.resolve() for p in input_paths if p is not None}:
            raise ValueError("Output must differ from every input file.")
        try:
            info = source.prepare(threading.Event(), lambda **_: None)
            revision, selected_id, selected = None, None, None
            if args.annotations:
                store = AnnotationStore(args.annotations.resolve(), 8 * 1024**2)
                store.initialize(info)
                document = store.snapshot()["document"]
                matches = [
                    s for s in document["segments"] if s["id"] == args.segment_id
                ]
                if not args.all_segments and len(matches) != 1:
                    raise ValueError(
                        "--segment-id must select exactly one existing annotation."
                    )
                if args.all_segments:
                    first, last = 0, info.frame_count - 1
                else:
                    selected = matches[0]
                    core = selected.get("core_interval", selected)
                    first, last = core["start_frame"], core["end_frame"]
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
            scene="" if args.scene is None else str(args.scene),
            object_id=args.object_id,
            participant=args.participant,
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
        annotated_skill = (
            None if selected is None else _canonical_skill_id(selected["label"])
        )
        if annotated_skill is not None and annotated_skill != skill:
            raise ValueError("The fitted skill must match the selected annotation.")
        if args.all_segments:
            from ._episode import fit_episode

            arrays, report = fit_episode(
                inputs,
                document,
                lambda first, last: (
                    clock[first : last + 1],
                    durations[first : last + 1],
                ),
            )
        else:
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
                    annotation_segment=selected,
                    annotation_video_interval=(
                        None
                        if selected is None
                        else {
                            key: selected[key] for key in ("start_frame", "end_frame")
                        }
                    ),
                ),
            )
        source.check_unchanged()
        protected = set(report.get("scene_files_sha256", {}))
        protected.update(
            s["urdf_path"]
            for s in report.get("robot_profile", {}).get("solvers", {}).values()
        )
        for fit in report.get("fits", ()):
            protected.update(fit.get("scene_files_sha256", {}))
        if str(args.output.resolve()) in protected:
            raise ValueError(
                "Output must differ from reconstructed scene and hand geometry inputs."
            )
        if (
            args.annotations is not None
            and file_digest(args.annotations) != annotation_digest
        ):
            raise ValueError("Annotation revision changed during fitting.")
        save_skill_fit(args.output, arrays, report)
        if args.package_output:
            from ._bundle import pack_inputs

            pack_inputs(
                args.package_output,
                inputs,
                document,
                clock,
                durations,
                video=args.package_video,
                video_stream=args.video_stream,
            )
        run_task = None
        if args.run_task_output and report["accepted"]:
            from ._export import export_run_task

            run_task = export_run_task(
                args.run_task_output.resolve(), arrays, report, inputs
            )
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "run_task": None if run_task is None else str(run_task),
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
