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

"""Annotate the supplied cam_high HDF5 demo through the common frame interface.

Run from the repository root in conda environment py312:
    embodichain run-example toolkits/trajectory_annotation
Override --hdf5 and --output-dir for another recording and output directory.
Use --image-key or --fps to override camera selection and timing. Ctrl+C stops the app.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from embodichain.toolkits.trajectory_annotator.app import launch_annotation
from embodichain.toolkits.trajectory_annotator.hdf5 import Hdf5FrameSource


def main() -> None:
    """Launch cam_high annotation without converting or modifying the recording."""
    parser = argparse.ArgumentParser(
        prog="embodichain run-example toolkits/trajectory_annotation",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=Path(
            "/home/oem/piper-sim-data/label_down/episode_20261022T073344_323022Z/episode.hdf5"
        ),
        help="HDF5 recording; omit to use the supplied local demo.",
    )
    parser.add_argument(
        "--image-key", default=None, help="Auto-detect cam_high when omitted."
    )
    parser.add_argument(
        "--fps", type=float, default=None, help="Default: recorded FPS metadata."
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/trajectory_annotation"),
        help="Directory for <hdf5-stem>.<camera>.segments.json; created on save.",
    )
    output.add_argument(
        "--annotations",
        type=Path,
        help="Explicit annotation JSON path, as an alternative to --output-dir.",
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not args.hdf5.is_file():
        parser.error(f"HDF5 file does not exist: {args.hdf5}")
    if args.image_key is not None and not args.image_key.strip("/"):
        parser.error("--image-key must name an image dataset.")
    if args.annotations is None:
        if args.output_dir.exists() and not args.output_dir.is_dir():
            parser.error(f"--output-dir must be a directory: {args.output_dir}")
        camera = (args.image_key or "cam_high").strip("/").rsplit("/", 1)[-1]
        annotations = args.output_dir / f"{args.hdf5.stem}.{camera}.segments.json"
    else:
        annotations = args.annotations
    launch_annotation(
        lambda: Hdf5FrameSource(args.hdf5, image_key=args.image_key, fps=args.fps),
        annotations,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
