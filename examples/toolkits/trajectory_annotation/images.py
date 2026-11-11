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

"""Annotate an existing RGB image array, independent of HDF5 and video codecs.

Run: embodichain run-example toolkits/trajectory_annotation/images
Replace the generated images with your N×H×W×3 RGB uint8 array or image list.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from embodichain.toolkits.trajectory_annotator.app import launch_annotation
from embodichain.toolkits.trajectory_annotator.frames import ArrayFrameSource


def main() -> None:
    """Build a small moving-square sequence and pass its array to the annotator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("outputs/trajectory_annotation/images.segments.json"),
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    images = np.full((90, 160, 320, 3), (20, 28, 38), dtype=np.uint8)
    for index, image in enumerate(images):
        x = 12 + index * 3
        image[55:105, x : x + 32] = (45, 210, 160)
    # Caller owns the arrays and keeps them unchanged until launch_annotation returns.
    launch_annotation(
        lambda: ArrayFrameSource(images, fps=30, name="Moving square RGB frames"),
        args.annotations,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
