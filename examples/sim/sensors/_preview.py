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

"""Save plots of sensor samples with a frame from the simulation timeline."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from matplotlib.animation import FFMpegWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def write_preview(
    output: Path,
    times: list[float],
    draw: Callable[[Figure, int], None],
    *,
    still: int = -1,
    video: bool = True,
) -> None:
    """Render a PNG and optional MP4 from the same captured samples."""
    if video and shutil.which("ffmpeg") is None:
        raise RuntimeError("Install ffmpeg to export MP4 or use --no-video.")
    output.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(10, 5.5), dpi=120, layout="constrained")
    FigureCanvasAgg(figure)
    draw(figure, still % len(times))
    figure.savefig(output / "preview.png")
    if not video:
        return
    writer = FFMpegWriter(
        fps=16,
        codec="h264",
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    with writer.saving(figure, str(output / "preview.mp4"), dpi=120):
        for index in range(len(times)):
            figure.clear()
            draw(figure, index)
            writer.grab_frame()
