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

"""Compact world-space trajectory panel for recorded motion playback."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=16)
def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load the playback font with Pillow's bundled fallback."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


class TrajectoryHUD:
    """Draw a fixed-scale XY map with recent execution and future reference."""

    size = (300, 350)
    pale = (174, 211, 188)
    green = (19, 115, 76)
    red = (215, 65, 65)
    ink = (36, 53, 65)
    muted = (105, 123, 130)
    background = (249, 252, 251)

    def __init__(self, actual: np.ndarray, reference: np.ndarray, *, dt: float) -> None:
        self.actual = np.asarray(actual)[:, :2]
        self.reference = np.asarray(reference)[:, :2]
        bounds = np.concatenate((self.actual, self.reference))
        self.center = (bounds.min(0) + bounds.max(0)) / 2
        self.width = max(float(np.ptp(bounds, axis=0).max()) * 1.15, 1.0)
        self.dt = dt
        self.trail = round(3 / dt)
        self.lead = round(2 / dt)
        self.title_font = load_font(17)
        self.font = load_font(14)
        self.small = load_font(12)
        self.plot_size = 224
        self.plot_origin = (54, 64)
        self.base = Image.new("RGB", (self.plot_size, self.plot_size), self.background)
        draw = ImageDraw.Draw(self.base)
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            xy = round(fraction * (self.plot_size - 1))
            draw.line(((xy, 0), (xy, self.plot_size - 1)), fill=(224, 234, 229))
            draw.line(((0, xy), (self.plot_size - 1, xy)), fill=(224, 234, 229))
        context = self._points(self.reference[:: max(1, len(self.reference) // 2000)])
        if len(context) > 1:
            draw.line([tuple(p) for p in context], fill=self.pale, width=2)

    def _points(self, xy: np.ndarray) -> np.ndarray:
        return (np.asarray(xy) - self.center) / self.width * (self.plot_size - 1) * [
            1,
            -1,
        ] + (self.plot_size - 1) / 2

    def draw(self, image: Image.Image, frame: int, body_error_m: float) -> None:
        """Overlay the map; body error is RMS tracked-link error after alignment."""
        panel = Image.new("RGBA", self.size)
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle(
            (0, 0, self.size[0] - 1, self.size[1] - 1),
            radius=12,
            fill=(*self.background, 246),
            outline=(208, 222, 216, 255),
        )
        draw.text((14, 11), "World XY (m)", font=self.title_font, fill=self.ink)
        draw.text(
            (286, 15),
            f"{self.width:.1f} m wide",
            anchor="ra",
            font=self.small,
            fill=self.muted,
        )
        available = min(self.lead, len(self.reference) - frame - 1) * self.dt
        for x, color, label in (
            (14, self.pale, "Ref path"),
            (105, self.red, "Robot 3s"),
            (203, self.green, f"Ref +{available:.1f}s"),
        ):
            draw.line(((x, 45), (x + 10, 45)), fill=color, width=3)
            draw.text((x + 15, 37), label, font=self.small, fill=self.ink)
        plot = self.base.copy()
        pd = ImageDraw.Draw(plot)
        actual = self._points(self.actual[max(0, frame - self.trail) : frame + 1])
        target = self._points(self.reference[frame : frame + self.lead + 1])
        for points, color in ((target, self.green), (actual, self.red)):
            if len(points) > 1:
                pd.line([tuple(p) for p in points], fill=color, width=2)
        a, r = actual[-1], target[0]
        for k in range(0, 10, 2):
            pd.line(
                (tuple(r + (a - r) * k / 10), tuple(r + (a - r) * (k + 1) / 10)),
                fill=self.muted,
            )
        if len(target) > 1:
            direction = target[-1] - target[max(0, len(target) - 11)]
            length = float(np.linalg.norm(direction))
            if length > 1:
                u = direction / length
                side = np.array([-u[1], u[0]])
                pd.polygon(
                    [
                        tuple(target[-1]),
                        tuple(target[-1] - 7 * u + 3 * side),
                        tuple(target[-1] - 7 * u - 3 * side),
                    ],
                    fill=self.green,
                )
        pd.ellipse(
            (r[0] - 4, r[1] - 4, r[0] + 4, r[1] + 4),
            fill=self.background,
            outline=self.green,
            width=2,
        )
        pd.ellipse(
            (a[0] - 3, a[1] - 3, a[0] + 3, a[1] + 3), fill=self.red, outline="white"
        )
        panel.paste(plot, self.plot_origin)
        px, py = self.plot_origin
        for fraction in (0, 0.5, 1):
            x = self.center[0] + (fraction - 0.5) * self.width
            y = self.center[1] + (0.5 - fraction) * self.width
            draw.text(
                (px + fraction * (self.plot_size - 1), py + self.plot_size + 5),
                f"{x:.1f}",
                anchor="mt",
                font=self.small,
                fill=self.muted,
            )
            draw.text(
                (px - 7, py + fraction * (self.plot_size - 1)),
                f"{y:.1f}",
                anchor="rm",
                font=self.small,
                fill=self.muted,
            )
        draw.line(((14, 313), (286, 313)), fill=(221, 231, 226))
        xy_error = float(np.linalg.norm(self.actual[frame] - self.reference[frame]))
        draw.text(
            (14, 324), f"XY  {xy_error * 100:.1f} cm", font=self.font, fill=self.ink
        )
        draw.text(
            (157, 324),
            f"Body  {body_error_m * 100:.1f} cm",
            font=self.font,
            fill=self.ink,
        )
        scale = min(image.width / 1920, image.height / 1080)
        panel = panel.resize(
            tuple(max(1, round(n * scale)) for n in self.size), Image.Resampling.LANCZOS
        )
        image.paste(panel, (round(18 * scale), round(72 * scale)), panel)

    def metadata(self) -> dict:
        """Return map dimensions, coordinate bounds and time-window semantics."""
        return {
            "panel_pixels_at_1080p": list(self.size),
            "map_center_xy_m": self.center.tolist(),
            "map_width_m": self.width,
            "history_seconds": 3,
            "future_seconds": 2,
            "coordinates": "unaligned world root XY",
            "body_error": "RMS tracked-link position error after root heading/XY alignment",
        }
