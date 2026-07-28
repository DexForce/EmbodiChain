# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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

"""Tests for the vendored depth quantization helpers."""

from __future__ import annotations

import numpy as np
import pytest

from embodichain.data_pipeline.depth_video import (
    DEPTH_METER_UNIT,
    DEPTH_MILLIMETER_UNIT,
    DEPTH_QMAX,
    dequantize_depth,
    quantize_depth,
)


class TestQuantizeDepth:
    """Quantization math tests."""

    def test_code_range_float_metres(self):
        """Quantized codes must lie in [0, DEPTH_QMAX]."""
        depth = np.linspace(0.0, 20.0, 64 * 80, dtype=np.float32).reshape(64, 80)
        codes = quantize_depth(depth, video_backend=None, input_unit=DEPTH_METER_UNIT)
        assert codes.dtype == np.uint16
        assert codes.min() >= 0
        assert codes.max() <= DEPTH_QMAX

    def test_out_of_range_clamps_to_endpoints(self):
        """Depths below depth_min map to 0; above depth_max map to DEPTH_QMAX."""
        depth = np.array([0.0, 0.05, 5.0, 100.0], dtype=np.float32)
        codes = quantize_depth(
            depth,
            depth_min=0.05,
            depth_max=5.0,
            use_log=False,
            video_backend=None,
            input_unit=DEPTH_METER_UNIT,
        )
        assert codes[0] == 0  # below depth_min
        assert codes[1] == 0  # at depth_min
        assert codes[2] == DEPTH_QMAX  # at depth_max
        assert codes[3] == DEPTH_QMAX  # above depth_max

    def test_video_backend_returns_av_frame(self):
        """``video_backend="pyav"`` returns an av.VideoFrame with gray12le."""
        import av

        depth = np.full((32, 48), 1.0, dtype=np.float32)
        frame = quantize_depth(depth, video_backend="pyav", input_unit=DEPTH_METER_UNIT)
        assert isinstance(frame, av.VideoFrame)
        assert frame.format.name == "gray12le"
        assert frame.width == 48 and frame.height == 32

    def test_invalid_input_unit_raises(self):
        depth = np.zeros((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="input_unit"):
            quantize_depth(depth, input_unit="km", video_backend=None)

    def test_log_invalid_shift_raises(self):
        """Log mode requires depth_min + shift > 0."""
        depth = np.full((4, 4), 1.0, dtype=np.float32)
        with pytest.raises(ValueError, match="depth_min \\+ shift"):
            quantize_depth(
                depth,
                depth_min=0.1,
                shift=-0.2,
                use_log=True,
                video_backend=None,
                input_unit=DEPTH_METER_UNIT,
            )

    def test_uint16_millimetre_input(self):
        """Integer input is interpreted as millimetres by auto inference."""
        depth_mm = np.array([100, 1000, 5000], dtype=np.uint16)  # 0.1, 1.0, 5.0 m
        codes = quantize_depth(
            depth_mm,
            depth_min=0.1,
            depth_max=5.0,
            use_log=False,
            video_backend=None,
            input_unit="auto",
        )
        assert codes[0] == 0  # 0.1 m == depth_min
        assert codes[2] == DEPTH_QMAX  # 5.0 m == depth_max


@pytest.mark.parametrize("use_log", [True, False])
def test_roundtrip_error_bounded(use_log):
    """quantize -> dequantize error stays within a small bound over the range."""
    depth_min, depth_max, shift = 0.1, 3.0, 1.0
    depth = np.linspace(depth_min, depth_max, 64 * 80, dtype=np.float32).reshape(64, 80)
    codes = quantize_depth(
        depth,
        depth_min=depth_min,
        depth_max=depth_max,
        shift=shift,
        use_log=use_log,
        video_backend=None,
        input_unit=DEPTH_METER_UNIT,
    )
    back = dequantize_depth(
        codes,
        depth_min=depth_min,
        depth_max=depth_max,
        shift=shift,
        use_log=use_log,
        output_unit=DEPTH_METER_UNIT,
        output_tensor=False,
    ).squeeze()

    # Endpoints are reconstructed exactly by construction.
    assert back[0, 0] == pytest.approx(depth_min, abs=1e-5)
    assert back[-1, -1] == pytest.approx(depth_max, abs=1e-5)
    # 12-bit log/linear quantization keeps the max error well under 5 mm over
    # a 0.1-3.0 m range.
    max_err = float(np.abs(back - depth).max())
    assert max_err < 5e-3, f"max abs error {max_err} exceeds 5 mm"


def test_dequantize_invalid_output_unit_raises():
    codes = np.zeros((4, 4), dtype=np.uint16)
    with pytest.raises(ValueError, match="output_unit"):
        dequantize_depth(codes, output_unit="km")
