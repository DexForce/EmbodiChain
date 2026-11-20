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

"""Exercise standalone robot cleanup without provider or simulation calls."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
from PIL import Image
import pytest
import requests

from embodichain.cli.main import main
from embodichain.gen_sim._robot_cleanup import clean_robot_image
from embodichain.gen_sim.services import GenSimServices, SegmentationResult
from embodichain.gen_sim.services.contracts import SegmentationRequest

pytestmark = pytest.mark.no_sim

_SIZE = (100, 80)
_ARMS = ((3, 33, 15, 68), (81, 30, 93, 67))
_GRIPPERS = ((9, 30, 13, 38), (82, 28, 86, 37))
_PLATE = ((35, 66, 65, 68), (35, 67, 40, 80), (60, 67, 65, 80))
_PROP = (46, 60, 54, 63)


def _rle(*rectangles: tuple[int, int, int, int]) -> dict:
    width, height = _SIZE
    image = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in rectangles:
        image[y0:y1, x0:x1] = True
    changes = np.flatnonzero(np.diff(np.r_[False, image.ravel()]))
    counts = np.diff(np.r_[0, changes, image.size]).tolist()
    return {"size": [height, width], "counts": counts}


class _Segmenter:
    def __init__(self, mount_source: str) -> None:
        self.detections = {
            "robot arm": tuple(_rle(arm) for arm in _ARMS),
            "robot gripper": tuple(_rle(jaw) for jaw in _GRIPPERS),
            "camera mounting bracket": (_rle(_PROP),),
        }
        if mount_source == "generic":
            self.detections["camera mounting bracket"] += (_rle(*_PLATE, _PROP),)
        elif mount_source == "fallback":
            self.detections["black rectangular base at bottom center of image"] = (
                _rle(*_PLATE),
            )

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        return SegmentationResult(self.detections.get(request.prompt, ()), _SIZE)


class _Editor:
    def __init__(self, *, no_op: bool = False) -> None:
        self._no_op = no_op

    def generate(self, request: object) -> None:
        pytest.fail("The cleanup must call image-to-image editing.")

    def generate_image_by_image(
        self, *, image_path: Path, output_path: Path, **kwargs: object
    ) -> Path:
        if self._no_op:
            copyfile(image_path, output_path)
        else:
            Image.new("RGB", _SIZE, "blue").save(output_path)
        return output_path


def _frame(path: Path) -> Path:
    image = Image.new("RGB", _SIZE, "white")
    image.paste((255, 0, 0), _PROP)
    image.save(path)
    return path


@pytest.mark.parametrize("mount_source", ["generic", "fallback"])
def test_masks_preserve_tabletop_prop_and_exterior(
    tmp_path: Path, mount_source: str
) -> None:
    source = _frame(tmp_path / "head.png")
    output = tmp_path / "result"
    with GenSimServices(
        injected={
            "image_segmentation": _Segmenter(mount_source),
            "image_generation": _Editor(),
        }
    ) as services:
        report = clean_robot_image(source, output, services=services)
        assert services.records()[-1]["operation"] == "generate_image_by_image"

    with (
        Image.open(source) as original,
        Image.open(output / "input.png") as cleaned,
        Image.open(output / "robot-mask.png") as mask,
        Image.open(output / "grippers-mask.png") as grippers,
        Image.open(output / "camera-support-mask.png") as mount,
    ):
        outside = np.asarray(mask) == 0
        assert np.array_equal(
            np.asarray(cleaned)[outside], np.asarray(original)[outside]
        )
        assert cleaned.getpixel((50, 62)) == (255, 0, 0)
        x0, y0, x1, y1 = _PROP
        assert not np.asarray(mask)[y0:y1, x0:x1].any()
        assert grippers.getpixel((10, 34)) == grippers.getpixel((83, 34)) == 255
        assert mount.getpixel((50, 75)) == 0
        assert cleaned.getpixel((50, 75)) == (0, 0, 255)
    assert report["outside_mask_pixels_changed"] == 0
    assert report["gripper_box_source"] == "arm_masks"


def test_editor_echo_fails_without_publishing_cleaned_frame(tmp_path: Path) -> None:
    output = tmp_path / "result"
    with (
        GenSimServices(
            injected={
                "image_segmentation": _Segmenter("generic"),
                "image_generation": _Editor(no_op=True),
            }
        ) as services,
        pytest.raises(ValueError, match="did not change any detected robot"),
    ):
        clean_robot_image(_frame(tmp_path / "head.png"), output, services=services)

    assert not (output / "input.png").exists()
    assert not (output / "preparation.json").exists()


def test_custom_support_prompt_cannot_bypass_mount_geometry(tmp_path: Path) -> None:
    output = tmp_path / "result"
    with GenSimServices(
        injected={
            "image_segmentation": _Segmenter("invalid"),
            "image_generation": _Editor(),
        }
    ) as services:
        with pytest.raises(ValueError, match="bottom-center camera support"):
            clean_robot_image(
                _frame(tmp_path / "head.png"),
                output,
                services=services,
                camera_support_prompt="camera mounting bracket",
            )
        assert not any(
            record["role"] == "image_generation" for record in services.records()
        )
    assert not (output / "input.png").exists()


def test_cli_uses_configured_authenticated_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles = tmp_path / "model-services.json"
    profiles.write_text(
        json.dumps(
            {
                "model_services": {
                    "profiles": {
                        "segmenter": {
                            "adapter": "segmentation_http",
                            "base_url": "https://segmentation.invalid",
                            "api_key": "segmentation-test-key",
                            "options": {
                                "health_path": "/health",
                                "segment_by_prompt_path": "/masks",
                            },
                        },
                        "editor": {
                            "adapter": "image_http",
                            "base_url": "https://editing.invalid",
                            "api_key": "editing-test-key",
                            "options": {
                                "health_path": "/health",
                                "generate_image_by_prompt_path": "/generate",
                                "generate_image_by_image_path": "/edit-frame",
                            },
                        },
                    },
                    "roles": {
                        "image_segmentation": "segmenter",
                        "image_generation": "editor",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EMBODICHAIN_MODEL_SERVICES_CONFIG", str(profiles))
    detections = _Segmenter("generic").detections
    png = BytesIO()
    Image.new("RGB", _SIZE, "blue").save(png, format="PNG")
    edits = []

    def send(
        self: requests.adapters.HTTPAdapter,
        request: requests.PreparedRequest,
        **kwargs: object,
    ) -> requests.Response:
        response = requests.Response()
        response.status_code = 200
        if request.method == "GET":
            assert request.url in {
                "https://segmentation.invalid/health",
                "https://editing.invalid/health",
            }
            response.headers["Content-Type"] = "application/json"
            response._content = b'{"ok": true}'
        elif request.url == "https://segmentation.invalid/masks":
            assert request.headers["Authorization"] == "Bearer segmentation-test-key"
            prompt = next(
                prompt
                for prompt in detections
                if f"\r\n\r\n{prompt}\r\n".encode() in request.body
            )
            response.headers["Content-Type"] = "application/json"
            response._content = json.dumps(
                {"ok": True, "masks": detections[prompt]}
            ).encode()
        else:
            assert request.url == "https://editing.invalid/edit-frame"
            assert request.headers["Authorization"] == "Bearer editing-test-key"
            assert b'filename="original.png"' in request.body
            response.headers["Content-Type"] = "image/png"
            response._content = png.getvalue()
            edits.append(request.url)
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    output = tmp_path / "result"
    main(
        [
            "gensim",
            "image",
            "clean-robot",
            "--image",
            str(_frame(tmp_path / "head.png")),
            "--output",
            str(output),
        ]
    )
    assert len(edits) == 1
    assert json.loads(capsys.readouterr().out)["outside_mask_pixels_changed"] == 0
    assert (output / "input.png").is_file()
