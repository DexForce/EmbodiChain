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

"""User-deployed service validation and current HTTP deployment diagnostics."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
from threading import Thread

import pytest
from PIL import Image

from embodichain.cli.main import main
from embodichain.gen_sim.services import (
    GenSimServices,
    GeometryRequest,
    ImageGenerationRequest,
    SegmentationRequest,
)

pytestmark = pytest.mark.no_sim


def _write_current_profiles(root: Path, base_url: str) -> tuple[Path, Path]:
    config = root / "model_services.json"
    config.write_text(
        json.dumps(
            {
                "model_services": {
                    "profiles": {
                        "images": {
                            "adapter": "image_http",
                            "base_url": {"env": "TEST_IMAGE_URL"},
                            "options": {
                                "health_path": "/health",
                                "generate_image_by_prompt_path": (
                                    "/generate_image_by_prompt"
                                ),
                            },
                        },
                        "segmentation": {
                            "adapter": "segmentation_http",
                            "base_url": {"env": "TEST_SEGMENTATION_URL"},
                            "options": {
                                "health_path": "/health",
                                "segment_by_prompt_path": "/segment_by_prompt",
                            },
                        },
                        "geometry": {
                            "adapter": "geometry_http",
                            "base_url": {"env": "TEST_GEOMETRY_URL"},
                            "options": {
                                "health_path": "/health",
                                "generate_objects_path": "/generate_multiple_objects",
                            },
                        },
                    },
                    "roles": {
                        "image_generation": "images",
                        "image_segmentation": "segmentation",
                        "geometry_reconstruction": "geometry",
                    },
                }
            }
        )
    )
    env = root / "services.env"
    env.write_text(
        f"TEST_IMAGE_URL={base_url}\n"
        f"TEST_SEGMENTATION_URL={base_url}\n"
        f"TEST_GEOMETRY_URL={base_url}\n"
    )
    return config, env


def test_validate_resolves_current_profiles_without_network(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, env = _write_current_profiles(tmp_path, "http://127.0.0.1:1")
    main(
        [
            "gensim",
            "services",
            "validate",
            "--config",
            str(config),
            "--env-file",
            str(env),
            "--json",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert {item["adapter"] for item in report["roles"]} == {
        "image_http",
        "segmentation_http",
        "geometry_http",
    }


def test_validate_text_report_lists_configured_roles(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, env = _write_current_profiles(tmp_path, "http://127.0.0.1:1")
    main(
        [
            "gensim",
            "services",
            "validate",
            "--config",
            str(config),
            "--env-file",
            str(env),
        ]
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "model services: ok"
    assert len(lines) == 4
    assert all(": configured [" in line for line in lines[:-1])


def test_validate_rejects_an_incomplete_adapter_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "model_services.json"
    config.write_text(
        json.dumps(
            {
                "model_services": {
                    "profiles": {
                        "geometry": {
                            "adapter": "geometry_http",
                            "base_url": "http://127.0.0.1:1",
                        }
                    },
                    "roles": {"geometry_reconstruction": "geometry"},
                }
            }
        )
    )
    with pytest.raises(SystemExit, match="1"):
        main(
            [
                "gensim",
                "services",
                "validate",
                "--config",
                str(config),
                "--json",
            ]
        )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert report["error"] == "configuration"


def test_doctor_accepts_current_deployment_health_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    health_requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass

        def do_GET(self) -> None:
            health_requests.append(self.path)
            payload = b'{"ok": true, "status": "ready"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config, env = _write_current_profiles(
        tmp_path, f"http://127.0.0.1:{server.server_port}"
    )
    try:
        main(
            [
                "gensim",
                "services",
                "doctor",
                "--config",
                str(config),
                "--env-file",
                str(env),
                "--json",
            ]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert {item["status"] for item in report["roles"]} == {"ready"}
    assert health_requests == ["/health"] * 3


def test_current_deployment_wire_contract_runs_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested_paths: list[str] = []
    generated = BytesIO()
    Image.new("RGB", (4, 3), color="white").save(generated, format="PNG")
    generated_png = generated.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass

        def _send(self, content: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:
            requested_paths.append(self.path)
            if self.path == "/health":
                self._send(b'{"ok": true, "status": "ready"}', "application/json")
            elif self.path == "/assets/job/cup.glb":
                self._send(b"glTF-current-contract", "model/gltf-binary")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            requested_paths.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path == "/generate_image_by_prompt":
                self._send(generated_png, "image/png")
            elif self.path == "/segment_by_prompt":
                self._send(
                    b'{"ok":true,"result":{"masks":[{"size":[3,4],"counts":[12]}]}}',
                    "application/json",
                )
            elif self.path == "/generate_multiple_objects":
                payload = {
                    "ok": True,
                    "request_id": "job",
                    "result": {
                        "objects": [
                            {
                                "name": "cup",
                                "mesh": "/assets/job/cup.glb",
                                "pose_coordinate_system": "glb_y_up",
                                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                                "translation": [0.1, 0.2, 0.3],
                                "scale": [1.0, 1.0, 1.0],
                            }
                        ]
                    },
                }
                self._send(json.dumps(payload).encode(), "application/json")
            else:
                self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config, env = _write_current_profiles(
        tmp_path, f"http://127.0.0.1:{server.server_port}"
    )
    monkeypatch.setenv("EMBODICHAIN_MODEL_SERVICES_CONFIG", str(config))
    monkeypatch.setenv("EMBODICHAIN_ENV_FILE", str(env))
    image = tmp_path / "input.png"
    mask = tmp_path / "cup.png"
    Image.new("RGB", (4, 3), color="white").save(image)
    Image.new("L", (4, 3), color="white").save(mask)
    try:
        with GenSimServices() as services:
            image_result = services.get("image_generation").generate(
                ImageGenerationRequest("a cup", tmp_path / "generated.png")
            )
            segmentation = services.get("image_segmentation").segment(
                SegmentationRequest(image, "cup")
            )
            geometry = services.get("geometry_reconstruction").reconstruct(
                GeometryRequest(image, (("cup", mask),), tmp_path / "geometry")
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert image_result.path.read_bytes() == generated_png
    assert segmentation.masks[0]["size"] == [3, 4]
    assert geometry.objects[0].quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert geometry.objects[0].mesh_path.read_bytes() == b"glTF-current-contract"
    assert {
        "/generate_image_by_prompt",
        "/segment_by_prompt",
        "/generate_multiple_objects",
        "/assets/job/cup.glb",
    }.issubset(requested_paths)
