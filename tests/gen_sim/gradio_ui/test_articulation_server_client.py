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

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

GRADIO_UI_ROOT = (
    Path(__file__).resolve().parents[3] / "embodichain" / "gen_sim" / "gradio_ui"
)
sys.path.insert(0, str(GRADIO_UI_ROOT))

from _articulation_server_client import (  # noqa: E402
    ArticulationServerClient,
    ArticulationServerError,
)

SERVER_URL = "http://articulation.test:18688"
REQUEST_ID = "0123456789abcdef"


class _FakeOpener:
    """Return queued responses while recording outgoing requests."""

    def __init__(self, *responses: io.BytesIO | BaseException) -> None:
        self.responses = list(responses)
        self.requests: list[Request] = []

    def open(self, request: Request, *, timeout: float) -> io.BytesIO:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _FailingResponse(io.BytesIO):
    """Write one chunk before simulating an interrupted artifact stream."""

    def __init__(self) -> None:
        super().__init__(b"partial")
        self.read_count = 0

    def read(self, size: int = -1) -> bytes:
        self.read_count += 1
        if self.read_count > 1:
            raise OSError("connection lost")
        return super().read(size)


def _json_response(payload: object) -> io.BytesIO:
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


@pytest.mark.parametrize(
    "timeout_seconds",
    [float("nan"), float("inf"), float("-inf")],
)
def test_client_rejects_non_finite_timeout(timeout_seconds: float) -> None:
    with pytest.raises(ValueError, match="finite positive number"):
        ArticulationServerClient(SERVER_URL, timeout_seconds=timeout_seconds)


def test_submit_text_uses_json_without_authorization_header() -> None:
    opener = _FakeOpener(_json_response({"request_id": REQUEST_ID}))
    client = ArticulationServerClient(SERVER_URL)
    client.opener = opener

    result = client.submit("a hinged cabinet")

    request = opener.requests[0]
    assert result == {"request_id": REQUEST_ID}
    assert request.full_url == f"{SERVER_URL}/generate_articulation"
    assert request.method == "POST"
    assert request.get_header("Authorization") is None
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data or b"") == {"prompt": "a hinged cabinet"}


def test_submit_image_uses_multipart_body(tmp_path: Path) -> None:
    image = tmp_path / "reference.png"
    image.write_bytes(b"png-bytes")
    opener = _FakeOpener(_json_response({"request_id": REQUEST_ID}))
    client = ArticulationServerClient(SERVER_URL)
    client.opener = opener

    client.submit("a service bell", image=image)

    request = opener.requests[0]
    content_type = request.get_header("Content-type") or ""
    assert content_type.startswith("multipart/form-data; boundary=articulation-")
    assert b'name="prompt"' in (request.data or b"")
    assert b"a service bell" in (request.data or b"")
    assert b'filename="reference.png"' in (request.data or b"")
    assert b"png-bytes" in (request.data or b"")


def test_download_writes_artifact_atomically(tmp_path: Path) -> None:
    artifact_url = f"/tasks/{REQUEST_ID}/artifacts/usdc"
    opener = _FakeOpener(
        _json_response(
            {
                "status": "succeeded",
                "result": {"artifacts": {"usdc": artifact_url}},
            }
        ),
        io.BytesIO(b"usdc-data"),
    )
    client = ArticulationServerClient(SERVER_URL)
    client.opener = opener
    destination = tmp_path / "result" / "model.usdc"

    downloaded = client.download(REQUEST_ID, "usdc", destination)

    assert downloaded == destination.resolve()
    assert downloaded.read_bytes() == b"usdc-data"
    assert not (destination.parent / ".model.usdc.part").exists()
    assert opener.requests[-1].full_url == f"{SERVER_URL}{artifact_url}"
    assert opener.requests[-1].get_header("Accept") == "*/*"


def test_download_rejects_absolute_artifact_url(tmp_path: Path) -> None:
    opener = _FakeOpener(
        _json_response(
            {
                "status": "succeeded",
                "result": {
                    "artifacts": {"usdc": "https://untrusted.example/model.usdc"}
                },
            }
        )
    )
    client = ArticulationServerClient(SERVER_URL)
    client.opener = opener

    with pytest.raises(ArticulationServerError, match="must be relative"):
        client.download(REQUEST_ID, "usdc", tmp_path / "model.usdc")


def test_download_removes_partial_file_after_network_error(tmp_path: Path) -> None:
    artifact_url = f"/tasks/{REQUEST_ID}/artifacts/usdc"
    opener = _FakeOpener(
        _json_response(
            {
                "status": "succeeded",
                "result": {"artifacts": {"usdc": artifact_url}},
            }
        ),
        _FailingResponse(),
    )
    client = ArticulationServerClient(SERVER_URL)
    client.opener = opener
    destination = tmp_path / "model.usdc"

    with pytest.raises(OSError, match="connection lost"):
        client.download(REQUEST_ID, "usdc", destination)

    assert not destination.exists()
    assert not (tmp_path / ".model.usdc.part").exists()


def test_http_error_includes_server_detail() -> None:
    response = io.BytesIO(b'{"detail":"invalid token"}')
    error = HTTPError(SERVER_URL, 401, "Unauthorized", {}, response)
    client = ArticulationServerClient(SERVER_URL)
    client.opener = _FakeOpener(error)

    with pytest.raises(ArticulationServerError, match="HTTP 401: invalid token"):
        client.health()


def test_http_error_without_response_body_is_wrapped() -> None:
    client = ArticulationServerClient(SERVER_URL)
    client.opener = _FakeOpener(HTTPError(SERVER_URL, 503, "Unavailable", {}, None))

    with pytest.raises(ArticulationServerError, match="HTTP 503: Unavailable"):
        client.health()


def test_network_error_is_wrapped() -> None:
    client = ArticulationServerClient(SERVER_URL)
    client.opener = _FakeOpener(URLError("connection refused"))

    with pytest.raises(ArticulationServerError, match="could not reach"):
        client.health()
