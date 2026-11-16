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

"""Keep profile credentials on their configured HTTP origin."""

from __future__ import annotations

from urllib.parse import urlsplit
import requests

__all__: list[str] = []


def _origin(url: str) -> tuple[str, str | None, int | None]:
    parts = urlsplit(url)
    port = parts.port
    if port is None:
        port = {"https": 443, "http": 80}.get(parts.scheme)
    return parts.scheme, parts.hostname, port


class _OriginBoundSession(requests.Session):
    def __init__(self, base_url: str, api_key: str) -> None:
        super().__init__()
        self._service_origin = _origin(base_url)
        self._authorization = f"Bearer {api_key}"

    def prepare_request(self, request: requests.Request) -> requests.PreparedRequest:
        prepared = super().prepare_request(request)
        if _origin(prepared.url) == self._service_origin:
            prepared.headers["Authorization"] = self._authorization
        else:
            prepared.headers.pop("Authorization", None)
        return prepared

    def rebuild_auth(
        self, prepared_request: requests.PreparedRequest, response: requests.Response
    ) -> None:
        super().rebuild_auth(prepared_request, response)
        # Requests permits HTTP-to-HTTPS auth forwarding on the same host;
        # profile credentials remain restricted to the exact configured origin.
        if _origin(prepared_request.url) != self._service_origin:
            prepared_request.headers.pop("Authorization", None)
