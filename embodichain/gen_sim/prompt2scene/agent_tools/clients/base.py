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

import time
from pathlib import Path
from typing import Callable

import requests

from embodichain.gen_sim.prompt2scene.agent_tools.clients.common import (
    ClientError,
    build_client_error,
)
from embodichain.gen_sim.prompt2scene.agent_tools.clients.config import (
    load_client_config,
)
from embodichain.gen_sim.prompt2scene.utils.log import (
    log_api_request_start,
    log_info,
    log_warning,
)

__all__ = ["BaseHttpClient"]


class BaseHttpClient:
    """Shared HTTP client behavior for agent-tool service clients."""

    def __init__(
        self,
        *,
        config_key: str,
        server_name: str,
        base_url: str | None = None,
        timeout_s: int | None = None,
        config_path: Path | None = None,
        session: requests.Session | None = None,
        trust_env: bool = True,
    ) -> None:
        """Initialize common service client fields from config."""
        self.config = load_client_config(config_key, config_path)
        self.server_name = server_name
        self.base_url = (base_url or str(self.config["base_url"])).rstrip("/")
        self.timeout_s = int(timeout_s or self.config.get("timeout_s", 120))
        self.health_path = str(self.config.get("health_path", "/health"))
        self.session = session or requests.Session()
        self.session.trust_env = trust_env
        log_info(f"{self.server_name} client initialized for {self.base_url}")

    def health_check(self) -> bool:
        """Check whether the configured service is healthy."""
        try:
            response = self.session.get(
                f"{self.base_url}{self.health_path}",
                timeout=5,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            log_warning(f"{self.server_name} health check failed: {exc}")
            return False

    def post_with_retries(
        self,
        request_fn: Callable[[], requests.Response],
        *,
        max_retries: int,
        error_cls: type[ClientError] = ClientError,
        request_label: str | None = None,
    ) -> requests.Response | ClientError:
        """Run a POST request function with retry and HTTP error handling."""
        for attempt in range(max_retries):
            try:
                if request_label is not None:
                    log_api_request_start(
                        step=self.server_name,
                        request=request_label,
                        attempt=attempt + 1,
                    )
                response = request_fn()
                response.raise_for_status()
                return response

            except requests.exceptions.ConnectionError as exc:
                if attempt < max_retries - 1:
                    log_warning(
                        f"{self.server_name} connection failed; retrying "
                        f"({attempt + 1}/{max_retries})."
                    )
                    time.sleep(min(2**attempt, 60))
                    continue
                raise ConnectionError(
                    f"Failed to connect to {self.server_name} at {self.base_url}"
                ) from exc

            except requests.exceptions.HTTPError as exc:
                response = exc.response
                if response is None:
                    raise RuntimeError(f"{self.server_name} HTTP request failed.") from exc
                if response.status_code >= 500 and attempt < max_retries - 1:
                    log_warning(
                        f"{self.server_name} server error; retrying "
                        f"({attempt + 1}/{max_retries})."
                    )
                    time.sleep(min(2**attempt, 60))
                    continue
                return build_client_error(
                    response,
                    server_name=self.server_name,
                    error_cls=error_cls,
                )

            except requests.exceptions.Timeout as exc:
                raise TimeoutError(f"{self.server_name} request timed out.") from exc

        raise RuntimeError(f"{self.server_name} request failed unexpectedly.")
