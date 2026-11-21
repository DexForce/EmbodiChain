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


"""Shared authenticated checkpoint downloads for model-specific entry points."""

from __future__ import annotations

import os
from huggingface_hub import hf_hub_download

__all__: list[str] = []

_HF_ENDPOINT = "https://huggingface.co"


def _download_checkpoint(
    repo_id: str,
    filename: str,
    token: str | None,
    endpoint: str,
) -> str:
    """Apply one proxy, authentication and error policy to checkpoint downloads."""
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if https_proxy:
        os.environ.setdefault("https_proxy", https_proxy)
        os.environ.setdefault("HTTPS_PROXY", https_proxy)

    # Allow callers to pass the token explicitly; otherwise fall back to
    # HF_TOKEN (huggingface_hub also reads this automatically, but being
    # explicit makes the fallback order transparent).
    if token is None:
        token = os.environ.get("HF_TOKEN") or None

    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            endpoint=endpoint,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download '{filename}' from '{repo_id}'.\n"
            "The repository is gated and requires an authenticated HuggingFace account.\n"
            "To fix this:\n"
            f"  1. Accept the model license at {_HF_ENDPOINT}/{repo_id}\n"
            "  2. Create an access token at https://huggingface.co/settings/tokens\n"
            "  3. Export the token:  export HF_TOKEN=<your_token>\n"
            "     or run:            huggingface-cli login\n"
            f"Original error: {exc}"
        ) from exc
