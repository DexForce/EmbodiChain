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

import os

from huggingface_hub import hf_hub_download

# HuggingFace endpoint. Mirrors (e.g. hf-mirror.com) often redirect to the
# real hub without forwarding the required commit-hash response headers, so we
# default to the canonical endpoint and rely on the system proxy when needed.
_HF_ENDPOINT = "https://huggingface.co"

__all__ = ["download_neural_ik_checkpoint"]


def download_neural_ik_checkpoint(
    repo_id: str = "DexForceAI/neural_ik_solver",
    filename: str = "franka.pt",
    token: str | None = None,
    endpoint: str = _HF_ENDPOINT,
) -> str:
    """Download a neural IK solver checkpoint from HuggingFace.

    The repository is gated. Either set the ``HF_TOKEN`` environment variable or
    run ``huggingface-cli login`` before calling this function.

    If your network requires an HTTP proxy, set ``HTTPS_PROXY`` or
    ``https_proxy`` in the environment before launching Python.

    Args:
        repo_id: HuggingFace repository ID.
        filename: Checkpoint filename to download.
        token: HuggingFace API token. Falls back to the ``HF_TOKEN``
            environment variable or the cached token from
            ``huggingface-cli login``.
        endpoint: HuggingFace-compatible endpoint URL. Defaults to
            ``https://huggingface.co``. Mirrors that merely redirect to the
            real hub are not supported.

    Returns:
        str: Local path to the downloaded checkpoint file.

    Raises:
        RuntimeError: If the download fails, with authentication instructions.
    """
    # Normalize proxy env vars: the ``requests`` library on Linux requires the
    # lowercase form (``https_proxy``), but users typically export the uppercase
    # form (``HTTPS_PROXY``).
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
            "  1. Accept the model license at https://huggingface.co/DexForceAI/neural_ik_solver\n"
            "  2. Create an access token at https://huggingface.co/settings/tokens\n"
            "  3. Export the token:  export HF_TOKEN=<your_token>\n"
            "     or run:            huggingface-cli login\n"
            f"Original error: {exc}"
        ) from exc
