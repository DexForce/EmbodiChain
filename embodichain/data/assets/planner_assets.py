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

from ._checkpoint_download import _HF_ENDPOINT, _download_checkpoint

__all__ = ["download_neural_planner_checkpoint"]


def download_neural_planner_checkpoint(
    repo_id: str = "DexForceAI/neural_motion_generator",
    filename: str = "franka/franka.pt",
    token: str | None = None,
    endpoint: str = _HF_ENDPOINT,
) -> str:
    """Download a neural planner checkpoint from HuggingFace.

    The repository is gated. Either set the ``HF_TOKEN`` environment variable or
    run ``huggingface-cli login`` before calling this function.

    If your network requires an HTTP proxy, set ``HTTPS_PROXY`` or
    ``https_proxy`` in the environment before launching Python.

    Args:
        repo_id: HuggingFace repository ID.
        filename: Checkpoint path in the repo, e.g. ``franka/franka.pt``.
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
    return _download_checkpoint(repo_id, filename, token, endpoint)
