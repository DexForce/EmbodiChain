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
    filename: str = "v1/franka.onnx",
    token: str | None = None,
    endpoint: str = _HF_ENDPOINT,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> str:
    """Download a neural planner deployment model from Hugging Face.

    Public weights can be downloaded anonymously. For a private or gated
    repository, set ``HF_TOKEN`` or log in with the Hugging Face CLI.

    If your network requires an HTTP proxy, set ``HTTPS_PROXY`` or
    ``https_proxy`` in the environment before launching Python.

    Args:
        repo_id: HuggingFace repository ID.
        filename: Model path in the repo, e.g. ``v1/franka.onnx`` or ``v1/piper.onnx``.
        token: HuggingFace API token. Falls back to the ``HF_TOKEN``
            environment variable or the cached token from
            ``huggingface-cli login``.
        endpoint: HuggingFace-compatible endpoint URL. Defaults to
            ``https://huggingface.co``. Mirrors that merely redirect to the
            real hub are not supported.
        revision: Optional commit hash, tag or branch. Pin a commit for releases.
        local_files_only: Resolve an already cached file without network access.

    Returns:
        str: Local path to the downloaded model file.

    Raises:
        RuntimeError: If the download fails, with network/cache/authentication guidance.
    """
    return _download_checkpoint(
        repo_id,
        filename,
        token,
        endpoint,
        revision=revision,
        local_files_only=local_files_only,
    )
