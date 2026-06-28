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
from pathlib import Path

from huggingface_hub import hf_hub_download

from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

# HuggingFace endpoint. Mirrors (e.g. hf-mirror.com) often redirect to the
# real hub without forwarding the required commit-hash response headers, so we
# default to the canonical endpoint and rely on the system proxy when needed.
_HF_ENDPOINT = "https://huggingface.co"
_NEURAL_PLANNER_LOCAL_CHECKPOINT = Path(
    "checkpoints/dexforce/neural_motion_generator/franka/franka.pt"
)
NEURAL_PLANNER_CHECKPOINT_ENV = "EMBODICHAIN_NEURAL_PLANNER_CHECKPOINT"

__all__ = [
    "NEURAL_PLANNER_CHECKPOINT_ENV",
    "download_neural_planner_checkpoint",
    "get_default_neural_planner_checkpoint_path",
]


def get_default_neural_planner_checkpoint_path(
    data_root: str | os.PathLike[str] | None = None,
) -> str:
    """Return the default local NeuralPlanner checkpoint path."""
    root = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
    return str(Path(root).expanduser() / _NEURAL_PLANNER_LOCAL_CHECKPOINT)


def _normalize_existing_checkpoint_path(
    checkpoint_path: str | os.PathLike[str],
) -> str:
    path = os.path.abspath(os.path.expanduser(os.fspath(checkpoint_path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"NeuralPlanner checkpoint not found: {path}")
    return path


def _resolve_local_neural_planner_checkpoint(
    checkpoint_path: str | os.PathLike[str] | None,
) -> str | None:
    if checkpoint_path is not None:
        return _normalize_existing_checkpoint_path(checkpoint_path)

    env_checkpoint_path = os.environ.get(NEURAL_PLANNER_CHECKPOINT_ENV)
    if env_checkpoint_path:
        return _normalize_existing_checkpoint_path(env_checkpoint_path)

    default_checkpoint_path = get_default_neural_planner_checkpoint_path()
    if os.path.isfile(default_checkpoint_path):
        return default_checkpoint_path

    return None


def download_neural_planner_checkpoint(
    repo_id: str = "dexforce/neural_motion_generator",
    filename: str = "franka/franka.pt",
    token: str | None = None,
    endpoint: str = _HF_ENDPOINT,
    checkpoint_path: str | os.PathLike[str] | None = None,
) -> str:
    """Resolve or download a neural planner checkpoint.

    Local checkpoint resolution is tried first, in this order:

    1. The explicit ``checkpoint_path`` argument.
    2. The ``EMBODICHAIN_NEURAL_PLANNER_CHECKPOINT`` environment variable.
    3. ``~/.cache/embodichain_data/checkpoints/dexforce/neural_motion_generator/franka/franka.pt``
       unless ``EMBODICHAIN_DATA_ROOT`` overrides the data root.

    If no local checkpoint is found, the checkpoint is downloaded from
    HuggingFace. The repository is gated. Either set the ``HF_TOKEN``
    environment variable or run ``huggingface-cli login`` before calling this
    function.

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
        checkpoint_path: Optional local checkpoint path. If provided, this path
            must exist and HuggingFace is not used.

    Returns:
        str: Local path to the downloaded checkpoint file.

    Raises:
        RuntimeError: If the download fails, with authentication instructions.
    """
    local_checkpoint_path = _resolve_local_neural_planner_checkpoint(checkpoint_path)
    if local_checkpoint_path is not None:
        return local_checkpoint_path

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
            "  1. Accept the model license at https://huggingface.co/dexforce/neural_motion_generator\n"
            "  2. Create an access token at https://huggingface.co/settings/tokens\n"
            "  3. Export the token:  export HF_TOKEN=<your_token>\n"
            "     or run:            huggingface-cli login\n"
            f"Original error: {exc}"
        ) from exc
