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

"""Download versioned native policy bundles from the official model repository."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from requests.exceptions import RequestException

__all__ = ["download_pretrained_policy"]

_REPO_ID = "DexForceAI/embodichain_model"
_DEFAULT_REVISION = "28ed627f35dd5af38dedf7bee25ab44cad170e63"
_BUNDLE_FILES = (
    "run-manifest.json",
    "checkpoint.pt",
    "configs/train.yaml",
    "configs/env.yaml",
    "evaluation.json",
)


def download_pretrained_policy(
    model_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[Path, dict[str, str]]:
    """Download one official native policy and identify its repository snapshot.

    Files are materialized inside one revision-specific directory so relative
    RunManifest paths stay within the downloaded run. Robot assets continue
    to use the task's normal asset resolver. Bundles contain ``checkpoint.pt``,
    ``run-manifest.json``, ``configs/train.yaml``, ``configs/env.yaml`` and
    ``evaluation.json``. Commit-pinned files reuse the Hub's local cache.

    Args:
        model_id: Model directory name from the official repository's index.
        revision: Hub commit, tag or branch; defaults to a tested commit.
        cache_dir: Model cache root; defaults to ``~/.cache/embodichain/policies``.

    Returns:
        Local RUN directory and source fields (repo_id, revision, model_id).

    Raises:
        ValueError: The model ID, index schema or policy format is unsupported.
        ImportError: The model's task package is not installed.
        RuntimeError: The model files cannot be downloaded.
    """
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", model_id) is None:
        raise ValueError("Model ID must use lowercase letters, digits and hyphens")
    selected_revision = revision if revision is not None else _DEFAULT_REVISION
    cache_root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path.home() / ".cache" / "embodichain" / "policies"
    )
    repository_cache = cache_root.resolve() / _REPO_ID.replace("/", "--")
    try:
        if re.fullmatch(r"[0-9a-f]{40}", selected_revision) is None:
            selected_revision = (
                HfApi().model_info(_REPO_ID, revision=selected_revision).sha
            )
        if (
            not isinstance(selected_revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", selected_revision) is None
        ):
            raise ValueError("The model revision did not resolve to a Hub commit")
        index_file = hf_hub_download(
            repo_id=_REPO_ID,
            filename="index.json",
            revision=selected_revision,
            cache_dir=repository_cache / ".hub",
        )
        index = json.loads(Path(index_file).read_text(encoding="utf-8"))
        _validate_model(index, model_id)
        snapshot = repository_cache / selected_revision
        for filename in _BUNDLE_FILES:
            hf_hub_download(
                repo_id=_REPO_ID,
                filename=f"policies/{model_id}/{filename}",
                revision=selected_revision,
                local_dir=snapshot,
            )
    except (OSError, RequestException) as error:
        raise RuntimeError(
            f"Unable to download pretrained policy {model_id!r} "
            f"from {_REPO_ID}: {error}"
        ) from error
    return Path(snapshot) / "policies" / model_id, {
        "repo_id": _REPO_ID,
        "revision": selected_revision,
        "model_id": model_id,
    }


def _validate_model(index: object, model_id: str) -> None:
    if not isinstance(index, dict) or index.get("schema_version") != 1:
        raise ValueError("Unsupported pretrained policy index schema")
    models = index.get("models")
    if not isinstance(models, dict):
        raise ValueError("Pretrained policy index must contain a models mapping")
    if model_id not in models:
        raise ValueError(
            f"Unknown pretrained policy {model_id!r}. Available: "
            + ", ".join(sorted(models))
        )
    entry = models[model_id]
    if not isinstance(entry, dict) or entry.get("format") != "embodichain-rl-run-v1":
        raise ValueError(f"Unsupported pretrained policy format for {model_id!r}")
    task_package = entry.get("task_package")
    if not isinstance(task_package, str) or not task_package.isidentifier():
        raise ValueError("Pretrained policy must declare its task package")
    if importlib.util.find_spec(task_package) is None:
        raise ImportError(f"Install the {task_package!r} task package for {model_id!r}")
