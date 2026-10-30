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

"""Bind experiment reports to executed sources, dependencies and actual assets."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import ctypes
import json
import platform
from typing import Any

import torch

from .._prepare import sha256


def evidence(checkpoint: Path, root: Path, backend: str) -> dict[str, Any]:
    """Describe the executed installation and the immutable experiment inputs."""
    import embodichain.learning.sl as sl
    import embodichain_models.dynamics.rigidformer as family

    dependencies = {}
    for package in (
        "torch",
        "einops",
        "roma",
        "numpy",
        "pyarrow",
        "trimesh",
        "flash-attn",
    ):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    sources = {}
    for label, directory in (
        ("rigidformer", Path(family.__file__).parent),
        ("sl", Path(sl.__file__).parent),
        ("showcase", Path(__file__).parents[1]),
    ):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (
                path.suffix in (".py", ".yaml")
                or (label == "rigidformer" and path.suffix == ".json")
            ):
                sources[f"{label}/{path.relative_to(directory)}"] = sha256(path)
    corpus = json.loads((root / "manifest.json").read_text())
    nccl = None
    if torch.distributed.is_nccl_available():
        loaded = ctypes.c_int()
        ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(loaded))
        nccl = [loaded.value // 10000, loaded.value % 10000 // 100, loaded.value % 100]
    return {
        "python": platform.python_version(),
        "dependencies": dependencies,
        "torch_cuda": torch.version.cuda,
        "nccl_runtime": nccl,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "attention_backend": backend,
        "executed_source_sha256": sources,
        "checkpoint_sha256": sha256(checkpoint),
        "data_manifest_sha256": sha256(root / "manifest.json"),
        "dataset": corpus.get("dataset"),
        "scope": corpus.get("scope"),
        "scene_counts": corpus.get("scene_counts"),
    }
