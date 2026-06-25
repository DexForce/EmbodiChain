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

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLEAN_SCRIPT = _REPO_ROOT / "docs" / "scripts" / "clean_docs_artifact_paths.py"
_MERGE_SCRIPT = _REPO_ROOT / "docs" / "scripts" / "merge_published_site.py"
_VALIDATE_SCRIPT = _REPO_ROOT / "docs" / "scripts" / "validate_docs_site.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_clean = _load_module("clean_docs_artifact_paths", _CLEAN_SCRIPT)
_merge = _load_module("merge_published_site", _MERGE_SCRIPT)
_validate = _load_module("validate_docs_site", _VALIDATE_SCRIPT)
clean_docs_artifact_paths = _clean.clean_docs_artifact_paths
load_versions_manifest = _merge.load_versions_manifest
merge_published_site = _merge.merge_published_site
validate_docs_site = _validate.validate_docs_site
