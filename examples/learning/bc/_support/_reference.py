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

"""Load pinned reference definitions without restoring retired core exports."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
from types import ModuleType

__all__: list[str] = []


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_reference(root: Path) -> tuple[ModuleType, ModuleType]:
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != "51f1f0cd5a7602f23358c142997e9045323f547a":
        raise ValueError("BC reference must be the pinned PR828 source revision")
    encoder = _load_module(
        "bc_reference_encoder", root / "embodichain/learning/models/resnet_encoder.py"
    )
    models = _load_module(
        "bc_reference_targets", root / "examples/ods/simple_bc_demo/models.py"
    )
    path = root / "examples/ods/data_stream_demo/visual_policy.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    # Supply both historical imports from their pinned source definitions.
    # No current implementation is substituted into the reference network.
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module
            in {
                "embodichain.learning.models",
                "examples.ods.simple_bc_demo.models",
            }
        )
    ]
    visual = ModuleType("bc_reference_visual")
    visual.__dict__.update(
        ResNet10Encoder=encoder.ResNet10Encoder, pose_targets=models.pose_targets
    )
    exec(compile(tree, str(path), "exec"), visual.__dict__)
    return models, visual
