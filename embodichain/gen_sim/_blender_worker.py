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

"""Standalone worker: deliberately do not import the EmbodiChain package."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

__all__ = []

_OPERATIONS = {
    "front_top": "_render_object_front_top_views",
    "pose_candidate": "_render_grounded_pose_candidate",
    "visual_yaw": "_render_canonical_oblique_view",
    "remesh_bake": "_blender_remesh_bake",
    "process_obj": "process_obj",
}


def _main() -> None:
    arguments = (
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    )
    request_path, response_path = map(Path, arguments)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    function = _OPERATIONS[request["operation"]]
    path = Path(__file__).with_name("_blender_operations.py")
    spec = importlib.util.spec_from_file_location(
        "_embodichain_blender_operations", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Blender operations: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = getattr(module, function)(**request["arguments"])
    response_path.write_text(
        json.dumps({"result": result}, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    _main()
