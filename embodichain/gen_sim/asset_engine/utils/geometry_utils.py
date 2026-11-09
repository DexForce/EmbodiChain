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

from pathlib import Path
from embodichain.gen_sim._blender_runtime import _run_blender_job

__all__ = ["process_obj"]


def process_obj(
    input_path: str | Path,
    output_path: str | Path,
    ratio: float = 0.5,
    weld_distance: float = 0.0001,
    merge_dist: float = 1e-5,
    remove_non_manifold: bool = True,
    triangulate: bool = False,
    collapse_triangulate: bool = True,
) -> None:
    """Clean and simplify an OBJ using the isolated Blender runtime.

    Args:
        input_path: Input OBJ asset.
        output_path: Destination OBJ asset.
        ratio: Decimation ratio.
        weld_distance: Vertex welding distance.
        merge_dist: Mesh cleaning merge distance.
        remove_non_manifold: Remove non-manifold vertices during cleanup.
        triangulate: Triangulate cleaned faces.
        collapse_triangulate: Triangulate during decimation.
    """
    _run_blender_job(
        "process_obj",
        {
            "input_path": str(Path(input_path).resolve()),
            "output_path": str(Path(output_path).resolve()),
            "ratio": ratio,
            "weld_distance": weld_distance,
            "merge_dist": merge_dist,
            "remove_non_manifold": remove_non_manifold,
            "triangulate": triangulate,
            "collapse_triangulate": collapse_triangulate,
        },
    )
