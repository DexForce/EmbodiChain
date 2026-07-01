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

from typing import Any

from .utils import (
    _center_xy_aabb_layout,
    _footprint_layout_diagnostics,
    _layout_text_objects_grid,
    _object_scenes_xy_aabb_manifest,
    _optimize_text_layout_slp,
    _settle_and_pack_object_footprints,
    _xy_aabb_overlap,
    _xy_union_area,
    _xy_union_bounds,
)


class LayoutManager:
    """Public API for layout planning and footprint analysis.

    Tools should compose these methods instead of importing private helpers from
    ``layout_manager.utils`` directly.  The utils module remains an internal
    implementation detail for shared math and optimization routines.
    """

    @staticmethod
    def center_xy_aabb_layout(**kwargs: Any) -> Any:
        return _center_xy_aabb_layout(**kwargs)

    @staticmethod
    def footprint_layout_diagnostics(**kwargs: Any) -> Any:
        return _footprint_layout_diagnostics(**kwargs)

    @staticmethod
    def layout_text_objects_grid(**kwargs: Any) -> Any:
        return _layout_text_objects_grid(**kwargs)

    @staticmethod
    def object_scenes_xy_aabb_manifest(**kwargs: Any) -> Any:
        return _object_scenes_xy_aabb_manifest(**kwargs)

    @staticmethod
    def optimize_text_layout_slp(**kwargs: Any) -> Any:
        return _optimize_text_layout_slp(**kwargs)

    @staticmethod
    def settle_and_pack_object_footprints(**kwargs: Any) -> Any:
        return _settle_and_pack_object_footprints(**kwargs)

    @staticmethod
    def xy_aabb_overlap(**kwargs: Any) -> Any:
        return _xy_aabb_overlap(**kwargs)

    @staticmethod
    def xy_union_area(bounds: Any) -> float:
        return _xy_union_area(bounds)

    @staticmethod
    def xy_union_bounds(**kwargs: Any) -> Any:
        return _xy_union_bounds(**kwargs)
