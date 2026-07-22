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

from embodichain.gen_sim.prompt2scene.agent_tools.managers.layout_manager.scene_edit_sa import (
    optimize_scene_edit_layout_with_sa_node3_5,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.layout_manager.scene_edit_support import (
    build_xy_footprint,
    clamp_center_to_support_region,
    compute_simready_glb_xy_size,
    support_region_default_center,
    support_region_grid_center,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.layout_manager.utils import (
    _layout_text_objects_grid,
    _object_scenes_xy_aabb_manifest,
    _optimize_text_layout_slp,
    _settle_and_pack_object_footprints,
)


class LayoutManager:
    """Public API for layout planning and footprint analysis.

    Tools should compose these methods instead of importing private helpers from
    ``layout_manager.utils`` directly.  The utils module remains an internal
    implementation detail for shared math and optimization routines.
    """

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
    def optimize_scene_edit_layout_with_sa_node3_5(**kwargs: Any) -> Any:
        return optimize_scene_edit_layout_with_sa_node3_5(**kwargs)

    @staticmethod
    def compute_simready_glb_xy_size(**kwargs: Any) -> Any:
        return compute_simready_glb_xy_size(**kwargs)

    @staticmethod
    def build_xy_footprint(**kwargs: Any) -> Any:
        return build_xy_footprint(**kwargs)

    @staticmethod
    def clamp_center_to_support_region(**kwargs: Any) -> Any:
        return clamp_center_to_support_region(**kwargs)

    @staticmethod
    def support_region_default_center(**kwargs: Any) -> Any:
        return support_region_default_center(**kwargs)

    @staticmethod
    def support_region_grid_center(**kwargs: Any) -> Any:
        return support_region_grid_center(**kwargs)
