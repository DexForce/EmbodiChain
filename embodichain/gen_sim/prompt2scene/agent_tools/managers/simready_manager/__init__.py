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

from embodichain.gen_sim.prompt2scene.agent_tools.managers.simready_manager.manager import (
    METRIC_SCALE_ENABLED,
    SimreadyManager,
)
from embodichain.gen_sim.prompt2scene.agent_tools.managers.simready_manager.schemas import (
    EstimateMetricScalesRequest,
    EstimateMetricScalesResult,
    GlobalMetricScaleRequest,
    MakeAssetSimreadyRequest,
    MakeAssetSimreadyResult,
    MakeTableSimreadyRequest,
    MakeTableSimreadyResult,
    MetricScaleObjectInput,
)
from embodichain.gen_sim.prompt2scene.prompts.builders import (
    build_image_metric_scale_messages,
)
from embodichain.gen_sim.prompt2scene.prompts.schemas import (
    IMAGE_METRIC_SCALE_JSON_SCHEMA,
)

MetricScaleManager = SimreadyManager

__all__ = [
    "EstimateMetricScalesRequest",
    "EstimateMetricScalesResult",
    "GlobalMetricScaleRequest",
    "IMAGE_METRIC_SCALE_JSON_SCHEMA",
    "MakeAssetSimreadyRequest",
    "MakeAssetSimreadyResult",
    "MakeTableSimreadyRequest",
    "MakeTableSimreadyResult",
    "METRIC_SCALE_ENABLED",
    "MetricScaleManager",
    "MetricScaleObjectInput",
    "SimreadyManager",
    "build_image_metric_scale_messages",
]
