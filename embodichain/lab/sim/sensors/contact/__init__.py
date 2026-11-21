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

"""Contact sensors, raw evidence sampling and bounded physical histories.

Grasp interpretation and semantic integration remain with their own layers.
"""

from __future__ import annotations

from .sensor import (
    ArticulationContactFilterCfg,
    ContactSensor,
    ContactSensorCfg,
    ContactSamplingObserver,
)
from .sampling import (
    CONTACT_ACTOR_IDS,
    CONTACT_VALID_MASK,
    CONTACT_NORMALS,
    CONTACT_IMPULSES,
    CONTACT_POSITIONS,
    CONTACT_DISTANCES,
    DEFAULT_CHANNELS,
    ContactEvidenceSample,
    ContactEvidenceSampler,
    ContactSourceStamp,
    native_contact_revision,
)
from .timeline import (
    ContactTimeline,
    ContactTimelineCfg,
)
from .observation import JointContactObservation, JointContactObserver
from .history import (
    ContactHistory,
)
from .permissions import ContactPermissionChecker

__all__ = [
    "ArticulationContactFilterCfg",
    "ContactSensor",
    "ContactPermissionChecker",
    "ContactSensorCfg",
    "ContactSamplingObserver",
    "CONTACT_ACTOR_IDS",
    "CONTACT_VALID_MASK",
    "CONTACT_NORMALS",
    "CONTACT_IMPULSES",
    "CONTACT_POSITIONS",
    "CONTACT_DISTANCES",
    "DEFAULT_CHANNELS",
    "ContactEvidenceSample",
    "ContactEvidenceSampler",
    "ContactSourceStamp",
    "native_contact_revision",
    "ContactTimeline",
    "ContactTimelineCfg",
    "ContactHistory",
    "JointContactObservation",
    "JointContactObserver",
]
