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
"""Native DexSim Default mimic compliance writes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dexsim.scene import SpawnedArticulation
    from embodichain.lab.sim.cfg import MimicComplianceCfg


def require_mimic_compliance_backend(backend: str) -> None:
    """Require a backend that exposes native compliance parameters."""
    if backend != "dexsim":
        raise NotImplementedError(
            f"Mimic compliance requires the DexSim Default backend. "
            f"Backend {backend!r} does not support natural frequency and damping ratio."
        )


def apply_mimic_compliance(
    uid: str,
    entities: Iterable[tuple[int, SpawnedArticulation]],
    settings: Mapping[str, MimicComplianceCfg],
) -> None:
    """Validate all selected masters in native cache order, then write."""
    if not settings:
        return

    writes = []
    for env_id, entity in entities:
        cache_ids = {
            name: index
            for index, name in enumerate(
                entity.get_actived_joint_names(in_cache_order=True)
            )
        }
        masters = set(entity.get_mimic_info().mimic_parent)
        groups: dict[tuple[float, float], dict[str, int]] = {}
        for name, compliance in settings.items():
            cache_id = cache_ids.get(name)
            if cache_id not in masters:
                raise ValueError(
                    f"Joint {name!r} is not an active mimic master of "
                    f"articulation {uid!r} in env {env_id}."
                )
            values = (
                float(compliance.natural_frequency),
                float(compliance.damping_ratio),
            )
            groups.setdefault(values, {})[name] = cache_id
        writes.append((env_id, entity, groups))

    for env_id, entity, groups in writes:
        for (frequency, ratio), joints in groups.items():
            status = entity.set_mimic_compliance(
                frequency, ratio, list(joints.values())
            )
            if status != 0:
                raise RuntimeError(
                    f"set_mimic_compliance failed for articulation {uid!r} "
                    f"in env {env_id}, joints {list(joints)}, with status {status}."
                )
