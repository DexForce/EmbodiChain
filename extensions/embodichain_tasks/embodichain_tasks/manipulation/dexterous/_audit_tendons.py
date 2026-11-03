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

"""Compare fixed-tendon topology and physical properties across MJCF export."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mujoco
import numpy as np

__all__ = ["audit_tendons"]


def audit_tendons(
    source: Any,
    imported: Any,
    joint_map: Mapping[int, int],
    *,
    atol: float = 2e-6,
    rtol: float = 2e-5,
) -> dict[str, Any]:
    """Compare every fixed tendon after mapping joint indices.

    Args:
        source: Source MuJoCo model.
        imported: MuJoCo model exported by the active solver.
        joint_map: Source joint index to imported joint index.
        atol: Absolute numeric tolerance.
        rtol: Relative numeric tolerance.

    Returns:
        JSON-compatible per-tendon address ranges, ordered joint/coefficient
        records, physical-property checks, and the overall status. A non-joint
        wrap is reported as unsupported instead of being treated as equivalent.
    """
    rows: list[dict[str, Any]] = []
    joint_wrap = int(mujoco.mjtWrap.mjWRAP_JOINT)
    imported_names = [imported.tendon(i).name for i in range(imported.ntendon)]
    used: set[int] = set()
    for sid in range(source.ntendon):
        name = source.tendon(sid).name
        candidates = [
            iid
            for iid, actual_name in enumerate(imported_names)
            if iid not in used
            and (
                actual_name == name
                or actual_name.endswith("_" + name)
                or actual_name.endswith("/" + name)
            )
        ]
        if len(candidates) != 1:
            rows.append(
                {
                    "name": name,
                    "matches": False,
                    "mapping_error": candidates,
                }
            )
            continue
        iid = candidates[0]
        used.add(iid)
        sa, ia = int(source.tendon_adr[sid]), int(imported.tendon_adr[iid])
        sn, inn = int(source.tendon_num[sid]), int(imported.tendon_num[iid])
        source_path, imported_path = [], []
        supported = True
        for model, start, count, destination, remap in (
            (source, sa, sn, source_path, True),
            (imported, ia, inn, imported_path, False),
        ):
            for wid in range(start, start + count):
                kind = int(model.wrap_type[wid])
                target = int(model.wrap_objid[wid])
                if kind != joint_wrap:
                    supported = False
                destination.append(
                    {
                        "wrap_index": wid,
                        "wrap_type": kind,
                        "joint_index": target,
                        "joint_name": (
                            model.joint(target).name if kind == joint_wrap else None
                        ),
                        "mapped_joint_index": (
                            joint_map.get(target, -1)
                            if remap and kind == joint_wrap
                            else target
                        ),
                        "coefficient": float(model.wrap_prm[wid]),
                    }
                )
        topology_matches = (
            supported
            and sn == inn
            and all(
                expected["mapped_joint_index"] == actual["mapped_joint_index"]
                and expected["wrap_type"] == actual["wrap_type"]
                and np.isclose(
                    expected["coefficient"],
                    actual["coefficient"],
                    atol=atol,
                    rtol=rtol,
                )
                for expected, actual in zip(source_path, imported_path)
            )
        )
        properties = []
        for field in (
            "tendon_limited",
            "tendon_range",
            "tendon_stiffness",
            "tendon_damping",
            "tendon_frictionloss",
            "tendon_margin",
            "tendon_solref_lim",
            "tendon_solimp_lim",
            "tendon_solref_fri",
            "tendon_solimp_fri",
            "tendon_lengthspring",
            "tendon_armature",
            "tendon_actfrclimited",
            "tendon_actfrcrange",
        ):
            expected = np.asarray(getattr(source, field)[sid])
            actual = np.asarray(getattr(imported, field)[iid])
            properties.append(
                {
                    "field": field,
                    "matches": bool(
                        np.allclose(expected, actual, atol=atol, rtol=rtol)
                    ),
                    "max_abs_error": float(
                        np.max(np.abs(expected.astype(float) - actual.astype(float)))
                    ),
                    "source": expected.tolist(),
                    "imported": actual.tolist(),
                }
            )
        rows.append(
            {
                "name": name,
                "source_index": sid,
                "imported_index": iid,
                "source_address_range": [sa, sa + sn],
                "imported_address_range": [ia, ia + inn],
                "fixed_joint_wraps_supported": supported,
                "source_path": source_path,
                "imported_path": imported_path,
                "topology_matches": bool(topology_matches),
                "properties": properties,
                "matches": bool(
                    topology_matches and all(item["matches"] for item in properties)
                ),
            }
        )
    matches = (
        source.ntendon == imported.ntendon
        and len(used) == imported.ntendon
        and all(row["matches"] for row in rows)
    )
    return {
        "status": "passed" if matches else "differences",
        "source_count": int(source.ntendon),
        "imported_count": int(imported.ntendon),
        "matched_count": len(used),
        "atol": atol,
        "rtol": rtol,
        "tendons": rows,
        "mismatch_names": [row["name"] for row in rows if not row["matches"]],
    }
