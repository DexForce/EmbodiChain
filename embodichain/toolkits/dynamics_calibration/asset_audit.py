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

"""EmbodiChain admission policy over DexSim's SimReady asset facts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


def audit_assets(
    assets: Iterable[str | Path], *, reference_links: Iterable[str] = ()
) -> tuple[Any, ...]:
    """Audit URDF inputs through DexSim without duplicating engine semantics.

    Args:
        assets: Robot asset paths to audit.
        reference_links: Exact link names intentionally allowed to omit
            inertial properties.

    Returns:
        DexSim SimReady reports in the same order as ``assets``.

    Raises:
        RuntimeError: If the installed DexSim does not expose SimReady auditing.
        ValueError: If an asset format has no V1 audit implementation.
    """
    try:
        from dexsim.simready import audit_urdf
    except ImportError as error:
        raise RuntimeError(
            "the installed DexSim does not provide dexsim.simready; install the "
            "matching DexSim SimReady release"
        ) from error

    reports = []
    for raw_asset in assets:
        asset = Path(raw_asset).expanduser().resolve()
        if asset.suffix.lower() != ".urdf":
            raise ValueError(
                f"V1 SimReady audit supports URDF assets only, received: {asset}"
            )
        reports.append(audit_urdf(asset, reference_links=reference_links))
    return tuple(reports)


def audits_admit_calibration(reports: Iterable[Any]) -> bool:
    """Return whether every DexSim report has no error-level diagnostics.

    Args:
        reports: SimReady reports to evaluate under EmbodiChain's admission
            policy.

    Returns:
        ``True`` when at least one report is present and every report is ready.
    """
    materialized = tuple(reports)
    return bool(materialized) and all(bool(report.ready) for report in materialized)


__all__ = ["audit_assets", "audits_admit_calibration"]
