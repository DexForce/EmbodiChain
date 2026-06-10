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

from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Asset:

    asset_id: str
    ingest_sha256: str = ""

    identity: Dict[str, Any] = field(default_factory=dict)
    asset_data: Dict[str, Any] = field(default_factory=dict)

    parsed: Dict[str, Any] = field(default_factory=dict)  # Visual, Geometry, Topology
    semantics: Dict[str, Any] = field(default_factory=dict)
    physics: Dict[str, Any] = field(default_factory=dict)
    simulation: Dict[str, Any] = field(default_factory=dict)
    affordance: Dict[str, Any] = field(default_factory=dict)
    usd: Dict[str, Any] = field(default_factory=dict)

    provenance: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    status: Dict[str, Any] = field(default_factory=dict)
    internal: Dict[str, Any] = field(default_factory=dict)

    ingest_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._init_simulation_defaults()
        self.touch()

    def _init_simulation_defaults(self) -> None:
        self.simulation.setdefault("articulation", None)
        self.simulation.setdefault("sim_ready", {})

    def touch(self) -> None:
        self.status["last_updated"] = datetime.now().isoformat()

    def _server_metadata(self) -> Dict[str, Any]:
        category = str(self.identity.get("category") or "").strip()
        object_name_generated = str(
            self.semantics.get("object_name_generated") or ""
        ).strip()
        name = object_name_generated or str(self.identity.get("name") or "").strip()

        return {
            "ingest_sha256": self.ingest_sha256,
            "name": name,
            "display_name": name,
            "object_name_generated": object_name_generated,
            "category": category,
            "type": category,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            **self._server_metadata(),
            "identity": self.identity,
            "asset_data": self.asset_data,
            "parsed": self.parsed,
            "quality": self.quality,
            "semantics": self.semantics,
            "physics": self.physics,
            "simulation": self.simulation,
            "usd": self.usd,
            "provenance": self.provenance,
            "status": self.status,
            "internal": self.internal,
            "affordance": self.affordance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Asset":
        identity = dict(data.get("identity") or {})
        if not identity.get("name") and (data.get("name") or data.get("display_name")):
            identity["name"] = data.get("name") or data.get("display_name")
        if not identity.get("category") and (data.get("category") or data.get("type")):
            identity["category"] = data.get("category") or data.get("type")

        semantics = dict(data.get("semantics") or {})
        if not semantics.get("object_name_generated") and data.get(
            "object_name_generated"
        ):
            semantics["object_name_generated"] = data.get("object_name_generated")

        return cls(
            asset_id=data["asset_id"],
            ingest_sha256=data.get("ingest_sha256", ""),
            identity=identity,
            asset_data=data.get("asset_data", []),
            parsed=data.get("parsed", {}),
            quality=data.get("quality", {}),
            semantics=semantics,
            physics=data.get("physics", {}),
            simulation=data.get("simulation", {}),
            usd=data.get("usd", {}),
            provenance=data.get("provenance", {}),
            status=data.get("status", {}),
            internal=data.get("internal", {}),
            affordance=data.get("affordance", {}),
        )
