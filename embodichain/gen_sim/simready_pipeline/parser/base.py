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

from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from embodichain.gen_sim.simready_pipeline.core.asset import Asset
from pathlib import Path


class AssetParser(ABC):
    """
    Parser = capability, no orchestration logic.
    """

    name: str

    @abstractmethod
    def parse(self, asset: Asset, asset_root: Path) -> None:
        """
        Mutate asset in-place.
        Must be idempotent.
        """
        raise NotImplementedError


from embodichain.gen_sim.simready_pipeline.parser.inspector import AssetInspector
from embodichain.gen_sim.simready_pipeline.parser.geometry import GeometryParser
from embodichain.gen_sim.simready_pipeline.parser.physics import PhysicsParser
from embodichain.gen_sim.simready_pipeline.parser.usd import UsdParser
from embodichain.gen_sim.simready_pipeline.parser.internal import InternalParser


class ParserManager:
    """
    Central parser dispatcher & pipeline owner.
    """

    DEFAULT_PIPELINE: List[str] = [
        "inspector",
        "geometry",
        "physics",
        "usd",
        "internal",
    ]

    def __init__(self):
        self._parsers: Dict[str, object] = {}

        self._register(
            AssetInspector(),
            GeometryParser(),
            PhysicsParser(),
            UsdParser(),
            InternalParser(),
        )

    def _register(self, *parsers):
        for p in parsers:
            if not getattr(p, "name", None):
                raise ValueError(f"Parser missing name: {p}")
            if p.name in self._parsers:
                raise ValueError(f"Duplicate parser: {p.name}")
            self._parsers[p.name] = p

    def parse(
        self,
        asset: Asset,
        asset_root: Path,
        pipeline: Optional[List[str]] = None,
    ) -> None:
        pipeline = pipeline or self.DEFAULT_PIPELINE

        for name in pipeline:
            self._run(name, asset, asset_root)
        asset.status["parsed"] = True

    def parse_one(self, name: str, asset: Asset, asset_root: Path) -> None:
        self._run(name, asset, asset_root)

    def _run(self, name: str, asset: Asset, asset_root: Path):
        parser = self._parsers.get(name)
        if not parser:
            raise KeyError(f"Parser not registered: {name}")
        parser.parse(asset, asset_root)
