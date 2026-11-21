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

"""Recoverable, source-bound input preparation, separate from published corpora."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .artifacts import (
    _package_version,
    _read_case_manifest,
    write_case_manifest,
    write_json,
)
from .config import stable_hash, suite_to_dict

if TYPE_CHECKING:
    from .config import SuiteCfg
    from .embodiments import EmbodimentResolution
    from .models import BenchmarkCase

__all__: list[str] = []


class _CasePreparation:
    """Retain immutable generation units under one exclusive preparation lease."""

    def __init__(
        self,
        path: Path,
        suite: SuiteCfg,
        embodiment: EmbodimentResolution,
        device: torch.device,
        revision: str,
    ) -> None:
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(
                "Recoverable preparation needs a nonempty generation_revision."
            )
        self.path, self.suite, self.embodiment, self.device = (
            path,
            suite,
            embodiment,
            device,
        )
        configuration = suite_to_dict(suite)
        configuration.pop("planners", None)
        self.identity = {
            "schema_version": 1,
            "generation_revision": revision,
            "suite_sha256": stable_hash(configuration),
            "embodiment": embodiment.to_metadata(),
            "device_type": device.type,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "dexsim_engine": _package_version("dexsim-engine"),
        }
        self.used: set[str] = set()
        self._lock = None

    def __enter__(self) -> _CasePreparation:
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = (self.path / "lease").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            identity = self.path / "identity.json"
            if identity.exists():
                if json.loads(identity.read_text()) != self.identity:
                    raise ValueError(
                        "Input preparation identity differs; use a new progress directory."
                    )
            else:
                self._publish(identity, lambda path: write_json(path, self.identity))
        except BaseException:
            self._lock.close()
            self._lock = None
            raise
        return self

    def __exit__(self, *exception: object) -> None:
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def _publish(self, target: Path, write: Callable[[Path], object]) -> None:
        with tempfile.NamedTemporaryFile(
            dir=self.path, prefix="pending-", delete=False
        ) as file:
            temporary = Path(file.name)
        try:
            write(temporary)
            with temporary.open("rb") as file:
                os.fsync(file.fileno())
            target.hardlink_to(temporary)
        finally:
            temporary.unlink(missing_ok=True)

    def get_or_create(
        self,
        key: str,
        generate: Callable[[], list[BenchmarkCase]],
    ) -> list[BenchmarkCase]:
        """Load one accepted unit, or publish it only after generation succeeds."""
        if self._lock is None:
            raise RuntimeError("Input preparation requires its exclusive lease.")
        name = "case-" + stable_hash(key) + ".json"
        if name in self.used:
            raise ValueError(f"Duplicate input preparation unit {key!r}.")
        self.used.add(name)
        path = self.path / name
        if path.exists():
            return _read_case_manifest(
                path, self.device, expected_embodiment=self.embodiment
            )
        cases = generate()
        if not cases:
            raise ValueError("Input preparation cannot checkpoint an empty unit.")
        self._publish(
            path,
            lambda output: write_case_manifest(
                output,
                cases,
                suite=self.suite,
                embodiment_resolution=self.embodiment,
            ),
        )
        return cases

    def validate_complete(self) -> None:
        """Do not silently drop saved inputs when finalizing a resumed population."""
        if {path.name for path in self.path.glob("case-*.json")} != self.used:
            raise ValueError("Input preparation has unconsumed saved units.")
