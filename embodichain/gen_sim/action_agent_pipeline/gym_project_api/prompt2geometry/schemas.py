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

from dataclasses import dataclass, field

__all__ = ["SelectedBox"]


@dataclass(frozen=True)
class SelectedBox:
    """One box prompt passed to the SAM3 segmentation service."""

    target_id: str
    target_kind: str
    phrase: str
    bbox_xyxy: list[float]
    source_candidate_ids: list[str] = field(default_factory=list)
    selection_reason: str | None = None

    def to_manifest(self) -> dict[str, object]:
        """Convert the selected box to JSON-safe data."""
        manifest: dict[str, object] = {
            "target_id": self.target_id,
            "target_kind": self.target_kind,
            "phrase": self.phrase,
            "bbox_xyxy": self.bbox_xyxy,
            "source_candidate_ids": self.source_candidate_ids,
        }
        if self.selection_reason is not None:
            manifest["selection_reason"] = self.selection_reason
        return manifest
