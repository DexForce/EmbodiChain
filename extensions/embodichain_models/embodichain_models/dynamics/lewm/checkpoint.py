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

"""Weights-only LeWM exports with explicitly bound data contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .model import LeWM, LeWMCfg

__all__ = ["export_lewm", "load_lewm"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_lewm(path: str | Path, model: LeWM, metadata: dict[str, Any]) -> None:
    """Write portable weights, model configuration and data identity.

    Args:
        path: Fresh export directory.
        model: Model whose parameters and action moments are exported.
        metadata: JSON-compatible source and controller contract.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    weights, manifest = path / "weights.pt", path / "manifest.json"
    if weights.exists() or manifest.exists():
        raise FileExistsError(f"LeWM export already exists: {path}")
    info = {
        "format": "embodichain.lewm.export.v1",
        "model": model.cfg.to_dict(),
        "metadata": metadata,
    }
    json.dumps(info, allow_nan=False)
    temporary = path / "weights.pt.tmp"
    torch.save(
        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, temporary
    )
    temporary.replace(weights)
    info["weights_sha256"] = _sha256(weights)
    temporary_manifest = path / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
    temporary_manifest.replace(manifest)


def load_lewm(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[LeWM, dict[str, Any]]:
    """Verify and load a weights-only export in evaluation mode.

    Args:
        path: Export directory produced by :func:`export_lewm`.
        device: Destination device for the model.

    Returns:
        Evaluation-mode model and its JSON manifest.
    """
    path = Path(path)
    info = json.loads((path / "manifest.json").read_text())
    if info.get("format") != "embodichain.lewm.export.v1" or _sha256(
        path / "weights.pt"
    ) != info.get("weights_sha256"):
        raise ValueError("unsupported LeWM export or weights checksum mismatch")
    model = LeWM(LeWMCfg(**info["model"]))
    model.load_state_dict(
        torch.load(path / "weights.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    return model.to(device).eval(), info
