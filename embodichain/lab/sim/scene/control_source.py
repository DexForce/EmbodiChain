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
"""Static source/annotation/mass lineage for a declared revolute control."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from embodichain.compute.geometry.cylinder import circular_prism_radius
from .control_mass_source import qualify_mass_source_copy
from .control_semantics import resolve_twist_semantics
from .revolute_inputs import RevoluteControlBinding
from .revolute_geometry import TwistGeometry

__all__ = [
    "RevoluteSourceSeal",
    "capture_revolute_source_seal",
    "qualify_revolute_grip_shell",
]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class RevoluteSourceSeal:
    """Captured byte identities, including required absence of optional sidecars.

    Source files/annotation/calibration identity and static body scale are kept
    separate from Robot, native instance, root pose and table-path qualification.
    A changed file is rejected rather than silently accepted as a new baseline.
    """

    source_path: Path
    scale: float
    source_sha256: str
    calibration_sha256: str
    mass_lineage_sha256: str
    files: tuple[tuple[Path, str | None], ...]

    def __post_init__(self) -> None:
        source = Path(self.source_path).resolve()
        if (
            type(self.scale) not in (int, float)
            or not math.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("Source seal scale must be finite and positive.")

        def digest(value: str) -> bool:
            return (
                type(value) is str
                and len(value) == 64
                and all(c in "0123456789abcdef" for c in value)
            )

        if (
            not digest(self.source_sha256)
            or not digest(self.calibration_sha256)
            or type(self.mass_lineage_sha256) is not str
            or self.mass_lineage_sha256
            and not digest(self.mass_lineage_sha256)
        ):
            raise ValueError("Source seal identities must be SHA-256 values.")
        files = tuple((Path(path).resolve(), expected) for path, expected in self.files)
        if len({path for path, _ in files}) != len(files) or any(
            expected is not None and not digest(expected) for _, expected in files
        ):
            raise ValueError(
                "Source seal entries must be unique with qualified identities."
            )
        table = dict(files)
        if (
            table.get(source) != self.source_sha256
            or source.with_suffix(".twist.json") not in table
            or source.with_suffix(".mass.json") not in table
            or table[source.with_suffix(".mass.json")]
            != (self.mass_lineage_sha256 or None)
        ):
            raise ValueError(
                "Source seal must cover its asset and optional sidecar presence."
            )
        object.__setattr__(self, "source_path", source)
        object.__setattr__(self, "files", files)

    def assert_current(
        self,
        *,
        source_path: str | Path,
        body_scale: Sequence[float],
        body_scale_mass_policy: str = "native",
    ) -> None:
        """Recheck explicit live declaration values and captured source bytes.

        Args:
            source_path: Currently selected asset source.
            body_scale: Current three-component deployment scale.
            body_scale_mass_policy: Current Lab mass import selection.

        Raises:
            ValueError: Source, scale, sidecar presence or byte identity drifted.
        """
        if Path(source_path).resolve() != self.source_path:
            raise ValueError("Revolute geometry source path changed.")
        scale = np.asarray(body_scale)
        if scale.shape != (3,) or not np.allclose(scale, self.scale, rtol=1e-7, atol=0):
            raise ValueError("Revolute articulation geometry scale changed.")
        if self.mass_lineage_sha256 and body_scale_mass_policy != "native":
            raise ValueError("A derived mass source cannot also use Lab conversion.")
        for path, expected in self.files:
            actual = _digest(path) if path.is_file() else None
            if actual != expected:
                raise ValueError(
                    f"Revolute source bytes or sidecar presence changed: {path}"
                )


def capture_revolute_source_seal(
    source_path: str | Path,
    binding: RevoluteControlBinding,
    *,
    body_scale_mass_policy: str = "native",
) -> RevoluteSourceSeal:
    """Qualify the original source roles and optional two-attribute mass copy.

    Args:
        source_path: Explicit selected runtime USD source.
        binding: Independently validated source-derived control binding.
        body_scale_mass_policy: Must remain native when using a derived mass source.

    Returns:
        Immutable identities for later freshness checks. Robot/native provenance
        and dynamic collision qualification remain with their owning adapters.
    """
    path = Path(source_path).resolve()
    semantics = resolve_twist_semantics(path)
    if (
        semantics.source_sha256 != binding.source_sha256
        or semantics.semantics_sha256 != binding.calibration_sha256
    ):
        raise ValueError(
            "Revolute geometry semantics differ from the bound source roles."
        )
    entries = [
        (path, binding.source_sha256),
        (path.with_suffix(".twist.json"), semantics.sidecar_sha256),
    ]
    lineage = binding.mass_lineage_sha256
    manifest = path.with_suffix(".mass.json")
    entries.append((manifest, lineage or None))
    if lineage:
        mass = qualify_mass_source_copy(path, manifest_sha256=lineage)
        if mass.scale != binding.scale or body_scale_mass_policy != "native":
            raise ValueError(
                "Derived mass source has a conflicting scale or import policy."
            )
        entries.extend(
            (
                (mass.source, mass.source_sha256),
                (mass.source.with_suffix(".twist.json"), mass.source_sidecar_sha256),
            )
        )
    elif manifest.exists():
        raise ValueError("Mass lineage must be bound before assembly.")
    result = RevoluteSourceSeal(
        path,
        binding.scale,
        binding.source_sha256,
        binding.calibration_sha256,
        lineage,
        tuple(entries),
    )
    result.assert_current(
        source_path=path,
        body_scale=(binding.scale,) * 3,
        body_scale_mass_policy=body_scale_mass_policy,
    )
    return result


def qualify_revolute_grip_shell(
    geometry: TwistGeometry, binding: RevoluteControlBinding
) -> dict[str, object]:
    """Prove the source circular-prism boundary without reading live geometry.

    Args:
        geometry: Full source-qualified collider inputs in each owning link frame.
        binding: The same source/calibration identity and hinge geometry.

    Returns:
        Source-compatible radius and source digest; not native contact evidence.
    """
    if geometry.source_sha256 != binding.source_sha256:
        raise ValueError("Revolute geometry differs from its calibrated source.")
    radius = circular_prism_radius(
        geometry.grip.vertices,
        geometry.grip.faces,
        np.asarray(binding.axis),
        np.asarray(binding.origin),
        extra_colliders=tuple(
            mesh.vertices
            for mesh in geometry.target_collisions
            if mesh.path != geometry.grip.path
        ),
    )
    return {"radius_m": radius, "source_sha256": binding.source_sha256}
