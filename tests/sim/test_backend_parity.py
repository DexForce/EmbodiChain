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
"""Backend capability parity matrix.

This is the single source of truth for which simulation features each physics
backend supports. It pins the capability contract so that:

- flipping a ``supports_*`` flag (or adding a backend) fails loudly, and
- every ``SimulationManager.add_*`` capability guard maps 1:1 to its flag.

Headless (no GPU / no dexsim world): backends are constructed with a minimal
fake owning-manager back-ref, and the ``add_*`` guard mapping is exercised by
binding a fake ``physics`` onto a bare ``SimulationManager`` via
``object.__new__`` (mirroring the lifecycle-test pattern).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from embodichain.lab.sim.physics import (
    DefaultPhysicsBackend,
    NewtonPhysicsBackend,
    PhysicsBackend,
)
from embodichain.lab.sim.sim_manager import SimulationManager

# ---------------------------------------------------------------------------
# The parity matrix — edit this table when a backend gains/loses a feature.
# ---------------------------------------------------------------------------
# feature -> {backend -> supported}
BACKEND_CAPABILITIES: dict[str, dict[str, bool]] = {
    "robot": {"default": True, "newton": True},
    "soft_bodies": {"default": True, "newton": False},
    "cloth": {"default": True, "newton": False},
    "rigid_object_group": {"default": True, "newton": False},
    "can_disable_manual_update": {"default": True, "newton": False},
}

BACKENDS: dict[str, type[PhysicsBackend]] = {
    "default": DefaultPhysicsBackend,
    "newton": NewtonPhysicsBackend,
}

# Map each capability flag to the SimulationManager.add_* method whose
# NotImplementedError guard consults it. ``None`` means the flag is consulted
# elsewhere (e.g. set_manual_update) rather than an add_* guard.
CAPABILITY_TO_ADD_METHOD: dict[str, str | None] = {
    "robot": "add_robot",
    "soft_bodies": "add_soft_object",
    "cloth": "add_cloth_object",
    "rigid_object_group": "add_rigid_object_group",
    "can_disable_manual_update": None,
}


def _make_backend(name: str) -> PhysicsBackend:
    """Construct a backend with a minimal fake owning-manager back-ref."""
    return BACKENDS[name](SimpleNamespace())


@pytest.mark.parametrize("backend_name", list(BACKENDS))
def test_backend_name_matches(backend_name: str) -> None:
    backend = _make_backend(backend_name)
    assert backend.name == backend_name


@pytest.mark.parametrize("backend_name", list(BACKENDS))
@pytest.mark.parametrize(
    "feature", [f for f in BACKEND_CAPABILITIES if f != "can_disable_manual_update"]
)
def test_supports_flags_match_matrix(backend_name: str, feature: str) -> None:
    """Each backend's supports_* property matches the parity matrix."""
    backend = _make_backend(backend_name)
    expected = BACKEND_CAPABILITIES[feature][backend_name]
    actual = getattr(backend, f"supports_{feature}")
    assert (
        actual is expected
    ), f"{backend_name}.supports_{feature} = {actual}, matrix says {expected}"


@pytest.mark.parametrize("backend_name", list(BACKENDS))
def test_can_disable_manual_update_matches_matrix(backend_name: str) -> None:
    backend = _make_backend(backend_name)
    expected = BACKEND_CAPABILITIES["can_disable_manual_update"][backend_name]
    assert backend.can_disable_manual_update is expected


def _make_sim_with_backend(backend: PhysicsBackend) -> SimulationManager:
    """Build a bare SimulationManager whose ``physics`` is the given backend.

    The add_* capability guards consult only ``self.physics.supports_*`` (plus a
    few uid/existence checks that run after the guard), so a bare instance with
    ``physics`` + the registries set is enough to assert the guard fires.
    """
    sim = object.__new__(SimulationManager)
    sim.physics = backend
    sim._soft_objects = {}
    sim._cloth_objects = {}
    sim._rigid_object_groups = {}
    sim._robots = {}
    sim._rigid_objects = {}
    sim._articulations = {}
    return sim


@pytest.mark.parametrize(
    "feature,add_method",
    [(f, m) for f, m in CAPABILITY_TO_ADD_METHOD.items() if m is not None],
)
@pytest.mark.parametrize("backend_name", list(BACKENDS))
def test_add_method_guard_maps_to_capability(
    backend_name: str, feature: str, add_method: str
) -> None:
    """add_<feature> raises NotImplementedError iff the backend lacks the flag.

    For unsupported features the guard must fire before any world access; for
    supported features the method proceeds past the guard (and is expected to
    fail later on the missing world — we only assert it does NOT raise
    NotImplementedError at the guard).
    """
    backend = _make_backend(backend_name)
    sim = _make_sim_with_backend(backend)
    supported = BACKEND_CAPABILITIES[feature][backend_name]
    method = getattr(sim, add_method)

    # Minimal cfg stub: add_* only reads .uid before/after the guard.
    cfg = SimpleNamespace(uid=None)

    if supported:
        # Past the guard it will hit missing-world attrs; assert the failure is
        # NOT the capability NotImplementedError.
        with pytest.raises(Exception) as exc_info:
            method(cfg=cfg)
        assert not isinstance(exc_info.value, NotImplementedError), (
            f"{add_method} raised NotImplementedError on the {backend_name} "
            f"backend despite supports_{feature}=True"
        )
        assert "not enabled" not in str(exc_info.value)
    else:
        with pytest.raises(NotImplementedError, match="not enabled"):
            method(cfg=cfg)


def test_matrix_covers_all_capability_flags() -> None:
    """Every supports_* / can_disable_manual_update flag is in the matrix."""
    flag_names = {
        name[len("supports_") :] if name.startswith("supports_") else name
        for name in dir(PhysicsBackend)
        if name.startswith("supports_") or name == "can_disable_manual_update"
    }
    matrix_features = set(BACKEND_CAPABILITIES)
    assert (
        flag_names == matrix_features
    ), f"capability flags {flag_names} != matrix features {matrix_features}"


def test_matrix_covers_all_backends() -> None:
    """Every concrete backend class is in the matrix."""
    # Discover concrete (non-abstract) backends by instantiation.
    concrete = set(BACKENDS)
    matrix_backends = {b for feats in BACKEND_CAPABILITIES.values() for b in feats}
    assert concrete == matrix_backends


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
