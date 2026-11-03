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

"""Optional collision backends fail explicitly when their binding is absent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from embodichain.lab.sim.scene import _collision_backends as backends


@pytest.mark.parametrize(
    "failure", [ModuleNotFoundError("fcl"), OSError("missing native library")]
)
def test_missing_fcl_has_actionable_dependency_error_and_no_fallback(
    monkeypatch, failure
):
    requested = []

    def missing(name):
        requested.append(name)
        raise failure

    monkeypatch.setattr(backends, "import_module", missing)
    with pytest.raises(
        ImportError, match=r"embodichain\[planning-geometry\]"
    ) as caught:
        backends._require_fcl()
    assert requested == ["fcl"]
    assert caught.value.__cause__ is failure


def test_incomplete_fcl_binding_is_unavailable(monkeypatch):
    monkeypatch.setattr(backends, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(ImportError, match="No collision result is available"):
        backends._require_fcl()


def test_installed_fcl_preflight():
    pytest.importorskip(
        "fcl", reason="actual FCL backend is not installed; coverage is unqualified"
    )
    backends._require_fcl()
