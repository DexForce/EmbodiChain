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

"""The benchmark must discover the in-tree TOPPRA backend on a clean install."""

from __future__ import annotations

import importlib.util

import pytest


@pytest.mark.no_sim
def test_toppra_adapter_is_available_without_external_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = importlib.util.find_spec

    def without_external(name, *args, **kwargs):
        return None if name == "toppra" else original(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", without_external)
    from embodichain_benchmarks.motion_generation.planners.toppra import ToppraAdapter

    adapter = object.__new__(ToppraAdapter)
    assert adapter.availability() == (True, None)
