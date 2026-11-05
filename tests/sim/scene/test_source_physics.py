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

"""Source physical declarations retain independently frozen B0 behavior."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from embodichain.lab.sim.scene.source_physics import source_scene_object_declaration

_BASELINE = json.loads(
    (Path(__file__).parent / "data/source_physics_b0.json").read_text()
)


@pytest.mark.parametrize("case", _BASELINE["cases"], ids=lambda value: value["id"])
def test_public_source_object_declaration_preserves_b0_fields_and_error_order(case):
    source = deepcopy(case["config"])
    before = deepcopy(source)
    expected = case["expected"]
    if "error_type" in expected:
        with pytest.raises(Exception) as raised:
            source_scene_object_declaration(source, role=case["role"])
        assert type(raised.value).__name__ == expected["error_type"]
        assert str(raised.value) == expected["error"]
    else:
        result = source_scene_object_declaration(source, role=case["role"])
        assert (
            json.dumps(
                result, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
            == expected["serialized"]
        )
        if result.get("shape"):
            result["shape"]["caller-edit"] = True
        if result.get("attrs"):
            first = next(iter(result["attrs"].values()))
            if isinstance(first, dict):
                first["caller-edit"] = True
            else:
                result["attrs"]["caller-edit"] = True
    assert source == before


def test_source_physics_public_api_needs_no_gensim_or_native_runtime(tmp_path):
    script = """
import importlib.abc
import sys
class RejectProviderImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("embodichain.gen_sim"):
            raise AssertionError("Physical declaration imported its host: " + fullname)
sys.meta_path.insert(0, RejectProviderImports())
from embodichain.lab.sim.sim_manager import SimulationManager
from embodichain.lab.sim.objects.robot import Robot
def forbidden(*args, **kwargs):
    raise AssertionError("Physical declaration constructed a native runtime")
SimulationManager.__init__ = forbidden
Robot.__init__ = forbidden
from embodichain.lab.sim.scene.source_physics import source_scene_object_declaration
result = source_scene_object_declaration({"uid": "direct"}, role="rigid_object")
assert result["attrs"]["mass_props"]["mass"] == 0.1
assert result["body_type"] == "dynamic"
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_standalone_ingress_rejects_unknown_roles_before_emitting_dynamic_physics():
    with pytest.raises(ValueError, match="Source role"):
        source_scene_object_declaration({"uid": "source"}, role="unregistered")
