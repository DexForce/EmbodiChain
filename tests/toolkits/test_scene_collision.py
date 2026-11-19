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

"""Content invalidation, durable publication and concurrent scene preparation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

import numpy as np
import pytest
import yaml

from embodichain.toolkits.skill_fitting import _collision

pytestmark = pytest.mark.no_sim


@pytest.fixture
def collision_scene(tmp_path, monkeypatch):
    """Two known disjoint tetrahedra, with an isolated cooking oracle/cache."""
    mesh = tmp_path / "mesh.obj"
    content = (
        "o left\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
        "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
        "o right\nv 3 0 0\nv 4 0 0\nv 3 1 0\nv 3 0 1\n"
        "f 5 7 6\nf 5 6 8\nf 5 8 7\nf 6 7 8\n"
    )
    mesh.write_text(content)
    config = dict(
        objects=dict(
            target=dict(
                pose=np.eye(4).tolist(),
                shape=dict(
                    shape_type="Mesh",
                    fpath="./mesh.obj",
                    collision=dict(approximation="convex_decomposition", max_hulls=8),
                ),
            )
        )
    )
    scene = tmp_path / "scene.yaml"
    scene.write_text(yaml.safe_dump(config))
    control = dict(
        calls=0, entered=threading.Event(), release=threading.Event(), fail=False
    )
    control["release"].set()

    def cook(vertices, triangles, recipe, directory):
        control["calls"] += 1
        control["entered"].set()
        assert control["release"].wait(5), "Cooking oracle was not released."
        output = directory / "output.obj"
        output.write_text(content)
        if control["fail"]:
            raise RuntimeError("native cooking failed")
        return output

    monkeypatch.setattr(_collision, "_cook", cook)
    return dict(
        scene=scene, config=config, mesh=mesh, cache=tmp_path / "cache", control=control
    )


@pytest.mark.parametrize("change", ["mesh", "parameters", "engine", "corruption"])
def test_cache_reuses_local_parts_and_invalidates_changed_inputs(
    collision_scene, monkeypatch, change
):
    data = collision_scene
    prepare = lambda: _collision.prepare_scene_collision(
        data["scene"], cache_root=data["cache"]
    )
    first = prepare()
    assert first["hull_count"] == 2 and first["cache_hits"] == 0
    # Placement is not part of object-local collision cooking.
    data["config"]["objects"]["target"]["pose"][0][3] = 12.0
    data["scene"].write_text(yaml.safe_dump(data["config"]))
    assert prepare()["cache_hits"] == 1
    assert data["control"]["calls"] == 1
    if change == "mesh":
        data["mesh"].write_text(data["mesh"].read_text().replace("v 4 0 0", "v 5 0 0"))
    elif change == "parameters":
        data["config"]["objects"]["target"]["shape"]["collision"]["max_hulls"] = 4
        data["scene"].write_text(yaml.safe_dump(data["config"]))
    elif change == "engine":
        import dexsim

        monkeypatch.setattr(dexsim, "__version__", "different-engine-build")
    else:
        Path(first["objects"]["target"]["fpath"]).write_text("# interrupted file\n")
    changed = prepare()
    assert changed["cache_hits"] == 0 and changed["hull_count"] == 2
    assert data["control"]["calls"] == 2
    assert prepare()["cache_hits"] == 1
    assert data["control"]["calls"] == 2


def test_concurrent_scene_imports_cook_one_complete_entry(collision_scene):
    data = collision_scene
    data["control"]["release"].clear()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            _collision.prepare_scene_collision, data["scene"], cache_root=data["cache"]
        )
        assert data["control"]["entered"].wait(5)
        second = pool.submit(
            _collision.prepare_scene_collision, data["scene"], cache_root=data["cache"]
        )
        try:
            time.sleep(0.05)
            assert not second.done()
        finally:
            data["control"]["release"].set()
        results = [first.result(timeout=5), second.result(timeout=5)]
    assert data["control"]["calls"] == 1
    assert sorted(result["cache_hits"] for result in results) == [0, 1]
    assert all(result["hull_count"] == 2 for result in results)


def test_failed_cooking_cannot_publish_a_partial_cache(collision_scene):
    data = collision_scene
    data["control"]["fail"] = True
    with pytest.raises(RuntimeError, match="native cooking failed"):
        _collision.prepare_scene_collision(data["scene"], cache_root=data["cache"])
    assert not list(data["cache"].rglob("manifest.json"))
    data["control"]["fail"] = False
    assert (
        _collision.prepare_scene_collision(data["scene"], cache_root=data["cache"])[
            "hull_count"
        ]
        == 2
    )
    assert data["control"]["calls"] == 2
