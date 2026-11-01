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

"""Native joint-layout/installation task acceptance and persistence qualification."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

_WITHOUT_FCL = """
import importlib.abc, runpy, sys
class NoFCL(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'fcl' or fullname.startswith('fcl.'):
            raise ModuleNotFoundError('FCL is deliberately unavailable')
sys.meta_path.insert(0, NoFCL())
sys.argv = ['embodichain', *sys.argv[1:]]
runpy.run_module('embodichain', run_name='__main__')
"""


def _source_environment(root: Path) -> dict[str, str]:
    """Keep bundled extensions in the same checkout as the tested core."""
    env = dict(os.environ)
    sources = (
        root,
        root / "extensions",
        root / "extensions/embodichain_tasks",
        root / "extensions/embodichain_benchmarks",
        root / "extensions/embodichain_models",
    )
    env["PYTHONPATH"] = (
        os.pathsep.join(map(str, sources)) + os.pathsep + env.get("PYTHONPATH", "")
    )
    env["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    return env


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.gpu
@pytest.mark.slow
def test_native_joint_search_preserves_actual_start_and_measured_receipts(tmp_path):
    root = Path(__file__).resolve().parents[4]
    source = root / "examples/expansion/scene_robot_feasibility/search.ur5.yaml"
    config = yaml.safe_load(source.read_text())
    for key in ("deployment", "program"):
        config["task_program"][key] = str(
            (source.parent / config["task_program"][key]).resolve()
        )
    config["search"].update(
        max_candidates=1, max_regions=1, validation_samples=1, max_wall_time_s=180.0
    )
    path = tmp_path / "search.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "output"
    env = _source_environment(root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _WITHOUT_FCL,
            "search-scene-feasibility",
            "--config",
            str(path),
            "--output",
            str(output),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    from embodichain.lab.sim.scene.feasibility import load_feasibility_atlas

    atlas = load_feasibility_atlas(output / "atlas.json")
    assert len(atlas.regions) == 1 and atlas.regions[0].validation_complete
    assert len(atlas.observations) == 2
    assert all(
        o.evaluation.accepted and o.evaluation.committed for o in atlas.observations
    )
    files = list((output / "episodes").glob("*.npz"))
    assert len(files) == 2
    for file in files:
        with np.load(file, allow_pickle=False) as episode:
            metadata = json.loads(str(episode["metadata"]))
            physical = metadata["metadata"]["task_program"]["preparation"][
                "initial_state"
            ]
            robot = next(row for row in physical["entities"] if row["kind"] == "robot")
            assert np.allclose(
                episode["observation/joint_positions"][0],
                robot["qpos"],
                atol=1e-5,
                rtol=0,
            )
            assert (
                metadata["identity"]["initial_state_id"]
                == metadata["metadata"]["task_program"]["initial_state_id"]
            )
            assert (
                episode["actions"].shape[0] + 1
                == episode["observation/joint_positions"].shape[0]
            )
            assert "execution_seed" in metadata["metadata"]
            assert (
                metadata["metadata"]["collision_query"]["backend"]
                == "newton.CollisionPipeline"
            )
            assert (
                metadata["metadata"]["collision_query"]["clearance_scope"]
                == "capped_contact_band"
            )
    session = json.loads((output / "session.json").read_text())
    assert session["counts"]["committed"] == 2


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.gpu
@pytest.mark.slow
def test_native_bimanual_search_requires_transfer_place_and_independent_collision_checks(
    tmp_path,
):
    root = Path(__file__).resolve().parents[4]
    source = (
        root / "examples/expansion/scene_robot_feasibility/search.bar_dual_ur5.yaml"
    )
    config = yaml.safe_load(source.read_text())
    config["task_program"]["deployment"] = str(
        (source.parent / config["task_program"]["deployment"]).resolve()
    )
    config["search"].update(
        max_candidates=8, max_regions=1, validation_samples=1, max_wall_time_s=600.0
    )
    path = tmp_path / "bimanual.yaml"
    path.write_text(yaml.safe_dump(config))
    output = tmp_path / "bimanual"
    env = _source_environment(root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _WITHOUT_FCL,
            "search-scene-feasibility",
            "--config",
            str(path),
            "--output",
            str(output),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=750,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    from embodichain.lab.sim.scene.feasibility import load_feasibility_atlas

    atlas = load_feasibility_atlas(output / "atlas.json")
    accepted = [o for o in atlas.observations if o.evaluation.accepted]
    assert accepted and all(o.evaluation.committed for o in accepted)
    for row in accepted:
        validation = row.evaluation.evidence["task_program"]["measured_validation"]
        assert validation and all(c["status"] == "passed" for c in validation)
        events = next(c["detail"] for c in validation if c["check_id"] == "task.events")
        assert (
            "lift" in events
            and "right_holds_left_released" in events
            and "placed_and_released" in events
        )
        assert row.evaluation.evidence["first_runtime_collision"] is None
        assert (
            row.evaluation.evidence["collision_query"]["backend"]
            == "newton.CollisionPipeline"
        )
    files = list((output / "episodes").glob("*.npz"))
    assert len(files) == len(accepted)
    with np.load(files[0], allow_pickle=False) as episode:
        assert episode["actions"].shape[1] == 16
        assert (
            episode["actions"].shape[0] + 1
            == episode["observation/entity/bar"].shape[0]
        )
