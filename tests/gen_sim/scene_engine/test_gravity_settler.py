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

import pytest

from embodichain.gen_sim.scene_engine.core.scene_object import (
    ObjectPhysics,
    SceneObject,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.gravity_settler import (
    GravitySettleBody,
    GravitySettler,
)

_TABLE_ID = "table"
_ASSET_ID = "cube_001"
_IDENTITY_LAYOUT = {
    "rot": [0.0, 0.0, 0.0],
    "pos": [0.0, 0.0, 0.0],
    "scale": [1.0, 1.0, 1.0],
}


def _table_body() -> GravitySettleBody:
    return GravitySettleBody(
        scene_object=SceneObject(
            id=_TABLE_ID,
            kind="table",
            category="table",
            name="table",
            description="table",
        ),
        y_up_layout={"id": _TABLE_ID, **_IDENTITY_LAYOUT},
    )


def _asset_body() -> GravitySettleBody:
    return GravitySettleBody(
        scene_object=SceneObject(
            id=_ASSET_ID,
            kind="asset",
            category="cube",
            name="cube",
            description="cube",
        ),
        y_up_layout={"id": _ASSET_ID, **_IDENTITY_LAYOUT},
    )


def test_gravity_settler_returns_no_poses_without_dynamic_assets() -> None:
    settled_pose_by_id = GravitySettler(
        table_body=_table_body(),
        participant_bodies=[_asset_body()],
        dynamic_asset_ids=set(),
        static_asset_ids={_ASSET_ID},
    ).settle()

    assert settled_pose_by_id == {}


def test_gravity_settler_rejects_dynamic_assets_outside_participants() -> None:
    with pytest.raises(ValueError, match="exactly match participants"):
        GravitySettler(
            table_body=_table_body(),
            participant_bodies=[],
            dynamic_asset_ids={_ASSET_ID},
            static_asset_ids=set(),
        ).settle()


@pytest.mark.parametrize(
    ("max_hulls", "expected_approximation", "expected_max_hulls"),
    [
        (1, "convex_hull", None),
        (8, "convex_decomposition", 8),
    ],
)
def test_gravity_settler_normalizes_legacy_hull_budget(
    max_hulls: int,
    expected_approximation: str,
    expected_max_hulls: int | None,
) -> None:
    collision = GravitySettler._mesh_collision_cfg(
        ObjectPhysics(
            body_type="dynamic",
            attrs={"mass_props": {"mass": 1.0}},
            max_convex_hull_num=max_hulls,
        )
    )

    assert collision.approximation == expected_approximation
    assert collision.max_hulls == expected_max_hulls


@pytest.mark.parametrize("mode", ["fixed", "converged", "timeout"])
def test_gravity_settler_uses_shared_settling_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from types import SimpleNamespace
    import torch
    from embodichain.gen_sim.scene_engine.pipeline.utils import (
        gravity_settler as module,
    )
    from embodichain.lab.sim.scene.validation import SettlingCfg

    simulations = []
    cleanup = []

    class Simulation:
        num_envs = 1
        physics_backend = "default"
        device = torch.device("cpu")

        def __init__(self, cfg):
            self.rigid = {}
            self.steps = 0
            self.destroyed = False
            simulations.append(self)

        def get_rigid_object_uid_list(self):
            return list(self.rigid)

        def get_robot_uid_list(self):
            return []

        def get_articulation_uid_list(self):
            return []

        def get_rigid_object(self, uid):
            return self.rigid[uid]

        def prepare(self):
            pass

        def update(self, *, step):
            self.steps += step

        def destroy(self, *, exit_process):
            self.destroyed = True

        @staticmethod
        def flush_cleanup_queue():
            cleanup.append(True)

    def add_body(self, *, sim, object_id, body_info, physics, body_type):
        pose = torch.eye(4).unsqueeze(0)
        state = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        body = SimpleNamespace(
            body_type=body_type,
            body_state=state,
            get_local_pose=lambda **kwargs: pose,
        )
        sim.rigid[object_id] = body
        return body

    monkeypatch.setattr(module, "SimulationManager", Simulation)
    monkeypatch.setattr(
        module.GravitySettler,
        "_prepare_sim_body",
        lambda self, **kwargs: {"z_up_scale": [1.0, 1.0, 1.0]},
    )
    monkeypatch.setattr(module.GravitySettler, "_add_sim_body", add_body)
    settler = module.GravitySettler(
        table_body=_table_body(),
        participant_bodies=[_asset_body()],
        dynamic_asset_ids={_ASSET_ID},
        static_asset_ids=set(),
        config=module.GravitySettlerConfig(
            settle_steps=7,
            settling=(
                SettlingCfg(
                    mode="convergence",
                    max_steps=0 if mode == "timeout" else 5,
                    consecutive_steps=2,
                )
                if mode != "fixed"
                else None
            ),
        ),
    )
    if mode == "timeout":
        with pytest.raises(RuntimeError, match="convergence"):
            settler.settle()
        assert not settler.last_report.passed
        assert simulations[0].steps == 0
    else:
        result = settler.settle()
        assert result[_ASSET_ID]["pos"] == pytest.approx([0.0, 0.0, 0.0])
        assert settler.last_report.passed is (mode == "converged")
        assert simulations[0].steps == (2 if mode == "converged" else 7)
    assert simulations[0].destroyed and cleanup == [True]
