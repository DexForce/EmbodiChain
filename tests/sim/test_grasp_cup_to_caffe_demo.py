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

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.no_sim

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEMO_PATH = _REPOSITORY_ROOT / "examples/sim/demo/grasp_cup_to_caffe.py"
_INITIAL_PHYSICS_STEPS = 1
_IDLE_LOOP_PHYSICS_STEPS = 10


def _load_demo_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("grasp_cup_to_caffe_demo", _DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scene_perturbations_precede_first_physics_step(monkeypatch) -> None:
    demo = _load_demo_module()
    events: list[str] = []

    class FakeSimulation:
        def prepare(self) -> None:
            events.append("prepare")

        def update(self, step: int) -> None:
            events.append(f"update:{step}")
            if step == _IDLE_LOOP_PHYSICS_STEPS:
                raise KeyboardInterrupt

        def open_window(self) -> None:
            events.append("open_window")

    sim = FakeSimulation()
    robot = object()
    cup = object()
    caffe = object()
    monkeypatch.setattr(
        demo,
        "parse_arguments",
        lambda: SimpleNamespace(headless=True, seed=0),
    )
    monkeypatch.setattr(demo, "initialize_simulation", lambda _args: sim)
    monkeypatch.setattr(demo, "create_robot", lambda _sim: robot)
    monkeypatch.setattr(demo, "create_table", lambda _sim: object())
    monkeypatch.setattr(demo, "create_caffe", lambda _sim: caffe)
    monkeypatch.setattr(demo, "create_cup", lambda _sim: cup)
    monkeypatch.setattr(
        demo,
        "apply_random_xy_perturbation",
        lambda item, **_kwargs: events.append(
            "perturb:cup" if item is cup else "perturb:caffe"
        ),
    )
    monkeypatch.setattr(
        demo,
        "run_simulation",
        lambda *_args: events.append("run_simulation"),
    )
    monkeypatch.setattr(
        demo.np.random,
        "seed",
        lambda seed: events.append(f"seed:{seed}"),
    )

    demo.main()

    assert events[:5] == [
        "prepare",
        "seed:0",
        "perturb:cup",
        "perturb:caffe",
        f"update:{_INITIAL_PHYSICS_STEPS}",
    ]


def test_trajectory_uses_authored_hold_target_as_ik_seed(monkeypatch) -> None:
    demo = _load_demo_module()
    target_reads: list[bool] = []

    class FakeRobot:
        def get_joint_ids(self, name: str) -> list[int]:
            assert name == "right_arm"
            return [0, 1]

        def get_qpos(self, target: bool = False) -> torch.Tensor:
            target_reads.append(target)
            return torch.tensor([[0.25, -0.5]], dtype=torch.float32)

        def compute_fk(self, **_kwargs) -> torch.Tensor:
            return torch.eye(4, dtype=torch.float32).unsqueeze(0)

        def compute_ik(
            self, *, joint_seed: torch.Tensor, **_kwargs
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ones(1, dtype=torch.bool), joint_seed.clone()

    class FakeItem:
        def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
            assert to_matrix
            return torch.eye(4, dtype=torch.float32).unsqueeze(0)

    monkeypatch.setattr(
        demo,
        "interpolate_with_distance",
        lambda trajectory, **_kwargs: trajectory,
    )

    trajectory = demo.create_trajectory(
        SimpleNamespace(
            device=torch.device("cpu"), num_envs=1, is_newton_backend=False
        ),
        FakeRobot(),
        FakeItem(),
        FakeItem(),
    )

    assert target_reads == [True]
    assert trajectory.shape == (1, 10, 8)
