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

pytestmark = pytest.mark.no_sim

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_TUTORIAL_PATH = _REPOSITORY_ROOT / "scripts/tutorials/sim/open_drawer.py"


def _load_tutorial_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "open_drawer_tutorial", _TUTORIAL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("is_newton_backend", [False, True])
def test_create_scene_configures_newton_grasp_material_only_for_newton(
    monkeypatch,
    is_newton_backend,
) -> None:
    tutorial = _load_tutorial_module()
    robot = object()
    drawer = object()
    captured: dict[str, object] = {}

    class FakeSimulation:
        def __init__(self):
            self.is_newton_backend = is_newton_backend

        def add_robot(self, cfg):
            captured["robot_cfg"] = cfg
            return robot

        def add_articulation(self, cfg):
            captured["drawer_cfg"] = cfg
            return drawer

    monkeypatch.setattr(
        tutorial.FrankaPandaCfg,
        "from_dict",
        lambda _config: SimpleNamespace(
            joint_drive_props=SimpleNamespace(damping={}),
            link_attrs=None,
        ),
    )
    monkeypatch.setattr(tutorial, "get_data_path", lambda asset: asset)

    tutorial.create_scene(FakeSimulation())

    drawer_cfg = captured["drawer_cfg"]
    robot_cfg = captured["robot_cfg"]
    assert robot_cfg.joint_drive_props.damping == {}
    assert drawer_cfg.root_props.fixed_base is True
    if is_newton_backend:
        robot_material = robot_cfg.link_attrs[
            "newton_gripper_contacts"
        ].attrs.material_props
        drawer_override = drawer_cfg.link_attrs["newton_handle_contacts"]
        drawer_material = drawer_override.attrs.material_props
        assert robot_material.ke == pytest.approx(
            tutorial.NEWTON_GRASP_CONTACT_STIFFNESS
        )
        assert robot_material.kd == pytest.approx(tutorial.NEWTON_GRASP_CONTACT_DAMPING)
        assert drawer_override.link_names_expr == [tutorial.DRAWER_CONTACT_LINK_NAME]
        assert drawer_material.ke == pytest.approx(
            tutorial.NEWTON_GRASP_CONTACT_STIFFNESS
        )
        assert drawer_material.kd == pytest.approx(
            tutorial.NEWTON_GRASP_CONTACT_DAMPING
        )
    else:
        assert robot_cfg.link_attrs is None
        assert drawer_cfg.link_attrs is None


def test_tutorial_newton_physics_cfg_enables_multiccd() -> None:
    tutorial = _load_tutorial_module()

    cfg = tutorial._tutorial_physics_cfg("newton")

    assert cfg.num_substeps == 20
    assert cfg.solver_cfg == {
        "solver_type": "mujoco_warp",
        "njmax": 8192,
        "nconmax": 8192,
        "cone": "elliptic",
        "enable_multiccd": True,
    }


def test_main_opens_native_window_after_spawn_prepare(monkeypatch) -> None:
    tutorial = _load_tutorial_module()
    events: list[str] = []
    captured_cfg: dict[str, object] = {}
    args = SimpleNamespace(
        num_envs=1,
        hold_steps=0,
        record_fps=30,
        record_save_path=None,
        headless=False,
        viser=False,
        auto_start=True,
        physics="default",
        device="cpu",
        arena_space=2.0,
        renderer="hybrid",
    )

    class FakeSimulation:
        num_envs = 1

        def prepare(self) -> None:
            events.append("prepare")

        def open_window(self) -> None:
            events.append("open_window")

        def update(self, *, step: int) -> None:
            pass

        def is_window_recording(self) -> bool:
            return False

        def wait_window_record_saves(self) -> None:
            pass

        def destroy(self) -> None:
            pass

    monkeypatch.setattr(
        tutorial.argparse.ArgumentParser,
        "parse_args",
        lambda _parser: args,
    )
    monkeypatch.setattr(
        tutorial,
        "SimulationManagerCfg",
        lambda **kwargs: captured_cfg.update(kwargs) or kwargs,
    )
    monkeypatch.setattr(tutorial, "SimulationManager", lambda _cfg: FakeSimulation())
    monkeypatch.setattr(
        tutorial,
        "create_scene",
        lambda _sim: events.append("create_scene")
        or (SimpleNamespace(uid="robot"), object()),
    )
    monkeypatch.setattr(tutorial, "MotionGenerator", lambda *, cfg: object())
    monkeypatch.setattr(tutorial, "open_drawer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tutorial, "visualization_cfg_from_args", lambda _args: None)

    tutorial.main()

    assert captured_cfg["headless"] is True
    assert events == ["create_scene", "prepare", "open_window"]
