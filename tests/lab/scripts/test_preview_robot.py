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

"""Robot-preview input, orchestration, and resource-boundary contracts."""

from __future__ import annotations

from copy import deepcopy
import gc
import json
from pathlib import Path
import queue
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
import weakref
import xml.etree.ElementTree as ET

import pytest

from embodichain.lab.scripts import preview_robot
from embodichain import robot_registry as registry

pytestmark = pytest.mark.no_sim
_ROOT = Path(__file__).resolve().parents[3]


def _args(*args: str):
    return preview_robot._create_parser().parse_args(args)


@pytest.fixture
def robot_asset(tmp_path: Path) -> Path:
    path = tmp_path / "robot.urdf"
    path.write_text(
        '<robot name="preview"><link name="base"/><link name="tip"/>'
        '<joint name="joint" type="revolute"><parent link="base"/>'
        '<child link="tip"/><axis xyz="0 0 1"/>'
        '<limit lower="-1" upper="1" effort="10" velocity="1"/>'
        "</joint></robot>"
    )
    return path


@pytest.fixture
def provider(monkeypatch, robot_asset):
    from embodichain.lab.sim.cfg import RobotCfg
    from embodichain.utils import configclass

    @configclass
    class VendorCfg(RobotCfg):
        fpath: str = str(robot_asset)
        control_parts: dict[str, list[str]] = {"custom_arm": ["joint"]}

    spec = {
        "class_path": "preview_test_provider:VendorCfg",
        "defaults": {"uid": "catalog", "init_qpos": [0.1]},
        "variants": {"alternate": {"defaults": {"uid": "variant", "init_qpos": [0.5]}}},
    }
    entry = SimpleNamespace(name="preview", load=lambda: {"vendor.robot": spec})
    monkeypatch.setitem(
        sys.modules, "preview_test_provider", SimpleNamespace(VendorCfg=VendorCfg)
    )
    original_entry_points = registry.metadata.entry_points

    def entry_points(**kwargs):
        if kwargs.get("group") == "embodichain.robots":
            return [entry]
        return original_entry_points(**kwargs)

    monkeypatch.setattr(registry.metadata, "entry_points", entry_points)
    registry._robot_specs.cache_clear()
    yield spec
    registry._robot_specs.cache_clear()


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--robot", "ur5", "--robot-config", "robot.yaml"],
        ["--robot", "ur5", "--max-steps", "0"],
        ["--robot", "ur5", "--max-steps", "-1"],
        ["--robot", "ur5", "--max-steps", "1.5"],
        ["--robot", "ur5", "--robot-params", "[]"],
        ["--robot", "ur5", "--robot-params", "{invalid"],
        ["--robot", "ur5", "--robot-params", '{"class_type":"ur5"}'],
    ],
)
def test_invalid_cli_arguments_are_rejected(argv) -> None:
    with pytest.raises(SystemExit) as error:
        _args(*argv)
    assert error.value.code == 2


def test_preview_defaults_and_opt_outs() -> None:
    defaults = _args("--robot", "ur5")
    disabled = _args("--robot", "ur5", "--viser", "--no-gizmo", "--no-joint-control")
    assert not defaults.headless and not defaults.viser
    assert defaults.physics == "default" and defaults.device is None
    assert defaults.num_envs == 1 and defaults.gizmo and defaults.joint_control
    assert defaults.max_steps is None
    assert not disabled.gizmo and not disabled.joint_control


@pytest.mark.parametrize("extension", ["json", "yaml", "yml"])
def test_standalone_config_resolves_relative_assets_and_cli_overrides(
    robot_asset, extension
) -> None:
    from embodichain.utils.utility import save_config

    data = {
        "uid": "file",
        "fpath": robot_asset.name,
        "control_parts": {"custom_arm": ["joint"]},
    }
    path = robot_asset.parent / f"robot.{extension}"
    save_config(path, data)
    before = path.read_bytes()
    args = _args("--robot-config", str(path), "--robot-params", '{"uid":"caller"}')

    cfg = preview_robot.build_robot_cfg(args)

    assert cfg.fpath == str(robot_asset)
    assert cfg.uid == "caller"
    assert cfg.control_parts == {"custom_arm": ["joint"]}
    assert path.read_bytes() == before
    assert args.robot_params == {"uid": "caller"}


@pytest.mark.parametrize("sensors", [None, [], {}])
def test_component_asset_paths_resolve_from_config_directory(
    robot_asset, sensors
) -> None:
    path = robot_asset.parent / "assembled.json"
    path.write_text(
        json.dumps(
            {
                "uid": "assembly",
                "urdf_cfg": {
                    "sensors": sensors,
                    "components": [
                        {"component_type": "arm", "urdf_path": robot_asset.name}
                    ],
                },
                "control_parts": {"custom_arm": ["joint"]},
            }
        )
    )

    cfg = preview_robot.build_robot_cfg(_args("--robot-config", str(path)))

    assert cfg.urdf_cfg.components["arm"]["urdf_path"] == str(robot_asset)


def test_registered_asset_ids_are_not_rewritten(robot_asset) -> None:
    fields = {
        "fpath": "Franka/panda/pandaWithHand.urdf",
        "solver_cfg": {"custom_arm": {"urdf_path": robot_asset.name}},
    }
    preview_robot._resolve_local_assets(fields, robot_asset.parent)
    assert fields["fpath"] == "Franka/panda/pandaWithHand.urdf"
    assert fields["solver_cfg"]["custom_arm"]["urdf_path"] == str(robot_asset)


@pytest.mark.parametrize("sensor_format", ["list", "mapping"])
def test_relative_sensor_urdf_is_included_from_another_working_directory(
    robot_asset: Path, monkeypatch: pytest.MonkeyPatch, sensor_format: str
) -> None:
    hand = robot_asset.parent / "hand.urdf"
    hand.write_text('<robot name="hand"><link name="hand_base"/></robot>')
    sensor = robot_asset.parent / "sensor.urdf"
    sensor.write_text('<robot name="sensor"><link name="sensor_link"/></robot>')
    sensor_cfg = {
        "sensor_name": "camera",
        "sensor_source": sensor.name,
        "parent_component": "arm",
        "parent_link": "tip",
    }
    path = robot_asset.parent / "assembled.json"
    path.write_text(
        json.dumps(
            {
                "uid": "assembly",
                "urdf_cfg": {
                    "components": [
                        {"component_type": "arm", "urdf_path": robot_asset.name},
                        {"component_type": "hand", "urdf_path": hand.name},
                    ],
                    "sensors": (
                        [sensor_cfg]
                        if sensor_format == "list"
                        else {"camera": sensor_cfg}
                    ),
                    "fpath": "output/assembled.urdf",
                },
                "control_parts": {"custom_arm": ["joint"]},
            }
        )
    )
    original = path.read_bytes()
    working_dir = robot_asset.parent / "other"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)

    cfg = preview_robot.build_robot_cfg(_args("--robot-config", str(path)))
    assembled = ET.parse(cfg.urdf_cfg.assemble_urdf())

    assert cfg.urdf_cfg.sensors[0]["sensor_source"] == str(sensor)
    links = {link.get("name") for link in assembled.findall("link")}
    assert "camera_sensor_link" in links
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "source",
    [
        {"type": "camera", "name": "camera_link"},
        ET.fromstring('<robot name="sensor"><link name="camera_link"/></robot>'),
        ([ET.Element("link", name="camera_link")], []),
        "VendorSensors/camera.urdf",
    ],
)
def test_sensor_sources_keep_inline_definitions_and_registry_references(
    tmp_path: Path, source
) -> None:
    config = {"urdf_cfg": {"sensors": [{"sensor_source": source}]}}

    preview_robot._resolve_local_assets(config, tmp_path)

    assert config["urdf_cfg"]["sensors"][0]["sensor_source"] is source


@pytest.mark.parametrize("source", ["catalog", "file"])
def test_catalog_variant_file_and_cli_precedence(provider, robot_asset, source) -> None:
    original = deepcopy(provider)
    if source == "catalog":
        argv = ["--robot", "vendor.robot"]
        expected_qpos = [0.5]
    else:
        path = robot_asset.parent / "vendor.json"
        path.write_text(
            json.dumps(
                {"class_type": "vendor.robot", "uid": "file", "init_qpos": [0.7]}
            )
        )
        argv = ["--robot-config", str(path)]
        expected_qpos = [0.7]
    args = _args(*argv, "--variant", "alternate", "--robot-params", '{"uid":"caller"}')

    cfg = preview_robot.build_robot_cfg(args)

    assert cfg.uid == "caller"
    assert cfg.init_qpos == expected_qpos
    assert provider == original


@pytest.mark.parametrize("payload", [[], {}, {"class_type": None}])
def test_invalid_config_fails_before_creating_world(tmp_path, payload) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, TypeError)):
        preview_robot.build_robot_cfg(_args("--robot-config", str(path)))


def test_generic_robot_config_rejects_catalog_variant(robot_asset) -> None:
    path = robot_asset.parent / "generic.json"
    path.write_text(json.dumps({"fpath": robot_asset.name}))
    with pytest.raises(ValueError, match="requires a catalog ID"):
        preview_robot.build_robot_cfg(
            _args("--robot-config", str(path), "--variant", "alternate")
        )


def test_generic_file_accepts_null_optional_asset_fields(robot_asset) -> None:
    path = robot_asset.parent / "nullable.json"
    path.write_text(json.dumps({"fpath": str(robot_asset), "urdf_cfg": None}))

    cfg = preview_robot.build_robot_cfg(_args("--robot-config", str(path)))

    assert cfg.fpath == str(robot_asset)
    assert cfg.urdf_cfg is None


@pytest.mark.parametrize("arguments", [[], ["preview-robot", "--help"]])
def test_help_does_not_load_simulation_or_robot_modules(arguments) -> None:
    script = """
import sys
from embodichain.cli.main import main
try:
    main(sys.argv[1:] or ["--help"])
except SystemExit as error:
    assert error.code == 0
for prefix in ("torch", "dexsim", "embodichain.lab.sim", "embodichain.lab.gym", "embodichain.lab.task_program", "embodichain_robots"):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix
"""
    result = subprocess.run(
        [sys.executable, "-c", script, *arguments],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "preview-robot" in result.stdout


@pytest.fixture
def fake_simulation(monkeypatch, robot_asset):
    from embodichain.lab.sim import sim_manager
    from embodichain.lab.scripts import _preview
    from embodichain.lab.sim.cfg import RobotCfg

    events = []
    captured = {}
    robot = SimpleNamespace(
        uid="test", active_joint_ids=[0], control_parts={"custom_arm": ["joint"]}
    )
    controller = SimpleNamespace(update=lambda: events.append("joint"))

    class Manager:
        def __init__(self, cfg):
            events.append("create")
            captured["sim_cfg"] = cfg
            captured["sim"] = self

        def add_robot(self, *, cfg):
            events.append("robot")
            return robot

        def prepare(self):
            events.append("prepare")

        def open_window(self):
            events.append("window")
            return True

        def update_gizmos(self):
            events.append("gizmos")

        def get_gizmo_items(self):
            return ()

        def capture_visualization_safely(self, *, force):
            assert force
            events.append("publish")

        def update(self, *, step):
            assert step == 1
            events.append("physics")

        def destroy(self, *, exit_process):
            assert exit_process is False
            events.append("destroy")

        @staticmethod
        def flush_cleanup_queue():
            events.append("flush")

    def setup(sim, assets, args, *, follow_external_targets=False):
        captured["follow"] = follow_external_targets
        assert assets == [robot]
        if args.viser and args.joint_control:
            events.append("controls")
            return controller
        return None

    monkeypatch.setattr(sim_manager, "SimulationManager", Manager)
    monkeypatch.setattr(_preview, "_setup_viser_joint_control", setup)
    monkeypatch.setattr(
        preview_robot,
        "build_robot_cfg",
        lambda _: RobotCfg(uid="test", fpath=str(robot_asset)),
    )
    return events, captured, Manager


@pytest.mark.parametrize("mode", ["native", "viser", "headless"])
def test_viewer_setup_steps_and_cleanup_are_ordered(fake_simulation, mode) -> None:
    from embodichain.lab.sim.objects import GizmoCfg

    events, captured, _manager = fake_simulation
    flags = {"native": [], "viser": ["--viser"], "headless": ["--headless"]}[mode]
    preview_robot.main(_args("--robot", "ur5", *flags, "--max-steps", "2"))

    assert captured["sim_cfg"].headless is True
    assert isinstance(captured["sim_cfg"].robot_ik_gizmo, GizmoCfg)
    assert not captured["sim_cfg"].robot_ik_gizmo.ik_start_enabled
    assert events[:3] == ["create", "robot", "prepare"]
    assert ("window" in events) is (mode == "native")
    if mode == "viser":
        assert (
            events.index("gizmos") < events.index("controls") < events.index("publish")
        )
        assert captured["follow"] is True
        assert events.count("joint") == 2
    assert events.count("physics") == 2
    assert events[-2:] == ["destroy", "flush"]


def test_headless_check_does_not_step_and_gizmo_can_be_disabled(
    fake_simulation,
) -> None:
    events, captured, _ = fake_simulation
    preview_robot.main(_args("--robot", "ur5", "--headless", "--no-gizmo"))
    assert captured["sim_cfg"].robot_ik_gizmo is None
    assert captured["follow"] is False
    assert "physics" not in events and "window" not in events
    assert events[-2:] == ["destroy", "flush"]


@pytest.mark.parametrize("stage", ["add_robot", "prepare", "update", "open_window"])
def test_loading_or_update_failure_still_destroys_and_flushes(
    fake_simulation, monkeypatch, stage
) -> None:
    events, _captured, manager = fake_simulation
    original = RuntimeError("preview failure")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(manager, stage, fail)
    with pytest.raises(RuntimeError) as error:
        preview_robot.main(_args("--robot", "ur5", "--max-steps", "1"))
    assert error.value is original
    assert events[-2:] == ["destroy", "flush"]


def test_unwound_native_views_are_released_before_deferred_cleanup(
    fake_simulation, monkeypatch
) -> None:
    _events, _captured, manager = fake_simulation
    views = []

    class NativeView:
        pass

    def backend():
        view = NativeView()
        views.append(weakref.ref(view))
        raise ValueError("native backend cause")

    def fail(*_args):
        try:
            backend()
        except ValueError as cause:
            raise RuntimeError("native failure") from cause

    def flush():
        gc.collect()
        assert views[0]() is None

    monkeypatch.setattr(preview_robot, "_run_simulation", fail)
    monkeypatch.setattr(manager, "flush_cleanup_queue", flush)
    with pytest.raises(RuntimeError, match="native failure"):
        preview_robot.main(_args("--robot", "ur5"))


@pytest.mark.parametrize(
    "stage",
    ["config", "registry_reassignment", "manual_update", "activate", "spawn", "viser"],
)
def test_constructor_failure_releases_only_the_preview_world(
    monkeypatch: pytest.MonkeyPatch, robot_asset: Path, stage: str
) -> None:
    """The real constructor queues its partial state before unwinding."""
    from embodichain.lab.sim import sim_manager
    from embodichain.lab.sim.cfg import RobotCfg
    from embodichain.lab import visualization

    manager = sim_manager.SimulationManager
    live = SimpleNamespace(_world=MagicMock())
    replacement = SimpleNamespace(_world=MagicMock())
    instances = {0: live}
    cleanup_queue = queue.Queue()
    monkeypatch.setattr(manager, "_instances", instances)
    monkeypatch.setattr(manager, "_cleanup_queue", cleanup_queue)
    monkeypatch.setattr(manager, "_initialized_process_renderer", False)
    monkeypatch.setattr(manager, "has_native_renderer", property(lambda self: False))
    monkeypatch.setattr(sim_manager.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(sim_manager, "_initialize_warp_runtime", lambda *args: None)
    monkeypatch.setattr(manager, "enable_physics", lambda *args: None)
    monkeypatch.setattr(manager, "_declare_spawn_default_plane", lambda *args: None)
    monkeypatch.setattr(manager, "wait_scene_destruction", MagicMock())
    monkeypatch.setattr(
        preview_robot, "build_robot_cfg", lambda args: RobotCfg(fpath=str(robot_asset))
    )
    failure = RuntimeError(f"constructor {stage} failed")
    views = []
    created = []
    world = MagicMock()
    scene = MagicMock()
    scene.builder.prepare_arenas.return_value = []
    physics = MagicMock()

    class NativeView:
        pass

    def fail(*args, **kwargs):
        view = NativeView()
        views.append(weakref.ref(view))
        raise failure

    def make_world(cfg):
        created.append(manager._instances[1])
        return world

    runtime = MagicMock()
    if stage in ("config", "registry_reassignment"):

        def fail_config(*args):
            if stage == "registry_reassignment":
                instances[1] = replacement
            fail()

        monkeypatch.setattr(manager, "_convert_sim_config", fail_config)
    else:
        monkeypatch.setattr(manager, "_convert_sim_config", lambda *args: object())
    if stage == "manual_update":
        world.set_manual_update.side_effect = fail
    elif stage == "activate":
        physics.activate.side_effect = fail
    elif stage == "spawn":
        scene.builder.prepare_arenas.side_effect = fail
    elif stage == "viser":
        runtime.start.side_effect = fail
    monkeypatch.setattr(sim_manager, "make_physics_backend", lambda *args: physics)
    monkeypatch.setattr(sim_manager.dexsim, "World", make_world)
    monkeypatch.setattr(sim_manager, "SpawnScene", lambda *args, **kwargs: scene)
    monkeypatch.setattr(visualization, "SceneExporter", lambda *args: object())
    monkeypatch.setattr(visualization, "VisualizationRuntime", lambda *args: runtime)

    def quit_world():
        gc.collect()
        assert views[0]() is None, "Constructor frame still owns a native view"

    world.quit.side_effect = quit_world
    with pytest.raises(RuntimeError) as error:
        preview_robot.main(_args("--robot", "ur5", "--viser", "--headless"))

    assert error.value is failure
    assert instances == (
        {0: live, 1: replacement} if stage == "registry_reassignment" else {0: live}
    )
    assert cleanup_queue.empty()
    live._world.quit.assert_not_called()
    replacement._world.quit.assert_not_called()
    manager.wait_scene_destruction.assert_not_called()
    if stage in ("config", "registry_reassignment"):
        world.quit.assert_not_called()
        assert not created
    else:
        world.quit.assert_called_once_with()
        assert created[0]._world is None
        assert created[0]._env is None
    if stage in ("spawn", "viser"):
        scene.close.assert_called_once_with()
    if stage == "viser":
        runtime.stop.assert_called_once_with()


def test_cleanup_error_does_not_replace_original_runtime_error(
    fake_simulation, monkeypatch
) -> None:
    events, _captured, manager = fake_simulation
    original = RuntimeError("loading failed")

    def fail_loading(self):
        raise original

    def fail_destroy(self, *, exit_process):
        events.append("destroy-failed")
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(manager, "prepare", fail_loading)
    monkeypatch.setattr(manager, "destroy", fail_destroy)
    with pytest.raises(RuntimeError) as error:
        preview_robot.main(_args("--robot", "ur5"))
    assert error.value is original
    assert events[-2:] == ["destroy-failed", "flush"]


def test_interrupted_preview_still_releases_its_world(
    fake_simulation, monkeypatch
) -> None:
    events, _captured, manager = fake_simulation

    def interrupt(self, *, step):
        raise KeyboardInterrupt

    monkeypatch.setattr(manager, "update", interrupt)
    preview_robot.main(_args("--robot", "ur5"))
    assert events[-2:] == ["destroy", "flush"]
