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

from collections.abc import Iterator
from dataclasses import MISSING
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.sensors import (
    BaseSensor,
    CameraCfg,
    ContactSensorCfg,
    SensorCfg,
    SensorContext,
    SensorRegistration,
    StereoCameraCfg,
    discover_sensor_plugins,
    get_sensor_registration,
    list_sensor_types,
    register_sensor,
)
from embodichain.lab.sim.sensors import registry
from embodichain.utils import configclass

pytestmark = pytest.mark.no_sim
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "examples/sim/sensors/external_joint_encoder"


@configclass
class _Options:
    gain: float = 2.0


@configclass
class _ProbeCfg(SensorCfg):
    sensor_type: str = "TestProbe"
    options: _Options = _Options()
    offsets: list[_Options] = []
    asset_uid: str = MISSING


@configclass
class _RequiredOptions:
    gain: float = MISSING
    offset: float = 0.0


@configclass
class _CalibratedProbeCfg(_ProbeCfg):
    sensor_type: str = "CalibratedTestProbe"
    options: _RequiredOptions = _RequiredOptions(gain=5.0, offset=1.0)


@configclass
class _TypedProbeCfg(SensorCfg):
    sensor_type: str = "TypedTestProbe"
    options: _RequiredOptions = _RequiredOptions()
    required: _RequiredOptions = MISSING
    optional: _RequiredOptions | None = None
    chain: list[_RequiredOptions] = []


class _Probe(BaseSensor):
    def __init__(self, cfg: _ProbeCfg, *, context: SensorContext) -> None:
        super().__init__(cfg, context=context)

    def _build_sensor_from_config(self, cfg: _ProbeCfg, device: torch.device) -> None:
        self._data_buffer["reading"] = torch.full(
            (self.context.num_instances, 1), cfg.options.gain, device=device
        )

    def update(self, **kwargs: object) -> None:
        pass

    def get_arena_pose(self, to_matrix: bool = False) -> None:
        return None

    def get_local_pose(self, to_matrix: bool = False) -> None:
        return None

    def set_local_pose(self, pose: torch.Tensor, env_ids=None) -> None:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(registry, "_registrations", registry._registrations.copy())
    monkeypatch.setattr(registry, "_entry_points", {})
    monkeypatch.setattr(registry, "_discovered", False)
    monkeypatch.setattr(registry, "_loading", set())
    monkeypatch.setattr(
        registry, "metadata", SimpleNamespace(entry_points=lambda **kwargs: ())
    )
    yield


def _manager() -> SimulationManager:
    manager = object.__new__(SimulationManager)
    manager.sim_config = SimpleNamespace(num_envs=2)
    manager.device = torch.device("cpu")
    manager._sensors = {}
    manager._lights = {}
    manager._robots = {}
    manager._articulations = {}
    manager._visualization_topology_revision = 0
    manager._visualization_sim_step = 12
    manager._visualization_sim_time = 0.25
    manager.prepare = lambda: None
    return manager


def _registration() -> SensorRegistration:
    return SensorRegistration("TestProbe", _ProbeCfg, _Probe)


def test_explicit_registration_shares_nested_config_decode_and_construction() -> None:
    register_sensor(_registration())
    cfg = SensorCfg.from_dict(
        {
            "sensor_type": "TestProbe",
            "uid": "probe",
            "asset_uid": "robot",
            "options": {"gain": 3.0},
            "offsets": [{"gain": 4.0}],
        }
    )
    assert type(cfg) is _ProbeCfg
    assert cfg.options.gain == 3.0
    assert isinstance(cfg.offsets[0], _Options) and cfg.offsets[0].gain == 4.0
    manager = _manager()
    sensor = manager.add_sensor(cfg)
    torch.testing.assert_close(
        sensor.get_data()["reading"], torch.tensor([[3.0], [3.0]])
    )
    assert sensor.num_instances == 2
    assert sensor.context.clock.physics_step_index == 12
    assert "TestProbe" not in SimulationManager.SUPPORTED_SENSOR_TYPES
    assert type(SensorCfg.from_dict(cfg.to_dict())) is _ProbeCfg


def test_required_plugin_fields_fail_before_scene_preparation() -> None:
    register_sensor(_registration())
    manager = _manager()
    manager.prepare = lambda: pytest.fail("Incomplete config must fail before prepare")
    with pytest.raises(TypeError, match="asset_uid"):
        manager.add_sensor(_ProbeCfg(uid="probe"))
    assert manager._sensors == {}


@pytest.mark.parametrize("optional", [{"gain": 2.0}, None])
def test_nested_required_and_optional_configurations_decode_from_annotations(
    optional: dict | None,
) -> None:
    register_sensor(SensorRegistration("TypedTestProbe", _TypedProbeCfg, _Probe))
    cfg = SensorCfg.from_dict(
        {
            "sensor_type": "TypedTestProbe",
            "options": {"gain": 1.0},
            "required": {"gain": 3.0},
            "optional": optional,
            "chain": [{"gain": 4.0}],
        }
    )
    cfg.validate()
    assert isinstance(cfg.options, _RequiredOptions) and cfg.options.gain == 1.0
    assert isinstance(cfg.required, _RequiredOptions) and cfg.required.gain == 3.0
    assert isinstance(cfg.chain[0], _RequiredOptions) and cfg.chain[0].gain == 4.0
    if optional is None:
        assert cfg.optional is None
    else:
        assert isinstance(cfg.optional, _RequiredOptions) and cfg.optional.gain == 2.0
    assert SensorCfg.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_registered_camera_is_recognized_before_creating_the_world() -> None:
    from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
    from embodichain.lab.sim.sensors import Camera

    @configclass
    class ExternalCameraCfg(CameraCfg):
        sensor_type: str = "ExternalCamera"

    class ExternalCamera(Camera):
        pass

    register_sensor(
        SensorRegistration(
            "ExternalCamera", ExternalCameraCfg, ExternalCamera, legacy_factory=True
        )
    )
    env = object.__new__(EmbodiedEnv)
    env.cfg = EmbodiedEnvCfg(sensor=[ExternalCameraCfg(uid="external_camera")])
    env.sim_cfg = env.cfg.sim_cfg
    env.sim_cfg.headless = True
    assert env._requires_native_renderer()
    assert "ExternalCamera" not in SimulationManager.SUPPORTED_SENSOR_TYPES


@pytest.mark.parametrize("overrides,offset", [({}, 1.0), ({"offset": 9.0}, 9.0)])
def test_partial_nested_overrides_preserve_owned_calibration_defaults(
    overrides: dict,
    offset: float,
) -> None:
    register_sensor(
        SensorRegistration("CalibratedTestProbe", _CalibratedProbeCfg, _Probe)
    )
    cfg = SensorCfg.from_dict(
        {
            "sensor_type": "CalibratedTestProbe",
            "asset_uid": "robot",
            "options": overrides,
        }
    )
    cfg.validate()
    assert cfg.options.gain == 5.0
    assert cfg.options.offset == offset
    assert _CalibratedProbeCfg().options.offset == 1.0


def test_duplicate_registration_and_api_mismatch_are_explicit() -> None:
    register_sensor(_registration())
    with pytest.raises(ValueError, match="already registered"):
        register_sensor(_registration())
    with pytest.raises(ValueError, match="supports API 1"):
        register_sensor(
            SensorRegistration("TestProbe", _ProbeCfg, _Probe, api_version=2)
        )


def test_installed_names_are_discovered_without_loading_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    entry = SimpleNamespace(
        name="TestProbe",
        value="external.sensor:REGISTRATION",
        load=lambda: calls.append("load") or _registration(),
    )
    monkeypatch.setattr(registry.metadata, "entry_points", lambda **kwargs: [entry])
    discover_sensor_plugins()
    assert list_sensor_types() == [
        "Camera",
        "ContactSensor",
        "StereoCamera",
        "TestProbe",
    ]
    assert calls == []
    SensorCfg.from_dict({"sensor_type": "Camera"})
    assert calls == []
    assert get_sensor_registration("TestProbe").config_cls is _ProbeCfg
    get_sensor_registration("TestProbe")
    assert calls == ["load"]


@pytest.mark.parametrize("duplicate_name", ["Camera", "TestProbe"])
def test_discovery_rejects_collisions_without_partial_registration(
    monkeypatch: pytest.MonkeyPatch, duplicate_name: str
) -> None:
    entries = [
        SimpleNamespace(name=duplicate_name, value="package_a:REGISTRATION"),
        SimpleNamespace(name=duplicate_name, value="package_b:REGISTRATION"),
    ]
    monkeypatch.setattr(registry.metadata, "entry_points", lambda **kwargs: entries)
    with pytest.raises(ValueError, match=f"Duplicate sensor type '{duplicate_name}'"):
        discover_sensor_plugins()
    assert not registry._entry_points


def test_plugin_load_failure_identifies_entry_point_and_preserves_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load():
        raise ImportError("missing optional sensor SDK")

    entry = SimpleNamespace(
        name="TestProbe", value="external.sensor:REGISTRATION", load=load
    )
    monkeypatch.setattr(registry.metadata, "entry_points", lambda **kwargs: [entry])
    with pytest.raises(RuntimeError, match="external.sensor:REGISTRATION") as caught:
        get_sensor_registration("TestProbe")
    assert isinstance(caught.value.__cause__, ImportError)
    assert "TestProbe" not in registry._registrations


def test_unknown_type_has_the_same_installation_guidance_in_config_and_manager() -> (
    None
):
    with pytest.raises(ValueError, match="Install its package"):
        SensorCfg.from_dict({"sensor_type": "NotInstalled"})
    with pytest.raises(ValueError, match="Unsupported sensor type 'NotInstalled'"):
        _manager().add_sensor(SensorCfg(uid="missing", sensor_type="NotInstalled"))


@pytest.mark.parametrize(
    ("sensor_type", "cfg_type", "extra"),
    [
        ("Camera", CameraCfg, {"extrinsics": {"parent": "robot/wrist"}}),
        ("StereoCamera", StereoCameraCfg, {"enable_disparity": True}),
        (
            "ContactSensor",
            ContactSensorCfg,
            {
                "articulation_cfg_list": [
                    {"articulation_uid": "robot", "link_name_list": ["foot"]}
                ]
            },
        ),
    ],
)
def test_builtin_configurations_keep_nested_values_and_round_trip(
    sensor_type: str, cfg_type: type[SensorCfg], extra: dict
) -> None:
    cfg = SensorCfg.from_dict({"sensor_type": sensor_type, "uid": "sensor", **extra})
    assert type(cfg) is cfg_type
    assert SensorCfg.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    if sensor_type == "Camera":
        assert cfg.extrinsics.parent == "robot/wrist"
    elif sensor_type == "StereoCamera":
        assert cfg.enable_disparity
    else:
        assert cfg.articulation_cfg_list[0].link_name_list == ["foot"]


def test_explicit_legacy_factory_keeps_two_argument_calling_convention() -> None:
    manager = _manager()

    def legacy(cfg, device):
        assert device == torch.device("cpu")
        return _Probe(cfg, context=SensorContext.from_manager(manager))

    register_sensor(
        SensorRegistration("TestProbe", _ProbeCfg, legacy, legacy_factory=True)
    )
    sensor = manager.add_sensor(_ProbeCfg(uid="legacy", asset_uid="robot"))
    assert sensor.device == torch.device("cpu") and sensor.num_instances == 2


@pytest.mark.parametrize("physical_registry", ["_robots", "_articulations"])
def test_context_clock_and_owned_state_follow_only_the_selected_manager(
    physical_registry: str,
) -> None:
    manager = _manager()
    # Distinct state rows make accidental environment broadcasting visible.
    qpos = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    qvel = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    getattr(manager, physical_registry)["robot"] = SimpleNamespace(
        get_qpos=lambda: qpos, get_qvel=lambda: qvel
    )
    # Physical and sensor UIDs intentionally occupy separate namespaces.
    manager._sensors["robot"] = object()
    context = SensorContext.from_manager(manager)
    snapshot = context.state.get_joint_state("robot")
    assert context.clock.now() == 0.25
    manager._visualization_sim_step = 13
    assert context.clock.physics_step_index == 13
    qpos[0, 0] = 99.0
    torch.testing.assert_close(snapshot["qpos"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    snapshot["qvel"].zero_()
    assert qvel[0, 0] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="does not provide joint state"):
        context.state.get_joint_state("missing")
    manager._is_pending_kill = True
    with pytest.raises(RuntimeError, match="unavailable"):
        context.clock.now()


@pytest.mark.slow
def test_wheel_installed_external_sensor_supports_python_yaml_and_manager(
    tmp_path: Path,
) -> None:
    """Exercise actual distribution metadata, rather than adding example source to sys.path."""
    source = tmp_path / "source"
    shutil.copytree(EXAMPLE_DIR, source)
    wheels = tmp_path / "wheels"
    target = tmp_path / "installed"

    def run(args: list[str], **kwargs) -> None:
        result = subprocess.run(args, capture_output=True, text=True, **kwargs)
        assert result.returncode == 0, result.stdout + result.stderr

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--no-index",
            "--wheel-dir",
            str(wheels),
            str(source),
        ]
    )
    wheel = next(wheels.glob("*.whl"))
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--no-compile",
            "--target",
            str(target),
            str(wheel),
        ]
    )
    script = textwrap.dedent("""
        from types import SimpleNamespace
        import torch
        import yaml
        from embodichain.lab.sim import SimulationManager
        from embodichain.lab.sim.sensors import SensorCfg, list_sensor_types
        from embodichain.lab.gym.utils.gym_utils import config_to_cfg

        assert "ExampleJointEncoder" in list_sensor_types()
        values = yaml.safe_load("sensor_type: ExampleJointEncoder\\nuid: robot\\nasset_uid: robot")
        cfg = SensorCfg.from_dict(values)
        from embodichain_joint_encoder_example import JointEncoderCfg
        assert isinstance(cfg, JointEncoderCfg)
        env_cfg = config_to_cfg({"id": "EmbodiedEnv-v1", "physics": "default", "env": {}, "robot": {}, "sensor": [values]})
        assert isinstance(env_cfg.sensor[0], JointEncoderCfg)

        sim = object.__new__(SimulationManager)
        sim.sim_config = SimpleNamespace(num_envs=3)
        sim.device = torch.device("cpu")
        sim._sensors = {}
        sim._lights = {}
        sim._articulations = {}
        sim._visualization_topology_revision = 0
        sim.prepare = lambda: None
        position = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        velocity = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        sim._robots = {"robot": SimpleNamespace(get_qpos=lambda: position, get_qvel=lambda: velocity)}
        sensor = sim.add_sensor(cfg)
        torch.testing.assert_close(sensor.get_data()["qpos"], position)
        torch.testing.assert_close(sensor.get_data()["qvel"], velocity)
        selections = (
            ([0], [[0.0, 0.0], [3.0, 4.0], [5.0, 6.0]]),
            (torch.tensor([2, 0], dtype=torch.int32), [[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]]),
            (torch.arange(3), [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            (torch.tensor([], dtype=torch.long), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        )
        for rows, expected in selections:
            sensor.update(fetch_only=True)
            torch.testing.assert_close(sensor.get_data()["qpos"], position)
            sensor.reset(rows)
            torch.testing.assert_close(sensor.get_data()["qpos"], torch.tensor(expected))
            torch.testing.assert_close(position, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        other = sim.add_sensor(JointEncoderCfg(uid="python_encoder", asset_uid="robot"))
        torch.testing.assert_close(other.get_data()["qvel"], velocity)
    """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(target), str(REPO_ROOT))))
    run([sys.executable, "-c", script], env=env, cwd=tmp_path)
