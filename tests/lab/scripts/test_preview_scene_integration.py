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

import json
from pathlib import Path

import dexsim
from PIL import Image
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, UsdShade, Sdf, UsdUtils
import pytest

from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.scripts.preview_scene import preview_scene

pytestmark = [pytest.mark.requires_sim, pytest.mark.renderer]


def _source(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    visual = UsdGeom.Cube.Define(stage, "/World/Visual")
    visual.CreateSizeAttr(0.2)
    visual.AddTranslateOp().Set((-0.4, 0, 0.5))
    visual.CreateDisplayColorAttr([(0.8, 0.1, 0.1)])
    material = UsdShade.Material.Define(stage, "/World/Looks/Red")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Red/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        (0.8, 0.05, 0.05)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)
    floor = UsdGeom.Mesh.Define(stage, "/World/Floor")
    floor.CreatePointsAttr(
        [(-500, -500, 0), (500, -500, 0), (500, 500, 0), (-500, 500, 0)]
    )
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    dynamic = UsdGeom.Cube.Define(stage, "/World/Dynamic")
    dynamic.CreateSizeAttr(0.2)
    dynamic.AddTranslateOp().Set((0.4, 0, 0.5))
    UsdPhysics.RigidBodyAPI.Apply(dynamic.GetPrim())
    UsdPhysics.CollisionAPI.Apply(dynamic.GetPrim())
    UsdPhysics.MassAPI.Apply(dynamic.GetPrim()).CreateMassAttr(0.25)
    arm = UsdGeom.Xform.Define(stage, "/World/Assembly").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(arm)
    for name, z in (("Base", 0.2), ("Link", 0.4)):
        link = UsdGeom.Cube.Define(stage, "/World/Assembly/" + name)
        link.CreateSizeAttr(0.15)
        link.AddTranslateOp().Set((0, z, z))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        UsdPhysics.CollisionAPI.Apply(link.GetPrim())
        UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr(0.1)
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Assembly/Slide")
    joint.CreateBody0Rel().SetTargets(["/World/Assembly/Base"])
    joint.CreateBody1Rel().SetTargets(["/World/Assembly/Link"])
    joint.CreateLocalPos0Attr((0, 0.2, 0.2))
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(0.2)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Assembly/Fixed")
    fixed.CreateBody1Rel().SetTargets(["/World/Assembly/Base"])
    fixed.CreateLocalPos0Attr((0, 0.2, 0.2))
    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(1500)
    stage.GetRootLayer().Save()


@pytest.mark.parametrize("suffix", ["usda", "usd", "usdc", "usdz"])
def test_plain_usd_mixed_scene_loads_without_producer_metadata(
    tmp_path: Path, suffix: str
) -> None:
    source = tmp_path / "scene.usda"
    _source(source)
    if suffix == "usdz":
        destination = tmp_path / "scene.usdz"
        assert UsdUtils.CreateNewUsdzPackage(str(source), str(destination))
    elif suffix != "usda":
        destination = tmp_path / ("scene." + suffix)
        assert Usd.Stage.Open(str(source)).GetRootLayer().Export(str(destination))
    else:
        destination = source
    before = destination.read_bytes()
    report = tmp_path / "report.json"
    result = preview_scene(
        destination, sim_cfg=SimulationManagerCfg(headless=True), report=report
    )
    assert result["physics_steps"] == 0
    assert result["simulation_time"] == 0.0
    by_path = {row["prim_path"]: row for row in result["entities"]}
    assert by_path["/World/Visual"]["kind"] == "visual_object"
    assert by_path["/World/Dynamic"]["body_type"] == "dynamic"
    assert by_path["/World/Floor"]["body_type"] == "static"
    assert by_path["/World/Assembly"]["kind"] == "articulation"
    assert destination.read_bytes() == before
    assert json.loads(report.read_text())["status"] == "loaded"


def test_scene_screenshot_and_explicit_physics_have_separate_step_counts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scene.usda"
    _source(source)
    stage = Usd.Stage.Open(str(source))
    stage.GetPrimAtPath("/World").SetCustomDataByKey(
        "embodichain:scene_schema", "obsolete/private"
    )
    stage.GetPrimAtPath("/World/Assembly").SetCustomDataByKey(
        "embodichain:initial_qpos", "opaque-data"
    )
    stage.GetRootLayer().Save()
    image = tmp_path / "preview.png"
    paused = preview_scene(
        source,
        sim_cfg=SimulationManagerCfg(headless=True),
        screenshot=image,
        resolution=(64, 64),
    )
    assert paused["physics_steps"] == 0
    assert paused["simulation_time"] == 0.0
    with Image.open(image) as rendered:
        assert rendered.size == (64, 64)
        import numpy as np

        pixels = np.asarray(rendered).astype(float)
        red = (pixels[:, :, 0] > 1.5 * pixels[:, :, 1]) & (
            pixels[:, :, 0] > 1.5 * pixels[:, :, 2]
        )
        assert red.sum() > 2, "The explicitly red USD visual must be visible."
    advanced = preview_scene(
        source, sim_cfg=SimulationManagerCfg(headless=True), simulate=True, steps=3
    )
    assert advanced["physics_steps"] == 3
    assert advanced["simulation_time"] == pytest.approx(3 * 0.01)


def test_native_viser_preview_processes_step_and_stop_without_pose_holding(
    tmp_path: Path,
):
    import socket
    from embodichain.lab.visualization import VisualizationCfg, ViserServerCfg

    source = tmp_path / "scene.usda"
    _source(source)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    phase = [0]
    frames = [0]

    def commands(sim):
        assert sim._default_plane is None
        frames[0] += 1
        assert frames[0] < 100, "Preview commands must not stall."
        runtime = sim.visualization_runtime
        if phase[0] == 0:
            runtime._enqueue_preview_command("step")
            phase[0] = 1
        elif phase[0] == 1 and sim.simulation_time > 0:
            assert sim.simulation_time == pytest.approx(0.01)
            runtime._enqueue_preview_command("play")
            phase[0] = 2
        elif phase[0] == 2 and sim.simulation_time > 0.01:
            runtime._enqueue_preview_command("stop")
            phase[0] = 3
        return False

    result = preview_scene(
        source,
        sim_cfg=SimulationManagerCfg(
            headless=True,
            visualization=VisualizationCfg(
                backend="viser",
                allow_commands=True,
                viser_server=ViserServerCfg(port=port),
            ),
        ),
        on_frame=commands,
    )
    assert phase[0] == 3
    assert result["physics_steps"] == 2
    assert result["frontend"] == "viser"


def test_native_window_displays_initial_state_without_physics(tmp_path: Path):
    import os

    if not os.getenv("DISPLAY"):
        pytest.skip("Native window qualification needs an available display.")
    source = tmp_path / "scene.usda"
    _source(source)
    frames = [0]

    def stop(sim):
        assert sim._default_plane is None
        assert sim.simulation_time == 0
        assert sim.is_window_opened
        frames[0] += 1
        if frames[0] == 2:
            raise KeyboardInterrupt()
        return False

    result = preview_scene(
        source, sim_cfg=SimulationManagerCfg(headless=False), on_frame=stop
    )
    assert result["physics_steps"] == 0
    assert frames[0] == 2


def test_native_scene_reset_reimports_without_advancing_physics(tmp_path: Path):
    import socket
    from embodichain.lab.visualization import VisualizationCfg, ViserServerCfg

    source = tmp_path / "scene.usda"
    _source(source)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    visits = []

    def reset_then_stop(sim):
        runtime = sim.visualization_runtime
        if not visits:
            visits.append(sim.instance_id)
            runtime._enqueue_preview_command("reset")
        else:
            visits.append(sim.instance_id)
            runtime._enqueue_preview_command("stop")
        assert sim.simulation_time == 0
        return False

    result = preview_scene(
        source,
        sim_cfg=SimulationManagerCfg(
            headless=True,
            visualization=VisualizationCfg(
                backend="viser",
                allow_commands=True,
                viser_server=ViserServerCfg(port=port),
            ),
        ),
        on_frame=reset_then_stop,
    )
    assert len(visits) == 2
    assert result["resets"] == 1
    assert result["physics_steps"] == 0
