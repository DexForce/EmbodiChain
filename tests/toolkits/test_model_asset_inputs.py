# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Model source inventories share geometry registry without constructing geometry."""

from __future__ import annotations

from pathlib import Path
import pytest

from embodichain.toolkits.graspkit.pg_grasp import model_geometry as owner


@pytest.mark.parametrize(
    "model,side,reference",
    [
        ("franka_panda_hand", None, "Franka/Panda/PandaWithHand.urdf"),
        ("robotiq_2f85", None, "Robotiq/robotiq_arg2f_85/robotiq_arg2f_85.urdf"),
        ("robotiq_arg2f_140", None, "Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf"),
        ("dh_pgi_140_80", None, "DH_PGI_140_80/DH_PGI_140_80.urdf"),
        ("pika_gripper", None, "PikaGripper/PikaGripper.urdf"),
        ("tianji_marvin_gripper", "left", "TianjiMarvin/robot_with_ee_acd.urdf"),
        ("tianji_marvin_gripper", "right", "TianjiMarvin/robot_with_ee_acd.urdf"),
    ],
)
def test_model_inventory_resolves_actual_registered_source_without_geometry(
    tmp_path, monkeypatch, model, side, reference
):
    source = tmp_path / "model.urdf"
    mesh = tmp_path / "mesh.stl"
    mesh.write_bytes(b"original")
    source.write_text(
        '<robot><link name="root"><collision><geometry><mesh filename="mesh.stl"/></geometry></collision></link></robot>'
    )
    requests = []

    def resolve(value):
        requests.append(value)
        assert value == reference
        return str(source)

    def forbidden(*args, **kwargs):
        pytest.fail("asset inventory constructed collision geometry")

    monkeypatch.setattr(owner, "get_data_path", resolve)
    monkeypatch.setattr(owner.FixedGripperCollisionGeometry, "from_urdf", forbidden)
    assert set(owner.get_parallel_jaw_model_asset_files(model, side=side)) == {
        source,
        mesh,
    }
    assert requests == [reference]


def test_cobot_inventory_uses_same_explicit_archive_overrides(tmp_path, monkeypatch):
    source = tmp_path / "model.urdf"
    names = ["gripper_base_v100_acd.obj", "link7_acd.obj", "link8_acd.obj"]
    meshes = {name: tmp_path / name for name in names}
    for path in meshes.values():
        path.write_text("v 0 0 0\n")
    source.write_text(
        "<robot>"
        + "".join(f'<mesh filename="../Collision/{name}"/>' for name in names)
        + "</robot>"
    )

    def resolve(reference):
        if reference == "CobotMagicArm/CobotMagicWithGripperV100.urdf":
            return str(source)
        return str(meshes[Path(reference).name])

    monkeypatch.setattr(owner, "get_data_path", resolve)
    assert set(owner.get_parallel_jaw_model_asset_files("cobotmagic_v100_gripper")) == {
        source,
        *meshes.values(),
    }


@pytest.mark.parametrize(
    "model,side",
    [("missing", None), ("tianji_marvin_gripper", None), ("robotiq_arg2f_140", "left")],
)
def test_inventory_rejects_unknown_model_or_wrong_side_before_data_lookup(
    monkeypatch, model, side
):
    monkeypatch.setattr(
        owner,
        "get_data_path",
        lambda *_: pytest.fail("invalid model/side reached data"),
    )
    with pytest.raises(ValueError):
        owner.get_parallel_jaw_model_asset_files(model, side=side)
