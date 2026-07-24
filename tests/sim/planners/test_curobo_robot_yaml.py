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

"""Dependency-free unit tests for cuRobo robot-YAML mimic-joint detection.

cuRobo folds each URDF ``<mimic>`` joint into its active joint, so mimic joints
have no independent body and must be excluded from ``cspace``/``lock_joints`` or
cuRobo raises ``KeyError`` when locking them. These tests pin the detection that
:func:`generate_curobo_robot_yaml` relies on; they need no CUDA/cuRobo/DexSim.
"""

from __future__ import annotations

from embodichain.lab.sim.planners.curobo.curobo_yaml import _parse_mimic_joint_names

# Minimal URDF mirroring the Franka Panda hand: ``fr3_finger_joint2`` mimics
# ``fr3_finger_joint1``. The auto-generator must detect ``fr3_finger_joint2``
# (and only it) so cuRobo does not try to lock a joint with no body.
_MIMIC_URDF = """\
<?xml version="1.0"?>
<robot name="panda_hand">
  <link name="base"/>
  <link name="fr3_hand"/>
  <link name="fr3_leftfinger"/>
  <link name="fr3_rightfinger"/>
  <joint name="fr3_hand_joint" type="fixed">
    <parent link="base"/>
    <child link="fr3_hand"/>
  </joint>
  <joint name="fr3_finger_joint1" type="prismatic">
    <parent link="fr3_hand"/>
    <child link="fr3_leftfinger"/>
    <axis xyz="0 1 0"/>
    <limit effort="100" lower="0.0" upper="0.04" velocity="0.2"/>
  </joint>
  <joint name="fr3_finger_joint2" type="prismatic">
    <parent link="fr3_hand"/>
    <child link="fr3_rightfinger"/>
    <axis xyz="0 1 0"/>
    <limit effort="100" lower="0.0" upper="0.04" velocity="0.2"/>
    <mimic joint="fr3_finger_joint1"/>
  </joint>
</robot>
"""

_NO_MIMIC_URDF = """\
<?xml version="1.0"?>
<robot name="arm">
  <link name="base"/>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit effort="100" lower="-3.14" upper="3.14" velocity="2.0"/>
  </joint>
</robot>
"""


def test_parse_mimic_joint_names_detects_mimic_joint(tmp_path):
    """The mimic joint name is returned; the active joint it mimics is not."""
    urdf = tmp_path / "panda_hand.urdf"
    urdf.write_text(_MIMIC_URDF, encoding="utf-8")

    mimic_joints = _parse_mimic_joint_names(str(urdf))

    assert mimic_joints == {"fr3_finger_joint2"}
    assert "fr3_finger_joint1" not in mimic_joints


def test_parse_mimic_joint_names_returns_empty_without_mimic(tmp_path):
    """A URDF with no ``<mimic>`` tags yields an empty set."""
    urdf = tmp_path / "arm.urdf"
    urdf.write_text(_NO_MIMIC_URDF, encoding="utf-8")

    assert _parse_mimic_joint_names(str(urdf)) == set()


def test_parse_mimic_joint_names_handles_missing_file(tmp_path):
    """A missing/unreadable URDF degrades to an empty set rather than raising."""
    assert _parse_mimic_joint_names(str(tmp_path / "does_not_exist.urdf")) == set()
