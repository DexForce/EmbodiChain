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

"""Mounted robot topology for the atomic-skill example planning tests."""

from __future__ import annotations

import re
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from examples.sim.atomic_skill.support import create_tutorial_robot_cfg


@pytest.fixture
def mounted_robot():
    def build(model: str) -> SimpleNamespace:
        cfg = create_tutorial_robot_cfg(model)
        source = ET.parse(cfg.urdf_cfg.assemble_urdf())
        joints = [
            j.attrib["name"]
            for j in source.findall("joint")
            if j.get("type") != "fixed"
        ]
        return SimpleNamespace(
            cfg=cfg,
            joint_names=joints,
            all_joint_names=joints,
            get_joint_ids=lambda name: [
                i
                for i, j in enumerate(joints)
                if any(re.fullmatch(p, j) for p in cfg.control_parts[name])
            ],
            link_names=[link.attrib["name"] for link in source.findall("link")],
        )

    return build
