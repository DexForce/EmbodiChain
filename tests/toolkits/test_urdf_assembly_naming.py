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

import os
import textwrap
import xml.etree.ElementTree as ET
import pytest

from embodichain.toolkits.urdf_assembly.urdf_assembly_manager import (
    URDFAssemblyManager,
)
from embodichain.toolkits.urdf_assembly.component import (
    URDFComponentManager,
)
from embodichain.toolkits.urdf_assembly.signature import (
    URDFAssemblySignatureManager,
)
from embodichain.toolkits.urdf_assembly.mesh import URDFMeshManager


# ---------------------------------------------------------------------------
# Minimal URDF fixture helpers
# ---------------------------------------------------------------------------

_SIMPLE_URDF = textwrap.dedent(
    """\
    <?xml version="1.0"?>
    <robot name="test_robot">
      <link name="base_link"/>
      <link name="end_link"/>
      <joint name="base_joint" type="fixed">
        <parent link="base_link"/>
        <child link="end_link"/>
      </joint>
    </robot>
    """
)


def _write_urdf(tmp_path: str, filename: str = "component.urdf") -> str:
    """Write a minimal URDF file and return its path."""
    path = os.path.join(tmp_path, filename)
    with open(path, "w") as f:
        f.write(_SIMPLE_URDF)
    return path


# ---------------------------------------------------------------------------
# 1. Unknown component key raises ValueError
# ---------------------------------------------------------------------------


class TestUnknownComponentKeyRaises:
    """Setting component_prefix with a key that is not in the default list
    must raise a ``ValueError``."""

    def test_unknown_key_raises(self):
        manager = URDFAssemblyManager()
        with pytest.raises(ValueError, match="cannot introduce new component"):
            manager.component_prefix = [("nonexistent_part", "x_")]

    def test_multiple_items_with_one_unknown_raises(self):
        manager = URDFAssemblyManager()
        with pytest.raises(ValueError, match="cannot introduce new component"):
            manager.component_prefix = [
                ("arm", "my_"),
                ("totally_unknown", "bad_"),
            ]

    def test_non_string_component_name_raises(self):
        manager = URDFAssemblyManager()
        with pytest.raises(ValueError):
            # component name is an int, not a string
            manager.component_prefix = [(42, "x_")]

    def test_non_list_raises(self):
        manager = URDFAssemblyManager()
        with pytest.raises(ValueError):
            manager.component_prefix = {"arm": "my_"}  # dict, not list


# ---------------------------------------------------------------------------
# 2. Per-component prefix patches preserve default ordering
# ---------------------------------------------------------------------------


class TestComponentPrefixPreservesOrder:
    """Patching specific prefixes must not reorder the component list."""

    def _default_order(self) -> list[str]:
        m = URDFAssemblyManager()
        return [comp for comp, _ in m.component_order_and_prefix]

    def test_patch_does_not_reorder(self):
        manager = URDFAssemblyManager()
        default_order = [comp for comp, _ in manager.component_order_and_prefix]

        # Patch a subset of components – order must remain the same
        manager.component_prefix = [
            ("left_arm", "L_"),
            ("right_arm", "R_"),
        ]

        patched_order = [comp for comp, _ in manager.component_order_and_prefix]
        assert patched_order == default_order

    def test_prefix_updated_correctly(self):
        manager = URDFAssemblyManager()
        manager.component_prefix = [("arm", "robot_arm_")]

        prefix_map = dict(manager.component_order_and_prefix)
        assert prefix_map["arm"] == "robot_arm_"

    def test_unpatched_components_keep_original_prefix(self):
        manager = URDFAssemblyManager()
        original_prefix_map = dict(manager.component_order_and_prefix)

        # Only patch "head" – all others must keep their original prefix
        manager.component_prefix = [("head", "hd_")]

        patched_prefix_map = dict(manager.component_order_and_prefix)
        for comp, original_prefix in original_prefix_map.items():
            if comp == "head":
                assert patched_prefix_map[comp] == "hd_"
            else:
                assert patched_prefix_map[comp] == original_prefix

    def test_multiple_patches_are_cumulative(self):
        """Two successive patch calls must both take effect."""
        manager = URDFAssemblyManager()
        manager.component_prefix = [("left_arm", "L_")]
        manager.component_prefix = [("right_arm", "R_")]

        prefix_map = dict(manager.component_order_and_prefix)
        assert prefix_map["left_arm"] == "L_"
        assert prefix_map["right_arm"] == "R_"


# ---------------------------------------------------------------------------
# 3. name_case changes affect link / joint names via URDFComponentManager
# ---------------------------------------------------------------------------


class TestNameCaseAffectsNames:
    """Changing ``name_case`` must propagate to the managers that rename
    links and joints when processing URDF components."""

    def _make_manager_and_process(
        self,
        tmp_path: str,
        name_case: dict,
    ) -> tuple[list, list]:
        """Helper: create a component manager with given name_case, process a
        minimal URDF, and return (links, joints) lists."""
        urdf_path = _write_urdf(tmp_path)
        mesh_manager = URDFMeshManager(output_dir=tmp_path)

        comp_manager = URDFComponentManager(
            mesh_manager=mesh_manager,
            name_case=name_case,
        )

        # Use SimpleNamespace so the local urdf_path variable is captured correctly
        import types

        fake_comp = types.SimpleNamespace(
            urdf_path=urdf_path,
            params=None,
            transform=None,
        )

        links: list = []
        joints: list = []
        name_mapping: dict = {}
        base_points: dict = {}

        comp_manager.process_component(
            comp="chassis",
            prefix=None,
            comp_obj=fake_comp,
            name_mapping=name_mapping,
            base_points=base_points,
            links=links,
            joints=joints,
        )
        return links, joints

    def test_link_names_lowercase(self, tmp_path):
        links, _ = self._make_manager_and_process(
            str(tmp_path), name_case={"link": "lower", "joint": "none"}
        )
        for link in links:
            name = link.get("name")
            if name:
                assert name == name.lower(), f"Link name not lowercase: {name!r}"

    def test_link_names_uppercase(self, tmp_path):
        links, _ = self._make_manager_and_process(
            str(tmp_path), name_case={"link": "upper", "joint": "none"}
        )
        for link in links:
            name = link.get("name")
            if name:
                assert name == name.upper(), f"Link name not uppercase: {name!r}"

    def test_joint_names_uppercase(self, tmp_path):
        _, joints = self._make_manager_and_process(
            str(tmp_path), name_case={"link": "none", "joint": "upper"}
        )
        for joint in joints:
            name = joint.get("name")
            if name:
                assert name == name.upper(), f"Joint name not uppercase: {name!r}"

    def test_joint_names_lowercase(self, tmp_path):
        _, joints = self._make_manager_and_process(
            str(tmp_path), name_case={"link": "none", "joint": "lower"}
        )
        for joint in joints:
            name = joint.get("name")
            if name:
                assert name == name.lower(), f"Joint name not lowercase: {name!r}"

    def test_name_case_none_preserves_original(self, tmp_path):
        links, joints = self._make_manager_and_process(
            str(tmp_path), name_case={"link": "none", "joint": "none"}
        )
        link_names = {link.get("name") for link in links}
        joint_names = {joint.get("name") for joint in joints}
        # Original names from the fixture are mixed-case preserved exactly
        assert "base_link" in link_names
        assert "end_link" in link_names
        assert "base_joint" in joint_names

    def test_assembly_manager_name_case_setter_validation(self):
        manager = URDFAssemblyManager()

        with pytest.raises(ValueError, match="must be a dictionary"):
            manager.name_case = "lower"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="must contain keys"):
            manager.name_case = {"link": "lower"}  # missing "joint"

    def test_assembly_manager_name_case_propagates_to_component_manager(self):
        manager = URDFAssemblyManager()
        manager.name_case = {"joint": "lower", "link": "upper"}

        assert manager.component_manager._name_case == {
            "joint": "lower",
            "link": "upper",
        }


# ---------------------------------------------------------------------------
# 4. Signature changes when component_prefix or name_case changes
# ---------------------------------------------------------------------------


class TestSignatureChangesWithNamingSettings:
    """Assembly signatures must differ when either ``component_prefix`` or
    ``name_case`` differs, so stale URDF caches are correctly invalidated."""

    _BASE_COMPONENT_INFO: dict = {
        "__component_order_and_prefix__": [("arm", None), ("hand", None)],
        "__name_case__": {"joint": "upper", "link": "lower"},
    }

    def _sig(self, component_info: dict) -> str:
        mgr = URDFAssemblySignatureManager()
        return mgr.calculate_assembly_signature(component_info, "/tmp/out.urdf")

    def test_baseline_is_stable(self):
        """Same inputs must produce the same signature (determinism)."""
        assert self._sig(self._BASE_COMPONENT_INFO) == self._sig(
            self._BASE_COMPONENT_INFO
        )

    def test_prefix_change_invalidates_signature(self):
        info_a = {
            "__component_order_and_prefix__": [("arm", None), ("hand", None)],
            "__name_case__": {"joint": "upper", "link": "lower"},
        }
        info_b = {
            "__component_order_and_prefix__": [("arm", "robot_"), ("hand", None)],
            "__name_case__": {"joint": "upper", "link": "lower"},
        }
        assert self._sig(info_a) != self._sig(info_b)

    def test_name_case_change_invalidates_signature(self):
        info_a = {
            "__component_order_and_prefix__": [("arm", None)],
            "__name_case__": {"joint": "upper", "link": "lower"},
        }
        info_b = {
            "__component_order_and_prefix__": [("arm", None)],
            "__name_case__": {"joint": "lower", "link": "lower"},
        }
        assert self._sig(info_a) != self._sig(info_b)

    def test_order_change_invalidates_signature(self):
        info_a = {
            "__component_order_and_prefix__": [("arm", None), ("hand", None)],
            "__name_case__": {"joint": "upper", "link": "lower"},
        }
        info_b = {
            "__component_order_and_prefix__": [("hand", None), ("arm", None)],
            "__name_case__": {"joint": "upper", "link": "lower"},
        }
        assert self._sig(info_a) != self._sig(info_b)

    def test_assembly_manager_signature_reflects_name_case(self, tmp_path):
        """End-to-end: changing name_case on the manager changes the
        signature that would be used to gate cache invalidation."""
        sig_mgr = URDFAssemblySignatureManager()

        manager_a = URDFAssemblyManager()
        manager_b = URDFAssemblyManager()
        manager_b.name_case = {"joint": "lower", "link": "lower"}

        output_path = str(tmp_path / "robot.urdf")

        def _compute_sig(m: URDFAssemblyManager) -> str:
            component_info = m.component_registry.all().copy()
            component_info["__component_order_and_prefix__"] = list(
                m.component_order_and_prefix
            )
            component_info["__name_case__"] = dict(m._name_case)
            return sig_mgr.calculate_assembly_signature(component_info, output_path)

        assert _compute_sig(manager_a) != _compute_sig(manager_b)

    def test_assembly_manager_signature_reflects_component_prefix(self, tmp_path):
        """End-to-end: changing component_prefix on the manager changes the
        signature that would be used to gate cache invalidation."""
        sig_mgr = URDFAssemblySignatureManager()

        manager_a = URDFAssemblyManager()
        manager_b = URDFAssemblyManager()
        manager_b.component_prefix = [("arm", "robot_")]

        output_path = str(tmp_path / "robot.urdf")

        def _compute_sig(m: URDFAssemblyManager) -> str:
            component_info = m.component_registry.all().copy()
            component_info["__component_order_and_prefix__"] = list(
                m.component_order_and_prefix
            )
            component_info["__name_case__"] = dict(m._name_case)
            return sig_mgr.calculate_assembly_signature(component_info, output_path)

        assert _compute_sig(manager_a) != _compute_sig(manager_b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
