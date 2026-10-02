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

"""Provider-free tests for the MCP URDF assembly adapter."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from embodichain.mcp import MCPAdapterRegistry, URDFAssemblyAdapter

EXPECTED_LINK_COUNT = 5
EXPECTED_JOINT_COUNT = 4


def _write_component(path: Path, *, root: str, tip: str, joint: str) -> None:
    """Write a minimal two-link URDF component for an assembly test."""
    path.write_text(
        f"""<robot name=\"{root}\">
  <link name=\"{root}\"/>
  <link name=\"{tip}\"/>
  <joint name=\"{joint}\" type=\"fixed\">
    <parent link=\"{root}\"/>
    <child link=\"{tip}\"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )


@pytest.fixture
def adapter(tmp_path: Path) -> URDFAssemblyAdapter:
    """Return an adapter backed by two local fixture URDFs."""
    arm = tmp_path / "arm.urdf"
    hand = tmp_path / "hand.urdf"
    _write_component(arm, root="arm_base", tip="arm_tip", joint="arm_joint")
    _write_component(hand, root="hand_base", tip="finger", joint="hand_joint")
    assets = {"fixture/arm.urdf": arm, "fixture/hand.urdf": hand}
    return URDFAssemblyAdapter(
        output_root=tmp_path / "output",
        asset_resolver=assets.__getitem__,
        asset_catalog=[
            {"asset": "fixture/arm.urdf", "component_type": "arm"},
            {"asset": "fixture/hand.urdf", "component_type": "hand"},
        ],
    )


def _components() -> list[dict[str, str]]:
    """Return the arm-plus-end-effector fixture specification."""
    return [
        {"component_type": "arm", "asset": "fixture/arm.urdf"},
        {"component_type": "hand", "asset": "fixture/hand.urdf"},
    ]


def test_compose_returns_manifest_without_source_paths(adapter: URDFAssemblyAdapter):
    """Composition returns a valid model and topology manifest."""
    result = adapter.compose(_components(), assembly_name="fixture_robot")

    assert result["status"] == "succeeded"
    assert result["validation"]["valid"] is True
    assert result["validation"]["link_count"] == EXPECTED_LINK_COUNT
    assert result["validation"]["joint_count"] == EXPECTED_JOINT_COUNT
    assert result["validation"]["root_links"] == ["base_link"]
    assert all("resolved_path" not in component for component in result["components"])
    assert Path(result["model"]["path"]).is_file()


def test_compose_rejects_absolute_asset_paths(adapter: URDFAssemblyAdapter):
    """MCP asset references cannot escape the registered asset resolver."""
    with pytest.raises(ValueError, match="relative registered asset"):
        adapter.compose(
            [{"component_type": "arm", "asset": "/tmp/robot.urdf"}],
        )


def test_compose_rejects_unregistered_asset(adapter: URDFAssemblyAdapter):
    """MCP input cannot select files outside the explicit asset catalog."""
    with pytest.raises(ValueError, match="not registered"):
        adapter.compose([{"component_type": "arm", "asset": "secret.urdf"}])


def test_compose_rejects_duplicate_component_types(adapter: URDFAssemblyAdapter):
    """Component registry keys cannot silently overwrite another component."""
    with pytest.raises(ValueError, match="duplicate component_type"):
        adapter.compose(
            [
                {"component_type": "arm", "asset": "fixture/arm.urdf"},
                {"component_type": "arm", "asset": "fixture/hand.urdf"},
            ]
        )


def test_eviction_removes_old_generated_artifacts(tmp_path: Path):
    """The manifest bound also bounds generated assembly directories."""
    arm = tmp_path / "arm.urdf"
    _write_component(arm, root="arm_base", tip="arm_tip", joint="arm_joint")
    adapter = URDFAssemblyAdapter(
        output_root=tmp_path / "output",
        asset_resolver=lambda _: arm,
        asset_catalog=[{"asset": "fixture/arm.urdf", "component_type": "arm"}],
        max_assemblies=1,
    )

    first = adapter.compose(
        [{"component_type": "arm", "asset": "fixture/arm.urdf"}],
        assembly_name="first",
    )
    first_dir = Path(first["model"]["path"]).parent
    adapter.compose(
        [{"component_type": "arm", "asset": "fixture/arm.urdf"}],
        assembly_name="second",
    )

    assert not first_dir.exists()


def test_compose_failure_removes_partial_artifacts(tmp_path: Path):
    """Malformed source XML does not leave an untracked output directory."""
    malformed = tmp_path / "malformed.urdf"
    malformed.write_text("<robot", encoding="utf-8")
    output_root = tmp_path / "output"
    adapter = URDFAssemblyAdapter(
        output_root=output_root,
        asset_resolver=lambda _: malformed,
        asset_catalog=[{"asset": "fixture/malformed.urdf", "component_type": "arm"}],
    )

    with pytest.raises(ET.ParseError):
        adapter.compose(
            [{"component_type": "arm", "asset": "fixture/malformed.urdf"}],
            assembly_name="broken",
        )

    assert list(output_root.glob("assembly-*/")) == []


def test_verify_uses_injected_simulation_boundary(adapter: URDFAssemblyAdapter):
    """Simulation verification receives the generated model, not components."""
    captured: dict[str, object] = {}

    def verifier(path: Path, backend: str, seed: int | None) -> dict[str, object]:
        captured.update({"path": path, "backend": backend, "seed": seed})
        return {"status": "succeeded", "robot_id": "fixture_robot"}

    adapter._simulation_verifier = verifier
    result = adapter.compose(_components(), assembly_name="fixture_robot")
    verification = adapter.verify_in_simulation(
        str(result["assembly_id"]), backend="newton", seed=7
    )

    assert verification["status"] == "succeeded"
    assert captured["path"] == Path(result["model"]["path"])
    assert captured["backend"] == "newton"
    assert captured["seed"] == 7
    assert (
        adapter.inspect(str(result["assembly_id"]))["simulation_verification"][
            "robot_id"
        ]
        == "fixture_robot"
    )


def test_adapter_registry_rejects_duplicate_domain_names():
    """The project-level registry keeps domain registrations unambiguous."""
    first = URDFAssemblyAdapter()
    second = URDFAssemblyAdapter()
    registry = MCPAdapterRegistry([first])

    with pytest.raises(ValueError, match="already registered"):
        registry.add(second)


def test_adapter_registry_rejects_capability_collisions():
    """Two providers cannot silently shadow one MCP tool name."""

    class ConflictingAdapter:
        name = "conflicting"

        def capabilities(self) -> tuple[str, ...]:
            return ("urdf_compose",)

        def register(self, server, *, register_tool) -> None:
            del server, register_tool

        def close(self) -> None:
            return None

    registry = MCPAdapterRegistry([URDFAssemblyAdapter()])
    with pytest.raises(ValueError, match="collides"):
        registry.add(ConflictingAdapter())


def test_urdf_module_import_does_not_start_simulation_runtime():
    """Importing the pure adapter does not import the simulation package."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import embodichain.mcp.urdf; "
                "assert 'embodichain.lab.sim' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
