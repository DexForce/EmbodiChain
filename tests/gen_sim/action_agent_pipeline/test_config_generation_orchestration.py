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

from types import SimpleNamespace

import pytest

from embodichain.gen_sim.action_agent_pipeline.generation import action_agent_config
from embodichain.gen_sim.action_agent_pipeline.generation import bundle_finalization
from embodichain.gen_sim.action_agent_pipeline.generation import scene_transforms


@pytest.mark.parametrize(
    ("route", "spec_builder_name", "bundle_builder_name", "validator_name"),
    [
        (
            "stacking",
            "_build_stacking_spec_from_response",
            "_build_stacking_bundle",
            "_validate_stacking_bundle",
        ),
        (
            "arrangement_line",
            "_build_arrangement_line_spec_from_response",
            "_build_arrangement_line_bundle",
            "_validate_arrangement_bundle",
        ),
        (
            "object_manipulation",
            "_build_object_manipulation_spec_from_response",
            "_build_relative_placement_bundle",
            "_validate_relative_bundle",
        ),
    ],
)
def test_config_generation_dispatches_each_route_once(
    monkeypatch,
    tmp_path,
    route: str,
    spec_builder_name: str,
    bundle_builder_name: str,
    validator_name: str,
) -> None:
    calls: list[tuple[str, object]] = []
    source_path = tmp_path / "gym_config.json"
    source_path.write_text("{}", encoding="utf-8")
    semantic_spec = {"selected_route": route}
    spec = SimpleNamespace(route=route)
    bundle = {"gym_config": {}, "summary": {}}
    expected = object()

    monkeypatch.setattr(
        action_agent_config, "_raise_if_generated_files_exist", lambda *args: None
    )
    monkeypatch.setattr(
        action_agent_config,
        "resolve_robot_profile",
        lambda profile: calls.append(("profile", profile)) or "resolved_franka",
    )
    monkeypatch.setattr(
        action_agent_config, "_resolve_gym_config_path", lambda _: source_path
    )
    monkeypatch.setattr(action_agent_config, "_read_json", lambda _: {})
    monkeypatch.setattr(
        action_agent_config, "_infer_project_name", lambda *args: "task4_2"
    )
    monkeypatch.setattr(
        action_agent_config, "_collect_scene_objects", lambda _: ["scene_object"]
    )
    monkeypatch.setattr(
        action_agent_config,
        "GlbGeometryNormalizer",
        lambda **kwargs: SimpleNamespace(reports=[]),
    )
    monkeypatch.setattr(
        action_agent_config,
        "_interpret_task_with_llm",
        lambda **kwargs: calls.append(("interpret", kwargs))
        or SimpleNamespace(
            task_route=SimpleNamespace(
                route=route,
                reason="",
                to_summary=lambda: {"route": route},
            ),
            spec=semantic_spec,
        ),
    )
    monkeypatch.setattr(
        action_agent_config,
        spec_builder_name,
        lambda **kwargs: calls.append(("spec", kwargs)) or spec,
    )
    monkeypatch.setattr(
        action_agent_config,
        bundle_builder_name,
        lambda **kwargs: calls.append(("bundle", kwargs)) or bundle,
    )
    monkeypatch.setattr(
        action_agent_config,
        validator_name,
        lambda built, built_spec: calls.append(("validate", (built, built_spec))),
    )
    monkeypatch.setattr(
        action_agent_config,
        "_finalize_and_write_bundle",
        lambda built, **kwargs: calls.append(("finalize", (built, kwargs))) or expected,
    )

    result = action_agent_config.generate_action_agent_config_from_project(
        source_path,
        tmp_path / "output",
        task_name="task4_2",
        task_description="将罐头摆成一排",
        robot_profile="franka",
        overwrite=True,
    )

    assert result is expected
    assert [name for name, _ in calls] == [
        "profile",
        "interpret",
        "spec",
        "bundle",
        "validate",
        "finalize",
    ]
    assert calls[0] == ("profile", "franka")
    assert calls[2][1]["response"] is semantic_spec
    assert calls[3][1]["robot_profile"] == "resolved_franka"


def test_config_generation_preserves_unsupported_route_error(
    monkeypatch,
    tmp_path,
) -> None:
    source_path = tmp_path / "gym_config.json"
    source_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        action_agent_config, "_raise_if_generated_files_exist", lambda *args: None
    )
    monkeypatch.setattr(
        action_agent_config, "resolve_robot_profile", lambda profile: profile
    )
    monkeypatch.setattr(
        action_agent_config, "_resolve_gym_config_path", lambda _: source_path
    )
    monkeypatch.setattr(action_agent_config, "_read_json", lambda _: {})
    monkeypatch.setattr(
        action_agent_config, "_infer_project_name", lambda *args: "task"
    )
    monkeypatch.setattr(action_agent_config, "_collect_scene_objects", lambda _: [])
    monkeypatch.setattr(
        action_agent_config,
        "GlbGeometryNormalizer",
        lambda **kwargs: SimpleNamespace(reports=[]),
    )
    monkeypatch.setattr(
        action_agent_config,
        "_interpret_task_with_llm",
        lambda **kwargs: SimpleNamespace(
            task_route=SimpleNamespace(
                route="unsupported",
                reason="not supported",
            ),
            spec={},
        ),
    )

    with pytest.raises(ValueError, match="not supported"):
        action_agent_config.generate_action_agent_config_from_project(
            source_path,
            tmp_path / "output",
            task_description="unsupported task",
        )


def test_config_generation_preserves_unknown_route_error(
    monkeypatch,
    tmp_path,
) -> None:
    source_path = tmp_path / "gym_config.json"
    source_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        action_agent_config, "_raise_if_generated_files_exist", lambda *args: None
    )
    monkeypatch.setattr(
        action_agent_config, "resolve_robot_profile", lambda profile: profile
    )
    monkeypatch.setattr(
        action_agent_config, "_resolve_gym_config_path", lambda _: source_path
    )
    monkeypatch.setattr(action_agent_config, "_read_json", lambda _: {})
    monkeypatch.setattr(
        action_agent_config, "_infer_project_name", lambda *args: "task"
    )
    monkeypatch.setattr(action_agent_config, "_collect_scene_objects", lambda _: [])
    monkeypatch.setattr(
        action_agent_config,
        "GlbGeometryNormalizer",
        lambda **kwargs: SimpleNamespace(reports=[]),
    )
    monkeypatch.setattr(
        action_agent_config,
        "_interpret_task_with_llm",
        lambda **kwargs: SimpleNamespace(
            task_route=SimpleNamespace(route="future_route", reason=""),
            spec={},
        ),
    )

    with pytest.raises(ValueError, match="future_route"):
        action_agent_config.generate_action_agent_config_from_project(
            source_path,
            tmp_path / "output",
            task_description="future task",
        )


def test_shared_source_scene_transform_order_is_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        scene_transforms,
        "_source_objects_by_runtime_uid",
        lambda *args, **kwargs: {},
    )
    for name in (
        "_maybe_apply_source_scene_xy_scale",
        "_maybe_preserve_source_scene_vertical_contacts",
        "_maybe_apply_tabletop_z_placement",
        "_apply_scene_z_rotation",
    ):
        monkeypatch.setattr(
            scene_transforms,
            name,
            lambda *args, _name=name, **kwargs: calls.append(_name),
        )

    scene_transforms._apply_source_scene_transforms(
        {},
        runtime_uids={},
        by_uid={},
        table_top_z=0.0,
        preserve_source_scene_geometry=False,
        source_scene_body_scale_mode=None,
        source_scene_z_rotation_degrees=0.0,
        robot_profile=SimpleNamespace(),
    )

    assert calls == [
        "_maybe_apply_source_scene_xy_scale",
        "_maybe_preserve_source_scene_vertical_contacts",
        "_maybe_apply_tabletop_z_placement",
        "_apply_scene_z_rotation",
    ]


def test_bundle_finalization_order_and_summary_are_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: list[str] = []
    bundle = {"gym_config": {}, "summary": {"route": "relative"}}
    expected = object()
    monkeypatch.setattr(
        bundle_finalization,
        "_validate_acd_method",
        lambda method: calls.append("validate_acd") or method,
    )
    monkeypatch.setattr(
        bundle_finalization,
        "_apply_acd_method",
        lambda *args, **kwargs: calls.append("apply_acd"),
    )
    monkeypatch.setattr(
        bundle_finalization,
        "_attach_mesh_normalization_summary",
        lambda *args, **kwargs: calls.append("normalize_summary"),
    )
    monkeypatch.setattr(
        bundle_finalization,
        "_attach_body_scale_bake_summary",
        lambda *args, **kwargs: calls.append("bake_summary"),
    )
    monkeypatch.setattr(
        bundle_finalization,
        "_write_config_bundle",
        lambda **kwargs: calls.append("write") or expected,
    )

    result = bundle_finalization._finalize_and_write_bundle(
        bundle,
        output_dir=tmp_path,
        mesh_normalizer=SimpleNamespace(reports=[]),
        acd_method="vhacd",
        overwrite=True,
    )

    assert result is expected
    assert calls == [
        "validate_acd",
        "apply_acd",
        "normalize_summary",
        "bake_summary",
        "write",
    ]
    assert bundle["summary"] == {
        "route": "relative",
        "mesh_loading_mode": "baked_glb",
        "acd_method": "vhacd",
    }
