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

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ci.run_test_plan import build_commands
from scripts.ci import run_test_plan, select_tests
from scripts.ci.select_tests import (
    ChangedPath,
    build_plan,
    collect_changed_paths,
    load_manifest,
    path_matches,
)


def _write(root: Path, relative: str, content: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _manifest(*, rules=None, topics=None, full_pr_if=None, always=None):
    return {
        "version": 1,
        "rules": list(rules or []),
        "topics": dict(topics or {}),
        "full_pr_if": list(full_pr_if or []),
        "always": list(always or []),
    }


def test_path_matches_directory_and_glob_patterns() -> None:
    assert path_matches(
        "embodichain/lab/sim/objects/rigid.py", "embodichain/lab/sim/objects/**"
    )
    assert path_matches("tests/utils/test_nms.py", "tests/utils/test_*.py")
    assert not path_matches("tests/utils/test_nms.py", "tests/sim/**")


def test_repository_manifest_selectors_resolve() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(root / ".ci/test-impact.toml")
    configured = list(manifest.get("always", []))
    for entry in (*manifest.get("rules", []), *manifest.get("topics", {}).values()):
        configured.extend(entry.get("tests", []))
        configured.extend(entry.get("contracts", []))

    assert configured
    missing = [
        value for value in configured if not select_tests._expand_selector(root, value)
    ]
    assert not missing, f"unresolved impact selectors: {missing}"


@pytest.mark.parametrize(
    ("source_path", "expected_selector"),
    [
        (
            "embodichain/lab/scripts/preview_joint_control.py",
            "tests/lab/scripts/test_preview_joint_control.py",
        ),
        (
            "scripts/tutorials/atomic_action/pickup.py",
            "tests/sim/atomic_actions/test_core.py",
        ),
        (
            "scripts/tutorials/visualization/viser_scene.py",
            "tests/visualization/test_runtime.py",
        ),
    ],
)
def test_repository_manifest_maps_supported_scripts(
    source_path: str,
    expected_selector: str,
) -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(root / ".ci/test-impact.toml")

    plan = build_plan(
        root,
        [ChangedPath(source_path)],
        manifest,
        map_data={"topics": []},
    )

    assert plan.mode == "partial"
    assert expected_selector in plan.selectors


def test_repository_manifest_maps_ik_tutorial_to_its_behavior_contracts() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(root / ".ci/test-impact.toml")

    plan = build_plan(
        root,
        [ChangedPath("scripts/tutorials/sim/ik_manipulability_selection.py")],
        manifest,
        map_data={"topics": []},
    )

    assert plan.mode == "partial"
    assert {
        "tests/sim/test_ik_manipulability_tutorial.py",
        "tests/sim/motion/solvers/test_ik_manipulability_selection.py",
        "tests/visualization/test_example_tutorial_coverage.py",
    } <= set(plan.selectors)


@pytest.mark.parametrize(
    "source_path",
    [
        "embodichain/lab/scripts/preview_widgets/new_unknown.py",
        "scripts/tutorials/atomic_action/nested/new_unknown.py",
        "scripts/tutorials/visualization/nested/new_unknown.py",
    ],
)
def test_repository_manifest_keeps_nested_unknown_scripts_conservative(
    source_path: str,
) -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(root / ".ci/test-impact.toml")

    plan = build_plan(
        root,
        [ChangedPath(source_path)],
        manifest,
        map_data={"topics": []},
    )

    assert plan.mode == "full-pr"
    assert "no impact rule or topic" in (plan.fallback_reason or "")


def test_data_pipeline_rule_covers_dataset_functor_consumer() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(root / ".ci/test-impact.toml")
    rule = next(rule for rule in manifest["rules"] if rule.get("id") == "data-pipeline")

    assert any(
        path_matches(
            "tests/gym/envs/managers/test_dataset_functors.py",
            pattern,
        )
        for pattern in rule["tests"]
    )
    plan = build_plan(
        root,
        [ChangedPath("embodichain/data_pipeline/depth_video/codec.py")],
        manifest,
        map_data={},
    )
    assert "tests/gym/envs/managers/test_dataset_functors.py" in plan.selectors


def test_manifest_rejects_partial_rule_without_tests(tmp_path: Path) -> None:
    path = tmp_path / "test-impact.toml"
    path.write_text(
        'version = 1\n[[rules]]\npaths = ["embodichain/feature.py"]\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="missing test selectors"):
        load_manifest(path)


def test_collect_changed_paths_preserves_renames(tmp_path: Path, monkeypatch) -> None:
    output = "R100\0old/module.py\0new/module.py\0M\0tests/test_new.py\0"
    monkeypatch.setattr(select_tests, "_git_output", lambda *_args: output)

    changed = collect_changed_paths(tmp_path, "base", "head")

    assert changed == [
        ChangedPath("new/module.py", status="R", old_path="old/module.py"),
        ChangedPath("tests/test_new.py", status="M"),
    ]


def test_rule_and_reverse_imports_select_consumer_tests(tmp_path: Path) -> None:
    _write(tmp_path, "embodichain/utils/nms.py", "def pose_nms():\n    return []\n")
    _write(
        tmp_path,
        "embodichain/toolkits/grasp.py",
        "from importlib import import_module\n"
        "pose_nms = import_module('embodichain.utils.nms').pose_nms\n",
    )
    _write(
        tmp_path,
        "tests/utils/test_nms.py",
        "from embodichain.utils.nms import pose_nms\n\ndef test_nms():\n    pose_nms()\n",
    )
    _write(
        tmp_path,
        "tests/toolkits/test_grasp.py",
        "from embodichain.toolkits.grasp import pose_nms\n\ndef test_grasp():\n    pose_nms()\n",
    )
    manifest = _manifest(
        rules=[
            {
                "id": "pose-nms",
                "paths": ["embodichain/utils/nms.py"],
                "tests": ["tests/utils/test_nms.py"],
                "resource_hints": ["gpu"],
                "risk": "medium",
                "include_reverse": True,
            }
        ],
        always=["tests/utils/test_nms.py"],
    )

    plan = build_plan(
        tmp_path,
        [ChangedPath("embodichain/utils/nms.py")],
        manifest,
    )

    assert plan.mode == "partial"
    assert "tests/utils/test_nms.py" in plan.selectors
    assert "tests/toolkits/test_grasp.py" in plan.selectors
    assert "gpu" in plan.resource_hints
    assert "gpu" in plan.lanes
    assert "reverse-import" in plan.reasons["tests/toolkits/test_grasp.py"]


def test_global_path_forces_full_pr_plan(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_smoke.py", "def test_smoke(): pass\n")
    manifest = _manifest(full_pr_if=["pyproject.toml"])

    plan = build_plan(tmp_path, [ChangedPath("pyproject.toml")], manifest)

    assert plan.mode == "full-pr"
    assert plan.selectors == ["tests"]
    assert plan.fallback_reason == "global-risk path: pyproject.toml"
    assert set(plan.lanes) == {"docs", "fast", "sim", "distributed", "gpu"}


def test_unmapped_source_falls_back_to_full_pr(tmp_path: Path) -> None:
    _write(tmp_path, "embodichain/new_module.py", "VALUE = 1\n")
    plan = build_plan(
        tmp_path,
        [ChangedPath("embodichain/new_module.py", status="A")],
        _manifest(),
    )

    assert plan.mode == "full-pr"
    assert "no impact rule or topic" in (plan.fallback_reason or "")


def test_mixed_diff_does_not_hide_an_unmapped_source(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_feature.py", "def test_feature(): pass\n")
    manifest = _manifest(
        rules=[
            {
                "paths": ["embodichain/feature.py"],
                "tests": ["tests/test_feature.py"],
            }
        ],
        always=["tests/test_feature.py"],
    )

    plan = build_plan(
        tmp_path,
        [ChangedPath("embodichain/feature.py"), ChangedPath("embodichain/unknown.py")],
        manifest,
    )

    assert plan.mode == "full-pr"
    assert "embodichain/unknown.py" in (plan.fallback_reason or "")


def test_mixed_diff_keeps_topic_fallback_for_unruled_source(tmp_path: Path) -> None:
    _write(tmp_path, "embodichain/leaf.py", "VALUE = 1\n")
    _write(tmp_path, "embodichain/domain.py", "VALUE = 2\n")
    _write(tmp_path, "tests/test_leaf.py", "def test_leaf(): pass\n")
    _write(tmp_path, "tests/test_domain.py", "def test_domain(): pass\n")
    manifest = _manifest(
        rules=[
            {
                "paths": ["embodichain/leaf.py"],
                "tests": ["tests/test_leaf.py"],
            }
        ],
        topics={"domain": {"tests": ["tests/test_domain.py"]}},
    )
    map_data = {
        "topics": [{"id": "domain", "source_of_truth": ["embodichain/domain.py"]}]
    }

    plan = build_plan(
        tmp_path,
        [ChangedPath("embodichain/leaf.py"), ChangedPath("embodichain/domain.py")],
        manifest,
        map_data=map_data,
    )

    assert plan.mode == "partial"
    assert set(plan.selectors) == {"tests/test_leaf.py", "tests/test_domain.py"}


def test_topic_without_tests_does_not_pass_on_smoke_only(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_smoke.py", "def test_smoke(): pass\n")
    map_data = {
        "topics": [{"id": "new-domain", "source_of_truth": ["embodichain/new.py"]}]
    }

    plan = build_plan(
        tmp_path,
        [ChangedPath("embodichain/new.py")],
        _manifest(always=["tests/test_smoke.py"]),
        map_data=map_data,
    )

    assert plan.mode == "full-pr"
    assert "no runnable test mapping" in (plan.fallback_reason or "")


def test_deleted_test_falls_back_to_full_pr(tmp_path: Path) -> None:
    plan = build_plan(
        tmp_path,
        [ChangedPath("tests/test_removed.py", status="D")],
        _manifest(),
    )

    assert plan.mode == "full-pr"
    assert "deleted test path" in (plan.fallback_reason or "")


def test_changed_slow_test_gets_a_narrow_slow_lane(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/test_slow.py",
        "import pytest\n\n@pytest.mark.slow\ndef test_slow(): pass\n",
    )
    plan = build_plan(
        tmp_path,
        [ChangedPath("tests/test_slow.py")],
        _manifest(),
    )

    assert plan.mode == "partial"
    assert plan.slow_lanes == {"slow-fast": ["tests/test_slow.py"]}


def test_changed_slow_file_keeps_cpu_and_gpu_cases(tmp_path: Path) -> None:
    selector = "tests/test_mixed.py"
    _write(
        tmp_path,
        selector,
        "import pytest\n"
        "@pytest.mark.slow\ndef test_cpu(): pass\n"
        "@pytest.mark.slow\n@pytest.mark.gpu\ndef test_gpu(): pass\n",
    )

    plan = build_plan(tmp_path, [ChangedPath(selector)], _manifest())
    commands = dict(build_commands(plan.to_dict()))

    assert plan.mode == "partial"
    assert plan.slow_lanes == {"slow-fast": [selector], "slow-gpu": [selector]}
    assert "slow and not requires_sim and not gpu" in commands["slow-fast"]
    assert "slow and gpu" in commands["slow-gpu"]
    assert "--run-gpu" in commands["slow-gpu"]


def test_non_slow_gpu_marker_does_not_create_slow_gpu_lane(tmp_path: Path) -> None:
    selector = "tests/test_cpu_slow.py"
    _write(
        tmp_path,
        selector,
        "import pytest\n"
        "@pytest.mark.slow\ndef test_cpu(): pass\n"
        "@pytest.mark.gpu\ndef test_gpu_smoke(): pass\n",
    )

    plan = build_plan(tmp_path, [ChangedPath(selector)], _manifest())

    assert plan.slow_lanes == {"slow-fast": [selector]}


def test_parameter_level_slow_marker_gets_a_slow_lane(tmp_path: Path) -> None:
    selector = "tests/test_parameter_slow.py"
    _write(
        tmp_path,
        selector,
        "import pytest\n"
        "@pytest.mark.parametrize('value', [pytest.param(1, marks=pytest.mark.slow)])\n"
        "def test_parameter(value): pass\n",
    )

    plan = build_plan(tmp_path, [ChangedPath(selector)], _manifest())

    assert plan.slow_lanes == {"slow-fast": [selector]}


def test_sim_gpu_slow_case_skips_irrelevant_fast_lane(tmp_path: Path) -> None:
    selector = "tests/sim/test_sim_slow.py"
    _write(
        tmp_path,
        selector,
        "import pytest\n"
        "@pytest.mark.slow\n@pytest.mark.requires_sim\n@pytest.mark.gpu\n"
        "def test_sim_gpu(): pass\n",
    )

    plan = build_plan(tmp_path, [ChangedPath(selector)], _manifest())

    assert plan.slow_lanes == {"slow-gpu": [selector]}


def test_changed_slow_docs_test_stays_in_docs_lane(tmp_path: Path) -> None:
    selector = "tests/docs/test_slow.py"
    _write(
        tmp_path, selector, "import pytest\n@pytest.mark.slow\ndef test_slow(): pass\n"
    )

    plan = build_plan(tmp_path, [ChangedPath(selector)], _manifest())
    commands = dict(build_commands(plan.to_dict(), lanes=["docs"]))

    assert plan.mode == "docs-only"
    assert plan.slow_lanes == {"slow-docs": [selector]}
    assert "--confcutdir=tests/docs" in commands["slow-docs"]
    assert commands["slow-docs"][-2:] == ["-m", "slow"]


def test_slow_marker_in_fixture_text_does_not_force_full(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/test_selector.py",
        'MARKER_TEXT = "pytest.mark.slow"\n\ndef test_selector(): pass\n',
    )

    plan = build_plan(
        tmp_path,
        [ChangedPath("tests/test_selector.py")],
        _manifest(),
    )

    assert plan.mode == "partial"


def test_changed_slow_test_keeps_full_pr_with_global_change(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/test_slow.py",
        "import pytest\n\npytestmark = pytest.mark.slow\n\ndef test_slow(): pass\n",
    )

    plan = build_plan(
        tmp_path,
        [ChangedPath("pyproject.toml"), ChangedPath("tests/test_slow.py")],
        _manifest(full_pr_if=["pyproject.toml"]),
    )

    assert plan.mode == "full-pr"
    assert plan.slow_lanes == {"slow-fast": ["tests/test_slow.py"]}


def test_docs_only_plan_has_no_hardware_lanes(tmp_path: Path) -> None:
    _write(tmp_path, "tests/docs/test_docs.py", "def test_docs(): pass\n")
    plan = build_plan(
        tmp_path,
        [ChangedPath("docs/source/index.rst")],
        _manifest(),
    )

    assert plan.mode == "docs-only"
    assert plan.lanes == {"docs": ["tests/docs"]}
    assert plan.install == {"gensim": False, "curobo": False}


def test_design_document_is_docs_only(tmp_path: Path) -> None:
    _write(tmp_path, "tests/docs/test_docs.py", "def test_docs(): pass\n")

    plan = build_plan(
        tmp_path,
        [ChangedPath("design/ci.md")],
        _manifest(),
    )

    assert plan.mode == "docs-only"


def test_nested_readme_is_docs_only(tmp_path: Path) -> None:
    plan = build_plan(
        tmp_path,
        [ChangedPath("scripts/tutorials/visualization/README.md")],
        _manifest(),
        map_data={"topics": []},
    )

    assert plan.mode == "docs-only"


def test_changed_test_does_not_expand_a_whole_topic(tmp_path: Path) -> None:
    _write(tmp_path, "tests/sim/test_changed.py", "def test_changed(): pass\n")
    _write(tmp_path, "tests/sim/test_contract.py", "def test_contract(): pass\n")
    _write(tmp_path, "tests/sim/test_unrelated.py", "def test_unrelated(): pass\n")
    map_data = {
        "topics": [
            {
                "id": "simulation-system",
                "source_of_truth": ["embodichain/lab/sim/"],
                "watch_paths": ["tests/sim/"],
                "related_topics": [],
            }
        ]
    }
    manifest = _manifest(
        topics={
            "simulation-system": {
                "tests": ["tests/sim/**"],
                "contracts": ["tests/sim/test_contract.py"],
            }
        }
    )

    plan = build_plan(
        tmp_path,
        [ChangedPath("tests/sim/test_changed.py")],
        manifest,
        map_data=map_data,
    )

    assert "tests/sim/test_changed.py" in plan.selectors
    assert "tests/sim/test_contract.py" in plan.selectors
    assert "tests/sim/test_unrelated.py" not in plan.selectors


def test_related_topics_add_contracts_without_full_topic_tests(tmp_path: Path) -> None:
    _write(tmp_path, "embodichain/feature.py", "VALUE = 1\n")
    _write(tmp_path, "tests/test_feature.py", "def test_feature(): pass\n")
    _write(tmp_path, "tests/test_contract.py", "def test_contract(): pass\n")
    _write(tmp_path, "tests/test_unrelated.py", "def test_unrelated(): pass\n")
    map_data = {
        "topics": [
            {
                "id": "feature",
                "source_of_truth": ["embodichain/feature.py"],
                "related_topics": ["consumer"],
            },
            {"id": "consumer", "source_of_truth": [], "related_topics": []},
        ]
    }
    manifest = _manifest(
        topics={
            "feature": {"tests": ["tests/test_feature.py"]},
            "consumer": {
                "tests": ["tests/test_unrelated.py"],
                "contracts": ["tests/test_contract.py"],
            },
        }
    )

    plan = build_plan(
        tmp_path,
        [ChangedPath("embodichain/feature.py")],
        manifest,
        map_data=map_data,
    )

    assert "tests/test_feature.py" in plan.selectors
    assert "tests/test_contract.py" in plan.selectors
    assert "tests/test_unrelated.py" not in plan.selectors


def test_manifest_globs_expand_to_pytest_paths(tmp_path: Path) -> None:
    _write(tmp_path, "embodichain/lab/sim/objects/rigid.py", "VALUE = 1\n")
    _write(tmp_path, "tests/sim/objects/test_rigid.py", "def test_rigid(): pass\n")
    _write(
        tmp_path,
        "tests/sim/objects/test_articulation.py",
        "def test_articulation(): pass\n",
    )
    manifest = _manifest(
        rules=[
            {
                "id": "sim-objects",
                "paths": ["embodichain/lab/sim/objects/**"],
                "tests": ["tests/sim/objects/**"],
            }
        ]
    )

    plan = build_plan(
        tmp_path,
        [ChangedPath("embodichain/lab/sim/objects/rigid.py")],
        manifest,
    )

    assert "tests/sim/objects/test_rigid.py" in plan.selectors
    assert "tests/sim/objects/test_articulation.py" in plan.selectors
    assert all("*" not in selector for selector in plan.selectors)


def test_changed_registration_line_escalates_risk(tmp_path: Path) -> None:
    _write(tmp_path, "embodichain/feature.py", "def feature(): pass\n")
    manifest = _manifest(
        rules=[
            {
                "id": "feature",
                "paths": ["embodichain/feature.py"],
                "tests": [],
                "risk": "low",
            }
        ]
    )
    plan = build_plan(
        tmp_path,
        [ChangedPath("embodichain/feature.py")],
        manifest,
        diff_text="@@ -1 +1,2 @@\n+register_env('new')\n",
    )

    assert plan.mode == "full-pr"
    assert plan.risk == "high"


def test_import_graph_parse_warning_forces_full_pr(tmp_path: Path) -> None:
    _write(tmp_path, "embodichain/feature.py", "VALUE = 1\n")
    _write(tmp_path, "embodichain/broken.py", "def broken(:\n")
    _write(tmp_path, "tests/test_feature.py", "def test_feature(): pass\n")
    manifest = _manifest(
        rules=[
            {
                "id": "feature",
                "paths": ["embodichain/feature.py"],
                "tests": ["tests/test_feature.py"],
                "include_reverse": True,
            }
        ]
    )

    plan = build_plan(
        tmp_path,
        [ChangedPath("embodichain/feature.py")],
        manifest,
    )

    assert plan.mode == "full-pr"
    assert plan.risk == "high"
    assert plan.graph_warnings


def test_reverse_import_of_shared_fixture_forces_full_pr(tmp_path: Path) -> None:
    _write(tmp_path, "embodichain/feature.py", "VALUE = 1\n")
    _write(
        tmp_path,
        "tests/conftest.py",
        "from embodichain.feature import VALUE\n",
    )
    _write(tmp_path, "tests/test_feature.py", "def test_feature(): pass\n")
    map_data = {
        "topics": [
            {
                "id": "feature",
                "source_of_truth": ["embodichain/feature.py"],
                "watch_paths": [],
                "related_topics": [],
            }
        ]
    }
    manifest = _manifest(topics={"feature": {"tests": ["tests/test_feature.py"]}})

    plan = build_plan(
        tmp_path,
        [ChangedPath("embodichain/feature.py")],
        manifest,
        map_data=map_data,
    )

    assert plan.mode == "full-pr"
    assert "shared test fixture" in (plan.fallback_reason or "")


def test_force_full_includes_slow_lanes(tmp_path: Path) -> None:
    _write(tmp_path, "tests/test_smoke.py", "def test_smoke(): pass\n")
    plan = build_plan(
        tmp_path,
        [ChangedPath("embodichain/feature.py")],
        _manifest(),
        force_full=True,
    )

    assert plan.mode == "full"
    assert plan.slow_lanes == {}
    commands = dict(build_commands(plan.to_dict()))
    assert "not slow" not in commands["fast"]
    assert "not slow" not in commands["sim"]
    assert "slow or not slow" in commands["docs"]


def test_partial_commands_keep_resource_boundaries() -> None:
    plan = {
        "version": 1,
        "mode": "partial",
        "lanes": {
            "fast": ["tests/utils/test_nms.py"],
            "gpu": ["tests/utils/test_nms.py"],
        },
    }

    commands = dict(build_commands(plan))

    assert "not slow and not requires_sim and not gpu" in commands["fast"]
    assert "not slow and gpu" in commands["gpu"]
    assert "--run-gpu" in commands["gpu"]
    assert "-n" in commands["fast"]


def test_default_lane_order_runs_gpu_before_simulation() -> None:
    plan = {
        "version": 1,
        "mode": "partial",
        "lanes": {
            "sim": ["tests/sim/test_sample.py"],
            "distributed": ["tests/learning/test_rl_distributed.py"],
            "gpu": ["tests/utils/test_nms.py"],
        },
    }

    commands = build_commands(plan)

    assert [lane for lane, _command in commands] == [
        "distributed",
        "gpu",
        "sim",
    ]


def test_impacted_slow_lanes_run_after_regular_resource_lanes() -> None:
    plan = {
        "version": 1,
        "mode": "full-pr",
        "lanes": {
            "fast": ["tests"],
            "gpu": ["tests"],
            "sim": ["tests"],
        },
        "slow_lanes": {"slow-fast": ["tests/utils/test_nms.py"]},
    }

    commands = build_commands(plan)

    assert [lane for lane, _command in commands] == [
        "fast",
        "gpu",
        "sim",
        "slow-fast",
    ]


def test_partial_commands_run_only_impacted_slow_selectors() -> None:
    plan = {
        "version": 1,
        "mode": "full-pr",
        "lanes": {"fast": ["tests"]},
        "slow_lanes": {"slow-fast": ["tests/utils/test_nms.py"]},
    }

    commands = dict(build_commands(plan))

    assert "not slow" in " ".join(commands["fast"])
    assert "slow and not requires_sim and not gpu" in " ".join(commands["slow-fast"])
    assert "tests/utils/test_nms.py" in commands["slow-fast"]
    assert commands["slow-fast"][-5:] == [
        "--ignore=tests/docs",
        "-n",
        "4",
        "--dist",
        "load",
    ]


def test_slow_lane_can_run_without_a_regular_selector() -> None:
    plan = {
        "version": 1,
        "mode": "partial",
        "lanes": {},
        "slow_lanes": {"slow-gpu": ["tests/sim/test_slow_gpu.py"]},
    }

    commands = dict(build_commands(plan, lanes=["gpu"]))

    assert "slow and gpu" in " ".join(commands["slow-gpu"])
    assert "--run-gpu" in commands["slow-gpu"]


def test_runner_rejects_selectors_outside_tests() -> None:
    plan = {
        "version": 1,
        "mode": "partial",
        "lanes": {"fast": ["embodichain/private.py"]},
    }

    with pytest.raises(RuntimeError, match="outside tests"):
        build_commands(plan)


def test_runner_rejects_unexpanded_globs() -> None:
    plan = {
        "version": 1,
        "mode": "partial",
        "lanes": {"fast": ["tests/utils/test_*.py"]},
    }

    with pytest.raises(RuntimeError, match="unexpanded"):
        build_commands(plan)


def test_runner_distinguishes_empty_optional_lane_from_docs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        run_test_plan.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=5),
    )
    plan = {
        "version": 1,
        "mode": "partial",
        "selectors": ["tests/test_sample.py"],
        "lanes": {"sim": ["tests/test_sample.py"], "docs": ["tests/docs"]},
    }

    assert run_test_plan.run_plan(plan, root=tmp_path, lanes=["sim"]) == 0
    assert run_test_plan.run_plan(plan, root=tmp_path, lanes=["docs"]) == 5

    slow_plan = {
        "version": 1,
        "mode": "partial",
        "selectors": ["tests/test_sample.py"],
        "lanes": {"gpu": ["tests/test_sample.py"]},
        "slow_lanes": {"slow-gpu": ["tests/test_sample.py"]},
    }
    assert run_test_plan.run_plan(slow_plan, root=tmp_path, lanes=["gpu"]) == 0

    slow_docs_plan = {
        "version": 1,
        "mode": "docs-only",
        "selectors": ["tests/docs"],
        "lanes": {},
        "slow_lanes": {"slow-docs": ["tests/docs/test_sample.py"]},
    }
    assert run_test_plan.run_plan(slow_docs_plan, root=tmp_path, lanes=["docs"]) == 5
