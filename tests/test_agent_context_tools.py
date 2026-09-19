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

import importlib.util
import subprocess
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import yaml
import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_HELPER_PATH = (
    _REPOSITORY_ROOT / ".agents/skills/project-dev-context/scripts/context.py"
)


def _load_helper() -> ModuleType:
    assert _HELPER_PATH.is_file(), "the agent-context helper has not been implemented"
    spec = importlib.util.spec_from_file_location("agent_context_helper", _HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _topic(topic_id: str, **updates: object) -> dict[str, object]:
    topic: dict[str, object] = {
        "id": topic_id,
        "title": topic_id.replace("-", " ").title(),
        "aliases": [],
        "keywords": [],
        "paths": [f"topics/{topic_id}/overview.md"],
        "source_of_truth": [f"src/{topic_id}.py"],
        "related_topics": [],
        "status": "active",
    }
    topic.update(updates)
    return topic


def _make_repository(
    tmp_path: Path, topics: list[dict[str, object]]
) -> tuple[Path, dict]:
    context_root = tmp_path / "agent_context"
    (context_root / "conventions").mkdir(parents=True)
    (context_root / "conventions/writing.md").write_text(
        "# Writing\n", encoding="utf-8"
    )

    for topic in topics:
        for relative_path in topic["paths"]:
            path = context_root / str(relative_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {topic['title']}\n", encoding="utf-8")
        for relative_path in topic["source_of_truth"]:
            path = tmp_path / str(relative_path)
            if str(relative_path).endswith("/"):
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# source\n", encoding="utf-8")

    data = {
        "version": 1,
        "defaults": {"write_contexts": ["conventions/writing.md"]},
        "topics": topics,
    }
    (context_root / "MAP.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    return tmp_path, data


def test_load_and_validate_a_minimal_map(tmp_path: Path) -> None:
    helper = _load_helper()
    root, data = _make_repository(tmp_path, [_topic("simulation-system")])

    assert helper.load_map(root) == data
    assert helper.validate_map(root, data) == []


def test_validate_map_reports_schema_relations_and_deprecation_errors(
    tmp_path: Path,
) -> None:
    helper = _load_helper()
    duplicate = _topic("duplicate", status="paused", related_topics=["missing"])
    deprecated = _topic("old-topic", status="deprecated", replaced_by="duplicate")
    root, data = _make_repository(
        tmp_path,
        [
            _topic("duplicate", aliases="not-a-list"),
            duplicate,
            deprecated,
        ],
    )
    data["version"] = True
    data["defaults"] = {"contexts": ["conventions/writing.md"]}

    errors = helper.validate_map(root, data)

    assert any("version" in error for error in errors)
    assert any("defaults.write_contexts" in error for error in errors)
    assert any("defaults.contexts" in error for error in errors)
    assert any("aliases" in error and "list" in error for error in errors)
    assert any("duplicate topic id" in error for error in errors)
    assert any("invalid status" in error for error in errors)
    assert any("unknown related topic" in error for error in errors)
    assert any("active topic" in error and "replaced_by" in error for error in errors)


def test_validate_map_reports_unhashable_status_and_replacement_values(
    tmp_path: Path,
) -> None:
    helper = _load_helper()
    root, data = _make_repository(
        tmp_path,
        [
            _topic("bad-status-dict", status={"unexpected": True}),
            _topic("bad-status-list", status=["unexpected"]),
            _topic("bad-replacement", status="deprecated", replaced_by=[]),
        ],
    )

    errors = helper.validate_map(root, data)

    assert any(
        "bad-status-dict" in error and "invalid status" in error for error in errors
    )
    assert any(
        "bad-status-list" in error and "invalid status" in error for error in errors
    )
    assert any(
        "bad-replacement" in error and "requires replaced_by" in error
        for error in errors
    )


def test_validate_map_rejects_unsafe_missing_and_symlink_paths(tmp_path: Path) -> None:
    helper = _load_helper()
    topic = _topic(
        "unsafe",
        paths=["../outside.md"],
        source_of_truth=["missing.py"],
        watch_paths=["/absolute/path"],
    )
    root, data = _make_repository(tmp_path, [topic])
    (root / "missing.py").unlink()
    outside = tmp_path.parent / "outside-source.py"
    outside.write_text("# outside\n", encoding="utf-8")
    escaping_link = tmp_path / "escaping-link.py"
    escaping_link.symlink_to(outside)
    topic["source_of_truth"].append("escaping-link.py")
    topic["source_of_truth"].append("")

    errors = helper.validate_map(root, data)

    assert any("../outside.md" in error and "unsafe" in error for error in errors)
    assert any("missing.py" in error and "does not exist" in error for error in errors)
    assert any("/absolute/path" in error and "unsafe" in error for error in errors)
    assert any("escaping-link.py" in error and "escapes" in error for error in errors)
    assert any("empty path" in error for error in errors)


def test_validate_map_finds_broken_links_and_unreachable_topic_markdown(
    tmp_path: Path,
) -> None:
    helper = _load_helper()
    root, data = _make_repository(tmp_path, [_topic("linked")])
    overview = root / "agent_context/topics/linked/overview.md"
    overview.write_text(
        "# Linked\n\n[Details](details.md)\n[Broken](missing.md)\n",
        encoding="utf-8",
    )
    (overview.parent / "details.md").write_text("# Details\n", encoding="utf-8")
    (overview.parent / "orphan.md").write_text("# Orphan\n", encoding="utf-8")

    errors = helper.validate_map(root, data)

    assert any(
        "broken local Markdown link" in error and "missing.md" in error
        for error in errors
    )
    assert any(
        "orphan topic Markdown" in error and "orphan.md" in error for error in errors
    )
    assert not any(
        "orphan topic Markdown" in error and "details.md" in error for error in errors
    )


def test_validate_map_allows_repository_links_but_rejects_repository_escapes(
    tmp_path: Path,
) -> None:
    helper = _load_helper()
    root, data = _make_repository(tmp_path, [_topic("repository-links")])
    docs_root = root / "docs"
    docs_root.mkdir()
    (docs_root / "guide.md").write_text("# Guide\n", encoding="utf-8")
    outside = root.parent / "outside-guide.md"
    outside.write_text("# Outside\n", encoding="utf-8")
    (docs_root / "escaping-link.md").symlink_to(outside)
    overview = root / "agent_context/topics/repository-links/overview.md"
    overview.write_text(
        "# Links\n\n"
        "[Guide](../../../docs/guide.md)\n"
        "[Outside](../../../../outside-guide.md)\n"
        "[Symlink](../../../docs/escaping-link.md)\n",
        encoding="utf-8",
    )

    errors = helper.validate_map(root, data)

    assert not any("docs/guide.md" in error for error in errors)
    assert any(
        "outside-guide.md" in error and "escapes repository" in error
        for error in errors
    )
    assert any(
        "escaping-link.md" in error and "escapes repository" in error
        for error in errors
    )


def test_validate_map_requires_markdown_context_files(tmp_path: Path) -> None:
    helper = _load_helper()
    root, data = _make_repository(tmp_path, [_topic("wrong-format")])
    text_path = root / "agent_context/topics/wrong-format/overview.txt"
    text_path.write_text("plain text\n", encoding="utf-8")
    convention_path = root / "agent_context/conventions/writing.txt"
    convention_path.write_text("plain text\n", encoding="utf-8")
    data["topics"][0]["paths"] = ["topics/wrong-format/overview.txt"]
    data["defaults"]["write_contexts"] = ["conventions/writing.txt"]

    errors = helper.validate_map(root, data)

    assert any(
        "defaults.write_contexts" in error and "Markdown" in error for error in errors
    )
    assert any(
        "wrong-format.paths" in error and "Markdown" in error for error in errors
    )


def test_route_topics_applies_precedence_boundaries_and_specificity() -> None:
    helper = _load_helper()
    data = {
        "topics": [
            _topic(
                "sim",
                aliases=["simulation"],
                keywords=["sim", "SimulationManager", "仿真管理器"],
            ),
            _topic(
                "motion-basic",
                aliases=["motion", "sim"],
                keywords=["SceneManifest", "simulation"],
            ),
            _topic(
                "motion-planning",
                aliases=["motion planning"],
                keywords=["SceneManifest"],
            ),
            _topic(
                "sim-visualization",
                aliases=["browser view"],
                keywords=["SceneManifest", "visualization"],
            ),
            _topic(
                "task-programs",
                aliases=["task program"],
                keywords=["SceneManifest", "program.yaml"],
            ),
        ]
    }

    assert helper.route_topics(data, "sim") == ["sim"]
    assert helper.route_topics(data, "参考 motion-planning 上下文") == [
        "motion-planning"
    ]
    assert helper.route_topics(data, "compare sim and task-programs") == [
        "sim",
        "task-programs",
    ]
    assert helper.route_topics(data, "simulation defaults") == ["sim"]
    assert helper.route_topics(data, "simready generation") == []
    assert helper.route_topics(data, "motion planning defaults") == ["motion-planning"]
    assert helper.route_topics(data, "SceneManifest") == [
        "motion-basic",
        "motion-planning",
        "sim-visualization",
        "task-programs",
    ]
    assert helper.route_topics(data, "visualization SceneManifest") == [
        "sim-visualization"
    ]
    assert helper.route_topics(data, "仿真管理器配置在哪里") == ["sim"]
    assert helper.route_topics(data, "查找 program.yaml 的加载入口") == [
        "task-programs"
    ]


def test_affected_topics_uses_complete_path_segments() -> None:
    helper = _load_helper()
    data = {
        "topics": [
            _topic(
                "simulation-system",
                source_of_truth=["embodichain/lab/sim/"],
                watch_paths=["tests/lab/sim", "pyproject.toml"],
            ),
            _topic(
                "task-programs",
                source_of_truth=["embodichain/lab/task_program/schema.py"],
            ),
        ]
    }

    assert helper.affected_topics(
        data,
        [
            "embodichain/lab/sim/cfg.py",
            "tests/lab/sim/test_cfg.py",
            "pyproject.toml",
            "embodichain/lab/task_program/schema.py",
        ],
    ) == ["simulation-system", "task-programs"]
    assert helper.affected_topics(data, ["embodichain/lab/simple.py"]) == []
    assert helper.affected_topics(data, ["tests/lab/simulation/test_cfg.py"]) == []


def test_affected_topics_maps_context_details_and_global_routing_files() -> None:
    helper = _load_helper()
    data = {
        "topics": [
            _topic("simulation-system"),
            _topic("task-programs"),
            _topic("old-topic", status="deprecated", replaced_by="task-programs"),
        ]
    }

    assert helper.affected_topics(
        data, ["agent_context/topics/simulation-system/details.md"]
    ) == ["simulation-system"]
    assert helper.affected_topics(data, ["agent_context/MAP.yaml"]) == [
        "simulation-system",
        "task-programs",
    ]
    assert helper.affected_topics(
        data, [".agents/skills/project-dev-context/references/context-system.md"]
    ) == ["simulation-system", "task-programs"]


def test_cli_reports_route_ambiguity_and_default_paths(tmp_path: Path, capsys) -> None:
    helper = _load_helper()
    first = _topic("first", keywords=["shared"])
    second = _topic("second", keywords=["shared"])
    root, _ = _make_repository(tmp_path, [first, second])

    exit_code = helper.main(["route", "shared"], root=root)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "ambiguous" in output.casefold()
    assert "first" in output and "topics/first/overview.md" in output
    assert "second" in output and "topics/second/overview.md" in output


def test_cli_reports_a_no_match(tmp_path: Path, capsys) -> None:
    helper = _load_helper()
    root, _ = _make_repository(tmp_path, [_topic("known-topic")])

    exit_code = helper.main(["route", "unindexed phrase"], root=root)

    assert exit_code == 1
    assert capsys.readouterr().out.strip() == "no matching topic"


def test_cli_check_returns_nonzero_for_validation_errors(
    tmp_path: Path, capsys
) -> None:
    helper = _load_helper()
    root, data = _make_repository(tmp_path, [_topic("known-topic")])
    data["version"] = 2
    (root / "agent_context/MAP.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )

    exit_code = helper.main(["check"], root=root)

    assert exit_code == 1
    assert "version must be 1" in capsys.readouterr().err


def test_affected_base_includes_deleted_staged_and_untracked_paths(
    tmp_path: Path, capsys
) -> None:
    helper = _load_helper()
    topics = [
        _topic("deleted", source_of_truth=["src/deleted.py"]),
        _topic("staged", source_of_truth=["src/staged.py"]),
        _topic("untracked", watch_paths=["new/"], source_of_truth=["src/stable.py"]),
    ]
    root, _ = _make_repository(tmp_path, topics)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"], cwd=root, check=True
    )
    subprocess.run(["git", "config", "user.name", "Tests"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)

    (root / "src/deleted.py").unlink()
    (root / "src/staged.py").write_text("# changed\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/staged.py"], cwd=root, check=True)
    (root / "new").mkdir()
    (root / "new/untracked.py").write_text("# new\n", encoding="utf-8")

    exit_code = helper.main(["affected", "--base", "HEAD"], root=root)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.splitlines() == ["deleted", "staged", "untracked"]


def test_affected_base_clean_checkout_is_a_success(tmp_path: Path, capsys) -> None:
    helper = _load_helper()
    root, _ = _make_repository(tmp_path, [_topic("unchanged")])
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"], cwd=root, check=True
    )
    subprocess.run(["git", "config", "user.name", "Tests"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)

    exit_code = helper.main(["affected", "--base", "HEAD"], root=root)

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "no affected topics"


def test_affected_base_preserves_non_ascii_git_paths(tmp_path: Path, capsys) -> None:
    helper = _load_helper()
    root, _ = _make_repository(
        tmp_path, [_topic("unicode-source", source_of_truth=["src/中文.py"])]
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"], cwd=root, check=True
    )
    subprocess.run(["git", "config", "user.name", "Tests"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    (root / "src/中文.py").write_text("# changed\n", encoding="utf-8")

    exit_code = helper.main(["affected", "--base", "HEAD"], root=root)

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == ["unicode-source"]


def test_affected_base_diffs_from_merge_base(tmp_path: Path, capsys) -> None:
    helper = _load_helper()
    topics = [
        _topic("branch-change", source_of_truth=["src/branch.py"]),
        _topic("base-only", source_of_truth=["src/base-only.py"]),
    ]
    root, _ = _make_repository(tmp_path, topics)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"], cwd=root, check=True
    )
    subprocess.run(["git", "config", "user.name", "Tests"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "common"], cwd=root, check=True)
    current_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "checkout", "-qb", "comparison"], cwd=root, check=True)
    (root / "src/base-only.py").write_text("# changed on base\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/base-only.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "base only"], cwd=root, check=True)
    subprocess.run(["git", "checkout", "-q", current_branch], cwd=root, check=True)
    (root / "src/branch.py").write_text("# changed on branch\n", encoding="utf-8")

    exit_code = helper.main(["affected", "--base", "comparison"], root=root)

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == ["branch-change"]


def _commit_context_fixture(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"], cwd=root, check=True
    )
    subprocess.run(["git", "config", "user.name", "Tests"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "context fixture"], cwd=root, check=True)


def test_affected_map_change_selects_only_changed_entries() -> None:
    helper = _load_helper()
    previous = {"topics": [_topic("changed"), _topic("unchanged")]}
    current = deepcopy(previous)
    current["topics"][0]["aliases"] = ["new alias"]

    assert helper.affected_topics(
        current, ["agent_context/MAP.yaml"], previous_data=previous
    ) == ["changed"]


def test_affected_map_additions_removals_and_old_source_scopes() -> None:
    helper = _load_helper()
    previous = {
        "topics": [
            _topic("moved", source_of_truth=["src/old.py"]),
            _topic("removed"),
            _topic("unchanged"),
        ]
    }
    current = {
        "topics": [
            _topic("moved", source_of_truth=["src/new.py"]),
            _topic("added"),
            _topic("unchanged"),
        ]
    }

    assert helper.affected_topics(
        current, ["agent_context/MAP.yaml"], previous_data=previous
    ) == ["moved", "added", "removed"]
    assert helper.affected_topics(current, ["src/old.py"], previous_data=previous) == [
        "moved"
    ]


def test_affected_ignores_map_formatting_and_topic_order() -> None:
    helper = _load_helper()
    previous = {"topics": [_topic("first"), _topic("second")]}
    current = {"topics": list(reversed(previous["topics"]))}

    assert (
        helper.affected_topics(
            current, ["agent_context/MAP.yaml"], previous_data=previous
        )
        == []
    )


def test_affected_global_defaults_still_select_all_active_topics() -> None:
    helper = _load_helper()
    previous = {
        "version": 1,
        "defaults": {"write_contexts": []},
        "topics": [
            _topic("first"),
            _topic("second"),
            _topic("old", status="deprecated"),
        ],
    }
    current = deepcopy(previous)
    current["defaults"]["write_contexts"] = ["conventions/new.md"]

    assert helper.affected_topics(
        current, ["agent_context/MAP.yaml"], previous_data=previous
    ) == ["first", "second"]


def test_affected_cli_explains_map_and_previous_source_matches(
    tmp_path: Path, capsys
) -> None:
    helper = _load_helper()
    root, data = _make_repository(
        tmp_path, [_topic("moved", source_of_truth=["src/old.py"]), _topic("unchanged")]
    )
    _commit_context_fixture(root)
    (root / "src/old.py").unlink()
    data["topics"][0]["source_of_truth"] = ["src/new.py"]
    (root / "src/new.py").write_text("# moved\n", encoding="utf-8")
    (root / "agent_context/MAP.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")

    assert helper.main(["affected", "--base", "HEAD", "--explain"], root=root) == 0
    output = capsys.readouterr().out
    assert "moved" in output and "unchanged" not in output
    assert "MAP entry" in output
    assert "src/old.py" in output and "previous source_of_truth" in output
    assert "src/new.py" in output and "source_of_truth" in output


def test_affected_cli_explains_path_only_map_fallback(tmp_path: Path, capsys) -> None:
    helper = _load_helper()
    root, _ = _make_repository(tmp_path, [_topic("first"), _topic("second")])

    assert (
        helper.main(["affected", "agent_context/MAP.yaml", "--explain"], root=root) == 0
    )
    output = capsys.readouterr().out
    assert "first" in output and "second" in output
    assert "no baseline" in output and "--base" in output
    assert output.count("no baseline") == 1


def test_affected_base_ignores_comment_only_map_change(tmp_path: Path, capsys) -> None:
    helper = _load_helper()
    root, _ = _make_repository(tmp_path, [_topic("unchanged")])
    _commit_context_fixture(root)
    path = root / "agent_context/MAP.yaml"
    path.write_text("# comment only\n" + path.read_text(), encoding="utf-8")

    assert helper.main(["affected", "--base", "HEAD"], root=root) == 0
    assert capsys.readouterr().out.strip() == "no affected topics"


def test_affected_base_without_historical_map_is_conservative(
    tmp_path: Path, capsys
) -> None:
    helper = _load_helper()
    (tmp_path / "README.md").write_text("# Initial\n", encoding="utf-8")
    _commit_context_fixture(tmp_path)
    root, _ = _make_repository(tmp_path, [_topic("new")])

    assert helper.main(["affected", "--base", "HEAD", "--explain"], root=root) == 0
    output = capsys.readouterr().out
    assert "new" in output and "no baseline" in output


def test_stats_reports_overview_growth_and_total_detail_reduction(
    tmp_path: Path, capsys
) -> None:
    helper = _load_helper()
    root, _ = _make_repository(tmp_path, [_topic("sample")])
    overview = root / "agent_context/topics/sample/overview.md"
    overview.write_text("one two\n", encoding="utf-8")
    detail = overview.with_name("details.md")
    detail.write_text("three four five six\n", encoding="utf-8")
    _commit_context_fixture(root)
    overview.write_text("one two three\n", encoding="utf-8")
    detail.unlink()

    assert helper.main(["stats", "--base", "HEAD"], root=root) == 0
    output = capsys.readouterr().out
    assert "Whitespace words (not tokens)" in output
    assert "3 (+1)" in output and "topics/sample/overview.md" in output
    assert "Total Markdown words: 5 (-3)" in output


def test_stats_warns_about_large_overview_without_failing(
    tmp_path: Path, capsys
) -> None:
    helper = _load_helper()
    root, _ = _make_repository(tmp_path, [_topic("sample")])
    overview = root / "agent_context/topics/sample/overview.md"
    overview.write_text("word " * 1300, encoding="utf-8")

    assert helper.main(["stats"], root=root) == 0
    output = capsys.readouterr().out
    assert "REVIEW" in output and "topics/sample/overview.md" in output


@pytest.mark.parametrize(
    ("before", "after", "needs_review"),
    [(400, 500, False), (300, 399, False), (300, 400, True), (400, 501, True)],
)
def test_stats_growth_notice_requires_both_absolute_and_relative_growth(
    tmp_path: Path, capsys, before: int, after: int, needs_review: bool
) -> None:
    helper = _load_helper()
    root, _ = _make_repository(tmp_path, [_topic("sample")])
    overview = root / "agent_context/topics/sample/overview.md"
    overview.write_text("word " * before, encoding="utf-8")
    _commit_context_fixture(root)
    overview.write_text("word " * after, encoding="utf-8")

    assert helper.main(["stats", "--base", "HEAD"], root=root) == 0
    assert ("REVIEW" in capsys.readouterr().out) is needs_review


@pytest.mark.parametrize("command", ["affected", "stats"])
def test_baseline_commands_report_invalid_revision(
    tmp_path: Path, capsys, command: str
) -> None:
    helper = _load_helper()
    root, _ = _make_repository(tmp_path, [_topic("sample")])
    _commit_context_fixture(root)

    assert helper.main([command, "--base", "does-not-exist"], root=root) == 2
    assert "failed to inspect Git" in capsys.readouterr().err
