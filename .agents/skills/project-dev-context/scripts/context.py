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

"""Validate, route, and inspect change impact for the agent context map."""

from __future__ import annotations

import argparse
import posixpath
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote

import yaml

__all__ = ["affected_topics", "load_map", "main", "route_topics", "validate_map"]

_REQUIRED_TOPIC_FIELDS = {
    "id",
    "title",
    "aliases",
    "keywords",
    "paths",
    "source_of_truth",
    "related_topics",
    "status",
}
_LIST_FIELDS = {"aliases", "keywords", "paths", "source_of_truth", "related_topics"}
_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]*)\)")
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_EXTERNAL_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
_GLOBAL_CONTEXT_PATHS = (
    "agent_context/MAP.yaml",
    "agent_context/conventions",
    ".agents/skills/project-dev-context",
    ".claude/skills/project-dev-context",
    ".github/copilot/project-dev-context.md",
)


def load_map(root: str | Path) -> dict[str, Any]:
    """Load the context registry.

    Args:
        root: Repository root containing ``agent_context/MAP.yaml``.

    Returns:
        Parsed registry mapping, before schema validation.
    """
    map_path = Path(root) / "agent_context/MAP.yaml"
    with map_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{map_path} must contain a mapping")
    return data


def _safe_path(base: Path, value: str, label: str, errors: list[str]) -> Path | None:
    if not value.strip():
        errors.append(f"{label}: empty path is unsafe")
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        errors.append(f"{label}: unsafe relative path {value!r}")
        return None

    base_resolved = base.resolve()
    candidate = base / relative
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        errors.append(f"{label}: path {value!r} escapes {base}")
        return None
    return candidate


def _string_list(
    owner: dict[str, Any],
    field: str,
    label: str,
    errors: list[str],
    *,
    nonempty: bool = False,
) -> list[str]:
    value = owner.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        errors.append(f"{label}.{field} must be a list of strings")
        return []
    if nonempty and not value:
        errors.append(f"{label}.{field} must not be empty")
    return value


def _markdown_targets(markdown_path: Path) -> list[str]:
    targets: list[str] = []
    for match in _LINK_PATTERN.finditer(markdown_path.read_text(encoding="utf-8")):
        target = match.group(1).strip()
        if target.startswith("<") and ">" in target:
            target = target[1 : target.index(">")]
        else:
            target = target.split(maxsplit=1)[0] if target else ""
        target = unquote(target.partition("#")[0].partition("?")[0])
        if target and not target.startswith("#") and not _EXTERNAL_SCHEME.match(target):
            targets.append(target)
    return targets


def _validate_markdown_links(
    repository_root: Path, context_root: Path, errors: list[str]
) -> None:
    repository_resolved = repository_root.resolve()
    for markdown_path in context_root.rglob("*.md"):
        for target in _markdown_targets(markdown_path):
            linked_path = (markdown_path.parent / target).resolve()
            try:
                linked_path.relative_to(repository_resolved)
            except ValueError:
                errors.append(
                    f"{markdown_path.relative_to(context_root)}: broken local Markdown "
                    f"link {target!r} escapes repository"
                )
                continue
            if not linked_path.exists():
                errors.append(
                    f"{markdown_path.relative_to(context_root)}: broken local Markdown "
                    f"link {target!r}"
                )


def _validate_topic_reachability(
    context_root: Path, registered_paths: list[Path], errors: list[str]
) -> None:
    context_resolved = context_root.resolve()
    pending = [path.resolve() for path in registered_paths if path.is_file()]
    reachable: set[Path] = set()
    while pending:
        markdown_path = pending.pop()
        if markdown_path in reachable:
            continue
        reachable.add(markdown_path)
        for target in _markdown_targets(markdown_path):
            linked_path = (markdown_path.parent / target).resolve()
            try:
                linked_path.relative_to(context_resolved)
            except ValueError:
                continue
            if linked_path.is_file() and linked_path.suffix.casefold() == ".md":
                pending.append(linked_path)

    topics_root = context_root / "topics"
    if not topics_root.exists():
        return
    for markdown_path in topics_root.rglob("*.md"):
        if markdown_path.resolve() not in reachable:
            errors.append(
                f"orphan topic Markdown: {markdown_path.relative_to(context_root)}"
            )


def validate_map(root: str | Path, data: dict[str, Any]) -> list[str]:
    """Check registry structure and local references without importing project code.

    Args:
        root: Repository root used to resolve paths.
        data: Parsed context registry.

    Returns:
        Validation errors; an empty list means structural checks passed.
    """
    repository_root = Path(root)
    context_root = repository_root / "agent_context"
    errors: list[str] = []

    if not isinstance(data, dict):
        return ["MAP.yaml must contain a mapping"]
    if type(data.get("version")) is not int or data["version"] != 1:
        errors.append("version must be 1")

    defaults = data.get("defaults")
    if not isinstance(defaults, dict):
        errors.append("defaults must be a mapping")
        defaults = {}
    if "contexts" in defaults:
        errors.append("defaults.contexts is replaced by defaults.write_contexts")
    write_contexts = _string_list(defaults, "write_contexts", "defaults", errors)
    for value in write_contexts:
        path = _safe_path(context_root, value, "defaults.write_contexts", errors)
        if path is not None and not path.is_file():
            errors.append(
                f"defaults.write_contexts: {value!r} does not exist as a file"
            )
        elif path is not None and path.suffix.casefold() != ".md":
            errors.append(f"defaults.write_contexts: {value!r} must be a Markdown file")

    topics = data.get("topics")
    if not isinstance(topics, list):
        errors.append("topics must be a list")
        topics = []

    topic_entries: list[dict[str, Any]] = []
    topic_ids: list[str] = []
    registered_paths: list[Path] = []
    for index, topic in enumerate(topics):
        label = f"topics[{index}]"
        if not isinstance(topic, dict):
            errors.append(f"{label} must be a mapping")
            continue
        topic_entries.append(topic)
        topic_id = topic.get("id")
        if isinstance(topic_id, str) and topic_id:
            label = topic_id
            topic_ids.append(topic_id)
        else:
            errors.append(f"{label}.id must be a non-empty string")

        missing = sorted(_REQUIRED_TOPIC_FIELDS - topic.keys())
        if missing:
            errors.append(f"{label}: missing required fields {missing}")
        for field in ("title", "status"):
            if not isinstance(topic.get(field), str) or not topic.get(field):
                errors.append(f"{label}.{field} must be a non-empty string")
        lists = {
            field: _string_list(
                topic,
                field,
                label,
                errors,
                nonempty=field in {"paths", "source_of_truth"},
            )
            for field in _LIST_FIELDS
        }
        watch_paths: list[str] = []
        if "watch_paths" in topic:
            watch_paths = _string_list(topic, "watch_paths", label, errors)

        for value in lists["paths"]:
            path = _safe_path(context_root, value, f"{label}.paths", errors)
            if path is not None:
                if not path.is_file():
                    errors.append(f"{label}.paths: {value!r} does not exist as a file")
                elif path.suffix.casefold() != ".md":
                    errors.append(f"{label}.paths: {value!r} must be a Markdown file")
                else:
                    registered_paths.append(path)
        for field, values in (
            ("source_of_truth", lists["source_of_truth"]),
            ("watch_paths", watch_paths),
        ):
            for value in values:
                path = _safe_path(repository_root, value, f"{label}.{field}", errors)
                if path is not None and not path.exists():
                    errors.append(f"{label}.{field}: {value!r} does not exist")

        status = topic.get("status")
        if status not in ("active", "deprecated"):
            errors.append(f"{label}: invalid status {status!r}")
        if status == "deprecated" and not isinstance(topic.get("replaced_by"), str):
            errors.append(f"{label}: deprecated topic requires replaced_by")

    seen: set[str] = set()
    for topic_id in topic_ids:
        if topic_id in seen:
            errors.append(f"duplicate topic id: {topic_id}")
        seen.add(topic_id)

    by_id = {
        topic.get("id"): topic
        for topic in topic_entries
        if isinstance(topic.get("id"), str)
    }
    for topic in topic_entries:
        topic_id = topic.get("id", "<missing-id>")
        related_topics = topic.get("related_topics", [])
        if isinstance(related_topics, list):
            for related_id in related_topics:
                if isinstance(related_id, str) and related_id not in by_id:
                    errors.append(f"{topic_id}: unknown related topic {related_id!r}")
        if topic.get("status") == "deprecated":
            replacement = topic.get("replaced_by")
            if not isinstance(replacement, str):
                continue
            replacement_topic = by_id.get(replacement)
            if (
                replacement_topic is not None
                and replacement_topic.get("status") != "active"
            ):
                errors.append(
                    f"{topic_id}: replaced_by must reference an active topic, got {replacement!r}"
                )
            elif replacement_topic is None:
                errors.append(
                    f"{topic_id}: replaced_by references unknown topic {replacement!r}"
                )

    _validate_markdown_links(repository_root, context_root, errors)
    _validate_topic_reachability(context_root, registered_paths, errors)
    return errors


def _phrase_matches(query: str, phrase: str) -> bool:
    normalized_phrase = phrase.strip().casefold()
    if not normalized_phrase:
        return False
    if _CJK_PATTERN.search(normalized_phrase):
        return normalized_phrase in query
    pattern = rf"(?<![a-z0-9_]){re.escape(normalized_phrase)}(?![a-z0-9_])"
    return re.search(pattern, query) is not None


def _id_matches(query: str, topic_id: str) -> bool:
    pattern = rf"(?<![a-z0-9_-]){re.escape(topic_id.casefold())}(?![a-z0-9_-])"
    return re.search(pattern, query) is not None


def route_topics(data: dict[str, Any], query: str) -> list[str]:
    """Route a query by id, alias, then keyword, preserving equal-best matches.

    Alias and keyword candidates are scored by the lengths of all distinct
    matching phrases, longest first. Comparing those tuples prefers a more
    specific phrase, then uses additional qualifying phrases to break a tie.

    Args:
        data: Context registry.
        query: Natural-language request or explicit topic id.

    Returns:
        Best-matching topic ids in registry order, including unresolved ties.
    """
    topics = [topic for topic in data.get("topics", []) if isinstance(topic, dict)]
    normalized_query = query.strip().casefold()
    id_matches = [
        topic["id"]
        for topic in topics
        if isinstance(topic.get("id"), str)
        and _id_matches(normalized_query, topic["id"])
    ]
    if id_matches:
        return id_matches

    for field in ("aliases", "keywords"):
        candidates: list[tuple[str, tuple[int, ...]]] = []
        for topic in topics:
            phrases = topic.get(field, [])
            if not isinstance(phrases, list) or not isinstance(topic.get("id"), str):
                continue
            matched = {
                phrase.strip().casefold()
                for phrase in phrases
                if isinstance(phrase, str) and _phrase_matches(normalized_query, phrase)
            }
            if matched:
                score = tuple(sorted((len(phrase) for phrase in matched), reverse=True))
                candidates.append((topic["id"], score))
        if candidates:
            best_score = max(score for _, score in candidates)
            return [topic_id for topic_id, score in candidates if score == best_score]
    return []


def _normalize_repo_path(value: str) -> str:
    normalized = posixpath.normpath(value.replace("\\", "/"))
    return normalized.removeprefix("./").rstrip("/")


def _path_matches(changed_path: str, mapped_path: str) -> bool:
    changed = _normalize_repo_path(changed_path)
    mapped = _normalize_repo_path(mapped_path)
    return bool(mapped) and (changed == mapped or changed.startswith(f"{mapped}/"))


def affected_topics(data: dict[str, Any], paths: Sequence[str]) -> list[str]:
    """Identify topics to review after source or context changes.

    Args:
        data: Context registry with source and optional watch paths.
        paths: Changed repository-relative paths, including deleted paths.

    Returns:
        Candidate topic ids in registry order; this does not prove stale prose.
    """
    changed_paths = list(paths)
    global_context_change = any(
        _path_matches(changed_path, mapped_path)
        for changed_path in changed_paths
        for mapped_path in _GLOBAL_CONTEXT_PATHS
    )
    affected: list[str] = []
    for topic in data.get("topics", []):
        if not isinstance(topic, dict) or not isinstance(topic.get("id"), str):
            continue
        mapped_paths = [
            value
            for field in ("source_of_truth", "watch_paths")
            for value in topic.get(field, [])
            if isinstance(value, str)
        ]
        context_paths = [
            f"agent_context/{posixpath.dirname(value)}"
            for value in topic.get("paths", [])
            if isinstance(value, str) and posixpath.dirname(value)
        ]
        if (global_context_change and topic.get("status") == "active") or any(
            _path_matches(changed_path, mapped_path)
            for changed_path in changed_paths
            for mapped_path in (*mapped_paths, *context_paths)
        ):
            affected.append(topic["id"])
    return affected


def _git_changed_paths(root: Path, base: str) -> list[str]:
    merge_base = subprocess.run(
        ["git", "-C", str(root), "merge-base", base, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            merge_base,
            "--",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    untracked = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return list(
        dict.fromkeys(
            [
                path
                for output in (diff.stdout, untracked.stdout)
                for path in output.split("\0")
                if path
            ]
        )
    )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="validate MAP.yaml and local Markdown links")

    route_parser = subparsers.add_parser("route", help="route a context query")
    route_parser.add_argument("query", nargs="+", help="query text")

    affected_parser = subparsers.add_parser(
        "affected", help="list topics affected by repository paths"
    )
    affected_parser.add_argument("paths", nargs="*", help="changed repository paths")
    affected_parser.add_argument(
        "--base",
        help="include tracked changes and untracked files relative to a Git ref",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, root: str | Path | None = None) -> int:
    """Run the agent-context maintenance command-line interface.

    Args:
        argv: CLI arguments, or ``None`` to use the process arguments.
        root: Repository override, or ``None`` to use this script's repository.

    Returns:
        Zero for success, one for failed checks/no route, or two for input errors.
    """
    args = _parser().parse_args(argv)
    repository_root = Path(root) if root is not None else _repository_root()
    try:
        data = load_map(repository_root)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"failed to load context map: {error}", file=sys.stderr)
        return 2

    if args.command == "check":
        errors = validate_map(repository_root, data)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print("agent context map: ok")
        return 0

    if args.command == "route":
        topic_ids = route_topics(data, " ".join(args.query))
        if not topic_ids:
            print("no matching topic")
            return 1
        if len(topic_ids) > 1:
            print(f"ambiguous match: {', '.join(topic_ids)}")
        topics_by_id = {topic.get("id"): topic for topic in data.get("topics", [])}
        for topic_id in topic_ids:
            paths = topics_by_id[topic_id].get("paths", [])
            print(f"{topic_id}: {', '.join(paths)}")
        return 0

    if not args.paths and not args.base:
        print("affected requires PATH... and/or --base REF", file=sys.stderr)
        return 2
    changed_paths = list(args.paths)
    if args.base:
        try:
            changed_paths.extend(_git_changed_paths(repository_root, args.base))
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() if error.stderr else str(error)
            print(f"failed to inspect Git changes: {detail}", file=sys.stderr)
            return 2
    topic_ids = affected_topics(data, list(dict.fromkeys(changed_paths)))
    if not topic_ids:
        print("no affected topics")
    else:
        print("\n".join(topic_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
