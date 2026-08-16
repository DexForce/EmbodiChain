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

"""Legacy deterministic L1-L3 natural-language adapter.

This module intentionally contains the finite keyword and alias vocabulary used
by ``--instruction-parser deterministic``.  The default LLM path must not
import it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from .assembly import (
    GroundedTaskBuilder,
    GroundedTaskSpec,
    SceneEntity,
    SceneInventory,
)

__all__ = ["GroundedTaskSpec", "plan_grounded_task_spec"]


_Entity = SceneEntity


_COLORS = {
    "black": ("black", "黑色", "黑"),
    "blue": ("blue", "蓝色", "蓝"),
    "green": ("green", "绿色", "绿"),
    "orange": ("orange", "橙色", "橙", "橘色", "橘"),
    "purple": ("purple", "紫色", "紫"),
    "red": ("red", "红色", "红"),
    "white": ("white", "白色", "白"),
    "yellow": ("yellow", "黄色", "黄"),
}

_CATEGORIES = {
    "button": ("button", "按钮", "按键"),
    "drawer": ("drawer", "抽屉"),
    "knob": ("knob", "旋钮"),
    "tray": ("tray", "托盘", "盘子", "盘"),
    "basket": ("basket", "篮子", "筐"),
    "bowl": ("bowl", "碗", "脸盆", "盆"),
    "bucket": ("bucket", "桶", "爆米花桶"),
    "cup": ("cup", "杯子", "纸杯", "杯"),
    "bottle": ("bottle", "瓶子", "瓶"),
    "can": ("soda can", "beverage can", "can", "易拉罐", "罐头", "罐子"),
    "notebook": ("notebook", "笔记本"),
    "earbuds": ("earbuds", "earphone", "耳机", "耳机盒"),
    "apple": ("apple", "苹果"),
    "table": ("table", "桌子", "桌面", "工作台"),
    "pourable_container": ("pourable_container", "pourable container", "容器"),
}


class _LegacySceneResolver:
    """Finite-vocabulary selector layered over the language-neutral inventory."""

    def __init__(
        self,
        scene_objects: Sequence[Mapping[str, Any]],
        *,
        robot_profile: str,
    ) -> None:
        self.inventory = SceneInventory(scene_objects, robot_profile=robot_profile)
        self.profile = self.inventory.profile
        self.entities = self.inventory.entities
        self.by_uid = self.inventory.by_uid
        self.support = self.inventory.support
        self.movable = self.inventory.movable
        self._metadata = {
            entity.uid: _legacy_metadata(raw, entity)
            for raw in scene_objects
            if (
                entity := self.by_uid.get(
                    str(raw.get("runtime_uid", raw.get("uid", ""))).strip()
                )
            )
            is not None
        }

    def resolve_one(
        self,
        query: str,
        *,
        exclude: Sequence[str] = (),
        context: str,
        apply_side: bool = True,
    ) -> _Entity:
        candidates = self.resolve_many(
            query,
            exclude=exclude,
            context=context,
            apply_side=apply_side,
        )
        if len(candidates) != 1:
            raise ValueError(
                f"{context} is ambiguous; matched scene UIDs "
                f"{[item.uid for item in candidates]}."
            )
        return candidates[0]

    def resolve_many(
        self,
        query: str,
        *,
        exclude: Sequence[str] = (),
        context: str,
        apply_side: bool = True,
    ) -> list[_Entity]:
        lowered = query.lower()
        excluded = set(exclude)
        category = _mentioned_category(lowered)
        color = _mentioned_color(lowered)
        pool = list(self.entities if category == "table" else self.movable)
        pool = [item for item in pool if item.uid not in excluded]

        # Runtime UIDs are authoritative.  Alias matching is retained only for
        # the deterministic natural-language adapter and is token-boundary
        # aware so ``can_1`` cannot accidentally select ``can_10``.
        explicit = [
            item
            for item in pool
            if _contains_uid_token(lowered, item.uid)
            or any(
                _contains_uid_token(lowered, alias) for alias in _uid_aliases(item.uid)
            )
        ]
        if category is not None:
            pool = [item for item in pool if self.match_category(item) == category]
        if color is not None:
            pool = [item for item in pool if self.color(item) == color]
        if not pool:
            available = [
                {
                    "uid": item.uid,
                    "category": item.category,
                    "color": self.color(item),
                }
                for item in self.movable
                if item.uid not in excluded
            ]
            raise ValueError(
                f"{context} did not match a scene object for query {query!r}; "
                f"available candidates are {available}."
            )

        # ``left/right`` denotes a robot-relative half-space and must remain
        # conjunctive.  Do not silently choose one of several candidates in
        # that half-space; ``resolve_one`` will report the ambiguity.  Only an
        # explicit ordinal such as ``leftmost/rightmost`` is allowed to reduce
        # a set to one extreme, and ties are rejected rather than guessed.
        spatial_kind = "none"
        if apply_side:
            spatial_text = re.sub(
                r"(?:left|right)\s+(?:arm|hand)(?!\s*side)|(?:左|右)(?:臂|手)(?!边|侧)|\bupright\b",
                "",
                lowered,
                flags=re.I,
            )
            if _contains_any(spatial_text, ("最左", "最左边", "leftmost")):
                spatial_kind = "leftmost"
                scores = [self.left_score(item) for item in pool]
                extreme = max(scores)
                pool = [item for item in pool if self.left_score(item) == extreme]
                if len(pool) != 1:
                    raise ValueError(f"{context} has an ambiguous leftmost selector.")
            elif _contains_any(spatial_text, ("最右", "最右边", "rightmost")):
                spatial_kind = "rightmost"
                scores = [self.left_score(item) for item in pool]
                extreme = min(scores)
                pool = [item for item in pool if self.left_score(item) == extreme]
                if len(pool) != 1:
                    raise ValueError(f"{context} has an ambiguous rightmost selector.")
            elif _contains_any(spatial_text, ("左侧", "左边", "左手边")) or re.search(
                r"\bleft\b", spatial_text, flags=re.I
            ):
                spatial_kind = "left"
                pool = [item for item in pool if self.left_score(item) > 0.0]
            elif _contains_any(spatial_text, ("右侧", "右边", "右手边")) or re.search(
                r"\bright\b", spatial_text, flags=re.I
            ):
                spatial_kind = "right"
                pool = [item for item in pool if self.left_score(item) < 0.0]
        if explicit:
            explicit_uids = {item.uid for item in explicit}
            pool = [item for item in pool if item.uid in explicit_uids]
            if not pool and spatial_kind != "none":
                raise ValueError(
                    f"{context} explicit UID conflicts with robot-relative "
                    f"{spatial_kind} selector."
                )
        return sorted(pool, key=lambda item: item.uid)

    def side_pair(self, query: str, *, context: str) -> list[_Entity]:
        candidates = self.resolve_many(query, context=context)
        if len(candidates) < 2:
            raise ValueError(f"{context} requires objects on both sides.")
        return [
            max(candidates, key=self.left_score),
            min(candidates, key=self.left_score),
        ]

    def left_score(self, entity: _Entity) -> float:
        return self.inventory.left_score(entity)

    def match_category(self, entity: _Entity) -> str:
        return str(self._metadata[entity.uid]["match_category"])

    def color(self, entity: _Entity) -> str | None:
        value = self._metadata[entity.uid]["color"]
        return str(value) if value is not None else None


class _TaskBuilder(GroundedTaskBuilder):
    """Compatibility shim exposing the resolver under the historic name."""

    def __init__(
        self, task_id: str, instruction: str, index: _LegacySceneResolver
    ) -> None:
        super().__init__(
            task_id,
            instruction,
            index.inventory,
            planner="deterministic_explicit_v2",
        )
        self.index = index


def plan_grounded_task_spec(
    task_name: str,
    task_description: str,
    scene_objects: Sequence[Mapping[str, Any]],
    *,
    robot_profile: str,
) -> GroundedTaskSpec:
    """Plan an explicit L1-L3 instruction without allowing UID guesses."""
    task_id = str(task_name).strip()
    instruction = str(task_description).strip()
    if not task_id or not instruction:
        raise ValueError("task_name and task_description must be non-empty.")
    index = _LegacySceneResolver(scene_objects, robot_profile=robot_profile)
    builder = _TaskBuilder(task_id, instruction, index)
    lowered = instruction.lower()

    if _contains_any(
        lowered, ("摆成一排", "排成一排", "排成一行", "arrange in a line")
    ):
        _plan_line(builder, instruction)
        return builder.build()

    clauses = _split_clauses(instruction)
    for clause in clauses:
        _plan_clause(builder, clause)
    if not builder.instances:
        raise ValueError(
            "Deterministic L1-L3 planner found no supported E1-E9 task clause."
        )
    return builder.build()


def _plan_line(builder: _TaskBuilder, instruction: str) -> None:
    # A support phrase such as ``桌面上的东西`` constrains where the movable
    # objects come from; it must not turn the table itself into the selector.
    object_query = re.sub(
        r"桌(?:面|子|面上|子上)?上?的?|(?:objects?\s+)?on\s+the\s+table",
        "",
        instruction,
        flags=re.I,
    )
    objects = builder.index.resolve_many(object_query, context="line object selector")
    if len(objects) < 2:
        raise ValueError("E1 line arrangement requires at least two matching objects.")
    parent = "line_layout"
    instance_ids = []
    for slot, entity in enumerate(objects):
        instance_ids.append(
            builder.add(
                "E1",
                entity,
                params={
                    "target_role": "table",
                    "relation": "on",
                    "layout": "line",
                    "objects_roles": [],
                    "axis": "world_y",
                    "order_by": "explicit",
                    "order_direction": "given",
                    "order_constraint": "free",
                    "orientation_goal": "none",
                    "orientation_axis": "none",
                    "nominal_slot_index": slot,
                    "slot_constraint": "free_reassignable",
                    "parent_task_instance_id": parent,
                },
                depends_on=[],
            )
        )
    role_by_uid = {uid: role for role, uid in builder.role_bindings().items()}
    roles = [role_by_uid[entity.uid] for entity in objects]
    for instance_id in instance_ids:
        instance = next(item for item in builder.instances if item["id"] == instance_id)
        instance["params"]["objects_roles"] = roles


def _plan_clause(builder: _TaskBuilder, clause: str) -> None:
    lowered = clause.lower().strip(" ，,。")
    if not lowered:
        return
    if _is_handover_retreat_clause(lowered):
        _plan_handover_retreat(builder, clause)
        return
    if _contains_any(
        lowered,
        ("交接", "交给", "递给", "递交", "handover", "hand over", "transfer"),
    ):
        _plan_handover(builder, clause)
        return
    if _contains_any(lowered, ("扶正", "立起来", "stand upright", "upright")):
        _plan_orient(builder, clause)
        return
    if _contains_any(lowered, ("倒入", "倾倒", "pour")):
        _plan_binary(builder, clause, "E3")
        return
    if _contains_any(lowered, ("双臂", "两只手", "both arms")) and _contains_any(
        lowered,
        (
            "拿起",
            "抓起",
            "端起",
            "抬起",
            "搬",
            "移动",
            "移到",
            "挪动",
            "放下",
            "放到",
            "pick",
            "lift",
            "move",
            "transport",
            "place",
        ),
    ):
        _plan_coordinated_pickment(builder, clause)
        return
    if _contains_any(lowered, ("打开", "拉开", "open", "pull")) and _contains_any(
        lowered, ("抽屉", "drawer", "托盘", "tray")
    ):
        entity = builder.index.resolve_one(clause, context="E6 object selector")
        builder.add("E6", entity, params={"target_state": "open"})
        return
    if _contains_any(lowered, ("关闭", "推闭", "close", "push")):
        entity = builder.index.resolve_one(clause, context="E7 object selector")
        builder.add("E7", entity, params={"target_state": "closed"})
        return
    if _contains_any(lowered, ("旋钮", "knob")) and _contains_any(
        lowered, ("旋转", "转到", "turn", "rotate")
    ):
        entity = builder.index.resolve_one(clause, context="E8 object selector")
        builder.add("E8", entity, params={"target_setting": _integer(lowered, 1)})
        return
    if _contains_any(lowered, ("按下", "按压", "press")):
        entity = builder.index.resolve_one(clause, context="E9 object selector")
        builder.add("E9", entity, params={"terminal_state": "activated"})
        return
    if _contains_any(
        lowered,
        ("放到", "放在", "放入", "移到", "摆到", "置于", "叠放到", "place", "put"),
    ):
        _plan_binary(builder, clause, "E1")
        return
    # Some natural instructions omit only the preposition (for example,
    # ``then put it left of the orange can``).  Complete that omission only
    # when a previous source exists and one symbolic relation/target is
    # recoverable; otherwise fail instead of guessing.
    if builder.previous_object_uid is not None and _contains_any(
        lowered,
        (
            "左边",
            "左侧",
            "右边",
            "右侧",
            "前面",
            "前方",
            "后面",
            "后方",
            "left of",
            "right of",
            "front of",
            "behind",
        ),
    ):
        _plan_implicit_binary(builder, clause)
        return
    raise ValueError(f"Unsupported explicit task clause {clause!r}.")


def _plan_handover(builder: _TaskBuilder, clause: str) -> None:
    delimiter = re.search(
        r"交接|交给|递给|递交|handover|hand\s+over|transfer",
        clause,
        flags=re.I,
    )
    if delimiter is None:
        raise ValueError("E4 requires a handover predicate.")
    before = clause[: delimiter.start()]
    after = clause[delimiter.end() :]
    # English commonly puts the source after the verb and spells out both
    # arms in one ``from ... to ...`` phrase.  Keep the object selector and
    # arm mentions separate so ``left side`` never becomes an arm reference.
    ordered_arms = _arm_mentions(clause)
    if not before.strip() and re.search(
        r"\b(?:transfer|handover|hand\s+over)\b", clause, re.I
    ):
        body = after.strip()
        split = re.search(r"\bfrom\b|\bto\b", body, flags=re.I)
        if split is not None:
            before = body[: split.start()].strip()
            after = body[split.end() :]
        else:
            before = body
            after = body
    if _has_object_selector(before):
        entity = builder.index.resolve_one(before, context="E4 object selector")
    elif builder.previous_object_uid is not None:
        entity = builder.index.by_uid[builder.previous_object_uid]
    else:
        raise ValueError("E4 requires an explicit source object.")
    before_arm = _required_arm(before)
    after_arm = _required_arm(after)
    # For ``from left arm to right arm`` use the ordered pair.  For the
    # Chinese ``right arm ... 递给 left arm`` form, the prefix/suffix split is
    # authoritative.  An omitted source arm is completed only from the prior
    # holder or the opposite of an explicit receiver.
    if len(ordered_arms) >= 2:
        mentioned_transfer, explicit_receive = ordered_arms[0], ordered_arms[1]
    else:
        mentioned_transfer, explicit_receive = before_arm, after_arm
    if mentioned_transfer == "right_arm":
        transfer = "right_arm"
    elif mentioned_transfer == "left_arm":
        transfer = "left_arm"
    else:
        transfer = builder.previous_arm or (
            "right_arm" if explicit_receive == "left_arm" else "left_arm"
        )
    receive = explicit_receive or (
        "right_arm" if transfer == "left_arm" else "left_arm"
    )
    if transfer == receive:
        raise ValueError("E4 requires distinct transfer and receive arms.")
    builder.add(
        "E4",
        entity,
        params={
            "transfer_arm": transfer,
            "receive_arm": receive,
            "orientation_goal": (
                "upright"
                if _contains_any(clause.lower(), ("竖直", "直立", "upright"))
                else "none"
            ),
        },
    )


def _is_handover_retreat_clause(text: str) -> bool:
    """Return whether a clause asks an arm to withdraw without a new object goal."""
    return _contains_any(
        text,
        (
            "撤回",
            "撤退",
            "退回",
            "回到初始位置",
            "回到初始姿态",
            "retract",
            "retreat",
            "return to initial",
        ),
    )


def _plan_handover_retreat(builder: _TaskBuilder, clause: str) -> None:
    """Consume an explicit transfer-arm withdrawal as E4 recipe cleanup."""
    if not builder.instances or builder.instances[-1]["task_type"] != "E4":
        raise ValueError(
            "An explicit arm retreat is supported only immediately after an E4 handover."
        )
    handover = builder.instances[-1]
    transfer_arm = str(handover["params"].get("transfer_arm", ""))
    requested_arm = _required_arm(clause)
    if requested_arm is not None and requested_arm != transfer_arm:
        raise ValueError(
            f"Explicit retreat requests {requested_arm!r}, but the preceding "
            f"handover transfer arm is {transfer_arm!r}."
        )


def _arm_mentions(text: str) -> list[str]:
    """Return distinct arm mentions in textual order."""
    matches = []
    pattern = re.compile(
        r"左臂|左手(?!边|侧)|右臂|右手(?!边|侧)|"
        r"\bleft\s+(?:arm|hand)(?!\s*side)|"
        r"\bright\s+(?:arm|hand)(?!\s*side)",
        flags=re.I,
    )
    for match in pattern.finditer(text):
        value = match.group(0).lower()
        matches.append("left_arm" if value.startswith(("左", "left")) else "right_arm")
    return matches


def _plan_implicit_binary(builder: _TaskBuilder, clause: str) -> None:
    """Ground an E1 clause whose ``放到/put`` preposition was omitted."""
    source = (
        builder.index.by_uid[builder.previous_object_uid]
        if builder.previous_object_uid is not None
        else None
    )
    if source is None:
        raise ValueError("E1 omitted placement predicate but has no source object.")
    target = builder.index.resolve_one(
        _target_selector_query(clause),
        exclude=(source.uid,),
        context="E1 implicit target selector",
    )
    relation = _relation(clause, "E1")
    if relation == "none":
        raise ValueError(
            "E1 omitted placement predicate but no unambiguous relation was found."
        )
    builder.add(
        "E1",
        source,
        target=target,
        params={
            "relation": relation,
            "relation_frame": "robot",
            "orientation_goal": "none",
            "orientation_axis": "none",
        },
    )


def _plan_orient(builder: _TaskBuilder, clause: str) -> None:
    lowered = clause.lower()
    category_query = clause
    requested_count = _quantity(lowered)
    if _contains_any(lowered, ("两边", "两侧", "both sides")):
        entities = builder.index.side_pair(category_query, context="E2 side selector")
    elif requested_count is not None or _contains_any(lowered, ("所有", "全部", "all")):
        entities = builder.index.resolve_many(category_query, context="E2 set selector")
        if requested_count is not None and len(entities) != requested_count:
            raise ValueError(
                f"E2 requested {requested_count} objects but matched {len(entities)}."
            )
    else:
        entities = [
            builder.index.resolve_one(category_query, context="E2 object selector")
        ]
    for entity in entities:
        builder.add(
            "E2",
            entity,
            params={
                "orientation_goal": "upright",
                "support_role": "table",
                "upright_local_axis": "long_axis",
                **(
                    {"required_arm": arm}
                    if (arm := _required_arm(clause)) is not None
                    else {}
                ),
            },
            depends_on=[],
        )


def _plan_coordinated_pickment(builder: _TaskBuilder, clause: str) -> None:
    """Ground one explicit dual-arm pick/move request without coordinates."""
    delimiter = re.search(
        r"移动到|移到|搬到|放到|move\s+to|transport\s+to|place",
        clause,
        flags=re.I,
    )
    source_text = clause if delimiter is None else clause[: delimiter.start()]
    entity = builder.index.resolve_one(source_text, context="E5 object selector")
    terminal_behavior = (
        "place"
        if _contains_any(clause.lower(), ("放下", "放到", "place", "release"))
        else "hold"
    )
    params: dict[str, Any] = {
        "direction": "none" if terminal_behavior == "place" else "up",
        "terminal_behavior": terminal_behavior,
        "relation": "none",
        "relation_frame": "robot",
    }
    target = None
    if delimiter is not None:
        target_text = clause[delimiter.end() :]
        lowered_target = target_text.lower()
        relation = next(
            (
                relation_name
                for markers, relation_name in (
                    (("的后面", "的后方"), "behind"),
                    (("的前面", "的前方"), "front_of"),
                    (("的左边", "的左侧"), "left_of"),
                    (("的右边", "的右侧"), "right_of"),
                )
                if _contains_any(lowered_target, markers)
            ),
            _relation(clause, "E1"),
        )
        relation_tokens = {
            "left_of": r"左(?:边|侧|手边|手侧)|left(?:\s+of|_of)",
            "right_of": r"右(?:边|侧|手边|手侧)|right(?:\s+of|_of)",
            "front_of": r"前(?:面|方)|in\s+front(?:\s+of)?|front(?:\s+of)?",
            "behind": r"后(?:面|方)|behind",
        }
        token = relation_tokens.get(relation)
        if token is not None:
            target_text = re.sub(token, " ", target_text, count=1, flags=re.I)
        target = builder.index.resolve_one(
            target_text,
            exclude=(entity.uid,),
            context="E5 target selector",
        )
        params.update({"direction": "none", "relation": relation})
    else:
        lowered = clause.lower()
        for markers, direction in (
            (("向前", "往前", "forward", "front"), "front"),
            (("向后", "往后", "backward", "back"), "back"),
            (("向左", "往左", "leftward"), "left"),
            (("向右", "往右", "rightward"), "right"),
            (("向上", "抬高", "端起", "抬起", "upward", "lift"), "up"),
            (("向下", "downward"), "down"),
        ):
            if _contains_any(lowered, markers):
                params["direction"] = direction
                break
    builder.add("E5", entity, target=target, params=params)


def _plan_binary(builder: _TaskBuilder, clause: str, task_type: str) -> None:
    pattern = (
        r"倒入|倾倒|pour(?:\s+into)?"
        if task_type == "E3"
        else r"放到|放在|放入|移到|摆到|置于|叠放到|place|put"
    )
    parts = re.split(pattern, clause, maxsplit=1, flags=re.I)
    if len(parts) != 2:
        raise ValueError(f"{task_type} clause has no recognizable target relation.")
    before, after = parts
    if not before.strip() and re.match(r"\s*[A-Za-z]", after):
        before, after = _split_english_imperative_binary(after, task_type)
    requested_count = _quantity(before.lower())
    if _has_object_selector(before):
        sources = builder.index.resolve_many(
            before, context=f"{task_type} object selector"
        )
        if requested_count is not None and len(sources) != requested_count:
            raise ValueError(
                f"{task_type} requested {requested_count} objects but matched {len(sources)}."
            )
        all_requested = _contains_any(before.lower(), ("所有", "全部", "all"))
        if requested_count is None and not all_requested and len(sources) != 1:
            raise ValueError(
                f"{task_type} object selector is ambiguous; matched {[item.uid for item in sources]}."
            )
    elif builder.previous_object_uid is not None:
        sources = [builder.index.by_uid[builder.previous_object_uid]]
    else:
        raise ValueError(f"{task_type} requires an explicit source object.")
    target = builder.index.resolve_one(
        _target_selector_query(after),
        exclude=tuple(item.uid for item in sources),
        context=f"{task_type} target selector",
    )
    relation = _relation(clause, task_type)
    params: dict[str, Any] = {
        "relation": relation,
        "relation_frame": "robot",
        "orientation_goal": "none",
        "orientation_axis": "none",
    }
    required_arm = _required_arm(clause)
    if required_arm is not None:
        params["required_arm"] = required_arm
    for source in sources:
        builder.add(task_type, source, target=target, params=params)


def _split_clauses(instruction: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", instruction.strip())
    parts = re.split(
        r"\s*(?:然后|接着|随后|then|next|after that|，\s*再|,\s*再|，\s*(?=把|将|用)|,\s*(?=把|将|用))\s*",
        normalized,
        flags=re.I,
    )
    return [part.strip(" ，,。") for part in parts if part.strip(" ，,。")]


def _legacy_metadata(raw: Mapping[str, Any], entity: _Entity) -> dict[str, str | None]:
    """Build adapter-local aliases without changing the shared scene entity."""
    raw_category = raw.get("category", raw.get("object_category", ""))
    text = (
        f"{entity.uid} {raw.get('source_uid', '')} {entity.description} "
        f"{raw_category} {entity.name}"
    ).lower()
    inferred_category = _mentioned_category(text)
    match_category = _canonical_category(raw_category) or (
        entity.role
        if inferred_category == "table"
        else inferred_category or entity.role
    )
    raw_color = raw.get("color")
    if raw_color is None and isinstance(raw.get("attributes"), Mapping):
        raw_color = raw["attributes"].get("color")
    color = _canonical_color(raw_color) if raw_color not in (None, "") else None
    color = color or _mentioned_color(text)
    return {"text": text, "match_category": match_category, "color": color}


def _mentioned_color(text: str) -> str | None:
    matches = [
        color for color, aliases in _COLORS.items() if _contains_any(text, aliases)
    ]
    return matches[0] if len(matches) == 1 else None


def _mentioned_category(text: str) -> str | None:
    matches = [
        category
        for category, aliases in _CATEGORIES.items()
        if any(_contains_category_alias(text, alias) for alias in aliases)
    ]
    matches = list(dict.fromkeys(matches))
    non_support = [item for item in matches if item != "table"]
    if len(non_support) == 1:
        return non_support[0]
    return matches[0] if len(matches) == 1 else None


def _has_object_selector(text: str) -> bool:
    lowered = text.lower()
    return (
        _mentioned_category(lowered) is not None
        or _mentioned_color(lowered) is not None
        or _contains_any(
            lowered, ("东西", "物体", "object", "左侧", "右侧", "左边", "右边")
        )
    )


def _relation(clause: str, task_type: str) -> str:
    lowered = clause.lower()
    if task_type == "E3":
        return "above"
    if _contains_any(lowered, ("放入", "里面", "内部", "inside", "into")):
        return "inside"
    suffix = re.split(r"放到|放在|移到|摆到|置于|叠放到|place|put", lowered)[-1]
    if re.search(
        r"右(?:边|侧|手边|手侧)(?!\s*的)|\bright(?:\s+of|_of)\b",
        suffix,
        flags=re.I,
    ):
        return "right_of"
    if re.search(
        r"左(?:边|侧|手边|手侧)(?!\s*的)|\bleft(?:\s+of|_of)\b",
        suffix,
        flags=re.I,
    ):
        return "left_of"
    if _contains_any(suffix, ("前面", "前方", "in front", "front of")):
        return "front_of"
    if _contains_any(suffix, ("后面", "后方", "behind")):
        return "behind"
    return "on"


def _target_selector_query(text: str) -> str:
    """Remove a binary relation before resolving the target's own selector.

    A phrase such as ``left of the orange can`` describes the placement
    relation, not the orange can's robot-relative side.  Stripping that phrase
    lets ``resolve_one`` still enforce an actual target selector such as
    ``the left orange can`` without conflating the two meanings.
    """
    return re.sub(
        r"(?:\b(?:on|to)\s+the\s+)?\b(?:left|right|front)\s+of\b|\bbehind\b|"
        r"左(?:边|侧|手边|手侧)(?!\s*的)|右(?:边|侧|手边|手侧)(?!\s*的)|"
        r"前(?:面|方)|后(?:面|方)",
        " ",
        text,
        flags=re.I,
    )


def _split_english_imperative_binary(text: str, task_type: str) -> tuple[str, str]:
    """Split ``put/pour source relation target`` into two object selectors."""
    relation_pattern = (
        r"\b(?:into|in)\b"
        if task_type == "E3"
        else (
            r"\b(?:to\s+the\s+(?:left|right)\s+of|"
            r"(?:left|right|front)\s+of|behind|on\s+top\s+of|on|onto|"
            r"inside|into|in|above)\b"
        )
    )
    match = re.search(relation_pattern, text, flags=re.I)
    if match is None:
        raise ValueError(
            f"{task_type} English imperative requires source, relation, and target."
        )
    source = text[: match.start()].strip()
    target = text[match.end() :].strip()
    if not source or not target:
        raise ValueError(
            f"{task_type} English imperative requires source, relation, and target."
        )
    return source, target


def _uid_aliases(uid: str) -> tuple[str, ...]:
    values = {uid, uid.removeprefix("interact_")}
    return tuple(value.replace("_", " ") for value in values if len(value) >= 4)


def _integer(text: str, default: int) -> int:
    match = re.search(r"\d+", text)
    return int(match.group()) if match else default


def _quantity(text: str) -> int | None:
    """Return an explicit object count, or ``None`` when no count is stated."""
    for marker in ("所有", "全部", "all"):
        if marker in text:
            return None
    numeric = re.search(
        r"(?<![A-Za-z0-9_])(\d+)\s*(?:个|只|件|枚|瓶|罐|杯|盒|块|"
        r"objects?|items?|cans?|cups?|bottles?|drawers?|buttons?|knobs?|"
        r"trays?|baskets?|bowls?)(?![A-Za-z0-9_])",
        text,
        flags=re.I,
    )
    if numeric:
        return int(numeric.group(1))
    english_numerals = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    english = re.search(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:objects?|items?|cans?|cups?|bottles?|drawers?|buttons?|knobs?|"
        r"trays?|baskets?|bowls?)\b",
        text,
        flags=re.I,
    )
    if english:
        return english_numerals[english.group(1).lower()]
    numerals = {
        "一": 1,
        "两": 2,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    for marker, value in numerals.items():
        if marker in text and any(
            unit in text
            for unit in (
                "个",
                "只",
                "件",
                "枚",
                "瓶",
                "罐",
                "杯",
                "盒",
                "块",
                "objects",
                "items",
                "cans",
                "cups",
                "bottles",
                "物体",
            )
        ):
            return value
    return None


def _required_arm(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"左臂|左手(?!边|侧)|\bleft\s+(?:arm|hand)(?!\s*side)", lowered):
        return "left_arm"
    if re.search(r"右臂|右手(?!边|侧)|\bright\s+(?:arm|hand)(?!\s*side)", lowered):
        return "right_arm"
    return None


def _contains_any(text: str, values: Sequence[str]) -> bool:
    return any(value.lower() in text for value in values)


def _contains_category_alias(text: str, alias: str) -> bool:
    lowered_alias = alias.lower()
    if not lowered_alias.isascii():
        return lowered_alias in text
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(lowered_alias)}(?:s|es)?(?![a-z0-9])",
            text,
        )
    )


def _canonical_color(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none" or text in {"无", "没有"}:
        return None
    lowered = text.lower()
    matches = [
        canonical for alias, canonical in _COLOR_ALIAS_TABLE.items() if alias in lowered
    ]
    if len(set(matches)) != 1:
        return None
    return matches[0]


def _canonical_category(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none" or text in {"无", "没有"}:
        return None
    lowered = text.lower()
    matches = [
        canonical
        for alias, canonical in _CATEGORY_ALIAS_TABLE.items()
        if _contains_category_alias(lowered, alias)
    ]
    if len(set(matches)) == 1:
        return matches[0]
    # Scene categories are an open vocabulary.  Aliases above only support the
    # deterministic language adapter; an unfamiliar exported category remains
    # useful semantic evidence and must not collapse to the structural role.
    return lowered


_COLOR_ALIAS_TABLE = {
    alias.lower(): canonical
    for canonical, aliases in _COLORS.items()
    for alias in aliases
}
_CATEGORY_ALIAS_TABLE = {
    alias.lower(): canonical
    for canonical, aliases in _CATEGORIES.items()
    for alias in aliases
}


def _contains_uid_token(text: str, uid: str) -> bool:
    token = str(uid).strip().lower()
    if not token:
        return False
    if re.fullmatch(r"[a-z0-9_.-]+", token):
        return (
            re.search(rf"(?<![a-z0-9_.-]){re.escape(token)}(?![a-z0-9_.-])", text)
            is not None
        )
    return token in text
