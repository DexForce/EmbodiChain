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

"""Compile a backend-neutral graph of qualified physical predicates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field as dataclass_field, replace
from hashlib import sha256
import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from . import EvaluationStatus

if TYPE_CHECKING:
    import torch

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class PredicateSpec:
    """A registered predicate with logical input roles bound to field IDs."""

    kind: str
    bindings: Mapping[str, str]
    params: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BooleanRule:
    """Combine named rules using ``all``, ``any`` or unary ``not``."""

    op: str
    refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TemporalRule:
    """Apply a bounded control-slot operator to a rule.

    Offsets are relative to each row's start. Both ends are inclusive.
    ``hold_for`` requires consecutive satisfying slots, while ``eventually``
    requires one and ``always`` requires every slot.
    """

    op: str
    ref: str
    end: int
    start: int = 0
    steps: int = 1


@dataclass(frozen=True, slots=True)
class MilestoneRule:
    """Witness all referenced rules in one observation after all predecessors."""

    refs: tuple[str, ...]
    after: tuple[str, ...] = ()


_Rule = PredicateSpec | BooleanRule | TemporalRule | MilestoneRule


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """Named rules and public roots of an acyclic evaluation graph."""

    rules: Mapping[str, _Rule]
    roots: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationSpec:
        """Decode inert declarations with closed keys and registered kind IDs."""
        _closed(value, {"rules", "roots"})
        if not isinstance(value["rules"], Mapping):
            raise ValueError("Evaluation rules must be a mapping.")
        rules = {}
        for name, item in value["rules"].items():
            if not isinstance(item, Mapping) or not _identifier(item.get("kind")):
                raise ValueError(f"Rule {name!r} needs an exact kind.")
            kind = item["kind"]
            if kind in ("all", "any", "not"):
                _closed(item, {"kind", "refs"})
                rules[name] = BooleanRule(kind, _refs(item["refs"]))
            elif kind in ("hold_for", "always", "eventually"):
                _closed(item, {"kind", "ref", "end"}, {"start", "steps"})
                rules[name] = TemporalRule(
                    kind,
                    item["ref"],
                    item["end"],
                    item.get("start", 0),
                    item.get("steps", 1),
                )
            elif kind == "milestone":
                _closed(item, {"kind", "refs"}, {"after"})
                rules[name] = MilestoneRule(
                    _refs(item["refs"]), _refs(item.get("after", ()), empty=True)
                )
            else:
                _closed(item, {"kind", "bindings"}, {"params"})
                if not isinstance(item["bindings"], Mapping) or not isinstance(
                    item.get("params", {}), Mapping
                ):
                    raise ValueError(
                        f"Rule {name!r} needs mapping bindings and params."
                    )
                rules[name] = PredicateSpec(
                    kind, dict(item["bindings"]), dict(item.get("params", {}))
                )
        return cls(rules, _refs(value["roots"]))


@dataclass(frozen=True, slots=True)
class PredicateMeasurement:
    """Return physical truth, evidence availability and batched metric tensors."""

    satisfied: torch.Tensor
    valid: torch.Tensor
    metrics: Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class PredicateDefinition:
    """Trusted predicate implementation and versioned input contract.

    ``normalize`` validates inert parameters and returns JSON-compatible data.
    ``measure`` receives values by role and immutable normalized parameters.
    """

    revision: int
    roles: tuple[str, ...]
    normalize: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    measure: Callable[
        [Mapping[str, torch.Tensor], Mapping[str, Any]], PredicateMeasurement
    ]
    input_kinds: tuple[str, ...] | None = None
    shared_layout: bool = False


@dataclass(frozen=True, slots=True)
class _BoundPredicate:
    definition: PredicateDefinition
    bindings: Mapping[str, str]
    params: Mapping[str, Any]
    kind: str


def _identifier(value: object) -> bool:
    return type(value) is str and bool(value) and value == value.strip()


def _closed(value, required, optional=()):
    if (
        not isinstance(value, Mapping)
        or not required <= value.keys()
        or value.keys() - (required | set(optional))
    ):
        raise ValueError(
            f"Expected evaluation keys {sorted(required)!r} with optional {sorted(optional)!r}."
        )


def _refs(value: object, *, empty: bool = False) -> tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or (not empty and not value)
        or any(not _identifier(ref) for ref in value)
        or len(value) != len(set(value))
    ):
        raise ValueError("Evaluation references must be unique exact identifiers.")
    return tuple(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _check_input_contract(name, definition, bindings, field_kinds, field_layouts):
    if field_kinds is not None:
        for role, kind in zip(definition.roles, definition.input_kinds or ()):
            field = bindings[role]
            actual = field_kinds.get(field)
            if actual != kind:
                raise ValueError(
                    f"Rule {name!r} role {role!r} requires {kind!r}, "
                    f"got {field!r} ({actual!r})."
                )
    if definition.shared_layout and field_layouts is not None:
        layouts = {field_layouts.get(field) for field in bindings.values()}
        if len(layouts) != 1 or None in layouts:
            raise ValueError(f"Rule {name!r} requires inputs from one joint layout.")


class PredicateRegistry:
    """Register trusted predicates and compile a closed, content-addressed graph."""

    def __init__(self) -> None:
        self._definitions: dict[str, PredicateDefinition] = {}

    def register(self, kind: str, definition: PredicateDefinition) -> None:
        """Register one predicate kind. Duplicate registrations are rejected."""
        if not _identifier(kind) or kind in self._definitions:
            raise ValueError(f"Invalid or duplicate predicate kind {kind!r}.")
        if type(definition.revision) is not int or definition.revision < 1:
            raise ValueError("Predicate revision must be a positive integer.")
        _refs(definition.roles)
        if definition.input_kinds is not None and (
            len(definition.input_kinds) != len(definition.roles)
            or any(not _identifier(kind) for kind in definition.input_kinds)
        ):
            raise ValueError("Predicate input kinds must describe each role.")
        if not callable(definition.normalize) or not callable(definition.measure):
            raise TypeError("Predicate registration requires trusted callables.")
        self._definitions[kind] = replace(
            definition,
            roles=tuple(definition.roles),
            input_kinds=(
                None
                if definition.input_kinds is None
                else tuple(definition.input_kinds)
            ),
        )

    def compile(
        self,
        spec: EvaluationSpec,
        *,
        available_fields: set[str] | None = None,
        field_kinds: Mapping[str, str] | None = None,
        field_layouts: Mapping[str, str] | None = None,
    ) -> CompiledEvaluation:
        """Resolve rules, validate bindings and compute their content identity."""
        roots = _refs(spec.roots)
        nodes = {}
        dependencies = {}
        payload = {}
        fields = set()
        predicates = {}
        for name, rule in spec.rules.items():
            if not _identifier(name):
                raise ValueError(f"Invalid evaluation rule ID {name!r}.")
            if isinstance(rule, PredicateSpec):
                definition = self._definitions.get(rule.kind)
                if definition is None:
                    raise ValueError(f"Unknown predicate {rule.kind!r} in {name!r}.")
                if set(rule.bindings) != set(definition.roles) or any(
                    not _identifier(field) for field in rule.bindings.values()
                ):
                    raise ValueError(f"Invalid input bindings for rule {name!r}.")
                bindings = dict(rule.bindings)
                _check_input_contract(
                    name, definition, bindings, field_kinds, field_layouts
                )
                fields.update(bindings.values())
                params = json.loads(
                    json.dumps(dict(definition.normalize(rule.params)), allow_nan=False)
                )
                dependencies[name] = ()
                payload[name] = [
                    "predicate",
                    rule.kind,
                    definition.revision,
                    definition.input_kinds,
                    definition.shared_layout,
                    bindings,
                    params,
                ]
                key = json.dumps(payload[name], sort_keys=True, separators=(",", ":"))
                if key not in predicates:
                    predicates[key] = _BoundPredicate(
                        definition,
                        MappingProxyType(bindings),
                        _freeze(params),
                        rule.kind,
                    )
                nodes[name] = predicates[key]
            elif isinstance(rule, BooleanRule):
                refs = _refs(rule.refs)
                if rule.op not in ("all", "any", "not") or (
                    rule.op == "not" and len(refs) != 1
                ):
                    raise ValueError(f"Invalid boolean rule {name!r}.")
                nodes[name] = BooleanRule(rule.op, refs)
                dependencies[name] = refs
                payload[name] = ["boolean", rule.op, refs]
            elif isinstance(rule, TemporalRule):
                if (
                    rule.op not in ("hold_for", "always", "eventually")
                    or any(
                        type(value) is not int
                        for value in (rule.start, rule.end, rule.steps)
                    )
                    or not 0 <= rule.start <= rule.end
                    or not 1 <= rule.steps <= rule.end - rule.start + 1
                    or (rule.op != "hold_for" and rule.steps != 1)
                ):
                    raise ValueError(f"Invalid bounded temporal rule {name!r}.")
                dependencies[name] = _refs((rule.ref,))
                nodes[name] = rule
                payload[name] = [
                    "temporal",
                    rule.op,
                    rule.ref,
                    rule.start,
                    rule.end,
                    rule.steps,
                ]
            elif isinstance(rule, MilestoneRule):
                refs, after = _refs(rule.refs), _refs(rule.after, empty=True)
                if any(
                    not isinstance(spec.rules.get(ref), MilestoneRule) for ref in after
                ):
                    raise ValueError(f"Milestone {name!r} has an invalid predecessor.")
                nodes[name] = MilestoneRule(refs, after)
                dependencies[name] = refs + after
                payload[name] = ["milestone", refs, after]
            else:
                raise TypeError(f"Unsupported evaluation declaration for {name!r}.")

        if available_fields is not None and (missing := fields - available_fields):
            raise ValueError(f"Unresolved evaluation fields: {sorted(missing)!r}.")
        order = []
        visiting = set()
        visited = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError(f"Evaluation graph contains a cycle at {name!r}.")
            if name not in nodes:
                raise ValueError(f"Unknown evaluation reference {name!r}.")
            visiting.add(name)
            for ref in dependencies[name]:
                visit(ref)
            visiting.remove(name)
            visited.add(name)
            order.append(name)

        for name in (*roots, *sorted(nodes)):
            visit(name)
        stateful = set()
        for name in order:
            node = nodes[name]
            if isinstance(node, MilestoneRule) and stateful.intersection(node.refs):
                raise ValueError(
                    f"Milestone {name!r} must observe instantaneous achievement rules."
                )
            if isinstance(node, (MilestoneRule, TemporalRule)) or stateful.intersection(
                dependencies[name]
            ):
                stateful.add(name)
        identity = sha256(
            json.dumps(
                [1, roots, payload], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        compiled = CompiledEvaluation(
            roots,
            tuple(sorted(fields)),
            identity,
            tuple((name, nodes[name]) for name in order),
        )
        compiled._preset_joint_names()
        return compiled


@dataclass(frozen=True, slots=True)
class CompiledEvaluation:
    """Resolved immutable graph, observation requirements and stable identity."""

    roots: tuple[str, ...]
    required_fields: tuple[str, ...]
    identity: str
    _nodes: tuple[
        tuple[str, _BoundPredicate | BooleanRule | TemporalRule | MilestoneRule], ...
    ]

    def evaluate(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
        device: str | torch.device = "cpu",
        valid: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, RuleResult]:
        """Evaluate instantaneous rules. Stateful rules without a session are not run."""
        return self._evaluate(values, batch_size, device, valid or {}, None)

    @property
    def stateful(self) -> bool:
        """Return whether any compiled rule needs an advancing session."""
        return any(
            isinstance(node, (TemporalRule, MilestoneRule)) for _, node in self._nodes
        )

    @property
    def parameters(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the normalized parameter contract for every selected predicate."""
        return MappingProxyType(
            {
                name: node.params
                for name, node in self._nodes
                if isinstance(node, _BoundPredicate)
            }
        )

    def select(self, root: str) -> CompiledEvaluation:
        """Select one exported root and the dependencies it requires."""
        if root not in self.roots:
            raise ValueError(f"Evaluation root {root!r} is not exported.")
        required = {root}
        fields = set()
        selected = []
        for name, node in reversed(self._nodes):
            if name not in required:
                continue
            selected.append((name, node))
            if isinstance(node, _BoundPredicate):
                fields.update(node.bindings.values())
            elif isinstance(node, TemporalRule):
                required.add(node.ref)
            else:
                required.update(node.refs)
                if isinstance(node, MilestoneRule):
                    required.update(node.after)
        identity = sha256(json.dumps([self.identity, root]).encode()).hexdigest()
        return CompiledEvaluation(
            (root,), tuple(sorted(fields)), identity, tuple(reversed(selected))
        )

    def _check_input_contract(self, field_kinds, field_layouts):
        for name, rule in self._nodes:
            if isinstance(rule, _BoundPredicate):
                _check_input_contract(
                    name, rule.definition, rule.bindings, field_kinds, field_layouts
                )

    def _preset_joint_names(self):
        layouts = {}
        for _, node in self._nodes:
            if (
                isinstance(node, _BoundPredicate)
                and node.kind == "command_preset_match"
            ):
                field, names = node.bindings["position"], node.params["joint_names"]
                if field in layouts and layouts[field] != names:
                    raise ValueError(
                        "Command presets on one field require one native joint layout."
                    )
                layouts[field] = names
        return layouts

    def _bind_joint_names(self, layouts, *, cache=None):
        """Bind unresolved presets from a trusted native owner, never recorded data."""
        cache = {} if cache is None else cache
        nodes = []
        changed = {}
        for name, node in self._nodes:
            if (
                isinstance(node, _BoundPredicate)
                and node.kind == "command_preset_match"
            ):
                names = tuple(layouts[node.bindings["position"]])
                expected = node.params["joint_names"]
                if expected is not None and tuple(expected) != names:
                    raise ValueError(
                        "Command preset differs from the native joint layout."
                    )
                if expected is None:
                    key = (id(node), names)
                    if key not in cache:
                        params = node.definition.normalize(
                            {**node.params, "joint_names": names}
                        )
                        cache[key] = replace(node, params=_freeze(dict(params)))
                    node = cache[key]
                    changed[name] = names
            nodes.append((name, node))
        if not changed:
            return self
        identity = sha256(
            json.dumps([self.identity, changed], sort_keys=True).encode()
        ).hexdigest()
        return replace(self, identity=identity, _nodes=tuple(nodes))

    def _evaluate(self, values, size, device, valid, advance, *, measurements=None):
        import torch

        # The caller may share raw measurements across selected roots of one
        # frame, device, and row order. Qualification and owned results stay
        # local to each evaluation.
        if measurements is None:
            measurements = {}
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        results = {}
        for name, rule in self._nodes:
            if not isinstance(rule, _BoundPredicate):
                continue
            if not all(field in values for field in rule.bindings.values()):
                results[name] = _empty_result(
                    size, device, EvaluationStatus.UNAVAILABLE, parameters=rule.params
                )
                continue
            arguments = {role: values[field] for role, field in rule.bindings.items()}
            masks = [valid[field] for field in rule.bindings.values() if field in valid]
            if any(
                t.device != device or t.ndim < 1 or t.shape[0] != size
                for t in arguments.values()
            ):
                raise ValueError(
                    f"Evaluation fields for {name!r} must match the batch and device."
                )
            key = id(rule)
            measured = measurements.get(key)
            if measured is None:
                measured = rule.definition.measure(arguments, rule.params)
                measurements[key] = measured
            _validate_masks((measured.satisfied, measured.valid, *masks), size, device)
            available = measured.valid
            for mask in masks:
                available = available & mask
            status = _predicate_status(measured.satisfied, available)
            metrics = {}
            for key, metric in measured.metrics.items():
                if (
                    metric.device != device
                    or metric.ndim < 1
                    or metric.shape[0] != size
                ):
                    raise ValueError(
                        f"Evaluation metric {key!r} must match the batch and device."
                    )
                metrics[key] = _qualified_metric(metric, available)
            results[name] = RuleResult(status, metrics, parameters=rule.params)
        for name, rule in self._nodes:
            if name in results:
                continue
            if isinstance(rule, BooleanRule):
                results[name] = RuleResult(
                    _combine(rule.op, [results[ref].status for ref in rule.refs]), {}
                )
            elif advance is not None:
                results[name] = advance(name, rule, results)
            else:
                results[name] = _empty_result(size, device, EvaluationStatus.NOT_RUN)
        return results


@dataclass(frozen=True, slots=True)
class RuleResult:
    """Owned status, measurements and optional witness or violation boundaries.

    Boundaries are batched pairs of control slots or acquisition revisions.
    A missing boundary uses -1. Status tensors use ``EvaluationStatus`` codes.
    """

    status: torch.Tensor
    metrics: Mapping[str, torch.Tensor]
    witness_steps: torch.Tensor | None = None
    witness_revisions: torch.Tensor | None = None
    violation_steps: torch.Tensor | None = None
    violation_revisions: torch.Tensor | None = None
    parameters: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def reasons(self) -> tuple[str, ...]:
        """Return a diagnostic reason for each row, in status order."""
        return tuple(EvaluationStatus(code).reason for code in self.status.tolist())


def _qualified_metric(metric, available):
    shape = (len(available),) + (1,) * (metric.ndim - 1)
    missing = float("nan") if metric.is_floating_point() else 0
    return metric.detach().masked_fill(~available.reshape(shape), missing)


def _validate_masks(masks, size, device):
    import torch

    if any(
        mask.dtype != torch.bool or mask.shape != (size,) or mask.device != device
        for mask in masks
    ):
        raise ValueError("Predicate masks must match the evaluation batch and device.")


def _empty_result(size, device, status, *, parameters=None):
    import torch

    return RuleResult(
        torch.full((size,), status, device=device, dtype=torch.long),
        {},
        parameters={} if parameters is None else parameters,
    )


def _predicate_status(satisfied, valid):
    import torch

    status = torch.where(satisfied, EvaluationStatus.SATISFIED, EvaluationStatus.FAILED)
    return status.masked_fill_(~valid, EvaluationStatus.UNAVAILABLE)


def _combine(op, statuses):
    import torch

    code = EvaluationStatus
    if op == "not":
        source = statuses[0]
        return torch.where(
            source == code.SATISFIED,
            code.FAILED,
            torch.where(source == code.FAILED, code.SATISFIED, source),
        )
    batch = torch.stack(statuses)
    passed = batch == code.SATISFIED
    failed = batch == code.FAILED
    combined = torch.full_like(batch[0], code.NOT_RUN)
    combined.masked_fill_((batch == code.UNKNOWN).any(0), code.UNKNOWN)
    combined.masked_fill_((batch == code.UNAVAILABLE).any(0), code.UNAVAILABLE)
    if op == "all":
        combined.masked_fill_(passed.all(0), code.SATISFIED)
        return combined.masked_fill_(failed.any(0), code.FAILED)
    combined.masked_fill_(failed.all(0), code.FAILED)
    return combined.masked_fill_(passed.any(0), code.SATISFIED)
