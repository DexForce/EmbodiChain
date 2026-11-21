// ----------------------------------------------------------------------------
// Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ----------------------------------------------------------------------------

import { tr, type Locale } from "./i18n";
import type {
  Control,
  Mode,
  Predicate,
  Status,
  TaskGroup,
  TaskNode,
  ViewerData,
  Layout,
} from "./types";

export const good = (status: string) =>
  status === "succeeded" || status === "satisfied";
export function aggregate(
  values: (string | undefined)[],
  pass = "succeeded",
): Status {
  return values.includes("failed")
    ? "failed"
    : values.length && values.every((value) => value === pass)
      ? (pass as Status)
      : values.includes("unavailable")
        ? "unavailable"
        : "unknown";
}
/** Project recorded facts exactly; the view never derives a physical verdict. */
export function createModel(
  data: ViewerData,
  environment: string,
  locale: Locale,
) {
  const t = (key: string) => tr(key, locale);
  const calls = new Map(data.graph.nodes.map((node) => [node.id, node]));
  const groups = new Map(
    data.graph.task_groups.map((group) => [group.id, group]),
  );
  const assets = new Map(
    (data.presentation.assets || []).map((asset) => [asset.uid, asset]),
  );
  const envRows = (data.report?.environments || []).filter(
    (row) => environment === "all" || String(row.env_id) === environment,
  );
  const name = (uid?: string) =>
    data.presentation.localizations?.assets?.[uid || ""]?.name?.[locale] ||
    assets.get(uid || "")?.name ||
    uid ||
    "—";
  const subject = (node: TaskNode) =>
    node.call.object || node.call.arguments?.object;
  const subjects = (group: TaskGroup) => [
    ...new Set(
      group.node_ids
        .map((id) => subject(calls.get(id)!))
        .filter((uid): uid is string => Boolean(uid)),
    ),
  ];
  const lane = (node: TaskNode): Control => {
    const values = Object.values(node.call.resources || {});
    return values.includes("left") && values.includes("right")
      ? "coordinated"
      : values.includes("left")
        ? "left"
        : values.includes("right")
          ? "right"
          : "auto";
  };
  const groupLane = (group: TaskGroup): Control => {
    const nodes = group.node_ids.map((id) => calls.get(id)!),
      primary = nodes.filter((node) => node.role === "primary"),
      lanes = new Set((primary.length ? primary : nodes).map(lane));
    return lanes.has("coordinated") || (lanes.has("left") && lanes.has("right"))
      ? "coordinated"
      : lanes.has("auto")
        ? "auto"
        : lanes.has("left")
          ? "left"
          : lanes.has("right")
            ? "right"
            : "auto";
  };
  const callStatus = (node: TaskNode): Status => {
    const segment = data.call_evidence[node.id],
      ids = segment?.metadata?.runtime?.env_ids;
    if (
      !Array.isArray(ids) ||
      new Set(ids).size !== ids.length ||
      !Array.isArray(segment?.successes) ||
      !Array.isArray(segment?.active) ||
      ids.length !== segment.successes.length ||
      ids.length !== segment.active.length ||
      ids.some(
        (id) =>
          !Number.isInteger(id) ||
          !data.report?.environments.some((row) => row.env_id === id),
      ) ||
      segment.active.some((value) => typeof value !== "boolean")
    )
      return "unknown";
    const states = envRows.map((row) => {
      const index = ids.indexOf(row.env_id);
      return index < 0 || segment.active![index] !== true
        ? null
        : segment.successes![index];
    });
    return states.some((value) => value === false)
      ? "failed"
      : states.length && states.every((value) => value === true)
        ? "succeeded"
        : "unknown";
  };
  const groupStatus = (group: TaskGroup) =>
    aggregate(
      envRows.map((row) =>
        row.semantic_success?.[group.id] === true
          ? "succeeded"
          : row.semantic_success?.[group.id] === false
            ? "failed"
            : "unknown",
      ),
    );
  const goalRows = (goal: Predicate) =>
    (data.task_spec_evaluation?.environments || [])
      .filter(
        (row) => environment === "all" || String(row.env_id) === environment,
      )
      .map((row) => row.goal.find((value) => value.predicate_id === goal.id));
  const goalStatus = (goal: Predicate) =>
    aggregate(
      goalRows(goal).map((row) => row?.status || "unavailable"),
      "satisfied",
    );
  const events = (id: string) =>
    data.call_evidence[id]?.metadata?.runtime?.events || [];
  const eventWindow = (ids: string[]) => {
    const values = ids
      .flatMap((id) => events(id).map((event) => event.timestamp))
      .filter(
        (value): value is number =>
          typeof value === "number" && Number.isFinite(value),
      );
    return values.length
      ? ([Math.min(...values), Math.max(...values)] as const)
      : null;
  };
  const steps = (ids: string[]) => {
    const values = ids
      .map((id) => data.call_evidence[id])
      .filter(Boolean)
      .map((segment) =>
        Number.isInteger(segment.start_step) &&
        Number.isInteger(segment.end_step)
          ? segment.end_step! - segment.start_step!
          : null,
      );
    return values.length === ids.length &&
      values.every((value): value is number => typeof value === "number")
      ? values.reduce((a, b) => a + b, 0)
      : null;
  };
  const callLabel = (node: TaskNode) => {
    const id = node.call.call_id || node.call.kind;
    if (id === "pick" || id.startsWith("gen_sim.pick.")) return t("pick");
    if (id.includes("place")) return t("place");
    if (id.includes("park")) return t("park");
    if (id === "hand_over") return t("handover");
    if (id.includes("move_held")) return t("move");
    if (id.includes("pour")) return t("pour");
    if (id.startsWith("gen_sim.press_prepare.")) return t("pressPrepare");
    if (id.includes("press")) return t("press");
    if (id.includes("twist")) return t("twist");
    if (id.includes("align")) return t("upright");
    if (id.includes("coordinated")) return t("coordinatedAction");
    return id;
  };
  const groupLabel = (group: TaskGroup) => {
    const action = {
      E1: "place",
      E2: "upright",
      E3: "pour",
      E4: "handover",
      E5: "coordinatedAction",
      E6: "joint",
      E8: "twist",
      E9: "press",
    }[group.task_type];
    const object = name(subjects(group)[0]);
    return action
      ? t(action) +
          (locale === "en" ? " " + object.toLocaleLowerCase("en-US") : object)
      : group.task_type;
  };
  const goalLabel = (goal: Predicate) => {
    const ids =
        data.validation?.coverage.goals.find((value) => value.id === goal.id)
          ?.groups || [],
      group = ids.length === 1 ? groups.get(ids[0]) : null;
    const relation = goal.relation || goal.arguments?.relation;
    if (group && (relation === "on" || relation === "inside")) {
      const node = group.node_ids
        .map((id) => calls.get(id)!)
        .find(
          (node) =>
            node.call.arguments?.reference &&
            node.call.arguments.relation === relation,
        );
      if (node)
        return (
          name(subject(node)) +
          (locale === "en" ? " " : "") +
          t(relation) +
          (locale === "en" ? " " : "") +
          name(node.call.arguments?.reference)
        );
    }
    return (
      (goal.predicate === "semantic_goal"
        ? t("semanticGoal")
        : goal.predicate) +
      " / " +
      goal.id
    );
  };
  const normativeRows = (data.task_spec_evaluation?.environments || []).filter(
    (row) => environment === "all" || String(row.env_id) === environment,
  );
  const strict =
    data.task_spec &&
    normativeRows.length &&
    normativeRows.every(
      (row) => row.status === "satisfied" && row.certificate_ready === true,
    )
      ? "satisfied"
      : normativeRows.some((row) => row.status === "failed")
        ? "failed"
        : "unavailable";

  return {
    t,
    locale,
    calls,
    groups,
    assets,
    envRows,
    name,
    subject,
    subjects,
    lane,
    groupLane,
    callStatus,
    groupStatus,
    goalRows,
    goalStatus,
    events,
    eventWindow,
    steps,
    callLabel,
    groupLabel,
    goalLabel,
    strict: strict as Status,
  };
}
export type Model = ReturnType<typeof createModel>;

/** Preserve dependency depth and logical arm lanes, independently of viewport. */
export function layoutGraph(
  data: ViewerData,
  model: Model,
  mode: Mode,
  query: string,
  statusFilter: string,
  selected: string,
): Layout {
  const definitions =
    mode === "groups" ? data.graph.task_groups : data.graph.nodes;
  const levels = new Map<string, number>(),
    occupied = new Map<string, number>();
  const counts = { left: 0, auto: 0, right: 0 },
    slots = { left: 1, auto: 1, right: 1 };
  const prefix = mode === "groups" ? "group:" : "call:";
  const previews = definitions.map((row) => {
    const object =
      mode === "groups"
        ? model.subjects(row as TaskGroup)[0]
        : model.subject(row as TaskNode);
    const asset = model.assets.get(object || "");
    const preview = (["render", "cutout"] as const).find(
      (kind) =>
        typeof asset?.[kind] === "string" &&
        /^data:image\/(png|jpeg|webp);base64,/.test(asset[kind]!),
    );
    return {
      object,
      kind: preview,
      source: preview ? asset![preview]! : undefined,
    };
  });
  const rowHeight = previews.some((preview) => preview.source) ? 248 : 148;
  const rowStride = rowHeight + 22;
  const rows = definitions.map((row, index) => {
    const level = Math.max(
      0,
      ...row.depends_on.map((id) => (levels.get(id) || 0) + 1),
    );
    levels.set(row.id, level);
    const group =
      mode === "groups"
        ? (row as TaskGroup)
        : model.groups.get((row as TaskNode).task_instance_id)!;
    const control =
      mode === "groups" ? model.groupLane(group) : model.lane(row as TaskNode);
    const track = control === "coordinated" ? "auto" : control;
    const slot = occupied.get(level + ":" + track) || 0;
    occupied.set(level + ":" + track, slot + 1);
    counts[track]++;
    slots[track] = Math.max(slots[track], slot + 1);
    const title =
      mode === "groups"
        ? model.groupLabel(group)
        : model.callLabel(row as TaskNode);
    const status =
      mode === "groups"
        ? model.groupStatus(group)
        : model.callStatus(row as TaskNode);
    const { object, kind, source: thumbnail } = previews[index];
    const source =
      mode === "groups"
        ? { group: row, calls: group.node_ids.map((id) => model.calls.get(id)) }
        : row;
    const related =
      !selected.startsWith("goal:") ||
      (
        data.validation?.coverage.goals.find(
          (goal) => goal.id === selected.slice(5),
        )?.groups || []
      ).includes(group.id);
    const text =
      JSON.stringify(source) +
      title +
      (mode === "groups"
        ? model.subjects(group).map(model.name).join(" ")
        : model.name(object));
    return {
      key: prefix + row.id,
      id: row.id,
      title,
      index:
        mode === "groups"
          ? row.task_type + " / " + String(index + 1).padStart(2, "0")
          : row.id.split("__").at(-1)!,
      subtitle:
        mode === "groups"
          ? group.node_ids.length +
            " " +
            model.t("calls") +
            " / " +
            (model
              .steps(group.node_ids)
              ?.toLocaleString(model.locale === "zh" ? "zh-CN" : "en-US") ??
              model.t("missing")) +
            " steps"
          : model.name(object) +
            (object && model.name(object) !== object ? " / " + object : ""),
      status,
      thumbnail,
      thumbnailKind: kind,
      thumbnailLabel: thumbnail
        ? model.name(object) + " / " + object
        : undefined,
      track,
      control,
      x: 112 + level * 330,
      y: slot * rowStride,
      width: 260,
      height: rowHeight,
      nodeIds: mode === "groups" ? group.node_ids : [row.id],
      faded:
        !related ||
        !text.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()) ||
        (statusFilter !== "all" && statusFilter !== status),
    };
  });
  let y = 42;
  const lanes: Layout["lanes"] = [];
  for (const track of ["left", "auto", "right"] as const)
    if (counts[track]) {
      const height = rowHeight + 42 + (slots[track] - 1) * rowStride;
      for (const row of rows) if (row.track === track) row.y += y;
      const controls = new Set(
        rows.filter((row) => row.track === track).map((row) => row.control),
      );
      const label =
        track !== "auto"
          ? track
          : controls.size === 1 && controls.has("coordinated")
            ? "coordinated"
            : controls.has("coordinated")
              ? "mixedChannel"
              : "auto";
      lanes.push({ track, y: y - 24, height, width: 0, label: model.t(label) });
      y += height + 28;
    }
  const width = Math.max(650, ...rows.map((row) => row.x + row.width + 32));
  lanes.forEach((lane) => (lane.width = width - 24));
  const edges = definitions.flatMap((row) =>
    row.depends_on.map((id) => ({
      id: prefix + id + "→" + prefix + row.id,
      source: prefix + id,
      target: prefix + row.id,
    })),
  );
  return { rows, lanes, edges, width, height: y + 12 };
}
