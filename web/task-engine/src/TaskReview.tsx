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

import { useEffect, useMemo, useState, useRef } from "react";
import {
  ReactFlow,
  Background,
  Handle,
  Position,
  MarkerType,
  useReactFlow,
  useStore,
  type Node,
  type NodeProps,
  type Edge,
} from "@xyflow/react";
import { GraphViewportControls } from "@embodichain/graph-ui";
import {
  Box,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleX,
  GitBranch,
  Maximize2,
  Moon,
  MousePointer2,
  ScanLine,
  Sun,
  Target,
  X,
} from "lucide-react";
import {
  IntentFacts,
  IntentStepDetails,
  intentArmBadge,
  intentFactChip,
  intentLabel,
  intentStep,
  intentStepRoleIds,
  type IntentFact,
} from "./IntentStepDetails";
import { createModel } from "./model";
import type { ViewerData, Predicate, TaskNode } from "./types";
import type { Locale } from "./i18n";
import "./review.css";

type Role = {
  id: string;
  kind: string;
  affordances: string[];
  properties: Record<string, unknown>;
  cardinality: string;
  count?: number;
};
type Objective = { id: string; achieve: Predicate[]; after?: string[] };
type Variation = {
  mode: string;
  dimension: string;
  role_id?: string;
  scope?: string;
  count?: number;
  cross_with?: string[];
  quote: string;
};
type Support = {
  subject: string;
  reference: string;
  footprint_margin_m: number;
  vertical_gap_m: number;
  satisfied: boolean;
};
type Observation = {
  holders?: Record<string, string>;
  terminal?: Record<string, Support>;
  tray_lift_m?: number;
  success: boolean | null;
  failure_reason: string | null;
  start_step: number;
  end_step: number;
  issues: {
    kind: string;
    message: string;
    timestamp: number;
    env_mask?: boolean[] | null;
  }[];
  env_ids: number[];
  successes: boolean[];
  active: boolean[];
  failure_reasons?: (string | null)[];
  start_steps?: number[];
  end_steps?: number[];
};
type Attempt = {
  status: string;
  verdict: string;
  physical: string;
  step: number;
  missing: string[];
  calls: Record<string, Observation>;
  identity: Record<string, string>;
};
export type Review = {
  replayVideo?: boolean;
  displayNames?: Record<string, { zh: string; en: string }>;
  source_image?: string;
  spec: {
    inputs: { instruction: string };
    configuration: {
      roles: Role[];
      descriptions: Record<string, string>;
      requirements?: { consumer: string; key: string; value: unknown }[];
    };
    objectives: {
      goal: Predicate[];
      milestones: Objective[];
      init: Predicate[];
      invariants: Predicate[];
      temporal: unknown[];
    };
    generalization: Variation[];
    interpretation: {
      steps?: {
        task_type: string;
        transfer_arm?: string;
        receive_arm?: string;
        required_arm?: string;
        terminal_behavior?: string;
      }[];
      clauses?: { id: string; quote: string }[];
      references: {
        role_id: string;
        entities: {
          id: string;
          description: string;
          provenance: string;
          bbox?: number[];
        }[];
      }[];
    };
  };
  pairing: {
    role_entities: Record<string, string[]>;
    milestone_groups: Record<string, string>;
    goal_groups: Record<string, string[]>;
  };
  pending: number;
  qualifiers?: { clause_id: string; status: string; reason: string }[];
  attempt: Attempt | { error: string } | null;
};
type Part = Node<{ content: React.ReactNode }, "reviewPart">;
function SpecSummary({
  review,
  t,
  name,
  predicate,
  show,
  locate,
}: {
  review: Review;
  t: (zh: string, en: string) => string;
  name: (id: string) => string;
  predicate: (p: Predicate) => string;
  show: (key: string) => void;
  locate: (id: string) => void;
}) {
  const { spec, pairing } = review,
    steps = spec.interpretation.steps || [];
  const arm = (id?: string) =>
    id === "left_arm"
      ? t("左臂", "left arm")
      : id === "right_arm"
        ? t("右臂", "right arm")
        : id;
  const operations: Record<string, string> = {
    E1: t("放置", "place"),
    E2: t("扶正", "set upright"),
    E3: t("倾倒", "pour"),
    E4: t("交接", "handover"),
    E5: t("双臂协同", "coordinated manipulation"),
    E6: t("移动滑动关节", "slide joint"),
    E7: t("开门", "open door"),
    E8: t("转动旋钮", "turn knob"),
    E9: t("按压", "press"),
  };
  const concepts = [
    ...new Set(
      steps.map((s) => {
        if (
          s.task_type === "E4" &&
          s.transfer_arm &&
          s.receive_arm &&
          ![s.transfer_arm, s.receive_arm].some((a) =>
            ["auto", "none"].includes(a),
          )
        )
          return t(
            arm(s.transfer_arm) + "递给" + arm(s.receive_arm),
            arm(s.transfer_arm) + " hands over to " + arm(s.receive_arm),
          );
        if (s.task_type === "E5" && s.terminal_behavior === "place")
          return t("双臂搬运并放下", "both arms carry and place");
        return [
          s.required_arm && !["auto", "none"].includes(s.required_arm)
            ? arm(s.required_arm)
            : "",
          operations[s.task_type] || s.task_type,
        ]
          .filter(Boolean)
          .join(" ");
      }),
    ),
  ];
  const arms = spec.configuration.requirements?.find(
    (r) => r.consumer === "embodiment" && r.key === "required_arms",
  )?.value;
  const pending = new Map(spec.objectives.milestones.map((m) => [m.id, m])),
    done = new Set<string>(),
    layers: Objective[][] = [];
  while (pending.size) {
    const ready = [...pending.values()].filter((m) =>
      (m.after || []).every((id) => done.has(id)),
    );
    if (!ready.length) break;
    layers.push(ready);
    for (const m of ready) {
      pending.delete(m.id);
      done.add(m.id);
    }
  }
  return (
    <div className="rv-spec-summary">
      <section>
        <h3>{t("任务输入", "Input")}</h3>
        <p>{spec.inputs.instruction}</p>
        {review.source_image && (
          <details>
            <summary>{t("查看来源图", "View source image")}</summary>
            <img
              src={review.source_image}
              alt={t("任务来源图", "Source image")}
            />
          </details>
        )}
      </section>
      <section>
        <h3>{t("任务理解", "Interpretation")}</h3>
        <p>
          {concepts.join(t("；", "; ")) ||
            t("未提供步骤解析", "No interpreted steps")}
        </p>
        {spec.interpretation.clauses?.map((c) => (
          <p key={c.id}>{c.quote}</p>
        ))}
      </section>
      <section>
        <h3>{t("任务构型", "Configuration")}</h3>
        {Array.isArray(arms) && (
          <p>
            {t("所需机械臂", "Required arms")} ·{" "}
            {arms.map((a) => arm(String(a))).join(" / ")}
          </p>
        )}
        <div className="rv-chips">
          {spec.configuration.roles.map((r) => (
            <button key={r.id} onClick={() => show("role:" + r.id)}>
              {name(r.id)}{" "}
              <small>
                ×{" "}
                {r.cardinality === "count"
                  ? r.count
                  : r.cardinality === "one"
                    ? 1
                    : r.cardinality}
              </small>
            </button>
          ))}
        </div>
      </section>
      <section>
        <h3>{t("目标与里程碑", "Objectives & milestones")}</h3>
        <h4>
          {t("最终要同时满足", "All final objectives")} ·{" "}
          {spec.objectives.goal.length}
        </h4>
        <ul>
          {spec.objectives.goal.map((p) => (
            <li key={p.id}>
              <button onClick={() => show("goal:" + p.id)}>
                {predicate(p)}
              </button>
            </li>
          ))}
        </ul>
        <h4>{t("里程碑 · 按依赖顺序", "Milestones · dependency order")}</h4>
        <ol>
          {layers.map((layer, i) => (
            <li key={layer[0].id}>
              <span>{String(i + 1).padStart(2, "0")}</span>
              <div>
                {layer.map((m) => (
                  <button
                    key={m.id}
                    title={(m.after || []).join(" + ") + " → " + m.id}
                    disabled={!pairing.milestone_groups[m.id]}
                    onClick={() => locate(pairing.milestone_groups[m.id])}
                  >
                    {m.achieve.map(predicate).join(" + ")}
                  </button>
                ))}
              </div>
            </li>
          ))}
        </ol>
        {(["init", "invariants", "temporal"] as const)
          .filter((k) => spec.objectives[k].length)
          .map((k) => (
            <details key={k}>
              <summary>{k}</summary>
              <Raw value={spec.objectives[k]} />
            </details>
          ))}
      </section>
      <section>
        <h3>{t("泛化范围", "Generalization")}</h3>
        {spec.generalization.length ? (
          spec.generalization.map((v, i) => (
            <button key={i} onClick={() => show("variation:" + i)}>
              {v.mode === "cover"
                ? t("必须覆盖", "Required")
                : t("允许变化", "Allowed")}{" "}
              · {v.dimension} {v.count && "× " + v.count}
            </button>
          ))
        ) : (
          <p>{t("未声明泛化要求", "No generalization requested")}</p>
        )}
      </section>
    </div>
  );
}
function ReviewPart({ data }: NodeProps<Part>) {
  return (
    <>
      <Handle id="l-in" type="target" position={Position.Left} />
      <Handle id="r-in" type="target" position={Position.Right} />
      <Handle id="t-in" type="target" position={Position.Top} />
      {data.content}
      <Handle id="r-out" type="source" position={Position.Right} />
      <Handle id="l-out" type="source" position={Position.Left} />
      <Handle id="b-out" type="source" position={Position.Bottom} />
    </>
  );
}
const nodeTypes = { reviewPart: ReviewPart };
function FrameGraph({ width, height }: { width: number; height: number }) {
  const flow = useReactFlow();
  const w = useStore((s) => s.width),
    h = useStore((s) => s.height);
  useEffect(() => {
    if (!flow.viewportInitialized || !w || !h) return;
    const zoom = Math.min(
      1.15,
      Math.max(0.12, Math.min((w - 48) / width, (h - 64) / height)),
    );
    void flow.setViewport({
      x: (w - width * zoom) / 2,
      y: (h - height * zoom) / 2,
      zoom,
    });
  }, [flow, w, h, width, height]);
  return null;
}
function Raw({ value }: { value: unknown }) {
  return <pre>{JSON.stringify(value, null, 2)}</pre>;
}

const ids = (value: unknown): string[] =>
  Array.isArray(value)
    ? value.filter((id): id is string => typeof id === "string")
    : [];

export default function TaskReview({
  data,
  review,
  onAdvanced,
}: {
  data: ViewerData;
  review: Review;
  onAdvanced: () => void;
}) {
  const [locale, language] = useState<Locale>(() => {
    try {
      return localStorage.getItem("gensim.graph.language") === "en"
        ? "en"
        : "zh";
    } catch {
      return "zh";
    }
  });
  const [dark, theme] = useState(
    () => new URLSearchParams(location.search).get("theme") === "dark",
  );
  const zh = locale === "zh",
    t = (a: string, b: string) => (zh ? a : b);
  const { spec, pairing } = review,
    groups = data.graph.task_groups,
    calls = data.graph.nodes;
  const authoredPreview =
    !review.attempt && calls.some((node) => intentStep(node.call));
  const taskTitle =
    data.presentation.localizations?.title?.[locale] ||
    data.graph.task_id
      .replace(/[_-]+/g, " ")
      .replace(/^./, (letter) => letter.toUpperCase());
  const taskSpecLabel = t("任务规范", "Task specification");
  const unverifiedQualifiers = (review.qualifiers || []).filter(
    (q) => q.status !== "covered",
  );
  const run =
    review.attempt && "calls" in review.attempt ? review.attempt : null;
  const initialFailure = calls.find((c) => run?.calls[c.id]?.success === false);
  const [open, setOpen] = useState<Set<string>>(
    () =>
      new Set(
        initialFailure
          ? [initialFailure.task_instance_id]
          : groups.length === 1
            ? [groups[0].id]
            : [],
      ),
  );
  const [selected, select] = useState(() =>
      location.hash === "#spec" ? "spec" : "",
    ),
    [env, setEnv] = useState("all"),
    [wrappedLayout, setWrappedLayout] = useState(true),
    [imageLightbox, setImageLightbox] = useState(false);
  const canWrapGroups =
    groups.length >= 5 &&
    groups.every(
      (g, i) =>
        i === 0 ||
        (g.depends_on.length === 1 && g.depends_on[0] === groups[i - 1].id),
    );
  const serpentine = canWrapGroups && wrappedLayout;
  const opener = useRef<HTMLElement | null>(null),
    closeButton = useRef<HTMLButtonElement>(null),
    imageOpener = useRef<HTMLButtonElement>(null),
    imageDialog = useRef<HTMLDialogElement>(null);
  const show = (key: string, source?: HTMLElement) => {
    if (!selected)
      opener.current = source ?? (document.activeElement as HTMLElement);
    select(key);
  };
  const close = () => {
    select("");
    requestAnimationFrame(() => {
      const key = opener.current?.dataset.key;
      const target = key
        ? document.querySelector<HTMLElement>(
            `button[data-key="${CSS.escape(key)}"]`,
          )
        : opener.current;
      target?.focus({ preventScroll: true });
    });
  };
  useEffect(() => {
    if (!selected) return;
    closeButton.current?.focus();
    const fn = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !imageDialog.current?.open) close();
    };
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, [selected]);
  useEffect(() => {
    const dialog = imageDialog.current;
    if (imageLightbox && dialog && !dialog.open) dialog.showModal();
  }, [imageLightbox]);
  const closeImage = () => {
    imageDialog.current?.close();
    setImageLightbox(false);
    imageOpener.current?.focus({ preventScroll: true });
  };
  useEffect(() => {
    document.documentElement.lang = zh ? "zh-CN" : "en";
    document.title = authoredPreview ? taskTitle : spec.inputs.instruction;
  }, [zh, authoredPreview, taskTitle, spec.inputs.instruction]);
  const model = useMemo(
    () => createModel(data, env, locale),
    [data, env, locale],
  );
  const flow = useReactFlow();
  const name = (id?: string) =>
    id
      ? review.displayNames?.[id]?.[locale] ||
        spec.configuration.descriptions[id] ||
        model.name(id)
      : "";
  const shortName = (id: string) =>
    review.displayNames?.[id]?.[locale] || model.name(id).replaceAll("_", " ");
  const predicate = (p: Predicate) => {
    const args = p.arguments;
    if (authoredPreview && p.predicate === "semantic_goal" && args) {
      const label = args.value ?? args.kind ?? args.verb;
      if (typeof label === "string")
        return [
          intentLabel(label, locale),
          ids(args.subjects).slice(0, 2).map(shortName).join(", "),
        ]
          .filter(Boolean)
          .join(" · ");
      return t("完成末步骤", "Finish the last step");
    }
    const subject = name(p.subject || String(p.arguments?.subject || "")),
      ref = name(p.reference || String(p.arguments?.reference || ""));
    const relation = p.relation || p.arguments?.relation;
    if (p.predicate === "held_by_both_grippers")
      return t("双臂共同持有", "Both arms hold") + " " + subject;
    if (p.predicate === "handover_complete")
      return subject + " · " + t("交接完成", "handover complete");
    if (
      (p.predicate === "relation" || p.predicate === "semantic_goal") &&
      (relation === "on" || relation === "inside")
    )
      return zh
        ? subject + " 在 " + ref + (relation === "on" ? " 上" : " 内")
        : subject + (relation === "on" ? " on " : " inside ") + ref;
    return [subject, p.predicate, ref].filter(Boolean).join(" · ");
  };
  const status = (c: TaskNode) => {
    const s = run?.calls[c.id];
    if (!s) return null;
    if (env === "all") return s.success;
    const i = s.env_ids.indexOf(Number(env));
    return i >= 0 && s.active?.[i] ? (s.successes?.[i] ?? null) : null;
  };
  const envs = [
    ...new Set(Object.values(run?.calls || {}).flatMap((s) => s.env_ids)),
  ];
  const failure = calls.find((c) => status(c) === false);
  const selectedRole = spec.configuration.roles.find(
    (r) => selected === "role:" + r.id,
  );
  const selectedGroup = groups.find((g) => selected === "group:" + g.id);
  const stageCall =
    selectedGroup?.node_ids.length === 1
      ? model.calls.get(selectedGroup.node_ids[0])
      : undefined;
  const selectedCall = intentStep(stageCall?.call)
    ? stageCall
    : calls.find((c) => selected === "call:" + c.id);
  const selectedIntent = intentStep(selectedCall?.call);
  const selectedGoal = spec.objectives.goal.find(
    (p) => selected === "goal:" + p.id,
  );
  const selectedGoalFact =
    authoredPreview && selectedGoal?.predicate === "semantic_goal"
      ? selectedGoal.arguments
      : undefined;
  const measuredGoal =
    selectedGoal &&
    Object.values(run?.calls || {}).find((c) => c.terminal?.[selectedGoal.id])
      ?.terminal?.[selectedGoal.id];
  const variation = selected.startsWith("variation:")
    ? spec.generalization[Number(selected.slice(10))]
    : null;
  const seg = selectedCall && run?.calls[selectedCall.id];
  const rowIndex = seg ? seg.env_ids.indexOf(Number(env)) : -1;
  const rowActive = rowIndex >= 0 && seg?.active?.[rowIndex] === true;
  const startStep =
    seg &&
    (env === "all"
      ? seg.start_step
      : rowActive
        ? seg.start_steps?.[rowIndex]
        : null);
  const endStep =
    seg &&
    (env === "all"
      ? seg.end_step
      : rowActive
        ? seg.end_steps?.[rowIndex]
        : null);
  const events = (s: Observation) =>
    s.issues.filter(
      (issue) =>
        env === "all" ||
        (issue.env_mask
          ? issue.env_mask[Number(env)] === true
          : envs.length === 1),
    );
  const anns = (data.validation?.source?.annotations || {}) as Record<
    string,
    { object?: string; reference?: string }
  >;
  const matchesRole = (id: string) => {
    if (!selectedRole) return true;
    const step = intentStep(model.calls.get(id)?.call);
    return step
      ? intentStepRoleIds(step).has(selectedRole.id)
      : pairing.role_entities[selectedRole.id]?.some((uid) =>
          [anns[id]?.object, anns[id]?.reference].includes(uid),
        ) === true;
  };
  const armName = (id: string) =>
    id === "left"
      ? t("左臂", "Left arm")
      : id === "right"
        ? t("右臂", "Right arm")
        : id;
  const armLabel = ({ call }: TaskNode) => {
    const { primary, source, destination, left, right } = call.resources || {};
    if (call.kind === "hand_over" && source && destination)
      return zh
        ? armName(source) + "递给" + armName(destination)
        : armName(source) + " → " + armName(destination);
    if (left && right) return t("双臂协同", "Both arms together");
    return primary ? armName(primary) : "";
  };
  const titles = (c: TaskNode) => {
    const object = name(c.call.object || c.call.arguments?.object);
    if (c.call.call_id === "simulation.coordinated_transport")
      return t("双臂搬", "Carry together: ") + object;
    if (c.call.kind === "place" && c.call.resources?.primary === "left")
      return t("左臂放", "Left arm places ") + object;
    if (c.call.kind === "hand_over") return t("交接", "Hand over ") + object;
    return null;
  };
  const titleFor = (c: TaskNode) =>
    intentStep(c.call)
      ? intentLabel(c.task_type, locale)
      : titles(c) ||
        model.callLabel(c) +
          " " +
          name(
            c.call.object || c.call.arguments?.object || anns[c.id]?.object,
          ) +
          (c.call.kind === "place" && anns[c.id]?.reference
            ? " → " + name(anns[c.id]?.reference)
            : "");
  const toggle = (id: string) =>
    setOpen((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const verbTone = (taskType: string) => {
    if (taskType === "pick" || taskType === "grasp") return "tone-pick";
    if (taskType === "place" || taskType === "release") return "tone-place";
    if (taskType === "return_arms" || taskType === "open_grippers")
      return "tone-return";
    if (
      [
        "actuate",
        "articulate",
        "press",
        "turn",
        "push",
        "transfer_contents",
        "handover",
      ].includes(taskType)
    )
      return "tone-action";
    if (["move_held", "move_gripper", "orient"].includes(taskType))
      return "tone-move";
    return "tone-default";
  };
  const nodes: Part[] = [],
    edges: Edge[] = [];
  const groupLevels = new Map<string, number>(),
    groupSizes = new Map<string, { w: number; h: number }>(),
    colWidths: number[] = [],
    nextY: number[] = [],
    serpCols = 4,
    serpMeta = new Map<string, { band: number; gridCol: number }>(),
    serpColWidths: number[] = [],
    serpBandHeights: number[] = [];
  const local = new Map<string, { x: number; y: number }>();
  for (const [gIndex, g] of groups.entries()) {
    const levels = new Map<string, number>(),
      rows: number[] = [];
    for (const id of g.node_ids) {
      const c = model.calls.get(id)!;
      const level = Math.max(
        0,
        ...c.depends_on
          .filter((d) => levels.has(d))
          .map((d) => levels.get(d)! + 1),
      );
      levels.set(id, level);
      const wrap =
        !!review.displayNames &&
        groups.length === 1 &&
        g.node_ids.every(
          (id, i) =>
            i === 0 ||
            model.calls.get(id)?.depends_on.join() === g.node_ids[i - 1],
        );
      const band = wrap ? Math.floor(level / 3) : 0;
      local.set(id, {
        x: 32 + (wrap ? (band % 2 ? 2 - (level % 3) : level % 3) : level) * 278,
        y: 98 + (band + (rows[level] || 0)) * 112,
      });
      rows[level] = (rows[level] || 0) + 1;
    }
    const count =
      spec.objectives.milestones
        .filter((m) => pairing.milestone_groups[m.id] === g.id)
        .reduce((sum, m) => sum + m.achieve.length, 0) +
      spec.objectives.goal.filter((p) =>
        pairing.goal_groups[p.id]?.includes(g.id),
      ).length;
    const singleIntent =
      authoredPreview && g.node_ids.length === 1
        ? intentStep(model.calls.get(g.node_ids[0])?.call)
        : undefined;
    const baseCollapsedH = singleIntent ? 136 : 100;
    const collapsedW = serpentine ? 332 : 370;
    const size = {
      w: open.has(g.id)
        ? Math.max(
            420,
            Math.max(...g.node_ids.map((id) => local.get(id)!.x)) + 246,
          )
        : collapsedW,
      h:
        (open.has(g.id)
          ? Math.max(...g.node_ids.map((id) => local.get(id)!.y)) + 132
          : baseCollapsedH) +
        Math.max(1, count) * 34,
    };
    const col = Math.max(
      0,
      ...g.depends_on.map((d) => (groupLevels.get(d) ?? -1) + 1),
    );
    groupLevels.set(g.id, col);
    groupSizes.set(g.id, size);
    colWidths[col] = Math.max(colWidths[col] || 0, size.w);
    const band = Math.floor(gIndex / serpCols);
    const slot = gIndex % serpCols;
    const gridCol = band % 2 === 0 ? slot : serpCols - 1 - slot;
    serpMeta.set(g.id, { band, gridCol });
    serpColWidths[gridCol] = Math.max(serpColWidths[gridCol] || 0, size.w);
    serpBandHeights[band] = Math.max(serpBandHeights[band] || 0, size.h);
  }
  const groupIdForKey = (key: string) =>
    key.startsWith("group:")
      ? key.slice(6)
      : model.calls.get(key.slice(5))?.task_instance_id;
  const activeGroupId =
    selectedGroup?.id || selectedCall?.task_instance_id || "";
  const edge = (source: string, target: string) => {
    const sg = groupIdForKey(source);
    const tg = groupIdForKey(target);
    let sourceHandle = "r-out";
    let targetHandle = "l-in";
    if (serpentine && sg && tg && sg !== tg) {
      const sp = serpMeta.get(sg);
      const tp = serpMeta.get(tg);
      if (sp && tp) {
        if (sp.band === tp.band) {
          if (tp.gridCol < sp.gridCol) {
            sourceHandle = "l-out";
            targetHandle = "r-in";
          }
        } else {
          sourceHandle = "b-out";
          targetHandle = "t-in";
        }
      }
    }
    const isActive =
      !!activeGroupId && (sg === activeGroupId || tg === activeGroupId);
    const strokeColor = isActive
      ? dark
        ? "#38bdf8"
        : "#2563eb"
      : dark
        ? "#334155"
        : "#94a3b8";
    edges.push({
      id: JSON.stringify([source, target]),
      source,
      target,
      sourceHandle,
      targetHandle,
      type: "smoothstep",
      animated: isActive,
      zIndex: isActive ? 3 : 1,
      markerEnd: { type: MarkerType.ArrowClosed, color: strokeColor },
      style: { stroke: strokeColor, strokeWidth: isActive ? 2.4 : 1.65 },
    });
  };
  const part = (
    id: string,
    x: number,
    y: number,
    width: number,
    height: number,
    content: React.ReactNode,
    frame = false,
  ) =>
    nodes.push({
      id,
      type: "reviewPart",
      position: { x, y },
      data: { content },
      width,
      height,
      style: { width, height, pointerEvents: "all" },
      className: frame ? "review-frame-node" : "review-call-node",
      zIndex: frame ? 0 : 2,
      draggable: false,
      selectable: false,
      focusable: false,
    });
  const shared = spec.objectives.goal.filter(
    (p) => pairing.goal_groups[p.id]?.length !== 1,
  );
  for (const [index, g] of groups.entries()) {
    const col = groupLevels.get(g.id)!,
      size = groupSizes.get(g.id)!;
    let x: number, y: number;
    if (serpentine) {
      const sm = serpMeta.get(g.id)!;
      x = serpColWidths.slice(0, sm.gridCol).reduce((a, b) => a + b + 54, 0);
      y = serpBandHeights.slice(0, sm.band).reduce((a, b) => a + b + 44, 0);
    } else {
      x = colWidths.slice(0, col).reduce((a, b) => a + b + 80, 0);
      y = nextY[col] || 0;
      nextY[col] = y + size.h + 48;
    }
    const milestones = spec.objectives.milestones.filter(
      (m) => pairing.milestone_groups[m.id] === g.id,
    );
    const goals = spec.objectives.goal.filter(
      (p) =>
        pairing.goal_groups[p.id]?.length === 1 &&
        pairing.goal_groups[p.id][0] === g.id,
    );
    // Compare the complete semantic operands, retaining separate achievement and final scopes.
    const signature = ({ id, observation, ...p }: Predicate) =>
      JSON.stringify(p, (_, v) =>
        v && typeof v === "object" && !Array.isArray(v)
          ? Object.fromEntries(
              Object.keys(v)
                .sort()
                .map((k) => [k, v[k]]),
            )
          : v,
      );
    const matches = (p: Predicate) =>
      milestones.some((m) =>
        m.achieve.some((q) => signature(q) === signature(p)),
      );
    const assignments = [
      ...new Set(
        g.node_ids.map((id) => armLabel(model.calls.get(id)!)).filter(Boolean),
      ),
    ];
    const previewNode =
      authoredPreview && g.node_ids.length === 1
        ? model.calls.get(g.node_ids[0])
        : undefined;
    const previewIntent = intentStep(previewNode?.call);
    const groupRoleMatch = !selectedRole || g.node_ids.some(matchesRole);
    const isFrameSelected =
      activeGroupId === g.id ||
      (selectedGoal && pairing.goal_groups[selectedGoal.id]?.includes(g.id));
    part(
      "group:" + g.id,
      x,
      y,
      size.w,
      size.h,
      <div
        className={
          "rv-frame " +
          (authoredPreview ? verbTone(g.task_type) + " " : "") +
          (isFrameSelected ? "highlight " : "") +
          (selectedRole && groupRoleMatch ? "role-match " : "") +
          (groupRoleMatch ? "" : "faded")
        }
      >
        <div className="rv-frame-head">
          <button className="rv-stage" onClick={() => show("group:" + g.id)}>
            <span className="rv-stage-index">
              {String(index + 1).padStart(2, "0")}
            </span>
            <span className="rv-stage-verb">
              <i className="rv-verb-dot" />
              {authoredPreview
                ? intentLabel(g.task_type, locale)
                : model.groupLabel(g)}
            </span>
          </button>
          <button
            className="rv-toggle"
            aria-expanded={open.has(g.id)}
            onClick={() => toggle(g.id)}
          >
            {g.node_ids.length} {t("个动作", "actions")}{" "}
            <ChevronDown size={14} />
          </button>
        </div>
        {assignments.length > 0 && (
          <div className="rv-arm-line" title={assignments.join(" · ")}>
            {assignments.join(" · ")}
          </div>
        )}
        {previewIntent && !open.has(g.id) && (
          <button
            type="button"
            className="rv-frame-preview nodrag nopan"
            onClick={(event) => show("group:" + g.id, event.currentTarget)}
          >
            <div className="rv-preview-top">
              <strong className="rv-preview-subject">
                {previewIntent.subjects.length
                  ? previewIntent.subjects.map(shortName).join(zh ? "、" : ", ")
                  : intentLabel(g.task_type, locale)}
              </strong>
              <span className="rv-preview-arm">
                {intentArmBadge(previewIntent.arms, locale)}
              </span>
            </div>
            <div className="rv-preview-chips">
              {previewIntent.effects.slice(0, 3).map((fact, fi) => {
                const chip = intentFactChip(fact, locale, shortName);
                return (
                  <span
                    key={fi}
                    className={"rv-preview-chip chip-" + chip.tone}
                    title={chip.text}
                  >
                    {chip.text}
                  </span>
                );
              })}
            </div>
          </button>
        )}
        <div className="rv-conditions">
          {milestones.flatMap((m) =>
            m.achieve
              .filter((p) => !goals.some((q) => signature(q) === signature(p)))
              .map((p) => (
                <button key={m.id + p.id} onClick={() => show("group:" + g.id)}>
                  <Target size={14} />
                  <span title={predicate(p)}>{predicate(p)}</span>
                  <small>{t("里程碑", "Milestone")}</small>
                </button>
              )),
          )}
          {goals.map((p) => (
            <button key={p.id} onClick={() => show("goal:" + p.id)}>
              <Target size={14} />
              <span title={predicate(p)}>{predicate(p)}</span>
              <small>
                {matches(p)
                  ? t("里程碑 · 终态", "Milestone · final")
                  : t("终态", "Final")}
              </small>
            </button>
          ))}
        </div>
      </div>,
      true,
    );
    if (open.has(g.id))
      for (const [i, id] of g.node_ids.entries()) {
        const c = model.calls.get(id)!,
          pos = local.get(id)!,
          s = status(c);
        const related = matchesRole(id);
        part(
          "call:" + id,
          x + pos.x,
          y + pos.y,
          225,
          90,
          <button
            data-key={"call:" + id}
            className={
              "rv-action nodrag nopan " +
              (s === false ? "failed" : s === true ? "passed" : "planned") +
              (selected === "call:" + id ? " selected" : "") +
              (related ? "" : " faded")
            }
            onClick={(event) => show("call:" + id, event.currentTarget)}
          >
            <span className="rv-action-meta">
              {String(i + 1).padStart(2, "0")}
              <span>
                {s === false ? (
                  <>
                    <CircleX size={13} />
                    {t("失败", "Failed")}
                  </>
                ) : s === true ? (
                  <>
                    <Check size={14} />
                    {t("完成", "Completed")}
                  </>
                ) : (
                  t("计划动作", "Planned")
                )}
              </span>
            </span>
            <strong>{titleFor(c)}</strong>
            {armLabel(c) ? (
              <span className="rv-arm-badge">{armLabel(c)}</span>
            ) : (
              <span className="rv-action-id">{id}</span>
            )}
          </button>,
        );
      }
  }
  for (const c of calls)
    for (const dep of c.depends_on) {
      const pg = model.calls.get(dep)!.task_instance_id;
      const from = open.has(pg) ? "call:" + dep : "group:" + pg,
        to = open.has(c.task_instance_id)
          ? "call:" + c.id
          : "group:" + c.task_instance_id;
      if (
        from !== to &&
        !edges.some((e) => e.source === from && e.target === to)
      )
        edge(from, to);
    }
  for (const g of groups)
    for (const dep of g.depends_on)
      if (
        !edges.some((e) =>
          [g.id, dep].every((id) =>
            [e.source, e.target].some(
              (k) =>
                k === "group:" + id ||
                model.calls.get(k.slice(5))?.task_instance_id === id,
            ),
          ),
        )
      )
        edge("group:" + dep, "group:" + g.id);
  const localeButton = () => {
    const l = zh ? "en" : "zh";
    language(l);
    try {
      localStorage.setItem("gensim.graph.language", l);
    } catch {}
  };
  const sources = selectedRole
    ? spec.interpretation.references
        .filter((r) => r.role_id === selectedRole.id)
        .flatMap((r) => r.entities)
    : [];
  const reason = (c: TaskNode) => {
    const observation = run?.calls[c.id];
    const gate =
      observation &&
      events(observation).find((e) => e.kind === "phase_effect_gate_failed");
    return gate?.message?.includes("'source_released'")
      ? t("夹爪释放检查失败", "Gripper release check failed")
      : (env === "all"
          ? observation?.failure_reason
          : observation?.failure_reasons?.[
              observation.env_ids.indexOf(Number(env))
            ]) || t("执行失败", "Execution failed");
  };
  return (
    <main className="rv graph-ui" data-theme={dark ? "dark" : "light"}>
      <header className="rv-bar">
        <span className="rv-brand">
          <GitBranch size={19} />
          EmbodiChain
        </span>
        <span className="rv-bread">
          /　Task Engine　/　{t("任务审阅", "Task review")}
        </span>
        <div className="rv-tools">
          <button onClick={localeButton}>{zh ? "EN" : "中文"}</button>
          <button aria-label={t("主题", "Theme")} onClick={() => theme(!dark)}>
            {dark ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          <button onClick={onAdvanced}>
            {t("详细图与导出", "Graph & exports")}
          </button>
        </div>
      </header>
      <section
        className={
          authoredPreview ? "rv-summary rv-authored-summary" : "rv-summary"
        }
      >
        <div className="rv-summary-layout">
          <div className="rv-summary-main">
            <div className="rv-title-line">
              <h1>{authoredPreview ? taskTitle : spec.inputs.instruction}</h1>
              <div className="rv-kicker">
                <span />
                {run
                  ? t("运行回看", "Recorded attempt")
                  : t("计划预览", "Plan preview")}
              </div>
            </div>
            <div className="rv-summary-row">
              <div className="rv-chips">
                {spec.configuration.roles.map((r) => (
                  <button
                    key={r.id}
                    className={"rv-role-chip kind-" + r.kind}
                    title={authoredPreview ? name(r.id) : undefined}
                    aria-pressed={selectedRole?.id === r.id}
                    onClick={() =>
                      selectedRole?.id === r.id ? close() : show("role:" + r.id)
                    }
                  >
                    <Box size={13} />
                    {authoredPreview ? shortName(r.id) : name(r.id)}{" "}
                    <small>
                      × {r.cardinality === "one" ? 1 : r.count || r.cardinality}
                    </small>
                  </button>
                ))}
                {spec.generalization.map((v, i) => (
                  <button key={i} onClick={() => show("variation:" + i)}>
                    {v.mode === "cover"
                      ? t("必须覆盖", "Required coverage")
                      : t("允许变化", "Allowed variation")}{" "}
                    · {v.dimension}
                    {v.count ? " × " + v.count : ""}
                  </button>
                ))}
              </div>
              <div className="rv-state">
                {run ? (
                  <>
                    <span className={run.status === "failed" ? "bad" : "good"}>
                      {run.status === "failed" ? (
                        <CircleX size={14} />
                      ) : (
                        <Check size={14} />
                      )}{" "}
                      {t("执行", "Execution")} · {model.t(run.status)}
                    </span>
                    <span>
                      {t("任务验收", "Acceptance")} ·{" "}
                      {run.verdict === "not_evaluated"
                        ? t("未评估", "Not evaluated")
                        : model.t(run.physical)}
                    </span>
                  </>
                ) : (
                  <>
                    <span
                      className={unverifiedQualifiers.length ? "bad" : "good"}
                    >
                      {unverifiedQualifiers.length ? (
                        <CircleX size={14} />
                      ) : (
                        <Check size={14} />
                      )}
                      {unverifiedQualifiers.length
                        ? t("限定要求待核实", "Qualifiers unverified")
                        : t("计划已校验", "Plan validated")}
                    </span>
                    {review.pending > 0 && (
                      <button onClick={() => show("bindings")}>
                        {review.pending} {t("项待绑定", "pending bindings")}
                      </button>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
          {review.source_image && (
            <button
              ref={imageOpener}
              type="button"
              className="rv-scene-thumb"
              aria-label={t("查看输入场景原图", "View source scene image")}
              onClick={() => setImageLightbox(true)}
            >
              <img
                src={review.source_image}
                alt={t("任务来源图", "Source image")}
              />
              <span>{t("原图", "RGB")}</span>
            </button>
          )}
        </div>
      </section>
      <div className="rv-work">
        <section className="rv-graph">
          <div className="rv-graphbar">
            <strong>
              <GitBranch size={16} />
              {authoredPreview
                ? t("任务流程", "Task flow")
                : t("动作与达成条件", "Actions & objectives")}
            </strong>
            <small>
              {groups.length} {t("阶段", "stages")}　/　{calls.length}{" "}
              {t("动作", "actions")}
            </small>
            {authoredPreview && spec.objectives.goal.length > 0 && (
              <div className="rv-goals-inline">
                <span className="rv-goals-label">
                  <Target size={13} />
                  {t("完成条件", "Goals")}
                </span>
                {spec.objectives.goal.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    className={
                      "rv-goal-pill" +
                      (selectedGoal?.id === p.id ? " active" : "")
                    }
                    onClick={() => show("goal:" + p.id)}
                  >
                    {predicate(p)}
                  </button>
                ))}
              </div>
            )}
            <div className="rv-graphbar-actions">
              {canWrapGroups && (
                <div
                  className="rv-layout-switch"
                  role="group"
                  aria-label={t("画布排布", "Graph layout")}
                >
                  <button
                    type="button"
                    aria-pressed={wrappedLayout}
                    onClick={() => setWrappedLayout(true)}
                  >
                    {t("折返全览", "Wrapped")}
                  </button>
                  <button
                    type="button"
                    aria-pressed={!wrappedLayout}
                    onClick={() => setWrappedLayout(false)}
                  >
                    {t("单行横排", "Linear")}
                  </button>
                </div>
              )}
              <button
                className="rv-spec-trigger"
                aria-label={taskSpecLabel}
                onClick={() => show("spec")}
              >
                {taskSpecLabel}
                {authoredPreview &&
                  ` · ${spec.objectives.goal.length} ${t("目标", "goals")}`}
                <ChevronRight size={14} />
              </button>
            </div>
          </div>
          {!authoredPreview &&
            shared.map((p) => (
              <button key={p.id} onClick={() => show("goal:" + p.id)}>
                {predicate(p)}
              </button>
            ))}
          <div className="rv-canvas">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              minZoom={0.12}
              nodesConnectable={false}
              nodesDraggable={false}
              elementsSelectable={false}
              deleteKeyCode={null}
            >
              <FrameGraph
                width={Math.max(
                  1,
                  ...nodes.map((n) => n.position.x + n.width!),
                )}
                height={Math.max(
                  1,
                  ...nodes.map((n) => n.position.y + n.height!),
                )}
              />
              <Background
                gap={22}
                size={1.1}
                color={dark ? "#1e293b" : "#cbd5e1"}
              />
            </ReactFlow>
            <div className="rv-controls">
              <GraphViewportControls />
            </div>
            {authoredPreview && (
              <div className="rv-legend" aria-hidden="true">
                <span className="rv-leg-item">
                  <i className="swatch-pick" />
                  {t("抓取", "Pick")}
                </span>
                <span className="rv-leg-item">
                  <i className="swatch-place" />
                  {t("放置", "Place")}
                </span>
                <span className="rv-leg-item">
                  <i className="swatch-move" />
                  {t("移动", "Move")}
                </span>
                <span className="rv-leg-item">
                  <i className="swatch-action" />
                  {t("精操", "Actuate")}
                </span>
                <span className="rv-leg-item">
                  <i className="swatch-return" />
                  {t("回位", "Return")}
                </span>
              </div>
            )}
          </div>
          {failure ? (
            <button
              className="rv-alert"
              onClick={() => show("call:" + failure.id)}
            >
              <CircleX size={17} />
              <strong>{titleFor(failure)}</strong> {reason(failure)}{" "}
              <span>{t("查看证据", "View evidence")} →</span>
            </button>
          ) : review.attempt && "error" in review.attempt ? (
            <div role="alert">
              {t("运行记录无法关联", "Attempt unavailable")} ·{" "}
              {review.attempt.error}
            </div>
          ) : null}
        </section>
        {selected && (
          <aside className="rv-detail" aria-label={t("详情", "Details")}>
            <div className="rv-detailbar">
              <ScanLine size={16} />
              {selectedIntent
                ? t("步骤详情", "Step details")
                : selectedCall
                  ? t("动作证据", "Action evidence")
                  : t("任务详情", "Task details")}
              <div className="rv-detailbar-right">
                {selectedCall &&
                  calls.length > 1 &&
                  (() => {
                    const callIdx = calls.findIndex(
                      (c) => c.id === selectedCall.id,
                    );
                    const stepKey = (c: TaskNode) =>
                      intentStep(c.call) &&
                      groups.find((g) => g.id === c.task_instance_id)?.node_ids
                        .length === 1
                        ? "group:" + c.task_instance_id
                        : "call:" + c.id;
                    return (
                      <>
                        <button
                          type="button"
                          aria-label={t("上一步", "Previous step")}
                          disabled={callIdx <= 0}
                          onClick={() =>
                            callIdx > 0 && select(stepKey(calls[callIdx - 1]))
                          }
                        >
                          <ChevronLeft size={16} />
                        </button>
                        <button
                          type="button"
                          aria-label={t("下一步", "Next step")}
                          disabled={callIdx < 0 || callIdx >= calls.length - 1}
                          onClick={() =>
                            callIdx >= 0 &&
                            callIdx < calls.length - 1 &&
                            select(stepKey(calls[callIdx + 1]))
                          }
                        >
                          <ChevronRight size={16} />
                        </button>
                      </>
                    );
                  })()}
                <button
                  ref={closeButton}
                  aria-label={t("关闭", "Close")}
                  onClick={close}
                >
                  <X size={17} />
                </button>
              </div>
            </div>
            <div className="rv-content">
              {selectedCall ? (
                <>
                  <h2>{titleFor(selectedCall)}</h2>
                  <code>{selectedCall.id}</code>
                  {selectedIntent && (
                    <IntentStepDetails
                      step={selectedIntent}
                      verb={selectedCall.task_type}
                      locale={locale}
                      name={shortName}
                      releaseQuote={
                        spec.interpretation.clauses?.find(
                          (clause) =>
                            clause.id ===
                            "source." + selectedCall.id + ".release",
                        )?.quote
                      }
                    />
                  )}
                  <h3
                    className={
                      selectedIntent ? "rv-intent-execution" : undefined
                    }
                  >
                    {t("执行记录", "Execution record")}
                  </h3>
                  {seg ? (
                    <>
                      <div
                        className={
                          "rv-result " +
                          (status(selectedCall) === false ? "bad" : "")
                        }
                      >
                        {status(selectedCall) === false ? (
                          <CircleX size={16} />
                        ) : (
                          <MousePointer2 size={16} />
                        )}{" "}
                        {status(selectedCall) === false
                          ? reason(selectedCall)
                          : model.t(
                              status(selectedCall) === true
                                ? "succeeded"
                                : "unknown",
                            )}
                      </div>
                      <p>
                        {t("控制步", "Control steps")}　
                        {startStep ?? t("未记录", "Not recorded")} –{" "}
                        {endStep ?? t("未记录", "Not recorded")}
                      </p>
                      {seg.holders && (
                        <p>
                          {t("段末持有", "Held at segment end")} ·{" "}
                          {Object.entries(seg.holders)
                            .map(
                              ([arm, obj]) => armName(arm) + "：" + name(obj),
                            )
                            .join(" / ") ||
                            t("双手已释放", "Both hands released")}
                        </p>
                      )}
                      {seg.tray_lift_m != null && (
                        <p>
                          {t("搬盘抬升", "Tray lift")} ·{" "}
                          {(seg.tray_lift_m * 1000).toFixed(1)} mm
                        </p>
                      )}
                      {seg.terminal && (
                        <>
                          <h3>
                            {t(
                              "搬盘后 · 三项支撑关系",
                              "Support after transport",
                            )}
                          </h3>
                          {Object.entries(seg.terminal).map(([id, v]) => (
                            <div key={id} className="rv-event">
                              <strong>
                                {v.satisfied ? "✓" : "×"} {name(v.subject)} →{" "}
                                {name(v.reference)}
                              </strong>
                              <p>
                                {t("底面间隙", "Vertical gap")}{" "}
                                {(v.vertical_gap_m * 1000).toFixed(2)} mm ·{" "}
                                {t("边缘余量", "Edge margin")}{" "}
                                {(v.footprint_margin_m * 1000).toFixed(1)} mm
                              </p>
                            </div>
                          ))}
                        </>
                      )}
                      {review.replayVideo && (
                        <details>
                          <summary>
                            {t("查看同部署复跑录像", "Deployment replay")}
                          </summary>
                          <video
                            controls
                            preload="none"
                            width="100%"
                            src="./execution.mp4"
                          />
                        </details>
                      )}
                      {events(seg)
                        .slice(0, 8)
                        .map((v, i) => (
                          <div key={i} className="rv-event">
                            <code>{v.kind}</code>
                            <p>{v.message}</p>
                            <small>{v.timestamp} s</small>
                          </div>
                        ))}
                    </>
                  ) : (
                    <p>
                      {t("没有关联的执行证据", "No linked execution evidence")}
                    </p>
                  )}
                  {envs.length > 1 && (
                    <select
                      aria-label={t("环境", "Environment")}
                      value={env}
                      onChange={(e) => setEnv(e.target.value)}
                    >
                      <option value="all">
                        {t("全部环境", "All environments")}
                      </option>
                      {envs.map((i) => (
                        <option key={i}>{i}</option>
                      ))}
                    </select>
                  )}
                  <details>
                    <summary>{t("调用参数", "Call parameters")}</summary>
                    <Raw value={selectedCall.call} />
                  </details>
                </>
              ) : selectedRole ? (
                <>
                  <h2>{name(selectedRole.id)}</h2>
                  <code>{selectedRole.id}</code>
                  <h3>{t("必须满足", "Requirements")}</h3>
                  <p>
                    {[
                      selectedRole.kind,
                      ...selectedRole.affordances,
                      ...Object.entries(selectedRole.properties).map(
                        ([k, v]) => k + " = " + v,
                      ),
                    ].join(" · ")}
                  </p>
                  <h3>{t("来源观察", "Source observations")}</h3>
                  {sources.map((s, i) => (
                    <div key={i}>
                      <p>
                        {s.description} · {s.provenance}
                      </p>
                      {s.bbox &&
                        review.source_image &&
                        /^data:image\/jpeg;base64,/.test(
                          review.source_image,
                        ) && (
                          <figure className="rv-source">
                            <img
                              alt={s.description}
                              src={review.source_image}
                            />
                            <span
                              style={{
                                left: s.bbox[0] * 100 + "%",
                                top: s.bbox[1] * 100 + "%",
                                width: (s.bbox[2] - s.bbox[0]) * 100 + "%",
                                height: (s.bbox[3] - s.bbox[1]) * 100 + "%",
                              }}
                            />
                          </figure>
                        )}
                      <small>
                        {s.id}　{s.bbox?.join(", ")}
                      </small>
                    </div>
                  ))}
                </>
              ) : selectedGoal ? (
                <>
                  <h2>{predicate(selectedGoal)}</h2>
                  <p>
                    {t("最终目标", "Final objective")} · {selectedGoal.id}
                  </p>
                  {typeof selectedGoalFact?.kind === "string" && (
                    <div className="intent-step-details">
                      <IntentFacts
                        facts={[selectedGoalFact as IntentFact]}
                        locale={locale}
                        name={shortName}
                      />
                    </div>
                  )}
                  {pairing.goal_groups[selectedGoal.id]?.length > 0 && (
                    <p>
                      {t("对应阶段", "Linked stages")}　
                      {pairing.goal_groups[selectedGoal.id].join(", ")}
                    </p>
                  )}
                  <p>
                    {t("终态证据", "Final evidence")}　
                    {measuredGoal
                      ? t("搬盘后几何测量", "Post-transport geometry") +
                        " · " +
                        (measuredGoal.satisfied
                          ? t("满足", "Satisfied")
                          : t("未满足", "Failed")) +
                        " · " +
                        (measuredGoal.vertical_gap_m * 1000).toFixed(2) +
                        " mm"
                      : run?.verdict === "not_evaluated" || !run
                        ? t("未评估", "Not evaluated")
                        : t("未提供逐目标证据", "No per-goal evidence")}
                  </p>
                  <details>
                    <summary>{t("谓词", "Predicate")}</summary>
                    <Raw value={selectedGoal} />
                  </details>
                </>
              ) : selectedGroup ? (
                <>
                  <h2>{model.groupLabel(selectedGroup)}</h2>
                  {spec.objectives.milestones
                    .filter(
                      (m) =>
                        pairing.milestone_groups[m.id] === selectedGroup.id,
                    )
                    .map((m) => (
                      <p key={m.id}>
                        {t("里程碑", "Milestone")} · {m.id}
                        <br />
                        {m.achieve.map(predicate).join(" + ")}
                      </p>
                    ))}
                  {selectedGroup.node_ids.map((id) => (
                    <button key={id} onClick={() => show("call:" + id)}>
                      {titleFor(model.calls.get(id)!)} →
                    </button>
                  ))}
                </>
              ) : (
                <>
                  <h2>
                    {selected === "spec"
                      ? t("任务规范", "Task specification")
                      : selected === "bindings"
                        ? t("待绑定", "Pending bindings")
                        : t("泛化要求", "Generalization")}
                  </h2>
                  {variation ? (
                    <>
                      <h3>
                        {variation.mode === "cover"
                          ? t("必须覆盖", "Required coverage")
                          : t("允许变化", "Permitted variation")}
                      </h3>
                      <p>
                        {variation.dimension}　{name(variation.role_id)}
                      </p>
                      {variation.scope && (
                        <p>
                          {t("范围", "Scope")}　{variation.scope}
                        </p>
                      )}
                      {variation.count && (
                        <p>
                          {t("要求数量", "Requested count")}　{variation.count}
                        </p>
                      )}
                      {variation.cross_with?.length && (
                        <p>
                          {t("交叉覆盖", "Cross with")}　
                          {variation.cross_with.join(", ")}
                        </p>
                      )}
                      <blockquote>{variation.quote}</blockquote>
                    </>
                  ) : selected === "spec" ? (
                    <>
                      <SpecSummary
                        review={review}
                        t={t}
                        name={name}
                        predicate={predicate}
                        show={show}
                        locate={(id) => {
                          setOpen((current) => new Set([...current, id]));
                          show("group:" + id);
                          requestAnimationFrame(() =>
                            flow.fitView({
                              nodes: [{ id: "group:" + id }],
                              padding: 0.2,
                              maxZoom: 1.15,
                              duration: 200,
                            }),
                          );
                        }}
                      />
                    </>
                  ) : null}
                  <details open={selected === "bindings"}>
                    <summary>{t("原始声明", "Declaration")}</summary>
                    <Raw
                      value={
                        selected === "spec"
                          ? spec
                          : selected === "bindings"
                            ? data.validation?.source?.bindings
                            : spec.generalization[
                                Number(selected.split(":")[1])
                              ]
                      }
                    />
                  </details>
                </>
              )}
            </div>
          </aside>
        )}
      </div>
      <footer className="rv-foot">
        <span>TaskSpec v1　 /　 SemanticTaskGraph</span>
        <button onClick={() => flow.fitView({ padding: 0.17, maxZoom: 1.15 })}>
          <Maximize2 size={12} /> {t("适应画布", "Fit canvas")}
        </button>
      </footer>
      {imageLightbox && review.source_image && (
        <dialog
          ref={imageDialog}
          className="rv-lightbox"
          aria-label={t("输入场景原图", "Source scene image")}
          onCancel={(e) => {
            e.preventDefault();
            closeImage();
          }}
          onKeyDown={(e) => {
            // The modal has one focusable control; keep both Tab directions on it.
            if (e.key === "Tab") e.preventDefault();
          }}
          onClick={(e) => {
            if (e.target === e.currentTarget) closeImage();
          }}
        >
          <div className="rv-lightbox-card">
            <div className="rv-lightbox-head">
              <span>{t("输入场景原图", "Source scene image")}</span>
              <button
                type="button"
                aria-label={t("关闭预览", "Close preview")}
                onClick={closeImage}
              >
                <X size={16} />
              </button>
            </div>
            <img
              src={review.source_image}
              alt={t("任务来源图", "Source image")}
            />
          </div>
        </dialog>
      )}
    </main>
  );
}
