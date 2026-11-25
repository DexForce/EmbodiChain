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
  ArrowUpFromLine,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleCheck,
  CircleX,
  FileText,
  Maximize2,
  MousePointer2,
  ScanLine,
  Target,
  Workflow,
  X,
} from "lucide-react";
import {
  IntentFacts,
  IntentStepDetails,
  intentArmBadge,
  intentLabel,
  intentStep,
  intentStepRoleIds,
  type IntentFact,
} from "./IntentStepDetails";
import { createModel } from "./model";
import { ReviewDependencies } from "./ReviewDependencies";
import type { ViewerData, Predicate, TaskNode } from "./types";
import type { Locale } from "./i18n";
import "./review.css";
import "./reviewer-layout.css";

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
      clauses?: { id: string; quote: string; target?: string }[];
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
    milestone_calls?: Record<string, string>;
    goal_calls?: Record<string, string[]>;
  };
  annotations?: Record<string, { object?: string; reference?: string }>;
  pending: number;
  qualifiers?: { clause_id: string; status: string; reason: string }[];
  attempt: Attempt | { error: string } | null;
};
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
  const [theme, setTheme] = useState<"light" | "dark" | "system">(() => {
    const value = new URLSearchParams(location.search).get("theme");
    return value === "light" || value === "dark" ? value : "system";
  });
  const [systemDark, setSystemDark] = useState(
    () => matchMedia("(prefers-color-scheme: dark)").matches,
  );
  useEffect(() => {
    const preference = matchMedia("(prefers-color-scheme: dark)");
    const sync = () => setSystemDark(preference.matches);
    preference.addEventListener("change", sync);
    return () => preference.removeEventListener("change", sync);
  }, []);
  const zh = locale === "zh",
    t = (a: string, b: string) => (zh ? a : b);
  const { spec, pairing } = review,
    groups = data.graph.task_groups,
    calls = data.graph.nodes;
  const callNumbers = new Map(calls.map((node, index) => [node.id, index + 1]));
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
    [reviewView, setReviewView] = useState<"steps" | "dependencies">(
      "dependencies",
    ),
    [linkedGoal, setLinkedGoal] = useState(""),
    [imageLightbox, setImageLightbox] = useState(false);
  const opener = useRef<HTMLElement | null>(null),
    closeButton = useRef<HTMLButtonElement>(null),
    imageOpener = useRef<HTMLElement>(null),
    imageDialog = useRef<HTMLDialogElement>(null);
  const show = (key: string, source?: HTMLElement, goalId = "") => {
    if (source || !selected)
      opener.current = source ?? (document.activeElement as HTMLElement);
    setLinkedGoal(goalId);
    select(key);
  };
  const chooseCall = (
    node: TaskNode,
    source?: HTMLElement,
    goalId?: string,
  ) => {
    setOpen(new Set([node.task_instance_id]));
    show("call:" + node.id, source, goalId);
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
    if (!selected.startsWith("call:") || run) closeButton.current?.focus();
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
  const openImage = (source: HTMLElement) => {
    imageOpener.current = source;
    setImageLightbox(true);
  };
  const closeImage = () => {
    imageDialog.current?.close();
    setImageLightbox(false);
    imageOpener.current?.focus({ preventScroll: true });
  };
  useEffect(() => {
    document.documentElement.lang = zh ? "zh-CN" : "en";
    document.title = taskTitle;
  }, [zh, taskTitle]);
  const model = useMemo(
    () => createModel(data, env, locale),
    [data, env, locale],
  );
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
      return zh ? subject + "交接完成" : subject + " handed over";
    if (p.predicate === "pour_motion" && ref)
      return t(
        subject + "朝" + ref + "完成一次倾倒动作",
        subject + " performs a pour toward " + ref,
      );
    if (
      (p.predicate === "relation" || p.predicate === "semantic_goal") &&
      (relation === "on" || relation === "inside")
    )
      return zh
        ? subject + "在" + ref + (relation === "on" ? "上" : "内")
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
    (p) => selected === "goal:" + p.id || (selectedCall && linkedGoal === p.id),
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
  const anns = (review.annotations ??
    data.validation?.source?.annotations ??
    {}) as NonNullable<Review["annotations"]>;
  const matchesRole = (id: string) => {
    if (!selectedRole) return true;
    const call = model.calls.get(id)?.call;
    const step = intentStep(call);
    if (step) return intentStepRoleIds(step).has(selectedRole.id);
    const roleIds = new Set([
      selectedRole.id,
      ...(pairing.role_entities[selectedRole.id] || []),
    ]);
    const args = call?.arguments;
    const operands = [
      anns[id]?.object,
      anns[id]?.reference,
      call?.object,
      call?.on,
      call?.inside,
      call?.at,
      args?.object,
      args?.handle,
      args?.reference,
      args?.target,
    ];
    return operands.some((id) => typeof id === "string" && roleIds.has(id));
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
  const actionObject = (node: TaskNode) => {
    const object =
      node.call.object || node.call.arguments?.object || anns[node.id]?.object;
    return object ? shortName(object) : "";
  };
  const join = (label: string, object: string) =>
    object ? label + t("", " ") + object : label;
  const laneTitle = (node: TaskNode) => {
    const label = (
      {
        pick: t("抓取", "Pick"),
        place: t("放下", "Place"),
        hand_over: t("交接", "Pass"),
        "simulation.coordinated_transport": t("搬运", "Carry"),
      } as Record<string, string>
    )[node.call.call_id || node.call.kind];
    return label ? join(label, actionObject(node)) : titleFor(node);
  };
  const titleFor = (node: TaskNode) => {
    const step = intentStep(node.call);
    if (step)
      return join(
        intentLabel(node.task_type, locale),
        step.subjects.map(shortName).join(t("、", ", ")),
      );
    const object = actionObject(node);
    const resources = node.call.resources;
    if (
      node.call.kind === "hand_over" &&
      resources?.source &&
      resources?.destination &&
      object
    )
      return t(
        armName(resources.source) +
          "将" +
          object +
          "递给" +
          armName(resources.destination),
        armName(resources.source) +
          " passes " +
          object +
          " to " +
          armName(resources.destination),
      );
    const target = anns[node.id]?.reference || node.call.arguments?.reference;
    if (node.call.kind === "place" && object && target) {
      const destination = shortName(target);
      const inside =
        Boolean(node.call.inside) || node.call.arguments?.relation === "inside";
      return t(
        "将" + object + (inside ? "放入" : "放到") + destination + "并松开",
        "Place " +
          object +
          (inside ? " inside " : " at ") +
          destination +
          " and release",
      );
    }
    return (
      join(model.callLabel(node), object) +
      (target ? " → " + shortName(target) : "")
    );
  };
  const toggle = (id: string) =>
    setOpen((current) => new Set(current.has(id) ? [] : [id]));
  const activeGroups = new Set(
    selectedGoal
      ? pairing.goal_groups[selectedGoal.id] || []
      : [selectedGroup?.id || selectedCall?.task_instance_id || ""],
  );
  const selectedForGraph = (node: TaskNode) =>
    selectedCall
      ? selectedCall.id === node.id
      : activeGroups.has(node.task_instance_id);
  const stepNumber = (number: number) =>
    t("第 " + number + " 步", "Step " + number);
  const renderStages = () =>
    groups.map((group, index) => {
      const openGroup = open.has(group.id);
      const groupRoleMatch = group.node_ids.some(matchesRole);
      const numbers = group.node_ids
        .map((id) => callNumbers.get(id)!)
        .filter(Boolean);
      const first = numbers[0],
        last = numbers.at(-1);
      const range = !first
        ? ""
        : first === last
          ? stepNumber(first)
          : t(`第 ${first}–${last} 步`, `Steps ${first}–${last}`);
      const stageTitle = authoredPreview
        ? titleFor(model.calls.get(group.node_ids[0])!)
        : model.groupLabel(group);
      return (
        <section
          key={group.id}
          className="rv-review-stage"
          data-id={"group:" + group.id}
        >
          <div
            className={[
              "rv-frame",
              activeGroups.has(group.id) && "highlight",
              selectedRole && groupRoleMatch && "role-match",
              !groupRoleMatch && "faded",
            ]
              .filter(Boolean)
              .join(" ")}
          >
            <button
              type="button"
              className="rv-toggle"
              aria-expanded={openGroup}
              aria-controls={"review-stage-" + index}
              onClick={() => toggle(group.id)}
            >
              <span className="rv-stage-index">{index + 1}</span>
              <span className="rv-stage-verb">{stageTitle}</span>
              <small>{range}</small>
              <ChevronDown size={14} />
            </button>
            <ol
              className="rv-stage-calls"
              id={"review-stage-" + index}
              hidden={!openGroup}
            >
              {openGroup &&
                group.node_ids.map((id) => {
                  const node = model.calls.get(id)!;
                  return (
                    <li key={id}>
                      <button
                        type="button"
                        data-key={"call:" + id}
                        aria-pressed={selectedForGraph(node)}
                        className={[
                          "rv-step-action",
                          status(node) === false && "failed",
                          selectedForGraph(node) && "selected",
                          !matchesRole(id) && "faded",
                        ]
                          .filter(Boolean)
                          .join(" ")}
                        onClick={(event) =>
                          chooseCall(node, event.currentTarget)
                        }
                      >
                        <small>
                          {String(callNumbers.get(id)).padStart(2, "0")}
                        </small>
                        <span>{titleFor(node)}</span>
                      </button>
                    </li>
                  );
                })}
            </ol>
          </div>
        </section>
      );
    });
  const selectLanguage = (value: Locale) => {
    language(value);
    try {
      localStorage.setItem("gensim.graph.language", value);
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
  const goalCalls = (id: string) =>
    (
      pairing.goal_calls?.[id] ??
      groups.flatMap((group) =>
        pairing.goal_groups[id]?.includes(group.id) &&
        group.node_ids.length === 1
          ? group.node_ids
          : [],
      )
    ).filter((id) => model.calls.has(id));
  const reportGoals = [...spec.objectives.goal].sort(
    (a, b) =>
      Math.min(Infinity, ...goalCalls(a.id).map((id) => callNumbers.get(id)!)) -
      Math.min(Infinity, ...goalCalls(b.id).map((id) => callNumbers.get(id)!)),
  );
  const stepMilestones = selectedCall
    ? spec.objectives.milestones.filter(
        (m) =>
          pairing.milestone_calls?.[m.id] === selectedCall.id ||
          (!pairing.milestone_calls?.[m.id] &&
            stageCall?.id === selectedCall.id &&
            pairing.milestone_groups[m.id] === selectedCall.task_instance_id),
      )
    : [];
  const stepGoals = selectedCall
    ? spec.objectives.goal.filter((goal) =>
        goalCalls(goal.id).includes(selectedCall.id),
      )
    : [];
  const stepQuotes = selectedCall
    ? [
        ...new Set(
          [
            selectedIntent?.quote,
            ...(data.validation?.source?.requirements || [])
              .filter((row) => row.node_ids.includes(selectedCall.id))
              .map((row) => row.quote),
            ...(spec.interpretation.clauses || [])
              .filter((clause) =>
                stepMilestones.some((m) => m.id === clause.target),
              )
              .map((clause) => clause.quote),
          ].filter((quote): quote is string => !!quote),
        ),
      ]
    : [];
  const bindingRows =
    data.validation?.source?.bindings?.filter(
      (row) => row.node_id === selectedCall?.id,
    ) || [];
  const stepChecks = new Map<string, { scopes: Set<string>; hit: boolean }>();
  for (const [check, scope] of [
    ...stepMilestones.flatMap((m) =>
      m.achieve.map((fact) => [fact, t("里程碑", "Milestone")] as const),
    ),
    ...stepGoals.map(
      (fact) => [fact, t("完成条件", "Completion criterion")] as const,
    ),
  ]) {
    const label = predicate(check);
    const row = stepChecks.get(label) || {
      scopes: new Set<string>(),
      hit: false,
    };
    row.scopes.add(scope);
    row.hit ||= check.id === selectedGoal?.id;
    stepChecks.set(label, row);
  }
  const sceneBoxes = spec.interpretation.references.flatMap((reference) =>
    reference.entities.flatMap((entity) => {
      const box = entity.bbox;
      if (
        !box ||
        box.length !== 4 ||
        box.some((v) => !Number.isFinite(v) || v < 0 || v > 1)
      )
        return [];
      const [left, top, right, bottom] = box;
      if (left >= right || top >= bottom) return [];
      return [
        {
          role: reference.role_id,
          id: reference.role_id + ":" + entity.id,
          style: {
            left: left * 100 + "%",
            top: top * 100 + "%",
            width: (right - left) * 100 + "%",
            height: (bottom - top) * 100 + "%",
          },
        },
      ];
    }),
  );
  const markedRoles = new Set(sceneBoxes.map((box) => box.role));
  const toggleRole = (id: string, source: HTMLElement) =>
    selectedRole?.id === id ? close() : show("role:" + id, source);
  return (
    <main
      className="rv rv-reviewer graph-ui"
      data-theme={
        theme === "dark" || (theme === "system" && systemDark)
          ? "dark"
          : "light"
      }
    >
      <header className="rv-bar">
        <div className="rv-brand">
          <Workflow size={16} aria-hidden="true" /> <strong>EmbodiChain</strong>
          <span className="rv-bread">Generative Simulation - Task Engine</span>
        </div>
        <div className="rv-tools">
          <label>
            {t("语言", "Language")}
            <select
              value={locale}
              onChange={(e) => selectLanguage(e.target.value as Locale)}
            >
              <option value="zh" lang="zh-CN">
                中文
              </option>
              <option value="en" lang="en">
                English
              </option>
            </select>
          </label>
          <label>
            {t("主题", "Theme")}
            <select
              value={theme}
              onChange={(e) => setTheme(e.target.value as typeof theme)}
            >
              <option value="system">{t("跟随系统", "System")}</option>
              <option value="light">{t("浅色", "Light")}</option>
              <option value="dark">{t("深色", "Dark")}</option>
            </select>
          </label>
        </div>
      </header>
      <section className="rv-summary">
        <div className="rv-summary-layout">
          <div className="rv-title-line">
            <h1>{taskTitle}</h1>
            <small>
              {run
                ? t("运行回看", "Recorded attempt")
                : t("计划预览", "Plan preview")}
            </small>
          </div>
          <div className="rv-title-actions">
            <button
              type="button"
              className="rv-spec-trigger"
              aria-label={taskSpecLabel}
              onClick={(event) => show("spec", event.currentTarget)}
            >
              <FileText size={15} /> Task Spec
            </button>
            <button
              type="button"
              className="rv-spec-trigger"
              aria-label={t("详细图与导出", "Graph & exports")}
              onClick={onAdvanced}
            >
              <ArrowUpFromLine size={15} /> {t("导出", "Export")}
            </button>
          </div>
        </div>
        <dl
          className="rv-state"
          aria-label={t("计划与执行状态", "Plan and execution status")}
        >
          <div>
            <dt>{t("计划覆盖", "Plan coverage")}</dt>
            <dd className={unverifiedQualifiers.length ? "bad" : "good"}>
              {unverifiedQualifiers.length ? (
                <CircleX size={14} />
              ) : (
                <CircleCheck size={14} />
              )}
              {unverifiedQualifiers.length
                ? t("限定要求待核实", "Qualifiers unverified")
                : t("通过", "Passed")}
            </dd>
          </div>
          <div>
            <dt>{t("场景", "Scene")}</dt>
            <dd className="rv-scene-state">
              {review.pending > 0 ? (
                <button
                  type="button"
                  onClick={(e) => show("bindings", e.currentTarget)}
                >
                  {review.pending} {t("项待绑定", "pending bindings")}
                </button>
              ) : markedRoles.size && !run ? (
                markedRoles.size +
                "/" +
                spec.configuration.roles.length +
                t(" 已标注 · 待资格检查", " annotated · Awaiting qualification")
              ) : !run ? (
                t("仅计划 · 尚未绑定", "Plan only · Not bound")
              ) : (
                t("运行场景", "Run scene")
              )}
            </dd>
          </div>
          <div>
            <dt>{t("执行", "Execution")}</dt>
            <dd className={run?.status === "failed" ? "bad" : undefined}>
              {run ? model.t(run.status) : t("未执行", "Not run")}
            </dd>
          </div>
          {run && (
            <div>
              <dt>{t("任务验收", "Acceptance")}</dt>
              <dd>
                {run.verdict === "not_evaluated"
                  ? t("未评估", "Not evaluated")
                  : model.t(run.physical)}
              </dd>
            </div>
          )}
        </dl>
      </section>
      <div className="rv-work">
        <div className="rv-flow-column">
          <section className="rv-graph" aria-labelledby="rv-flow-heading">
            <div className="rv-graphbar">
              <h2 id="rv-flow-heading">{t("任务流程", "Task flow")}</h2>
              <small>
                {groups.length} {t("个阶段", "stages")} · {calls.length}{" "}
                {t("个动作", "actions")}
              </small>
            </div>
            <div
              className="rv-view-switch"
              role="group"
              aria-label={t("查看方式", "View")}
            >
              <button
                type="button"
                aria-pressed={reviewView === "steps"}
                onClick={() => setReviewView("steps")}
              >
                {t("步骤列表", "Steps")}
              </button>
              <button
                type="button"
                aria-pressed={reviewView === "dependencies"}
                onClick={() => setReviewView("dependencies")}
              >
                {t("依赖图", "Dependencies")}
              </button>
            </div>
            {reviewView === "steps" ? (
              <div className="rv-steps">{renderStages()}</div>
            ) : (
              <ReviewDependencies
                calls={calls}
                t={t}
                title={laneTitle}
                selected={selectedForGraph}
                related={matchesRole}
                status={status}
                onSelect={chooseCall}
              />
            )}
            {failure ? (
              <button
                type="button"
                className="rv-alert"
                onClick={() => show("call:" + failure.id)}
              >
                <CircleX size={17} />
                <strong>{titleFor(failure)}</strong> {reason(failure)}
                <span>{t("查看证据", "View evidence")}</span>
              </button>
            ) : review.attempt && "error" in review.attempt ? (
              <div role="alert">
                {t("运行记录无法关联", "Attempt unavailable")} ·{" "}
                {review.attempt.error}
              </div>
            ) : null}
          </section>
          {selectedCall && !run && (
            <aside
              className="rv-detail rv-inline-step"
              aria-label={t("详情", "Details")}
            >
              <div className="rv-review-heading rv-inline-heading">
                <h2>{titleFor(selectedCall)}</h2>
                <div className="rv-inline-tools">
                  <small>
                    {stepNumber(callNumbers.get(selectedCall.id)!)}
                    {t("详情", " details")}
                  </small>
                  <button
                    type="button"
                    ref={closeButton}
                    aria-label={t("关闭", "Close")}
                    onClick={close}
                  >
                    <X size={14} />
                  </button>
                </div>
              </div>
              <details className="rv-step-call" key={selectedCall.id}>
                <summary>
                  Semantic Call{" "}
                  <code>
                    {selectedCall.call.call_id || selectedCall.call.kind}
                  </code>
                </summary>
                <Raw value={selectedCall.call} />
              </details>
              <div className="rv-step-info">
                {stepQuotes.length > 0 && (
                  <>
                    <h3>{t("对应原文", "Source wording")}</h3>
                    {stepQuotes.map((quote) => (
                      <blockquote key={quote}>{quote}</blockquote>
                    ))}
                  </>
                )}
                {stepChecks.size > 0 && (
                  <>
                    <h3>{t("本步检查条件", "Checks for this step")}</h3>
                    <ul className="rv-step-checks">
                      {[...stepChecks].map(([label, row]) => (
                        <li key={label} className={row.hit ? "hit" : undefined}>
                          <span>{label}</span>
                          <small>{[...row.scopes].join(" · ")}</small>
                        </li>
                      ))}
                    </ul>
                  </>
                )}
                <details className="rv-step-tech" key={selectedCall.id}>
                  <summary>{t("工程信息", "Engineering information")}</summary>
                  <div className="rv-tech-links">
                    {selectedCall.depends_on.map((id) => {
                      const prerequisite = model.calls.get(id);
                      return (
                        prerequisite && (
                          <button
                            type="button"
                            key={id}
                            onClick={() => chooseCall(prerequisite)}
                          >
                            {t("前置步骤", "Prerequisite")} ·{" "}
                            {stepNumber(callNumbers.get(id)!)}
                          </button>
                        )
                      );
                    })}
                    <code>{selectedCall.id}</code>
                    {(selectedIntent || armLabel(selectedCall)) && (
                      <span>
                        {t("机械臂", "Arm")} ·{" "}
                        {selectedIntent
                          ? intentArmBadge(selectedIntent.arms, locale)
                          : armLabel(selectedCall)}
                      </span>
                    )}
                  </div>
                  {selectedIntent && (
                    <IntentStepDetails
                      key={selectedCall.id}
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
                  {!!bindingRows.length && (
                    <details>
                      <summary>
                        {t("该调用的绑定声明", "Binding declaration")}
                      </summary>
                      <Raw value={bindingRows} />
                    </details>
                  )}
                </details>
              </div>
            </aside>
          )}
          {selected && (!selectedCall || run) && (
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
                        groups.find((g) => g.id === c.task_instance_id)
                          ?.node_ids.length === 1
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
                            disabled={
                              callIdx < 0 || callIdx >= calls.length - 1
                            }
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
                        {t(
                          "没有关联的执行证据",
                          "No linked execution evidence",
                        )}
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
                            {t("要求数量", "Requested count")}　
                            {variation.count}
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
                            setReviewView("steps");
                            show("group:" + id);
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
          <section
            className="rv-goal-section"
            aria-labelledby="rv-goals-heading"
          >
            <div className="rv-review-heading">
              <h2 id="rv-goals-heading">
                {t("完成条件", "Completion criteria")}
              </h2>
              {!run && <small>{t("待执行评估", "Awaiting execution")}</small>}
            </div>
            {reportGoals.map((goal) => {
              const producers = pairing.goal_groups[goal.id] || [];
              const nodes = goalCalls(goal.id);
              const actualSteps = nodes
                .map((id) => callNumbers.get(id)!)
                .sort((a, b) => a - b);
              const indexes = groups.flatMap((g, i) =>
                producers.includes(g.id) ? [i + 1] : [],
              );
              return (
                <button
                  key={goal.id}
                  data-key={"goal:" + goal.id}
                  type="button"
                  aria-pressed={selectedGoal?.id === goal.id}
                  onClick={(event) => {
                    const node =
                      nodes.length === 1 && !run
                        ? model.calls.get(nodes[0])
                        : null;
                    if (!node) {
                      show("goal:" + goal.id, event.currentTarget);
                      return;
                    }
                    chooseCall(node, event.currentTarget, goal.id);
                    requestAnimationFrame(() =>
                      document
                        .querySelector(".rv-inline-step")
                        ?.scrollIntoView({ block: "nearest" }),
                    );
                  }}
                >
                  <span>{predicate(goal)}</span>
                  <small>
                    {actualSteps.length === 1
                      ? stepNumber(actualSteps[0])
                      : actualSteps.length
                        ? t("步骤 ", "Steps ") + actualSteps.join(" · ")
                        : indexes.length
                          ? t("阶段 ", "Stages ") + indexes.join(" · ")
                          : t("无配对证据", "No paired evidence")}
                    {producers.length || nodes.length ? " →" : ""}
                  </small>
                </button>
              );
            })}
          </section>
        </div>
        <aside className="rv-review-scene" aria-labelledby="rv-scene-heading">
          <div className="rv-review-heading">
            <h2 id="rv-scene-heading">{t("输入场景", "Source scene")}</h2>
            {review.source_image && (
              <button
                type="button"
                className="rv-enlarge"
                aria-haspopup="dialog"
                onClick={(event) => openImage(event.currentTarget)}
              >
                <Maximize2 size={16} /> {t("放大", "Enlarge")}
              </button>
            )}
          </div>
          <div className="rv-scene-body">
            {review.source_image && (
              <div className="rv-scene-photo">
                <button
                  type="button"
                  className="rv-scene-thumb"
                  aria-label={t("查看输入场景原图", "View source scene image")}
                  aria-haspopup="dialog"
                  onClick={(event) => openImage(event.currentTarget)}
                >
                  <img
                    src={review.source_image}
                    alt={t("任务来源图", "Source image")}
                  />
                </button>
                {sceneBoxes
                  .filter(
                    (box) => !selectedRole || box.role === selectedRole.id,
                  )
                  .map((box) => (
                    <button
                      key={box.id}
                      type="button"
                      style={box.style}
                      className={
                        "rv-scene-box" +
                        (selectedRole?.id === box.role ? " active" : "")
                      }
                      aria-label={t("定位", "Locate ") + shortName(box.role)}
                      aria-pressed={selectedRole?.id === box.role}
                      onClick={(event) =>
                        toggleRole(box.role, event.currentTarget)
                      }
                    >
                      {selectedRole?.id === box.role && (
                        <span>{shortName(box.role)}</span>
                      )}
                    </button>
                  ))}
              </div>
            )}
            <div className="rv-role-area">
              <h3>{t("任务对象", "Task objects")}</h3>
              <div
                className="rv-chips"
                role="group"
                aria-label={t(
                  "按任务对象定位动作",
                  "Locate actions by task object",
                )}
              >
                {spec.configuration.roles.map((role) => (
                  <button
                    key={role.id}
                    type="button"
                    className={"rv-role-chip kind-" + role.kind}
                    title={name(role.id)}
                    aria-pressed={selectedRole?.id === role.id}
                    onClick={(event) =>
                      toggleRole(role.id, event.currentTarget)
                    }
                  >
                    {shortName(role.id)}
                    {role.cardinality !== "one" && (
                      <small>× {role.count || role.cardinality}</small>
                    )}
                  </button>
                ))}
              </div>
              {selectedRole &&
                review.source_image &&
                !markedRoles.has(selectedRole.id) && (
                  <small className="rv-role-note">
                    {t(
                      "该对象尚无图像标注框",
                      "No source annotation for this object",
                    )}
                  </small>
                )}
            </div>
          </div>
          <section
            className="rv-review-instruction"
            aria-labelledby="rv-instruction-heading"
          >
            <h3 id="rv-instruction-heading">
              {t("原始指令", "Source instruction")}
            </h3>
            <blockquote>{spec.inputs.instruction}</blockquote>
          </section>
        </aside>
      </div>
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
