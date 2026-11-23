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
import type { Call } from "./types";

type IntentFact = {
  kind: string;
  value?: string;
  subjects?: string[];
  references?: string[];
  arms?: string[];
  part?: string;
  [key: string]: unknown;
};

type IntentStep = {
  subjects?: string[];
  arms?: string[];
  direction?: string;
  effects?: IntentFact[];
  maintain?: IntentFact[];
  quote?: string;
};

const targetKinds = new Set([
  "relation",
  "pose_match",
  "view_location",
  "gripper_relation",
  "hung_by",
  "transferred",
  "articulated",
  "ordered_row",
  "distance_compare",
  "distance_change",
]);

const zhLabels: Record<string, string> = {
  pick: "抓取",
  grasp: "抓持",
  place: "放置",
  move_held: "持物移动",
  move_gripper: "移动夹爪",
  orient: "调整姿态",
  return_arms: "机械臂回位",
  open_grippers: "打开夹爪",
  handover: "交接",
  transfer_contents: "转移内容物",
  push: "推移",
  press: "按压",
  turn: "旋拧",
  articulate: "移动关节",
  release: "释放",
  released: "释放",
  held: "保持抓持",
  grasped_at: "在指定部位抓持",
  arms_at_initial_pose: "返回各自初始位",
  grippers_open: "保持夹爪打开",
  on: "置于表面",
  at: "位于指定位置",
  inside: "放入内部",
  above: "位于上方",
  centered_above: "在正上方居中",
  near_center_of: "靠近中心",
  centered_on: "在表面正中",
  spans: "横跨",
  on_top_faces: "由顶面承托",
  supported_by: "承托于",
  near_left_end: "靠近左端",
  near_right_end: "靠近右端",
  left_of: "在左侧",
  right_of: "在右侧",
  next_to: "紧邻",
  touching: "接触",
  hung_by: "用指定部位悬挂",
  transferred: "转移",
  stable: "保持稳定",
  aligned_long_axis: "长轴对齐",
  upright_cross_section: "截面直立",
  upright: "直立",
  flat: "平放",
  level: "保持水平",
  left_to_right: "从左到右",
  spaced_apart: "留有间距",
  direction: "移动方向",
  up: "向上",
  down: "向下",
  toward: "朝向",
  away: "远离",
  frame: "参考系",
  table: "桌面",
  comparison: "比较",
  meters: "米",
  quantity: "数量",
};

export function intentLabel(key: string, locale: Locale): string {
  return locale === "zh" && Object.hasOwn(zhLabels, key)
    ? zhLabels[key]
    : key.replaceAll("_", " ");
}

export function intentStep(call?: Call): IntentStep | undefined {
  if (call?.call_id?.startsWith("task_engine.intent."))
    return call.arguments as IntentStep | undefined;
}

export function IntentStepDetails({
  step,
  verb,
  locale,
  name,
  releaseQuote,
}: {
  step: IntentStep;
  verb: string;
  locale: Locale;
  name: (id: string) => string;
  releaseQuote?: string;
}) {
  const t = (zh: string, en: string) => (locale === "zh" ? zh : en);
  const label = (key: string) => intentLabel(key, locale);
  const separator = t("、", ", ");
  const role = (id: string) => {
    const display = name(id);
    return display === id ? id.replaceAll("_", " ") : display;
  };
  const arms = (ids: string[] = [], join = separator) =>
    ids
      .map((id) =>
        id === "auto"
          ? t("自动分配（未绑定）", "Auto (unbound)")
          : tr(id, locale),
      )
      .join(join);
  const isTarget = (fact: IntentFact) =>
    targetKinds.has(fact.kind) ||
    (fact.kind === "orientation" && Boolean(fact.references?.length));
  const effects = step.effects ?? [];
  const relations = effects.filter(isTarget);
  const outcomes = effects.filter((fact) => !isTarget(fact));
  const quotes = [...new Set([step.quote, releaseQuote].filter(Boolean))];
  const factList = (facts: IntentFact[]) => (
    <ul>
      {facts.map(
        (
          { kind, value, subjects, references, part, arms: by, ...fields },
          index,
        ) => {
          const operands = [
            subjects?.map(role).join(separator),
            references
              ?.map(role)
              .join(kind === "transferred" ? " → " : separator),
          ].filter(Boolean);
          const qualifiers = [
            part ? t("部位", "Part") + ": " + role(part) : "",
            by?.length ? t("机械臂", "Arms") + ": " + arms(by) : "",
            ...Object.entries(fields)
              .filter(([, val]) => val != null)
              .map(([key, val]) => label(key) + ": " + label(String(val))),
          ].filter(Boolean);
          return (
            <li key={index}>
              <strong>{label(value ?? kind)}</strong>
              {operands.length > 0 && (
                <span>
                  {operands.join(kind === "transferred" ? " · " : " → ")}
                </span>
              )}
              {qualifiers.length > 0 && <small>{qualifiers.join(" · ")}</small>}
            </li>
          );
        },
      )}
    </ul>
  );
  return (
    <section
      className="intent-step-details"
      aria-label={t("步骤语义", "Step semantics")}
    >
      <dl>
        <dt>{t("操作对象", "Objects")}</dt>
        <dd>{step.subjects?.map(role).join(separator) || t("无", "None")}</dd>
        <dt>{t("机械臂", "Arms")}</dt>
        <dd>
          {arms(step.arms, verb === "handover" ? " → " : separator) || "—"}
        </dd>
      </dl>
      <h3>{t("目标关系", "Target relations")}</h3>
      {relations.length ? factList(relations) : <p>—</p>}
      <h3>{t("动作效果", "Action effects")}</h3>
      {step.direction && (
        <p>
          {label("direction")}: {label(step.direction)}
        </p>
      )}
      {outcomes.length
        ? factList(outcomes)
        : !step.direction && <p>{t("无额外效果", "No additional effects")}</p>}
      {!!step.maintain?.length && (
        <>
          <h3>{t("过程中保持", "Maintain during the step")}</h3>
          {factList(step.maintain)}
        </>
      )}
      <h3>{t("对应原文", "Source wording")}</h3>
      {quotes.length ? (
        quotes.map((quote) => <blockquote key={quote}>{quote}</blockquote>)
      ) : (
        <p>—</p>
      )}
    </section>
  );
}
