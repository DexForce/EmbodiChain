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

export type IntentFact = {
  kind: string;
  value?: string;
  subjects?: string[];
  references?: string[];
  arms?: string[];
  part?: string;
  [key: string]: unknown;
};

type IntentStep = {
  subjects: string[];
  arms: string[];
  direction?: string;
  effects: IntentFact[];
  maintain?: IntentFact[];
  quote: string;
};

const zhLabels: Record<string, string> = {
  grasp: "抓持",
  move_held: "持物移动",
  move_gripper: "移动夹爪",
  orient: "调整姿态",
  return_arms: "机械臂回位",
  open_grippers: "打开夹爪",
  transfer_contents: "转移内容物",
  push: "推移",
  turn: "旋拧",
  articulate: "移动关节",
  release: "释放",
  released: "释放",
  held: "保持抓持",
  grasped_at: "指定部位抓持",
  arms_at_initial_pose: "返回各自初始位",
  above: "位于上方",
  centered_above: "正上方居中",
  centered_on: "表面居中",
  near_center_of: "靠近中心",
  on_top_faces: "由顶面承托",
  supported_by: "承托于",
  left_of: "在左侧",
  right_of: "在右侧",
  next_to: "紧邻",
  touching: "接触",
  hung_by: "指定部位悬挂",
  stable: "保持稳定",
  aligned_long_axis: "长轴对齐",
  flat: "平放",
  level: "保持水平",
  up: "向上",
  down: "向下",
};

export const intentLabel = (key: string, locale: Locale) =>
  (locale === "zh" && Object.hasOwn(zhLabels, key)
    ? zhLabels[key]
    : tr(key, locale)
  ).replaceAll("_", " ");

export function intentStep(call?: Call): IntentStep | undefined {
  if (call?.call_id?.startsWith("task_engine.intent."))
    return call.arguments as IntentStep | undefined;
}

type Display = { locale: Locale; name: (id: string) => string };
const separator = (locale: Locale) => (locale === "zh" ? "、" : ", ");
const arms = (ids: string[], locale: Locale, join = separator(locale)) =>
  ids
    .map((arm) =>
      arm === "auto" ? tr("unbound", locale) + " (auto)" : tr(arm, locale),
    )
    .join(join);

export function IntentFacts({
  facts,
  name,
  locale,
}: Display & { facts: IntentFact[] }) {
  if (!facts.length)
    return <p>{locale === "zh" ? "无额外效果" : "No additional effects"}</p>;
  return (
    <ul>
      {facts.map(
        (
          {
            kind,
            value,
            subjects,
            references,
            part,
            arms: assigned,
            ...fields
          },
          index,
        ) => {
          const roles = [
            subjects?.map(name).join(separator(locale)),
            references?.map(name).join(separator(locale)),
          ]
            .filter(Boolean)
            .join(" → ");
          const qualifiers = [
            part && name(part),
            assigned && arms(assigned, locale),
            ...Object.entries(fields)
              .filter(([, val]) => val != null)
              .map(
                ([key, val]) =>
                  intentLabel(key, locale) +
                  ": " +
                  intentLabel(String(val), locale),
              ),
          ].filter(Boolean);
          return (
            <li key={index}>
              <strong>{intentLabel(value ?? kind, locale)}</strong>
              {roles && <span>{roles}</span>}
              {!!qualifiers.length && <small>{qualifiers.join(" · ")}</small>}
            </li>
          );
        },
      )}
    </ul>
  );
}

export function IntentStepDetails({
  step,
  verb,
  locale,
  name,
  releaseQuote,
}: Display & {
  step: IntentStep;
  verb: string;
  releaseQuote?: string;
}) {
  const t = (zh: string, en: string) => (locale === "zh" ? zh : en);
  const quotes = [...new Set([step.quote, releaseQuote].filter(Boolean))];
  return (
    <section
      className="intent-step-details"
      aria-label={t("步骤语义", "Step semantics")}
    >
      <dl>
        <dt>{t("操作对象", "Objects")}</dt>
        <dd>{step.subjects.map(name).join(separator(locale)) || "—"}</dd>
        <dt>{t("机械臂", "Arms")}</dt>
        <dd>
          {arms(step.arms, locale, verb === "handover" ? " → " : undefined)}
        </dd>
        {step.direction && (
          <>
            <dt>{t("移动方向", "Direction")}</dt>
            <dd>{intentLabel(step.direction, locale)}</dd>
          </>
        )}
      </dl>
      <h3>{t("动作效果", "Action effects")}</h3>
      <IntentFacts facts={step.effects} name={name} locale={locale} />
      {!!step.maintain?.length && (
        <>
          <h3>{t("过程中保持", "Maintain during the step")}</h3>
          <IntentFacts facts={step.maintain} name={name} locale={locale} />
        </>
      )}
      <h3>{t("对应原文", "Source wording")}</h3>
      {quotes.map((quote) => (
        <blockquote key={quote}>{quote}</blockquote>
      ))}
    </section>
  );
}
