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

export type IntentStep = {
  subjects: string[];
  arms: string[];
  direction?: string;
  effects: IntentFact[];
  maintain?: IntentFact[];
  quote: string;
};

const zhLabels: Record<string, string> = {
  pick: "抓取",
  place: "放置",
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
  actuate: "触发动作",
  press: "按压",
  handover: "交接",
  release: "释放",
  released: "释放",
  held: "保持抓持",
  grasped_at: "指定部位抓持",
  arms_at_initial_pose: "返回各自初始位",
  at: "位于",
  on: "置于表面",
  inside: "放入内部",
  not_inside: "不在内部",
  spans: "跨越搭接",
  above: "位于上方",
  below: "位于下方",
  between: "位于之间",
  in_front_of: "位于前方",
  behind: "位于后方",
  centered_above: "正上方居中",
  centered_on: "表面居中",
  near_center_of: "靠近中心",
  near_left_end: "靠近左端",
  near_right_end: "靠近右端",
  opposite_sides: "分列两侧",
  opposite_ends: "分居两端",
  spaced_apart: "彼此留缝",
  on_top_faces: "由顶面承托",
  supported_by: "承托于",
  left_of: "在左侧",
  right_of: "在右侧",
  next_to: "紧邻",
  near: "靠近",
  touching: "接触",
  aligned_with: "对齐于",
  fixed_to: "固定于",
  hung_by: "指定部位悬挂",
  stable: "保持稳定",
  intact: "保持完整",
  mutually_separate: "互不套叠",
  unoccupied: "空置无遮挡",
  movable: "可移动",
  untouched: "保持未触碰",
  untouched_except: "其余保持未触碰",
  seated: "坐合就位",
  seated_on: "就位套合",
  tightened: "旋拧紧固",
  tightened_on: "旋紧固定",
  grippers_open: "夹爪保持张开",
  actuated: "触发到位",
  articulated: "关节就位",
  relation: "空间关系",
  orientation: "姿态朝向",
  distance_compare: "距离比较",
  distance_change: "距离变化",
  gripper_relation: "夹爪位置",
  pose_match: "位姿匹配",
  ordered_row: "按序成排",
  transferred: "内容已转移",
  vertical_rise: "竖直抬升",
  view_location: "视野位置",
  clear_of: "避让脱离",
  compatible_with: "姿态兼容",
  open: "保持打开",
  closed: "保持关闭",
  turned: "旋拧到位",
  opening_up: "开口朝上",
  opening_down: "开口朝下",
  pointing_up: "朝上指向",
  pointing_down: "朝下指向",
  pointing_left: "朝左指向",
  pointing_right: "朝右指向",
  pointing_forward: "朝前指向",
  pointing_backward: "朝后指向",
  pointing_toward: "指向目标",
  pointing_away_from: "背向目标",
  pressed: "按压到位",
  separate_from: "彼此分离",
  tilted: "倾斜姿态",
  upright: "直立",
  vertical: "竖直",
  upright_cross_section: "截面直立",
  lying_on_side: "侧卧",
  left_to_right: "水平左→右",
  front_to_back: "前后朝向",
  aligned_long_axis: "长轴对齐",
  flat: "平放",
  level: "保持水平",
  up: "向上",
  down: "向下",
  toward: "朝向目标",
  away: "远离目标",
  frame: "参考系",
  head_camera: "头相机视角",
  table: "桌面参考系",
  robot: "机器人参考系",
  world: "世界参考系",
  reference: "参照物参考系",
  lateral: "横向位置",
  left: "左侧",
  right: "右侧",
  center: "居中",
  far_left: "最左侧",
  far_right: "最右侧",
  depth: "远近位置",
  far: "远处",
  middle: "中部",
  epoch: "参考时刻",
  initial: "初始状态",
  current: "当前状态",
  components: "匹配分量",
  position: "位置",
  comparison: "比较关系",
  eq: "等于",
  gt: "大于",
  ge: "不小于",
  lt: "小于",
  le: "不大于",
  meters: "高度阈值(m)",
  mechanism: "关节类型",
  hinge: "铰链",
  slide: "滑轨",
  material: "物质类型",
  solid: "固体",
  liquid: "液体",
  all: "全部转移",
  quantity: "数量",
  true: "是",
  false: "否",
};

export const intentLabel = (key: string, locale: Locale) => {
  if (locale === "zh" && Object.hasOwn(zhLabels, key)) {
    return zhLabels[key].replaceAll("_", " ");
  }
  if (key === "left" || key === "right") return key;
  return tr(key, locale).replaceAll("_", " ");
};

export function intentStep(call?: Call): IntentStep | undefined {
  if (call?.call_id?.startsWith("task_engine.intent."))
    return call.arguments as IntentStep | undefined;
}

export function intentStepRoleIds(step?: IntentStep): Set<string> {
  const out = new Set<string>();
  if (!step) return out;
  for (const id of step.subjects || []) out.add(id);
  for (const fact of [...(step.effects || []), ...(step.maintain || [])]) {
    for (const id of fact.subjects || []) out.add(id);
    for (const id of fact.references || []) out.add(id);
  }
  return out;
}

export function intentArmBadge(ids: string[], locale: Locale): string {
  const sep = locale === "zh" ? " · " : " / ";
  return ids
    .map((arm) =>
      arm === "auto"
        ? locale === "zh"
          ? "自动分配"
          : "auto"
        : tr(arm, locale),
    )
    .join(sep);
}

export function intentFactChip(
  fact: IntentFact,
  locale: Locale,
  name: (id: string) => string,
): { text: string; tone: "rel" | "ori" | "state" | "muted" } {
  const sep = locale === "zh" ? "、" : ", ";
  const refs = fact.references?.map(name).join(sep);
  const label = intentLabel(fact.value ?? fact.kind, locale);
  if (fact.kind === "held") return { text: label, tone: "state" };
  if (fact.kind === "released") return { text: label, tone: "muted" };
  if (fact.kind === "orientation" || fact.kind === "arms_at_initial_pose") {
    return { text: refs ? `${label} · ${refs}` : label, tone: "ori" };
  }
  if (fact.kind === "relation") {
    return { text: refs ? `${label} ${refs}` : label, tone: "rel" };
  }
  return { text: refs ? `${label} ${refs}` : label, tone: "rel" };
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
            <dd>
              {step.direction === "left"
                ? t("向左", "left")
                : step.direction === "right"
                  ? t("向右", "right")
                  : intentLabel(step.direction, locale)}
            </dd>
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
