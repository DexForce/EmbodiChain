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

import { useId, type CSSProperties, type MouseEvent } from "react";
import { intentStep } from "./IntentStepDetails";
import type { TaskNode } from "./types";

const rowHeight = 86;
const columnWidth = 100;
const laneOrder = ["left", "shared", "right"];

function laneFor({ call }: TaskNode): string {
  const r = call.resources || {};
  if ((r.source && r.destination) || (r.left && r.right)) return "shared";
  const arms = intentStep(call)?.arms || [];
  return arms.length > 1 ? "shared" : arms[0] || r.primary || "unassigned";
}

export function ReviewDependencies({
  calls,
  t,
  title,
  selected,
  related,
  status,
  onSelect,
}: {
  calls: TaskNode[];
  t: (zh: string, en: string) => string;
  title: (node: TaskNode) => string;
  selected: (node: TaskNode) => boolean;
  related: (id: string) => boolean;
  status: (node: TaskNode) => boolean | null | undefined;
  onSelect: (node: TaskNode, button: HTMLButtonElement) => void;
}) {
  const marker = "review-arrow-" + useId().replaceAll(":", "");
  const used = [...new Set(calls.map(laneFor))];
  const lanes = used.some((lane) => laneOrder.includes(lane))
    ? [...laneOrder, ...used.filter((lane) => !laneOrder.includes(lane))]
    : used.length
      ? used
      : ["unassigned"];
  const location = new Map(
    calls.map((node, row) => [
      node.id,
      { row, col: lanes.indexOf(laneFor(node)) },
    ]),
  );
  const style = {
    "--rv-lane-count": lanes.length,
    "--rv-lane-row": `${rowHeight}px`,
  } as CSSProperties;
  const label = (lane: string) =>
    lane === "left"
      ? t("左臂", "Left arm")
      : lane === "right"
        ? t("右臂", "Right arm")
        : lane === "shared"
          ? t("交接 / 协同", "Handover / coordinated")
          : lane === "auto" || lane === "unassigned"
            ? t("机械臂待分配", "Arm unassigned")
            : lane;
  const select = (e: MouseEvent<HTMLButtonElement>, node: TaskNode) =>
    onSelect(node, e.currentTarget);

  return (
    <div
      className="rv-lane-scroll"
      role="group"
      aria-label={t("语义调用及依赖", "Semantic calls and dependencies")}
    >
      <div className="rv-lane-board" style={style}>
        <div className="rv-lane-head">
          {lanes.map((lane) => (
            <span key={lane} data-lane={lane}>
              {label(lane)}
            </span>
          ))}
        </div>
        <div className="rv-lane-body">
          <div className="rv-lane-backgrounds" aria-hidden="true">
            {lanes.map((lane) => (
              <span key={lane} data-lane={lane} />
            ))}
          </div>
          <svg
            className="rv-lane-edges"
            aria-hidden="true"
            preserveAspectRatio="none"
            viewBox={`0 0 ${lanes.length * columnWidth} ${Math.max(1, calls.length) * rowHeight}`}
          >
            <defs>
              <marker
                id={marker}
                markerWidth="7"
                markerHeight="7"
                refX="6"
                refY="3.5"
                orient="auto"
                markerUnits="userSpaceOnUse"
              >
                <path d="M 0 0 L 7 3.5 L 0 7 z" fill="currentColor" />
              </marker>
            </defs>
            {calls.flatMap((node) =>
              node.depends_on.map((id) => {
                const from = location.get(id),
                  to = location.get(node.id);
                if (!from || !to) return null;
                const x1 = from.col * columnWidth + columnWidth / 2;
                const x2 = to.col * columnWidth + columnWidth / 2;
                const y1 = (from.row + 1) * rowHeight - 8;
                const y2 = to.row * rowHeight + 8;
                const bend = (y1 + y2) / 2;
                return (
                  <path
                    key={`${id}:${node.id}`}
                    data-from={id}
                    data-to={node.id}
                    d={`M ${x1} ${y1} V ${bend} H ${x2} V ${y2}`}
                    className={selected(node) ? "selected" : undefined}
                    markerEnd={`url(#${marker})`}
                  />
                );
              }),
            )}
          </svg>
          {calls.map((node) => {
            const { row, col } = location.get(node.id)!;
            const callName = node.call.call_id || node.call.kind;
            const active = selected(node);
            return (
              <button
                key={node.id}
                type="button"
                data-key={`call:${node.id}`}
                data-lane={lanes[col]}
                className={[
                  "rv-lane-action",
                  active && "selected",
                  !related(node.id) && "faded",
                  status(node) === false && "failed",
                ]
                  .filter(Boolean)
                  .join(" ")}
                aria-pressed={active}
                title={`${title(node)} · ${callName} · ${node.id}`}
                style={{ gridColumn: col + 1, gridRow: row + 1 }}
                onClick={(e) => select(e, node)}
              >
                <strong>{title(node)}</strong>
                <code>{callName}</code>
                <small>{node.id}</small>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
