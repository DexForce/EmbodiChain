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

import { useEffect, useId, useRef, useState } from "react";
import { intentStep } from "./IntentStepDetails";
import type { TaskNode } from "./types";

const laneOrder = ["left", "shared", "right"];
const laneLabelHeight = 34;

function laneFor({ call }: TaskNode): string {
  const r = call.resources || {};
  if ((r.source && r.destination) || (r.left && r.right)) return "shared";
  const arms = intentStep(call)?.arms || [];
  if (arms.length > 1) return "shared";
  const arm = arms[0] || r.primary;
  return !arm || arm === "auto" ? "unassigned" : arm;
}

type Box = { x: number; y: number; width: number; height: number };

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
  const board = useRef<HTMLDivElement>(null);
  const [available, setAvailable] = useState(0);
  useEffect(() => {
    const element = board.current;
    if (!element) return;
    const resize = () => setAvailable(Math.floor(element.clientWidth));
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const used = [...new Set(calls.map(laneFor))];
  const assigned = used.some((lane) => lane !== "unassigned");
  const lanes = used.some((lane) => laneOrder.includes(lane))
    ? [...laneOrder, ...used.filter((lane) => !laneOrder.includes(lane))]
    : used.length
      ? used
      : ["unassigned"];
  const width = Math.max(available, assigned ? lanes.length * 86 : 86);
  const span = width / lanes.length;
  const laneLabel = (lane: string) =>
    lane === "left"
      ? t("左臂", "Left arm")
      : lane === "right"
        ? t("右臂", "Right arm")
        : lane === "shared"
          ? t("交接 / 协同", "Handover / coordinated")
          : lane === "unassigned"
            ? t("机械臂待分配", "Arm unassigned")
            : lane;
  const laneNote = (node: TaskNode, compact: boolean) => {
    const r = node.call.resources || {};
    const label = (id: string) =>
      compact && id === "left"
        ? t("左", "L")
        : compact && id === "right"
          ? t("右", "R")
          : laneLabel(id);
    return r.source && r.destination
      ? label(r.source) + " → " + label(r.destination)
      : (r.left && r.right) || (intentStep(node.call)?.arms.length ?? 0) > 1
        ? t("双臂同时", "Both arms")
        : "";
  };
  const callName = (node: TaskNode) => node.call.call_id || node.call.kind;
  const placement = new Map<string, Box>();
  const pad = assigned && width >= 390 ? 6 : assigned ? 3 : 2;
  const gap = assigned ? 18 : 24;
  const cols = assigned
    ? lanes.length
    : Math.min(3, Math.max(1, Math.floor((width + gap - 2 * pad) / 252)));
  const cell = assigned
    ? span - 2 * pad
    : (width - (cols - 1) * gap - 2 * pad) / cols;
  const codeWrap = Math.max(1, Math.floor((cell - 16) / 7.5));
  const tall = (node: TaskNode) => {
    const textWidth = Array.from(title(node)).reduce(
      (n, c) => n + (c.charCodeAt(0) < 128 ? 7.5 : 14),
      0,
    );
    const lines = Math.max(1, Math.ceil(textWidth / Math.max(1, cell - 18)));
    const baseline = assigned ? (cell < 104 ? 72 : 54) : 84;
    return (
      baseline +
      18 * Math.ceil(callName(node).length / codeWrap) +
      (lines - 1) * 20
    );
  };
  let nextY = laneLabelHeight + (assigned ? 5 : pad);
  if (assigned) {
    for (const node of calls) {
      const x = span * lanes.indexOf(laneFor(node)) + pad;
      const height = tall(node);
      placement.set(node.id, { x, y: nextY, width: cell, height });
      nextY += height + gap;
    }
  } else {
    for (let start = 0; start < calls.length; start += cols) {
      const row = calls.slice(start, start + cols);
      const height = Math.max(...row.map(tall));
      row.forEach((node, i) => {
        const col = (start / cols) % 2 ? cols - 1 - i : i;
        placement.set(node.id, {
          x: pad + col * (cell + gap),
          y: nextY,
          width: cell,
          height,
        });
      });
      nextY += height + gap;
    }
  }
  const height = Math.max(laneLabelHeight, nextY - gap + (assigned ? 5 : pad));

  return (
    <div
      className="rv-lane-scroll"
      role="group"
      aria-label={t("语义调用及依赖", "Semantic calls and dependencies")}
      ref={board}
    >
      <div className="rv-lane-board" style={{ width, height }}>
        <div
          className="rv-lane-backgrounds"
          aria-hidden="true"
          style={{
            gridTemplateColumns: `repeat(${lanes.length}, minmax(0, 1fr))`,
          }}
        >
          {lanes.map((lane) => (
            <span key={lane} data-lane={lane}>
              {laneLabel(lane)}
            </span>
          ))}
        </div>
        <svg
          className="rv-lane-edges"
          aria-hidden="true"
          viewBox={`0 0 ${width} ${height}`}
        >
          <defs>
            <marker
              id={marker}
              viewBox="0 0 8 8"
              refX="7"
              refY="4"
              markerWidth="6"
              markerHeight="6"
              orient="auto"
            >
              <path d="M0 0L8 4L0 8Z" fill="currentColor" />
            </marker>
          </defs>
          {calls.flatMap((node) =>
            node.depends_on.map((id) => {
              const from = placement.get(id),
                to = placement.get(node.id);
              if (!from || !to) return null;
              let edge: string;
              if (from.y === to.y) {
                const forward = from.x < to.x;
                edge = `M ${forward ? from.x + from.width : from.x} ${from.y + from.height / 2} H ${forward ? to.x - 2 : to.x + to.width + 2}`;
              } else {
                const x1 = from.x + from.width / 2,
                  x2 = to.x + to.width / 2;
                const y1 = from.y + from.height,
                  y2 = to.y - 2;
                edge = `M ${x1} ${y1} V ${(y1 + y2) / 2} H ${x2} V ${y2}`;
              }
              return (
                <path
                  key={`${id}:${node.id}`}
                  data-from={id}
                  data-to={node.id}
                  d={edge}
                  className={selected(node) ? "selected" : undefined}
                  markerEnd={`url(#${marker})`}
                />
              );
            }),
          )}
        </svg>
        {calls.map((node, index) => {
          const box = placement.get(node.id)!;
          const active = selected(node);
          const compact = assigned && box.width < 155;
          return (
            <button
              key={node.id}
              type="button"
              data-key={`call:${node.id}`}
              data-lane={laneFor(node)}
              className={[
                "rv-lane-action",
                active && "selected",
                !related(node.id) && "faded",
                status(node) === false && "failed",
              ]
                .filter(Boolean)
                .join(" ")}
              aria-pressed={active}
              title={`${title(node)} · ${callName(node)} · ${node.id}`}
              style={{
                left: box.x,
                top: box.y,
                width: box.width,
                height: box.height,
              }}
              onClick={(e) => onSelect(node, e.currentTarget)}
            >
              <strong>{title(node)}</strong>
              <code>{callName(node)}</code>
              <small>
                {[
                  compact ? String(index + 1).padStart(2, "0") : node.id,
                  laneNote(node, compact),
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </small>
            </button>
          );
        })}
      </div>
    </div>
  );
}
