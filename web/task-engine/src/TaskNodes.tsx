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

import { memo } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { CheckCircle2, CircleAlert, CircleX, GitBranch } from "lucide-react";
import { good } from "./model";
import type { Lane, ViewRow } from "./types";
export type TaskFlowNode = Node<
  {
    row: ViewRow;
    active: boolean;
    statusLabel: string;
    onSelect: (key: string) => void;
  },
  "task"
>;
export type LaneFlowNode = Node<{ lane: Lane }, "lane">;
export const TaskNodeCard = memo(function TaskNodeCard({
  data,
}: NodeProps<TaskFlowNode>) {
  const { row, active } = data;
  const Icon = good(row.status)
    ? CheckCircle2
    : row.status === "failed"
      ? CircleX
      : CircleAlert;
  return (
    <button
      className={`task-graph-node nodrag ${active ? "active" : ""} ${row.faded ? "faded" : ""}`}
      data-key={row.key}
      data-status={row.status}
      aria-pressed={active}
      aria-label={row.id + " " + row.title}
      onClick={() => data.onSelect(row.key)}
    >
      <Handle type="target" position={Position.Left} />
      <span className="task-node-top">
        <GitBranch size={12} />
        <span>{row.index}</span>
        <span className={`channel-tag ${row.control}`}>
          {row.control === "coordinated" ? "L + R" : row.control.toUpperCase()}
        </span>
      </span>
      <strong title={row.title}>{row.title}</strong>
      <span className="task-node-sub" title={row.subtitle}>
        {row.subtitle}
      </span>
      {row.thumbnail && (
        <span className="task-node-media">
          <img
            className="task-node-image"
            src={row.thumbnail}
            alt={row.thumbnailLabel}
            data-image-kind={row.thumbnailKind}
          />
        </span>
      )}
      <span
        className={`task-node-status ${good(row.status) ? "good" : row.status === "failed" ? "bad" : "unknown"}`}
      >
        <Icon size={13} />
        {data.statusLabel}
      </span>
      <Handle type="source" position={Position.Right} />
    </button>
  );
});
export const TaskLane = memo(function TaskLane({
  data,
}: NodeProps<LaneFlowNode>) {
  return (
    <div
      className={"task-lane " + data.lane.track}
      style={{ width: data.lane.width, height: data.lane.height }}
    >
      <span>
        {data.lane.label}
        <small>
          {data.lane.track === "auto"
            ? "AUTO / COORD."
            : data.lane.track.toUpperCase()}
        </small>
      </span>
    </div>
  );
});
