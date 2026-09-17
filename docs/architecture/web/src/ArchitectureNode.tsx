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

import { memo, type CSSProperties } from 'react';
import { Handle, Position, type Node, type NodeProps } from '@xyflow/react';
import { Box, ArrowUpRight } from 'lucide-react';
import type { ArchitectureNode as Module } from './types';

export type ModuleNode = Node<
  {
    module: Module;
    color: string;
    faded: boolean;
    active: boolean;
    count: number;
    recordedCount: number;
  },
  'module'
>;
export const ArchitectureNode = memo(function ArchitectureNode({ data }: NodeProps<ModuleNode>) {
  const { module, color, faded, active, count, recordedCount } = data;
  return (
    <div
      title={`${module.label} — ${module.summary}`}
      className={`module-card ${active ? 'active' : ''} ${faded ? 'faded' : ''}`}
      style={{ '--module-color': color } as CSSProperties}
    >
      <Handle type="target" position={Position.Top} />
      <div className="module-topline">
        <Box size={12} />
        <span>{module.kind === 'class' ? 'CLASS' : 'MODULE'}</span>
        <span
          className="module-count"
          title={
            recordedCount
              ? `${count} visible of ${recordedCount} recorded relationships`
              : 'Relationships not yet mapped; dependencies may exist'
          }
        >
          {recordedCount ? `${count} links` : 'Not mapped'}
        </span>
      </div>
      <div className="module-name">{module.label}</div>
      <div className="module-summary">{module.summary}</div>
      {active && (
        <span className="selected-indicator">
          <ArrowUpRight size={12} />
        </span>
      )}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});
export function LayerNode({
  data,
}: NodeProps<Node<{ label: string; index: number; count: number; color: string }, 'layer'>>) {
  return (
    <div className="layer-label" style={{ '--module-color': data.color } as CSSProperties}>
      <span className="layer-number">{String(data.index + 1).padStart(2, '0')}</span>
      <span>{data.label}</span>
      <span className="layer-count">{data.count} modules</span>
    </div>
  );
}
