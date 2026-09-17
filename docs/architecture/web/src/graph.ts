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

import type { Edge, Node } from '@xyflow/react';
import { MarkerType } from '@xyflow/react';
import { relations, selectGraph } from './state';
import type { ArchitectureSnapshot, ExplorerState } from './types';
export const layerColors = ['#68d7b7', '#91a5ec', '#dcb574', '#66bcd1', '#ac9ccb'];
export function buildCanvas(
  data: ArchitectureSnapshot,
  state: ExplorerState,
): { nodes: Node[]; edges: Edge[] } {
  const graph = selectGraph(data, state);
  const nodes: Node[] = [];
  let y = 0;
  const nodeGroups = new Map<string, number>();
  const positions = new Map<string, { x: number; y: number }>();
  graph.view.groups.forEach((group, index) => {
    const rows = Math.ceil(group.node_ids.length / 5);
    const height = 34 + rows * 124 - 12 + 10;
    nodes.push({
      id: 'layer-' + group.id,
      type: 'layer',
      position: { x: 0, y },
      data: {
        label: group.label,
        index,
        count: group.node_ids.length,
        color: layerColors[index % layerColors.length],
      },
      style: { width: 1060, height },
      selectable: false,
      draggable: false,
      connectable: false,
      zIndex: -1,
    });
    group.node_ids.forEach((id, i) => {
      nodeGroups.set(id, index);
      positions.set(id, { x: 18 + (i % 5) * 208, y: y + 34 + Math.floor(i / 5) * 124 });
    });
    y += height + 14;
  });
  for (const module of graph.nodes) {
    const selected = module.id === state.nodeId;
    const count = graph.edges.filter(
      (e) => e.source === module.id || e.target === module.id,
    ).length;
    nodes.push({
      id: module.id,
      type: 'module',
      position: positions.get(module.id)!,
      data: {
        module,
        color: layerColors[nodeGroups.get(module.id)! % layerColors.length],
        active: selected,
        faded: !graph.focusedNodeIds.has(module.id) || !graph.matchedNodeIds.has(module.id),
        count,
      },
      selected,
      width: 192,
      height: 110,
      zIndex: 2,
      ariaLabel: module.label,
    });
  }
  const edges: Edge[] = graph.edges.map((edge) => {
    const incident = edge.source === state.nodeId || edge.target === state.nodeId;
    const matches = graph.matchedNodeIds.has(edge.source) && graph.matchedNodeIds.has(edge.target);
    const color = relations[edge.relation].color;
    const opacity = !matches ? 0.03 : state.nodeId ? (incident ? 0.88 : 0.035) : 0.19;
    return {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: 'default',
      animated: incident,
      style: { stroke: color, strokeWidth: incident ? 1.65 : 1, opacity },
      markerEnd: { type: MarkerType.ArrowClosed, color, width: 12, height: 12 },
      zIndex: incident ? 3 : 0,
      selectable: false,
    };
  });
  return { nodes, edges };
}
