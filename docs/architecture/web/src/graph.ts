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
export const layerColors = ['#4b6b86', '#6c6589', '#957444', '#4c7976', '#7d6c64'];
export const CARD_WIDTH = 240;
export const CARD_HEIGHT = 128;
export function buildCanvas(
  data: ArchitectureSnapshot,
  state: ExplorerState,
  columns = 3,
): { nodes: Node[]; edges: Edge[] } {
  const graph = selectGraph(data, state);
  const nodes: Node[] = [];
  let y = 0;
  const nodeGroups = new Map<string, number>();
  const positions = new Map<string, { x: number; y: number }>();
  graph.view.groups.forEach((group, index) => {
    group.node_ids.forEach((id) => nodeGroups.set(id, index));
  });
  if (state.focus && state.nodeId) {
    const ordered = [...graph.visibleNodes].sort((a, b) =>
      a.id === state.nodeId ? -1 : b.id === state.nodeId ? 1 : 0,
    );
    ordered.forEach((node, index) =>
      positions.set(node.id, {
        x: 18 + (index % columns) * (CARD_WIDTH + 20),
        y: 18 + Math.floor(index / columns) * (CARD_HEIGHT + 52),
      }),
    );
  } else {
    graph.view.groups.forEach((group, index) => {
      const rows = Math.ceil(group.node_ids.length / columns);
      const height = 38 + rows * (CARD_HEIGHT + 20);
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
        style: { width: 16 + columns * (CARD_WIDTH + 20), height },
        selectable: false,
        draggable: false,
        connectable: false,
        zIndex: -1,
      });
      group.node_ids.forEach((id, i) =>
        positions.set(id, {
          x: 18 + (i % columns) * (CARD_WIDTH + 20),
          y: y + 38 + Math.floor(i / columns) * (CARD_HEIGHT + 20),
        }),
      );
      y += height + 24;
    });
  }
  for (const module of graph.visibleNodes) {
    const selected = module.id === state.nodeId;
    const count = graph.visibleEdges.filter(
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
        recordedCount: data.edges.filter(
          (e) =>
            graph.view.edge_ids.includes(e.id) &&
            (e.source === module.id || e.target === module.id),
        ).length,
      },
      selected,
      width: CARD_WIDTH,
      height: CARD_HEIGHT,
      zIndex: 2,
      ariaLabel: module.label,
    });
  }
  const edges: Edge[] = graph.visibleEdges.map((edge) => {
    const incident = edge.source === state.nodeId || edge.target === state.nodeId;
    const matches = graph.matchedNodeIds.has(edge.source) && graph.matchedNodeIds.has(edge.target);
    const color = relations[edge.relation].color;
    const opacity = !matches ? 0.03 : state.nodeId ? (incident ? 1 : 0.1) : 0.42;
    return {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: 'default',
      animated: false,
      style: { stroke: color, strokeWidth: incident ? 1.65 : 1, opacity },
      markerEnd: { type: MarkerType.ArrowClosed, color, width: 12, height: 12 },
      zIndex: incident ? 3 : 0,
      selectable: false,
    };
  });
  return { nodes, edges };
}
