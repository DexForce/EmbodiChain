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

import type { ArchitectureSnapshot, Evidence, ExplorerState } from './types';

export const relations: Record<string, { label: string; color: string; verb: string }> = {
  calls: { label: '调用', color: '#76b6f5', verb: '调用' },
  constructs: { label: '构造', color: '#e5b36c', verb: '构造' },
  holds: { label: '持有', color: '#b69be5', verb: '持有' },
  reads: { label: '读取', color: '#65cdb4', verb: '读取' },
  writes: { label: '写入', color: '#53b7b7', verb: '写入' },
  produces: { label: '产出', color: '#91bd75', verb: '产出' },
  inherits: { label: '继承', color: '#8b9fea', verb: '继承' },
  imports: { label: '导入', color: '#8895ab', verb: '导入' },
  configures: { label: '配置', color: '#da96b0', verb: '配置' },
  implements: { label: '实现', color: '#8b9fea', verb: '实现' },
};

export function parseState(
  hash: string,
  data: ArchitectureSnapshot,
): { state: ExplorerState; notices: string[] } {
  const params = new URLSearchParams(hash.replace(/^#/, ''));
  const notices: string[] = [];
  const requestedView = params.get('view');
  const view = data.views.find((v) => v.id === requestedView) ?? data.views[0];
  if (requestedView && requestedView !== view.id) notices.push('该视图已不可用，已返回全局架构。');
  const requestedNode = params.get('node');
  const nodeId = requestedNode && view.node_ids.includes(requestedNode) ? requestedNode : null;
  if (requestedNode && !nodeId) notices.push('该模块不在当前视图中，已清除选择。');
  const available = Object.keys(relations);
  const selected = params.has('relations')
    ? params
        .get('relations')!
        .split(',')
        .filter((r) => available.includes(r))
    : available;
  return {
    state: {
      viewId: view.id,
      nodeId,
      relations: [...new Set(selected)],
      query: params.get('q') ?? '',
    },
    notices,
  };
}
export function serializeState(state: ExplorerState): string {
  const p = new URLSearchParams({ view: state.viewId });
  if (state.nodeId) p.set('node', state.nodeId);
  p.set('relations', state.relations.join(','));
  if (state.query) p.set('q', state.query);
  return '#' + p.toString();
}
export function selectGraph(data: ArchitectureSnapshot, state: ExplorerState) {
  const view = data.views.find((v) => v.id === state.viewId) ?? data.views[0];
  const nodes = data.nodes.filter((n) => view.node_ids.includes(n.id));
  const edges = data.edges.filter(
    (e) => view.edge_ids.includes(e.id) && state.relations.includes(e.relation),
  );
  const query = state.query.trim().toLocaleLowerCase();
  const matchedNodeIds = new Set(
    nodes
      .filter((n) =>
        [n.label, n.summary, ...n.topic_ids, ...n.evidence.map((e) => e.path)]
          .join(' ')
          .toLocaleLowerCase()
          .includes(query),
      )
      .map((n) => n.id),
  );
  const incident = edges.filter((e) => e.source === state.nodeId || e.target === state.nodeId);
  const focusedNodeIds = state.nodeId
    ? new Set([state.nodeId, ...incident.flatMap((e) => [e.source, e.target])])
    : new Set(nodes.map((n) => n.id));
  return { view, nodes, edges, matchedNodeIds, focusedNodeIds };
}
export function sourceUrl(
  revision: string,
  proof: Pick<Evidence, 'path' | 'start_line' | 'end_line'>,
): string {
  return `https://github.com/DexForce/EmbodiChain/blob/${revision}/${proof.path}#L${proof.start_line}-L${proof.end_line}`;
}
export function documentationSourceUrl(
  revision: string,
  docname: string,
  extension: string,
): string {
  return `https://github.com/DexForce/EmbodiChain/blob/${revision}/docs/source/${docname}${extension}`;
}
