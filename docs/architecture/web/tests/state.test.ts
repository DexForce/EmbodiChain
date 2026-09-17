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

import { describe, it, expect } from 'vitest';
import data from '../../generated/architecture.json';
import { buildCanvas } from '../src/graph';
import {
  parseState,
  serializeState,
  selectGraph,
  sourceUrl,
  documentationSourceUrl,
} from '../src/state';

describe('shareable exploration', () => {
  it('restores selection and distinguishes no relations from all relations', () => {
    const { state } = parseState('#view=task-program&node=atomic-engine&relations=&q=engine', data);
    expect(state).toEqual({
      viewId: 'task-program',
      nodeId: 'atomic-engine',
      focus: false,
      relations: [],
      query: 'engine',
    });
    expect(parseState(serializeState(state), data).state).toEqual(state);
  });
  it('recovers stale links without choosing a node outside the view', () => {
    const result = parseState('#view=unknown&node=step-clock&relations=made-up', data);
    expect(result.state.viewId).toBe('overview');
    expect(result.state.nodeId).toBeNull();
    expect(result.notices.length).toBeGreaterThan(0);
  });
  it('uses only selected relationships to highlight neighbours', () => {
    const { state } = parseState('#view=task-program&node=atomic-engine&relations=holds', data);
    const graph = selectGraph(data, state);
    expect(graph.focusedNodeIds.has('motion-generator')).toBe(true);
    expect(graph.focusedNodeIds.has('simulation-factory')).toBe(false);
    expect(graph.edges.map((e) => e.id)).toContain('engine-holds-motion');
    expect(graph.edges.map((e) => e.id)).not.toContain('factory-constructs-engine');
  });
  it('searches responsibility and source paths, with an explicit empty result', () => {
    const { state } = parseState('#view=overview&q=from input images', data);
    expect(selectGraph(data, state).matchedNodeIds).toEqual(new Set(['scene-generation']));
    expect(selectGraph(data, { ...state, query: 'no-match-xyz' }).matchedNodeIds.size).toBe(0);
  });
  it('restores local focus and keeps only incident edges after filtering', () => {
    const { state } = parseState(
      '#view=task-program&node=atomic-engine&focus=1&relations=holds',
      data,
    );
    expect(state.focus).toBe(true);
    expect(parseState(serializeState(state), data).state).toEqual(state);
    const graph = selectGraph(data, state);
    expect(graph.visibleNodes.map((n) => n.id).sort()).toEqual([
      'atomic-engine',
      'motion-generator',
      'semantic-executor',
    ]);
    expect(graph.visibleEdges.map((e) => e.id)).toEqual([
      'engine-holds-motion',
      'executor-holds-engine',
    ]);
    const empty = selectGraph(data, { ...state, relations: [] });
    expect(empty.visibleNodes.map((n) => n.id)).toEqual(['atomic-engine']);
    expect(empty.visibleEdges).toEqual([]);
    expect(parseState('#view=overview&focus=1', data).state.focus).toBe(false);
  });
  it('counts only displayed links on neighbour cards in a local diagram', () => {
    const { state } = parseState('#view=task-program&node=atomic-engine&focus=1', data);
    const canvas = buildCanvas(data, state, 2);
    const executor = canvas.nodes.find((n) => n.id === 'semantic-executor')!;
    expect(executor.data.count).toBe(3);
    expect(executor.data.recordedCount).toBeGreaterThan(3);
  });
  it('pins source and documentation evidence to the reviewed revision', () => {
    const p = { path: 'embodichain/cli/main.py', start_line: 24, end_line: 29 };
    expect(sourceUrl(data.revision, p)).toBe(
      `https://github.com/DexForce/EmbodiChain/blob/${data.revision}/embodichain/cli/main.py#L24-L29`,
    );
    expect(documentationSourceUrl(data.revision, 'overview/gym/env', '.md')).toContain(
      `/blob/${data.revision}/docs/source/overview/gym/env.md`,
    );
  });
});
