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

import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { describe, it, expect } from 'vitest';
import Ajv2020 from 'ajv/dist/2020';
import { parse } from 'yaml';
import data from '../../preview.snapshot.json';
import schema from '../../architecture.schema.json';
const root = fileURLToPath(new URL('../../../../', import.meta.url));
const read = (path: string) =>
  execFileSync('git', ['show', `${data.revision}:${path}`], { cwd: root, encoding: 'utf8' });

describe('reviewed preview data', () => {
  it('conforms to the architecture contract', () => {
    const validate = new Ajv2020({ allErrors: true }).compile(schema);
    expect(validate(data), JSON.stringify(validate.errors)).toBe(true);
  });
  it('has no broken graph, group, topic, or documentation references', () => {
    const nodes = new Set(data.nodes.map((n) => n.id));
    const edges = new Map(data.edges.map((e) => [e.id, e]));
    const topics = new Set(parse(read(data.topic_index)).topics.map((t: { id: string }) => t.id));
    expect(nodes.size).toBe(data.nodes.length);
    expect(edges.size).toBe(data.edges.length);
    for (const edge of data.edges) {
      expect(nodes.has(edge.source)).toBe(true);
      expect(nodes.has(edge.target)).toBe(true);
    }
    for (const node of data.nodes) {
      for (const topic of node.topic_ids) expect(topics.has(topic)).toBe(true);
      for (const doc of node.documentation) {
        const extension = doc.docname.startsWith('api_reference/') ? '.rst' : '.md';
        expect(read(`docs/source/${doc.docname}${extension}`).length).toBeGreaterThan(0);
      }
    }
    for (const view of data.views) {
      const members = view.groups.flatMap((g) => g.node_ids);
      expect(new Set(members).size).toBe(members.length);
      expect(new Set(members)).toEqual(new Set(view.node_ids));
      for (const id of view.edge_ids) {
        expect(edges.has(id)).toBe(true);
        expect(view.node_ids).toContain(edges.get(id)!.source);
        expect(view.node_ids).toContain(edges.get(id)!.target);
      }
    }
  });
  it('keeps every evidence excerpt accurate at the linked commit', () => {
    const cache = new Map<string, string>();
    for (const item of [...data.nodes, ...data.edges])
      for (const proof of item.evidence) {
        if (!cache.has(proof.path)) cache.set(proof.path, read(proof.path));
        expect(
          cache
            .get(proof.path)!
            .split('\n')
            .slice(proof.start_line - 1, proof.end_line)
            .join('\n'),
          `${item.id}: ${proof.path}:${proof.start_line}`,
        ).toBe(proof.excerpt);
      }
  });
});
