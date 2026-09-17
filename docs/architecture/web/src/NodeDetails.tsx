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

import {
  ArrowDownLeft,
  ArrowUpRight,
  BookOpen,
  Check,
  Code2,
  ExternalLink,
  FileCode2,
  MousePointer2,
  ShieldCheck,
  X,
} from 'lucide-react';
import { documentationUrl, resolveDocsRoot } from './links';
import { relations, sourceUrl, documentationSourceUrl } from './state';
import type {
  ArchitectureSnapshot,
  ArchitectureNode,
  ArchitectureEdge,
  ArchitectureView,
} from './types';

export default function NodeDetails({
  data,
  node,
  edges,
  view,
  color,
  onSelect,
  onClose,
}: {
  data: ArchitectureSnapshot;
  node: ArchitectureNode | undefined;
  edges: ArchitectureEdge[];
  view: ArchitectureView;
  color: string;
  onSelect: (id: string) => void;
  onClose: () => void;
}) {
  const docsRoot = resolveDocsRoot(new URL(location.href));
  const linked = edges.filter((e) => e.source === node?.id || e.target === node?.id);
  const allCount = data.edges.filter(
    (e) => view.edge_ids.includes(e.id) && (e.source === node?.id || e.target === node?.id),
  ).length;
  return (
    <aside className={`inspector ${node ? 'has-selection' : ''}`} aria-label="Node details">
      <div className="inspector-top">
        <span>
          <Code2 size={14} /> {node ? 'Module details' : 'Reader’s guide'}
        </span>
        {node ? (
          <button className="icon-button" onClick={onClose} aria-label="Close details">
            <X size={15} />
          </button>
        ) : (
          <span className="small-label">INSPECTOR</span>
        )}
      </div>
      <div className="inspector-scroll">
        {!node ? (
          <>
            <div className="eyebrow">ARCHITECTURE NOTES</div>
            <h2 className="welcome-title">From modules to systems.</h2>
            <p className="welcome-copy">
              Select a module to examine its role in EmbodiChain and follow relationships grounded
              in source code.
            </p>
            <div className="guide-steps">
              <div>
                <span>01</span>
                <p>
                  <strong>Locate a module</strong>Select a node in the diagram or module index.
                </p>
              </div>
              <div>
                <span>02</span>
                <p>
                  <strong>Trace its relationships</strong>Inspect calls, construction, and
                  ownership.
                </p>
              </div>
              <div>
                <span>03</span>
                <p>
                  <strong>Examine the evidence</strong>Each relationship cites a pinned source
                  revision.
                </p>
              </div>
            </div>
            <div className="note-card">
              <ShieldCheck size={16} />
              <div>
                <strong>Static source snapshot</strong>
                <p>
                  Relationships describe code structure, not execution order or complete runtime
                  behaviour.
                </p>
                <code>
                  {data.source_ref} · {data.revision.slice(0, 8)}
                </code>
              </div>
            </div>
            <div className="inspector-caption">
              <MousePointer2 size={13} /> Select any module to begin
            </div>
          </>
        ) : (
          <>
            <div className="node-kind" style={{ color }}>
              <span className="tiny-dot" style={{ background: color }} />
              {node.kind === 'class' ? 'Core object' : 'Module'}
              <span>{node.kind.toUpperCase()}</span>
            </div>
            <h2 className="node-title">{node.label}</h2>
            <p className="node-description">{node.summary}</p>
            <a
              className="source-path"
              href={sourceUrl(data.revision, node.evidence[0])}
              target="_blank"
              rel="noreferrer"
            >
              <FileCode2 size={14} />
              <span>{node.evidence[0].path}</span>
              <ExternalLink size={12} />
            </a>
            <a
              className="source-button"
              href={sourceUrl(data.revision, node.evidence[0])}
              target="_blank"
              rel="noreferrer"
            >
              View source <ArrowUpRight size={14} />
            </a>
            <section className="boundary">
              <div className="section-title">
                <ShieldCheck size={14} /> Responsibility boundary
              </div>
              {node.boundaries.map((text) => (
                <p key={text}>{text}</p>
              ))}
            </section>
            {node.topic_ids.length > 0 && (
              <div className="topic-tags">
                {node.topic_ids.map((t) => (
                  <span key={t}>{t}</span>
                ))}
              </div>
            )}
            <p className="coverage-note">
              Curated source sample. Relationship counts indicate recorded evidence, not complete
              dependency coverage.
            </p>
            <section className="relations-section">
              <div className="section-title">
                Direct relationships{' '}
                <span>
                  {linked.length} / {allCount}
                </span>
              </div>
              {!linked.length && (
                <div className="empty-relations">
                  {allCount
                    ? 'No direct relationships match these filters.'
                    : 'Relationships not yet mapped in this view.'}
                  <br />
                  {allCount
                    ? 'Adjust relationship types in the toolbar.'
                    : 'This does not mean the module has no dependencies.'}
                </div>
              )}
              {linked.map((edge) => {
                const outgoing = edge.source === node.id;
                const other = data.nodes.find(
                  (n) => n.id === (outgoing ? edge.target : edge.source),
                )!;
                const meta = relations[edge.relation];
                return (
                  <article
                    className="relation-card"
                    key={edge.id}
                    style={{ borderLeftColor: meta.color }}
                  >
                    <button
                      onClick={() => onSelect(other.id)}
                      aria-label={
                        outgoing
                          ? `${meta.verb} ${other.label}`
                          : `Incoming: ${other.label} ${meta.verb.toLowerCase()} this module`
                      }
                      className="relation-link"
                    >
                      <span className="relation-direction" style={{ color: meta.color }}>
                        {outgoing ? <ArrowUpRight size={14} /> : <ArrowDownLeft size={14} />}{' '}
                        {outgoing ? 'OUT' : 'IN'}
                      </span>
                      <strong>{other.label}</strong>
                      <span className="relation-type">{meta.label}</span>
                    </button>
                    <p>{edge.description}</p>
                    <details>
                      <summary>
                        <Code2 size={12} /> Source evidence{' '}
                        <span>
                          {edge.provenance === 'static-extracted'
                            ? 'Static extraction'
                            : 'Source reviewed'}
                        </span>
                      </summary>
                      <div className="evidence-detail">
                        <p>{edge.scope}</p>
                        {edge.evidence.map((proof, index) => (
                          <div key={index}>
                            <a
                              href={sourceUrl(data.revision, proof)}
                              target="_blank"
                              rel="noreferrer"
                            >
                              {proof.symbol} · L{proof.start_line}
                              <ExternalLink size={11} />
                            </a>
                            <pre>{proof.excerpt}</pre>
                          </div>
                        ))}
                      </div>
                    </details>
                  </article>
                );
              })}
            </section>
            {node.documentation.length > 0 && (
              <section className="documentation-links">
                <div className="section-title">
                  <BookOpen size={14} /> {docsRoot ? 'Documentation' : 'Documentation source'}
                </div>
                {node.documentation.map((doc) => (
                  <a
                    key={doc.docname}
                    href={
                      docsRoot
                        ? documentationUrl(doc.docname, docsRoot)
                        : documentationSourceUrl(
                            data.revision,
                            doc.docname,
                            doc.docname.startsWith('api_reference/') ? '.rst' : '.md',
                          )
                    }
                    target={docsRoot ? '_parent' : '_blank'}
                    rel="noreferrer"
                  >
                    {doc.label}
                    <ArrowUpRight size={13} />
                  </a>
                ))}
              </section>
            )}
            <div className="evidence-footer">
              <Check size={12} /> Source revision <code>{data.revision.slice(0, 8)}</code>
            </div>
          </>
        )}
      </div>
    </aside>
  );
}
