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
  Layers3,
  MousePointer2,
  ShieldCheck,
  X,
} from 'lucide-react';
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
  const linked = edges.filter((e) => e.source === node?.id || e.target === node?.id);
  const allCount = data.edges.filter(
    (e) => view.edge_ids.includes(e.id) && (e.source === node?.id || e.target === node?.id),
  ).length;
  return (
    <aside className={`inspector ${node ? 'has-selection' : ''}`} aria-label="节点详情">
      <div className="inspector-top">
        <span>
          <Code2 size={14} /> {node ? '模块详情' : '探索指南'}
        </span>
        {node ? (
          <button className="icon-button" onClick={onClose} aria-label="关闭详情">
            <X size={15} />
          </button>
        ) : (
          <span className="small-label">INSPECTOR</span>
        )}
      </div>
      <div className="inspector-scroll">
        {!node ? (
          <>
            <div className="welcome-art" aria-hidden="true">
              <div />
              <div />
              <div />
              <span>
                <Layers3 size={28} />
              </span>
            </div>
            <div className="eyebrow">SEE THE CONNECTIONS</div>
            <h2 className="welcome-title">从模块，理解系统。</h2>
            <p className="welcome-copy">
              选择一个模块，沿着真实的源码关系，探索它在 EmbodiChain 中的位置。
            </p>
            <div className="guide-steps">
              <div>
                <span>01</span>
                <p>
                  <strong>找到关注的模块</strong>通过画布或左侧目录选择节点
                </p>
              </div>
              <div>
                <span>02</span>
                <p>
                  <strong>沿着关系继续探索</strong>查看调用、构造与持有关系
                </p>
              </div>
              <div>
                <span>03</span>
                <p>
                  <strong>回到源码验证</strong>每条关系都有固定版本的证据
                </p>
              </div>
            </div>
            <div className="note-card">
              <ShieldCheck size={16} />
              <div>
                <strong>源码静态快照</strong>
                <p>关系描述代码结构，不代表实际执行顺序或完整运行时行为。</p>
                <code>
                  {data.source_ref} · {data.revision.slice(0, 8)}
                </code>
              </div>
            </div>
            <div className="inspector-caption">
              <MousePointer2 size={13} /> 点击画布中的任意模块开始
            </div>
          </>
        ) : (
          <>
            <div className="node-kind" style={{ color }}>
              <span className="tiny-dot" style={{ background: color }} />
              {node.kind === 'class' ? '核心对象' : '功能模块'}
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
              查看源码 <ArrowUpRight size={14} />
            </a>
            <section className="boundary">
              <div className="section-title">
                <ShieldCheck size={14} /> 职责边界
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
            <section className="relations-section">
              <div className="section-title">
                直接关系{' '}
                <span>
                  {linked.length} / {allCount}
                </span>
              </div>
              {!linked.length && (
                <div className="empty-relations">
                  当前筛选下没有直接关系。
                  <br />
                  可在工具栏调整关系类型。
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
                        outgoing ? `${meta.verb} ${other.label}` : `被 ${other.label} ${meta.verb}`
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
                        <Code2 size={12} /> 源码证据{' '}
                        <span>
                          {edge.provenance === 'static-extracted' ? '静态提取' : '源码核对'}
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
                  <BookOpen size={14} /> 文档源码
                </div>
                {node.documentation.map((doc) => (
                  <a
                    key={doc.docname}
                    href={documentationSourceUrl(
                      data.revision,
                      doc.docname,
                      doc.docname.startsWith('api_reference/') ? '.rst' : '.md',
                    )}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {doc.label}
                    <ArrowUpRight size={13} />
                  </a>
                ))}
              </section>
            )}
            <div className="evidence-footer">
              <Check size={12} /> 源码版本 <code>{data.revision.slice(0, 8)}</code>
            </div>
          </>
        )}
      </div>
    </aside>
  );
}
