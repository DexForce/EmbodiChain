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

import { useState, useMemo, useEffect, useCallback } from 'react';
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  MiniMap,
  Panel,
  useReactFlow,
  useViewport,
  applyNodeChanges,
  type Node,
  type NodeChange,
} from '@xyflow/react';
import {
  ArrowUpRight,
  Box,
  Check,
  ChevronDown,
  CircleHelp,
  GitBranch,
  Layers3,
  Link2,
  Maximize,
  Menu,
  Moon,
  Network,
  Plus,
  Minus,
  RotateCcw,
  Search,
  SlidersHorizontal,
  Sun,
  X,
} from 'lucide-react';
import snapshot from '../../preview.snapshot.json';
import { ArchitectureNode, LayerNode } from './ArchitectureNode';
import NodeDetails from './NodeDetails';
import { buildCanvas, layerColors } from './graph';
import { parseState, serializeState, selectGraph, relations } from './state';
import type { ExplorerState, ArchitectureSnapshot } from './types';
const data: ArchitectureSnapshot = snapshot;
const nodeTypes = { module: ArchitectureNode, layer: LayerNode };

function CanvasTools() {
  const flow = useReactFlow();
  const { zoom } = useViewport();
  return (
    <div className="canvas-tools">
      <button aria-label="缩小" onClick={() => flow.zoomOut({ duration: 180 })}>
        <Minus size={16} />
      </button>
      <span>{Math.round(zoom * 100)}%</span>
      <button aria-label="放大" onClick={() => flow.zoomIn({ duration: 180 })}>
        <Plus size={16} />
      </button>
      <i />
      <button aria-label="适应画布" onClick={() => flow.fitView({ padding: 0.07, duration: 350 })}>
        <Maximize size={15} />
      </button>
    </div>
  );
}
function FitOnView({ viewId }: { viewId: string }) {
  const flow = useReactFlow();
  useEffect(() => {
    const timer = setTimeout(() => {
      void flow.fitView({ padding: 0.07, duration: 300 });
    }, 50);
    return () => clearTimeout(timer);
  }, [viewId, flow]);
  return null;
}

function Explorer() {
  const initial = useMemo(() => parseState(location.hash, data), []);
  const [state, setState] = useState<ExplorerState>(initial.state);
  const [notice, setNotice] = useState(initial.notices.join(' '));
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    try {
      return localStorage.getItem('architecture-theme') === 'light' ? 'light' : 'dark';
    } catch {
      return 'dark';
    }
  });
  const [mobileNav, setMobileNav] = useState(false);
  const [help, setHelp] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const graph = useMemo(() => selectGraph(data, state), [state]);
  const canvas = useMemo(() => buildCanvas(data, state), [state]);
  const [nodes, setNodes] = useState<Node[]>(canvas.nodes);
  const flow = useReactFlow();
  useEffect(() => setNodes(canvas.nodes), [canvas.nodes]);
  useEffect(() => {
    const update = () => {
      const parsed = parseState(location.hash, data);
      setState(parsed.state);
      setNotice(parsed.notices.join(' '));
    };
    addEventListener('popstate', update);
    addEventListener('hashchange', update);
    return () => {
      removeEventListener('popstate', update);
      removeEventListener('hashchange', update);
    };
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem('architecture-theme', theme);
    } catch {
      /* Browser storage may be unavailable. */
    }
  }, [theme]);
  const navigate = useCallback((patch: Partial<ExplorerState>, replace = false) => {
    setState((previous) => {
      const next = { ...previous, ...patch };
      const hash = serializeState(next);
      if (location.hash !== hash) history[replace ? 'replaceState' : 'pushState'](null, '', hash);
      return next;
    });
    setNotice('');
  }, []);
  const selectNode = useCallback(
    (id: string) => {
      navigate({ nodeId: id });
      setMobileNav(false);
    },
    [navigate],
  );
  const switchView = (id: string) => {
    navigate({ viewId: id, nodeId: null, query: '' });
    setMobileNav(false);
  };
  const selected = data.nodes.find((n) => n.id === state.nodeId);
  const selectedGroup = graph.view.groups.findIndex((g) => g.node_ids.includes(state.nodeId ?? ''));
  const usedRelations = Object.keys(relations).filter((type) =>
    data.edges.some((e) => graph.view.edge_ids.includes(e.id) && e.relation === type),
  );
  const hasQuery = state.query.trim().length > 0;
  const share = async () => {
    try {
      await navigator.clipboard.writeText(location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setNotice('请复制浏览器地址分享当前视图。');
    }
  };
  return (
    <div className="app" data-theme={theme}>
      <header className="app-header">
        <a
          className="brand"
          href="#"
          onClick={(event) => {
            event.preventDefault();
            navigate({
              viewId: 'overview',
              nodeId: null,
              query: '',
              relations: Object.keys(relations),
            });
          }}
          aria-label="EmbodiChain 首页"
        >
          <span className="brand-mark">
            <span />
            <span />
            <span />
          </span>
          <span>
            Embodi<span className="brand-light">Chain</span>
          </span>
        </a>
        <span className="header-divider" />
        <span className="product-label">Architecture Explorer</span>
        <div className="header-right">
          <span className="preview-badge">
            交互预览 <span>01</span>
          </span>
          <a
            className="revision-badge"
            href={`https://github.com/DexForce/EmbodiChain/tree/${data.revision}`}
            target="_blank"
            rel="noreferrer"
          >
            <GitBranch size={13} />
            {data.source_ref}
            <span>{data.revision.slice(0, 8)}</span>
          </a>
          <button
            className="icon-button"
            aria-label={theme === 'dark' ? '切换浅色主题' : '切换深色主题'}
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          >
            {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
          </button>
          <button className="share-button" onClick={share}>
            {copied ? <Check size={14} /> : <Link2 size={14} />}
            <span>{copied ? '已复制' : '分享视图'}</span>
          </button>
        </div>
      </header>
      <div className="workspace">
        <aside className={`sidebar ${mobileNav ? 'mobile-open' : ''}`} aria-label="模块导航">
          <div className="sidebar-heading">
            <span>工作空间</span>
            <button
              className="mobile-only icon-button"
              aria-label="关闭模块列表"
              onClick={() => setMobileNav(false)}
            >
              <X size={15} />
            </button>
            <span className="small-label desktop-only">EXPLORER</span>
          </div>
          <nav className="view-nav" aria-label="架构视图">
            {data.views.map((view) => (
              <button
                key={view.id}
                aria-label={view.label}
                onClick={() => switchView(view.id)}
                className={state.viewId === view.id ? 'current' : ''}
                aria-current={state.viewId === view.id ? 'page' : undefined}
              >
                {view.id === 'overview' ? <Layers3 size={16} /> : <Network size={16} />}
                <span>{view.label}</span>
                <span className="view-count">{view.node_ids.length}</span>
              </button>
            ))}
          </nav>
          <div className="sidebar-separator" />
          <div className="module-index-header">
            <span>模块目录</span>
            <span>{hasQuery ? graph.matchedNodeIds.size : graph.nodes.length}</span>
          </div>
          <div className="module-index">
            {graph.view.groups.map((group, index) => (
              <section className="index-group" key={group.id}>
                <div className="index-group-title">
                  <span style={{ background: layerColors[index % layerColors.length] }} />
                  {group.label}
                  <span className="index-count">{group.node_ids.length}</span>
                </div>
                {group.node_ids
                  .filter((id) => graph.matchedNodeIds.has(id))
                  .map((id) => {
                    const node = data.nodes.find((n) => n.id === id)!;
                    return (
                      <button
                        key={id}
                        aria-label={`选择 ${node.label}`}
                        aria-pressed={state.nodeId === id}
                        title={node.label}
                        className={`index-node ${state.nodeId === id ? 'active' : ''}`}
                        onClick={() => selectNode(id)}
                      >
                        <Box size={11} />
                        <span>{node.label}</span>
                        {state.nodeId === id && <span className="active-dot" />}
                      </button>
                    );
                  })}
              </section>
            ))}
          </div>
          <div className="sidebar-footer">
            <span className="status-dot" />
            <div>
              源码静态快照<small>EmbodiChain v{data.package_version}</small>
            </div>
            <span className="readonly-label">只读</span>
          </div>
        </aside>
        <main className="main-area">
          <div className="view-heading">
            <div>
              <div className="breadcrumb">
                EmbodiChain <span>/</span> {graph.view.label}
              </div>
              <h1 aria-label="架构浏览器">
                架构浏览器 <span>{state.viewId === 'overview' ? '系统总览' : '任务编排'}</span>
              </h1>
              <p>{graph.view.description}</p>
            </div>
            <div className="view-metrics">
              <div>
                <strong>{graph.nodes.length}</strong>
                <span>模块</span>
              </div>
              <i />
              <div>
                <strong>{graph.edges.length}</strong>
                <span>关系</span>
              </div>
              <i />
              <div>
                <strong>{graph.view.groups.length}</strong>
                <span>分层</span>
              </div>
            </div>
          </div>
          <div className="toolbar">
            <button
              className="mobile-only icon-button"
              aria-label="显示模块列表"
              onClick={() => setMobileNav(true)}
            >
              <Menu size={18} />
            </button>
            <label className="search-field">
              <Search size={15} />
              <input
                type="search"
                aria-label="搜索模块"
                placeholder="搜索模块、职责或源码…"
                value={state.query}
                onChange={(event) => navigate({ query: event.target.value }, true)}
              />
              {state.query ? (
                <button aria-label="清空搜索" onClick={() => navigate({ query: '' }, true)}>
                  <X size={13} />
                </button>
              ) : (
                <kbd>⌕</kbd>
              )}
            </label>
            <span className="toolbar-divider" />
            <button
              className={`filter-trigger ${showFilters ? 'active' : ''}`}
              onClick={() => setShowFilters(!showFilters)}
              aria-expanded={showFilters}
            >
              <SlidersHorizontal size={14} />
              <span>关系类型</span>
              <ChevronDown size={12} />
            </button>
            <div className="inline-legend">
              {usedRelations.slice(0, 4).map((type) => (
                <span key={type}>
                  <i style={{ background: relations[type].color }} />
                  {relations[type].label}
                </span>
              ))}
            </div>
            <button
              className="reset-button"
              onClick={() => {
                navigate({ nodeId: null, query: '', relations: Object.keys(relations) });
                void flow.fitView({ padding: 0.07, duration: 300 });
              }}
            >
              <RotateCcw size={13} />
              <span>重置</span>
            </button>
            <button className="icon-button" aria-label="使用帮助" onClick={() => setHelp(!help)}>
              <CircleHelp size={16} />
            </button>
          </div>
          {showFilters && (
            <div className="filter-menu">
              <div className="filter-menu-title">
                显示关系{' '}
                <button
                  onClick={() =>
                    navigate({ relations: state.relations.length ? [] : Object.keys(relations) })
                  }
                  aria-label={state.relations.length ? '清空关系筛选' : '显示所有关系'}
                >
                  {state.relations.length ? '清空' : '全选'}
                </button>
              </div>
              {usedRelations.map((type) => (
                <button
                  key={type}
                  className={state.relations.includes(type) ? 'enabled' : ''}
                  aria-pressed={state.relations.includes(type)}
                  onClick={() =>
                    navigate({
                      relations: state.relations.includes(type)
                        ? state.relations.filter((r) => r !== type)
                        : [...state.relations, type],
                    })
                  }
                >
                  <span className="filter-checkbox">
                    {state.relations.includes(type) && <Check size={11} />}
                  </span>
                  <i style={{ background: relations[type].color }} />
                  {relations[type].label}
                  <code>{type}</code>
                </button>
              ))}
            </div>
          )}
          {notice && (
            <div className="notice" role="status">
              {notice}
              <button aria-label="关闭提示" onClick={() => setNotice('')}>
                <X size={13} />
              </button>
            </div>
          )}
          <div className="canvas-shell">
            <ReactFlow
              nodes={nodes}
              edges={canvas.edges}
              nodeTypes={nodeTypes}
              onNodesChange={useCallback(
                (changes: NodeChange[]) => setNodes((old) => applyNodeChanges(changes, old)),
                [],
              )}
              onNodeClick={(_, node) => {
                if (node.type === 'module') selectNode(node.id);
              }}
              onNodeDoubleClick={(_, node) => {
                if (node.type === 'module') {
                  void flow.setCenter(node.position.x + 96, node.position.y + 55, {
                    zoom: 1.15,
                    duration: 300,
                  });
                }
              }}
              onPaneClick={() => {
                setShowFilters(false);
                setHelp(false);
              }}
              fitView
              fitViewOptions={{ padding: 0.07 }}
              minZoom={0.25}
              maxZoom={1.8}
              nodesConnectable={false}
              deleteKeyCode={null}
              colorMode={theme}
              onlyRenderVisibleElements={false}
              proOptions={{ hideAttribution: false }}
            >
              <Background
                variant={BackgroundVariant.Dots}
                gap={20}
                size={1}
                color={theme === 'dark' ? '#27313b' : '#c3cdd2'}
              />
              <FitOnView viewId={state.viewId} />
              <Panel position="bottom-left">
                <CanvasTools />
              </Panel>
              <MiniMap
                pannable
                zoomable
                nodeColor={(node) =>
                  node.type === 'layer' ? 'transparent' : (node.data.color as string)
                }
                nodeStrokeColor="transparent"
                maskColor={theme === 'dark' ? '#090e1455' : '#dce5e555'}
                position="bottom-right"
              />
              <Panel position="top-left">
                <div className="canvas-caption">
                  <span className="status-dot" />
                  {state.nodeId ? '关联聚焦' : '空间拓扑'}
                  <span>·</span>
                  {state.nodeId ? selected?.label : '模块按职责分层'}
                </div>
              </Panel>
            </ReactFlow>
            {hasQuery && graph.matchedNodeIds.size === 0 && (
              <div className="empty-search">
                <Search size={23} />
                <strong>没有找到匹配的模块</strong>
                <span>试试 Robot、规划或源码路径</span>
                <button onClick={() => navigate({ query: '' }, true)}>清空搜索</button>
              </div>
            )}
            {help && (
              <div className="help-popover">
                <strong>探索架构</strong>
                <p>点击模块查看职责与源码证据。</p>
                <p>拖动画布平移，滚轮缩放；左侧目录支持键盘选择。</p>
                <p>通过关系筛选聚焦依赖，分享链接保留当前选择。</p>
                <button onClick={() => setHelp(false)}>知道了</button>
              </div>
            )}
          </div>
          <div className="canvas-footer">
            <span>
              <MouseIcon /> 双击模块放大 · 拖动平移 · 滚轮缩放
            </span>
            <span>
              静态关系，不代表运行时全貌 <ArrowUpRight size={11} />
            </span>
          </div>
        </main>
        <NodeDetails
          data={data}
          node={selected}
          view={graph.view}
          edges={graph.edges}
          color={layerColors[Math.max(0, selectedGroup) % layerColors.length]}
          onSelect={selectNode}
          onClose={() => navigate({ nodeId: null })}
        />
      </div>
      <div className="sr-only" aria-live="polite">
        {selected ? `已选择 ${selected.label}` : ''}
      </div>
    </div>
  );
}
function MouseIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor">
      <path d="m3 2 9 6-4 1-2 4Z" />
    </svg>
  );
}
export default function App() {
  return (
    <ReactFlowProvider>
      <Explorer />
    </ReactFlowProvider>
  );
}
