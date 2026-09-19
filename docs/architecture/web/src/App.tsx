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

import { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import {
  ReactFlow,
  ReactFlowProvider,
  MiniMap,
  Panel,
  useReactFlow,
  useViewport,
  useStore,
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
import { ArchitectureNode, LayerNode } from './ArchitectureNode';
import NodeDetails from './NodeDetails';
import { useDocumentTheme } from './useDocumentTheme';
import { buildCanvas, layerColors, CARD_WIDTH, CARD_HEIGHT } from './graph';
import { parseState, serializeState, selectGraph, relations } from './state';
import type { ExplorerState, ArchitectureSnapshot } from './types';
const nodeTypes = { module: ArchitectureNode, layer: LayerNode };

function CanvasTools() {
  const flow = useReactFlow();
  const { zoom } = useViewport();
  return (
    <div className="canvas-tools">
      <button aria-label="Zoom out" onClick={() => flow.zoomOut({ duration: 180 })}>
        <Minus size={16} />
      </button>
      <button
        className="reading-size"
        aria-label="Read at 100 percent"
        title="Read at 100%"
        onClick={() => flow.zoomTo(1, { duration: 180 })}
      >
        {Math.round(zoom * 100)}%
      </button>
      <button aria-label="Zoom in" onClick={() => flow.zoomIn({ duration: 180 })}>
        <Plus size={16} />
      </button>
      <i />
      <button
        aria-label="Fit to view"
        onClick={() => flow.fitView({ padding: 0.07, duration: 350 })}
      >
        <Maximize size={15} />
      </button>
    </div>
  );
}
function ReadingViewport({
  layoutKey,
  nodeId,
  focus,
}: {
  layoutKey: string;
  nodeId: string | null;
  focus: boolean;
}) {
  const flow = useReactFlow();
  const width = useStore((store) => store.width);
  const height = useStore((store) => store.height);
  useEffect(() => {
    const timer = setTimeout(() => {
      if (focus) {
        const bounds = flow.getNodesBounds(flow.getNodes());
        // Centre small diagrams; top-align large ones so the selected first card stays visible.
        void flow.setViewport(
          {
            x: Math.max(24, (width - bounds.width) / 2) - bounds.x,
            y: Math.max(24, (height - bounds.height) / 2) - bounds.y,
            zoom: 1,
          },
          { duration: 180 },
        );
      } else {
        const node = nodeId ? flow.getNode(nodeId) : undefined;
        if (node)
          void flow.setCenter(node.position.x + CARD_WIDTH / 2, node.position.y + CARD_HEIGHT / 2, {
            zoom: 1,
            duration: 180,
          });
        else void flow.setViewport({ x: 24, y: 24, zoom: 1 }, { duration: 180 });
      }
    }, 50);
    return () => clearTimeout(timer);
  }, [layoutKey, nodeId, focus, flow, width, height]);
  return null;
}

function Explorer({ data }: { data: ArchitectureSnapshot }) {
  const initial = useMemo(() => parseState(location.hash, data), []);
  const [state, setState] = useState<ExplorerState>(initial.state);
  const [notice, setNotice] = useState(initial.notices.join(' '));
  const { theme, toggleTheme, embedded, shareUrl } = useDocumentTheme(serializeState(state));
  const [mobileNav, setMobileNav] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [help, setHelp] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const graph = useMemo(() => selectGraph(data, state), [state]);
  const canvasElement = useRef<HTMLDivElement>(null);
  const [columns, setColumns] = useState(3);
  useEffect(() => {
    const element = canvasElement.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setColumns(
        Math.max(1, Math.min(5, Math.floor((entry.contentRect.width - 64) / (CARD_WIDTH + 20)))),
      );
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const canvas = useMemo(() => buildCanvas(data, state, columns), [state, columns]);
  const [nodes, setNodes] = useState<Node[]>(canvas.nodes);
  const flow = useReactFlow();
  useEffect(() => setNodes(canvas.nodes), [canvas.nodes]);
  useEffect(() => {
    const update = () => {
      const parsed = parseState(location.hash, data);
      setState(parsed.state);
      setInspectorOpen(true);
      setNotice(parsed.notices.join(' '));
    };
    addEventListener('popstate', update);
    addEventListener('hashchange', update);
    return () => {
      removeEventListener('popstate', update);
      removeEventListener('hashchange', update);
    };
  }, []);
  const navigate = useCallback((patch: Partial<ExplorerState>, replace = false) => {
    setState((previous) => {
      const next = { ...previous, ...patch };
      if (!next.nodeId) next.focus = false;
      const hash = serializeState(next);
      if (location.hash !== hash) history[replace ? 'replaceState' : 'pushState'](null, '', hash);
      return next;
    });
    setNotice('');
  }, []);
  const selectNode = useCallback(
    (id: string) => {
      navigate({ nodeId: id });
      setInspectorOpen(true);
      setMobileNav(false);
    },
    [navigate],
  );
  const switchView = (id: string) => {
    navigate({ viewId: id, nodeId: null, query: '', focus: false });
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
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setNotice('Copy the browser URL to share this view.');
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
          aria-label="EmbodiChain home"
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
            PREVIEW <span>01</span>
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
          {!embedded && (
            <button
              className="icon-button"
              aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
              onClick={toggleTheme}
            >
              {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
            </button>
          )}
          <button className="share-button" onClick={share}>
            {copied ? <Check size={14} /> : <Link2 size={14} />}
            <span>{copied ? 'Copied' : 'Share view'}</span>
          </button>
        </div>
      </header>
      <div className="workspace">
        <aside
          className={`sidebar ${mobileNav ? 'mobile-open' : ''}`}
          aria-label="Module navigation"
        >
          <div className="sidebar-heading">
            <span>CONTENTS</span>
            <button
              className="mobile-only icon-button"
              aria-label="Close module list"
              onClick={() => setMobileNav(false)}
            >
              <X size={15} />
            </button>
            <span className="small-label desktop-only">INDEX</span>
          </div>
          <nav className="view-nav" aria-label="Architecture views">
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
            <span>MODULE INDEX</span>
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
                        aria-label={`Select ${node.label}`}
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
              Static source snapshot<small>EmbodiChain v{data.package_version}</small>
            </div>
            <span className="readonly-label">READ ONLY</span>
          </div>
        </aside>
        <main className="main-area">
          <div className="view-heading">
            <div>
              <div className="breadcrumb">
                EmbodiChain <span>/</span> {graph.view.label}
              </div>
              <h1 aria-label="Architecture Explorer">
                Architecture Explorer{' '}
                <span>
                  FIG.{' '}
                  {String(data.views.findIndex((view) => view.id === state.viewId) + 1).padStart(
                    2,
                    '0',
                  )}
                </span>
              </h1>
              <p>{graph.view.description}</p>
            </div>
            <div className="view-metrics">
              <div>
                <strong>{graph.visibleNodes.length}</strong>
                <span>nodes</span>
              </div>
              <i />
              <div>
                <strong>{graph.visibleEdges.length}</strong>
                <span>edges</span>
              </div>
              <i />
              <div>
                <strong>
                  {
                    graph.view.groups.filter((group) =>
                      group.node_ids.some((id) =>
                        graph.visibleNodes.some((node) => node.id === id),
                      ),
                    ).length
                  }
                </strong>
                <span>layers</span>
              </div>
            </div>
          </div>
          <div className="toolbar">
            <button
              className="mobile-only icon-button"
              aria-label="Show module list"
              onClick={() => setMobileNav(true)}
            >
              <Menu size={18} />
            </button>
            <label className="search-field">
              <Search size={15} />
              <input
                type="search"
                aria-label="Search modules"
                placeholder="Search modules, roles, or paths…"
                value={state.query}
                onChange={(event) => navigate({ query: event.target.value }, true)}
              />
              {state.query ? (
                <button aria-label="Clear search" onClick={() => navigate({ query: '' }, true)}>
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
              aria-label="Relationships"
              aria-expanded={showFilters}
            >
              <SlidersHorizontal size={14} />
              <span>Relationships</span>
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
              <span>Reset</span>
            </button>
            <button className="icon-button" aria-label="Help" onClick={() => setHelp(!help)}>
              <CircleHelp size={16} />
            </button>
          </div>
          <div className="reading-bar">
            <button
              className={`focus-toggle ${state.focus ? 'active' : ''}`}
              aria-label="Direct neighbours only"
              aria-pressed={state.focus}
              disabled={!selected}
              title={
                selected
                  ? 'Show the selected module and its direct relationships only'
                  : 'Select a module first'
              }
              onClick={() => navigate({ focus: !state.focus })}
            >
              <Network size={14} /> Direct neighbours only
            </button>
            <span className="reading-context">
              {state.focus
                ? `${graph.visibleNodes.length} of ${graph.nodes.length} modules shown`
                : 'Reading view · scroll to explore'}
            </span>
            {selected && !inspectorOpen && (
              <button
                className="focus-toggle"
                aria-label="Open node details"
                onClick={() => setInspectorOpen(true)}
              >
                <Box size={13} /> Details
              </button>
            )}
            <button
              className="fit-overview"
              onClick={() => void flow.fitView({ padding: 0.07, duration: 250 })}
            >
              <Maximize size={13} /> Fit overview
            </button>
          </div>
          {showFilters && (
            <div className="filter-menu">
              <div className="filter-menu-title">
                Show relationships{' '}
                <button
                  onClick={() =>
                    navigate({ relations: state.relations.length ? [] : Object.keys(relations) })
                  }
                  aria-label={
                    state.relations.length ? 'Clear relationship filters' : 'Show all relationships'
                  }
                >
                  {state.relations.length ? 'Clear' : 'Select all'}
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
              <button aria-label="Dismiss notice" onClick={() => setNotice('')}>
                <X size={13} />
              </button>
            </div>
          )}
          <div className="canvas-shell" ref={canvasElement}>
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
                  void flow.setCenter(
                    node.position.x + CARD_WIDTH / 2,
                    node.position.y + CARD_HEIGHT / 2,
                    {
                      zoom: 1.15,
                      duration: 300,
                    },
                  );
                }
              }}
              onPaneClick={() => {
                setShowFilters(false);
                setHelp(false);
              }}
              panOnScroll
              zoomOnScroll={false}
              minZoom={0.25}
              maxZoom={1.8}
              nodesConnectable={false}
              deleteKeyCode={null}
              colorMode={theme}
              onlyRenderVisibleElements={false}
              proOptions={{ hideAttribution: false }}
            >
              <ReadingViewport
                layoutKey={`${state.viewId}:${columns}:${state.focus ? state.relations.join() : ''}`}
                nodeId={state.nodeId}
                focus={state.focus}
              />
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
            </ReactFlow>
            {hasQuery && graph.matchedNodeIds.size === 0 && (
              <div className="empty-search">
                <Search size={23} />
                <strong>No matching modules</strong>
                <span>Try Robot, planning, or a source path</span>
                <button onClick={() => navigate({ query: '' }, true)}>Clear search</button>
              </div>
            )}
            {help && (
              <div className="help-popover">
                <strong>Reading the diagram</strong>
                <p>Select a module to inspect its role and source evidence.</p>
                <p>
                  Drag or scroll to pan; use the zoom controls or pinch to scale. The module index
                  supports keyboard navigation.
                </p>
                <p>
                  Filter relationships to focus the diagram. Shared links preserve your selection.
                </p>
                <button onClick={() => setHelp(false)}>Got it</button>
              </div>
            )}
          </div>
          <div className="canvas-footer">
            <span>
              <MouseIcon /> Scroll to pan · Double-click a module to zoom
            </span>
            <span>
              Curated sample · Not a complete dependency graph <ArrowUpRight size={11} />
            </span>
          </div>
        </main>
        <NodeDetails
          data={data}
          node={inspectorOpen ? selected : undefined}
          view={graph.view}
          edges={graph.edges}
          color={layerColors[Math.max(0, selectedGroup) % layerColors.length]}
          onSelect={selectNode}
          onClose={() => setInspectorOpen(false)}
        />
      </div>
      <div className="sr-only" aria-live="polite">
        {selected ? `Selected ${selected.label}` : ''}
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
export default function App({ data }: { data: ArchitectureSnapshot }) {
  return (
    <ReactFlowProvider>
      <Explorer data={data} />
    </ReactFlowProvider>
  );
}
