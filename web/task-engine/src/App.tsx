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
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ReactFlow,
  Panel,
  Background,
  MiniMap,
  MarkerType,
  useReactFlow,
  useStore,
  useNodesState,
  type Node,
  type Edge,
} from "@xyflow/react";
import { GraphViewportControls } from "@embodichain/graph-ui";
import {
  Activity,
  ArrowDownToLine,
  ArrowUpRight,
  Box,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  CircleX,
  Clock3,
  Code2,
  FileJson,
  Layers3,
  Moon,
  PanelRightClose,
  Search,
  ShieldCheck,
  Sun,
  Workflow,
  X,
} from "lucide-react";
import { createModel, good, layoutGraph, type Model } from "./model";
import { tr, type Locale } from "./i18n";
import { TaskNodeCard, TaskLane } from "./TaskNodes";
import { download, graphSvg } from "./svg";
import type { Asset, Layout, Mode, Status, ViewerData, ViewRow } from "./types";

const nodeTypes = { task: TaskNodeCard, lane: TaskLane };
const format = (value: unknown, locale: Locale, digits = 1) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString(locale === "zh" ? "zh-CN" : "en-US", {
        maximumFractionDigits: digits,
      })
    : tr("missing", locale);
const safeImage = (value: unknown): string | undefined =>
  typeof value === "string" &&
  /^data:image\/(png|jpeg|webp);base64,/.test(value)
    ? value
    : undefined;

function RawJson({ label, value }: { label: string; value: unknown }) {
  const [open, setOpen] = useState(false);
  return (
    <details
      className="raw-json"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>
        <Code2 size={12} />
        {label}
      </summary>
      {open && <pre>{JSON.stringify(value ?? null, null, 2)}</pre>}
    </details>
  );
}
function Badge({
  status,
  children,
}: {
  status: string;
  children: React.ReactNode;
}) {
  const Icon = good(status)
    ? CheckCircle2
    : status === "failed"
      ? CircleX
      : CircleAlert;
  return (
    <span
      className={
        "status-badge " +
        (good(status) ? "good" : status === "failed" ? "bad" : "unknown")
      }
    >
      <Icon size={13} />
      {children}
    </span>
  );
}
function Facts({ values }: { values: [string, React.ReactNode][] }) {
  return (
    <dl className="facts">
      {values.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}
function Viewport({
  layout,
  mode,
  revision,
  onReady,
}: {
  layout: Layout;
  mode: Mode;
  revision: number;
  onReady: (ready: boolean) => void;
}) {
  const flow = useReactFlow();
  const viewportWidth = useStore((state) => state.width);
  const viewportHeight = useStore((state) => state.height);
  useEffect(() => {
    onReady(false);
    if (!viewportWidth || !viewportHeight) return;
    let cancelled = false;
    const operation =
      mode === "groups"
        ? flow.fitView({ padding: 0.12, maxZoom: 1, duration: 180 })
        : flow.setViewport({ x: 0, y: 18, zoom: 1 }, { duration: 180 });
    void operation.then(() => {
      if (!cancelled) onReady(true);
    });
    return () => {
      cancelled = true;
    };
  }, [
    viewportWidth,
    viewportHeight,
    layout.width,
    layout.height,
    mode,
    revision,
    flow,
    onReady,
  ]);
  return null;
}
function AssetDialog({
  asset,
  model,
  onClose,
}: {
  asset: Asset;
  model: Model;
  onClose: () => void;
}) {
  const [kind, setKind] = useState<"render" | "cutout">(
    safeImage(asset.render) ? "render" : "cutout",
  );
  const dialog = useRef<HTMLDialogElement>(null);
  useLayoutEffect(() => {
    const element = dialog.current!;
    const opener = document.activeElement;
    element.showModal();
    return () => {
      element.close();
      if (opener instanceof HTMLElement && opener.isConnected)
        opener.focus({ preventScroll: true });
    };
  }, []);
  return (
    <dialog
      ref={dialog}
      aria-label={model.name(asset.uid)}
      className="asset-modal"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape") event.stopPropagation();
      }}
      onClick={(event) => {
        if (event.target !== event.currentTarget) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        if (
          event.clientX < bounds.left ||
          event.clientX > bounds.right ||
          event.clientY < bounds.top ||
          event.clientY > bounds.bottom
        )
          onClose();
      }}
    >
      <header>
        <h2>
          {model.name(asset.uid)}
          <small> / {asset.uid}</small>
        </h2>
        <button autoFocus aria-label={model.t("close")} onClick={onClose}>
          <X size={18} />
        </button>
      </header>
      <div className="segmented">
        {(["render", "cutout"] as const).map((value) => (
          <button
            key={value}
            disabled={!safeImage(asset[value])}
            aria-pressed={kind === value}
            onClick={() => setKind(value)}
          >
            {model.t(value)}
          </button>
        ))}
      </div>
      {safeImage(asset[kind]) ? (
        <img src={safeImage(asset[kind])} alt={model.name(asset.uid)} />
      ) : (
        <p className="empty-state">{model.t("assetMissing")}</p>
      )}
      <p className="muted">
        {model.t(kind === "render" ? "sourceRender" : "sourceCutout")}
        {(asset[kind + "_source"] as string) || "—"}
      </p>
    </dialog>
  );
}
function Inspector({
  data,
  model,
  selected,
  locale,
  onSelect,
  onAsset,
}: {
  data: ViewerData;
  model: Model;
  selected: string;
  locale: Locale;
  onSelect: (value: string) => void;
  onAsset: (asset: Asset) => void;
}) {
  const t = model.t;
  if (selected === "overview")
    return (
      <>
        <span className="eyebrow">RUN / IDENTITY</span>
        <h2>{data.graph.task_id}</h2>
        <p className="muted">{t("sourceOnly")}</p>
        <Facts
          values={[
            [
              t("assemblyIdentity"),
              <code>{data.identities.integration_fingerprint}</code>,
            ],
            [t("graphIdentity"), <code>{data.identities.graph_digest}</code>],
            [t("mode"), data.presentation.acceptance_mode || t("missing")],
          ]}
        />
        <RawJson label={t("rawSpec")} value={data.task_spec} />
        <RawJson label={t("rawGraph")} value={data.graph} />
        <RawJson label={t("rawReport")} value={data.report} />
        <RawJson label={t("rawCoverage")} value={data.validation} />
      </>
    );
  if (selected.startsWith("asset:")) {
    const asset = model.assets.get(selected.slice(6));
    return asset ? (
      <>
        <span className="eyebrow">ASSET / SOURCE</span>
        <h2>{model.name(asset.uid)}</h2>
        <code className="sub-id">{asset.uid}</code>
        {safeImage(asset.render) || safeImage(asset.cutout) ? (
          <button
            className="asset-preview-trigger"
            aria-label={model.name(asset.uid)}
            onClick={() => onAsset(asset)}
          >
            <img
              className="inspector-image"
              src={safeImage(asset.render) || safeImage(asset.cutout)}
              alt={model.name(asset.uid)}
            />
          </button>
        ) : (
          <p className="muted">{t("assetMissing")}</p>
        )}
        <Facts
          values={[
            [t("originalName"), asset.name || "—"],
            [
              t("description"),
              data.presentation.localizations?.assets?.[asset.uid]
                ?.description?.[locale] ||
                asset.description ||
                "—",
            ],
            [t("sourceFile"), <code>{asset.glb_source || t("missing")}</code>],
            ["SHA256", <code>{asset.glb_sha256 || t("missing")}</code>],
          ]}
        />
        <RawJson
          label={t("rawAsset")}
          value={Object.fromEntries(
            Object.entries(asset).filter(
              ([key]) => !["cutout", "render"].includes(key),
            ),
          )}
        />
      </>
    ) : null;
  }
  if (selected.startsWith("goal:")) {
    const goal = data.task_spec?.goal.find(
      (value) => value.id === selected.slice(5),
    );
    return goal ? (
      <>
        <span className="eyebrow">TASKSPEC / FINAL GOAL</span>
        <h2>{model.goalLabel(goal)}</h2>
        <code>{goal.id}</code>
        <Badge status={model.goalStatus(goal)}>
          {t(model.goalStatus(goal))}
        </Badge>
        <Facts
          values={[
            [t("predicate"), goal.predicate],
            [
              t("groupId"),
              (
                data.validation?.coverage.goals.find(
                  (value) => value.id === goal.id,
                )?.groups || []
              ).join(", "),
            ],
          ]}
        />
        <RawJson label={t("rawGoal")} value={goal} />
        <RawJson label={t("observedGoal")} value={model.goalRows(goal)} />
      </>
    ) : null;
  }
  const isGroup = selected.startsWith("group:"),
    node = isGroup ? undefined : model.calls.get(selected.slice(5));
  const group = isGroup
    ? model.groups.get(selected.slice(6))
    : node && model.groups.get(node.task_instance_id);
  if (!group) return null;
  const ids = isGroup ? group.node_ids : [node!.id],
    state = isGroup ? model.groupStatus(group) : model.callStatus(node!);
  const control = isGroup ? model.groupLane(group) : model.lane(node!);
  const window = model.eventWindow(ids);
  const related = (data.task_spec?.goal || []).filter((goal) =>
    data.validation?.coverage.goals
      .find((value) => value.id === goal.id)
      ?.groups.includes(group.id),
  );
  return (
    <>
      <span className="eyebrow">
        {isGroup ? "STAGE" : "CALL"} / {group.task_type}
      </span>
      <h2>{isGroup ? model.groupLabel(group) : model.callLabel(node!)}</h2>
      <code className="sub-id">{isGroup ? group.id : node!.id}</code>
      <Badge status={state}>
        {t(isGroup ? "stagePassed" : "callEvidence")} · {t(state)}
      </Badge>
      <div className="evidence-stack">
        <div>
          <span>{t("runRecord")}</span>
          <Badge status={data.report?.status || "unknown"}>
            {t(data.report?.status || "unknown")}
          </Badge>
        </div>
        <div>
          <span>{t("stagePassed")}</span>
          <Badge status={model.groupStatus(group)}>
            {t(model.groupStatus(group))}
          </Badge>
        </div>
        {related.map((goal) => (
          <div key={goal.id}>
            <span>{goal.id}</span>
            <Badge status={model.goalStatus(goal)}>
              {t(model.goalStatus(goal))}
            </Badge>
          </div>
        ))}
      </div>
      <Facts
        values={[
          [t("channel"), t(control === "auto" ? "unbound" : control)],
          [t("controlSteps"), format(model.steps(ids), locale, 0)],
          [
            t("eventSpan"),
            window
              ? format(window[0], locale) +
                " – " +
                format(window[1], locale) +
                " s"
              : t("missing"),
          ],
          [
            t("directDeps"),
            (isGroup ? group.depends_on : node!.depends_on).join(", ") || "—",
          ],
        ]}
      />
      {isGroup ? (
        <>
          <h3>
            {t("callList")} <span className="count">{ids.length}</span>
          </h3>
          <div className="call-list">
            {ids.map((id) => {
              const value = model.calls.get(id)!;
              const span = model.eventWindow([id]);
              return (
                <button
                  key={id}
                  title={id}
                  onClick={() => onSelect("call:" + id)}
                >
                  <span className={"status-dot " + model.callStatus(value)} />
                  {model.callLabel(value)}
                  <small>
                    {span
                      ? format(span[1] - span[0], locale) + " s"
                      : t("missing")}
                  </small>
                  <ChevronRight size={13} />
                </button>
              );
            })}
          </div>
          <RawJson
            label={t("rawStage")}
            value={{ group, calls: ids.map((id) => model.calls.get(id)) }}
          />
        </>
      ) : (
        <>
          <h3>{t("events")}</h3>
          <div className="event-list">
            {model
              .events(node!.id)
              .filter((event) =>
                [
                  "skill_planned",
                  "trajectory_segment_entered",
                  "phase_effect_gate_satisfied",
                  "skill_completed",
                  "action_failed",
                ].includes(event.kind),
              )
              .map((event, index) => (
                <div key={index}>
                  <small>{format(event.timestamp, locale)} s</small>
                  <span>{event.segment_name || event.kind}</span>
                  <code>{event.kind}</code>
                </div>
              ))}
          </div>
          <RawJson label={t("rawCall")} value={data.call_evidence[node!.id]} />
          <RawJson label={t("rawNode")} value={node} />
        </>
      )}
    </>
  );
}

export default function App({ data }: { data: ViewerData }) {
  const [viewportReady, setViewportReady] = useState(false);
  const [viewportRevision, setViewportRevision] = useState(0);
  const [locale, setLocale] = useState<Locale>(() => {
    try {
      return localStorage.getItem("gensim.graph.language") === "en"
        ? "en"
        : "zh";
    } catch {
      return "zh";
    }
  });
  const [theme, setTheme] = useState<"light" | "dark">(() =>
    new URLSearchParams(window.location.search).get("theme") === "dark"
      ? "dark"
      : "light",
  );
  const [mode, setMode] = useState<Mode>("groups"),
    [environment, setEnvironment] = useState("all"),
    [query, setQuery] = useState(""),
    [filter, setFilter] = useState("all");
  const [selected, setSelected] = useState(
    data.graph.task_groups[0]
      ? "group:" + data.graph.task_groups[0].id
      : "overview",
  );
  const [inspectorOpen, setInspectorOpen] = useState(
      () => window.matchMedia("(min-width: 901px)").matches,
    ),
    [asset, setAsset] = useState<Asset | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const model = useMemo(
    () => createModel(data, environment, locale),
    [data, environment, locale],
  );
  const { t } = model;
  const layout = useMemo(
    () => layoutGraph(data, model, mode, query, filter, selected),
    [data, model, mode, query, filter, selected],
  );
  const onSelect = useCallback((key: string) => {
    setSelected(key);
    setInspectorOpen(true);
    setSidebarOpen(false);
  }, []);
  const nodeDefinitions = useMemo<Node[]>(
    () => [
      ...layout.lanes.map((lane) => ({
        id: "lane:" + lane.track,
        type: "lane",
        position: { x: 12, y: lane.y },
        data: { lane },
        style: { width: lane.width, height: lane.height },
        selectable: false,
        draggable: false,
        connectable: false,
        focusable: false,
        zIndex: -1,
      })),
      ...layout.rows.map((row) => ({
        id: row.key,
        type: "task",
        position: { x: row.x, y: row.y },
        data: {
          row,
          active: selected === row.key,
          statusLabel: t(row.status),
          onSelect,
        },
        style: { width: row.width, height: row.height },
        draggable: false,
        connectable: false,
        selected: selected === row.key,
      })),
    ],
    [layout, selected, t, onSelect],
  );
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  useLayoutEffect(() => {
    setNodes((current) => {
      const previous = new Map(current.map((node) => [node.id, node]));
      return nodeDefinitions.map((node) => {
        const old = previous.get(node.id);
        return old?.style?.width === node.style?.width &&
          old?.style?.height === node.style?.height
          ? { ...node, measured: old?.measured }
          : node;
      });
    });
  }, [nodeDefinitions, setNodes]);
  const edges = useMemo<Edge[]>(
    () =>
      layout.edges.map((edge) => ({
        ...edge,
        type: "smoothstep",
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: "#8b9aad",
          width: 14,
          height: 14,
        },
        style: { stroke: "#8b9aad", strokeWidth: 1.35 },
        interactionWidth: 0,
        focusable: false,
      })),
    [layout.edges],
  );
  const goals = data.task_spec?.goal || [],
    passed = goals.filter(
      (goal) => model.goalStatus(goal) === "satisfied",
    ).length;
  const localized = data.presentation.localizations;
  const title = localized?.title?.[locale] || data.graph.task_id;
  const instruction =
    localized?.instruction?.[locale] || data.graph.instruction;
  const setLanguage = (value: Locale) => {
    setLocale(value);
    try {
      localStorage.setItem("gensim.graph.language", value);
    } catch {
      /* File URLs may deny storage. */
    }
  };
  const exportSvg = () => {
    const css = getComputedStyle(document.querySelector(".task-app")!);
    const color = (name: string) => css.getPropertyValue("--" + name).trim();
    download(
      graphSvg(
        layout,
        selected,
        document.getElementById("viewer-style")?.textContent || "",
        JSON.parse(
          document.getElementById("font-license-data")?.textContent || '""',
        ),
        t,
        {
          name: theme,
          canvas: color("canvas"),
          panel: color("panel"),
          rail: color("accent-bg"),
          text: color("text"),
          muted: color("muted"),
          line: color("line"),
          accent: color("accent"),
          good: color("status-success"),
          bad: color("status-failure"),
          unknown: color("status-unknown"),
        },
      ),
      "semantic_graph.svg",
      "image/svg+xml",
    );
  };
  const reset = () => {
    setViewportReady(false);
    setViewportRevision((value) => value + 1);
    setQuery("");
    setFilter("all");
    setSelected(
      data.graph.task_groups[0]
        ? "group:" + data.graph.task_groups[0].id
        : "overview",
    );
    setMode("groups");
  };
  useEffect(() => {
    document.documentElement.lang = locale === "zh" ? "zh-CN" : "en";
    document.title = data.graph.task_id + " · " + t("graph");
  }, [locale, data.graph.task_id, t]);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setInspectorOpen(false);
        setAsset(null);
        setSidebarOpen(false);
      }
    };
    addEventListener("keydown", handler);
    return () => removeEventListener("keydown", handler);
  }, []);
  useEffect(() => {
    const target = window as unknown as { __graphViewer: unknown };
    target.__graphViewer = {
      data,
      counts: () => ({
        calls: data.graph.nodes.length,
        groups: data.graph.task_groups.length,
        edges: data.graph.nodes.reduce(
          (sum, node) => sum + node.depends_on.length,
          0,
        ),
      }),
      locale: () => locale,
      mode: () => mode,
      selected: () => selected,
      setLocale: setLanguage,
      select: onSelect,
      groupStatus: model.groupStatus,
      callStatus: model.callStatus,
      goalStatus: model.goalStatus,
      implementation: "react-typescript-reactflow",
    };
  });
  const video = data.presentation.video_href;
  const safeVideo =
    typeof video === "string" &&
    /^videos\/(?:env_\d+\/)?[^/\\:]+\.mp4$/.test(video) &&
    !video.endsWith(".partial.mp4")
      ? video.split("/").map(encodeURIComponent).join("/")
      : null;
  return (
    <div className="task-app graph-ui" data-theme={theme}>
      <header className="task-header">
        <div className="brand">
          <Workflow size={22} />
          <strong>EmbodiChain</strong>
          <span>/ {t("graph")}</span>
        </div>
        <span className="archive-chip">
          <Clock3 size={12} />
          {t(data.report ? "runRecord" : "staticGraph")}
        </span>
        <div className="header-actions">
          <div className="language-switch">
            {(["zh", "en"] as const).map((value) => (
              <button
                key={value}
                data-language={value}
                aria-pressed={locale === value}
                onClick={() => setLanguage(value)}
              >
                {value === "zh" ? "中" : "EN"}
              </button>
            ))}
          </div>
          <button
            className="icon-button"
            aria-label={locale === "zh" ? "切换主题" : "Toggle theme"}
            onClick={() => setTheme(theme === "light" ? "dark" : "light")}
          >
            {theme === "light" ? <Moon size={16} /> : <Sun size={16} />}
          </button>
          <button
            className="icon-button"
            id="provenance"
            title={t("identity")}
            onClick={() => onSelect("overview")}
          >
            <FileJson size={16} />
          </button>
        </div>
      </header>
      <section className="task-summary">
        <div>
          <div className="eyebrow">GENSIM / TASK ENGINE</div>
          <h1 title={title}>{title}</h1>
          <p>{instruction}</p>
          {localized?.instruction?.[locale] && (
            <details className="original-instruction">
              <summary>{t("originalInstruction")}</summary>
              {data.graph.instruction}
            </details>
          )}
        </div>
        <div className="summary-status">
          <Badge status={model.strict}>
            {t(
              model.strict === "satisfied"
                ? "strictPass"
                : model.strict === "failed"
                  ? "strictFail"
                  : "strictUnknown",
            )}
          </Badge>
          <Badge status={data.report?.status || "unknown"}>
            {t(
              data.report?.status === "succeeded"
                ? "completed"
                : data.report?.status || "unknown",
            )}
          </Badge>
        </div>
      </section>
      <div className="task-metrics">
        <div>
          <Clock3 />
          <span>{t("wallTime")}</span>
          <strong>
            {format(data.presentation.wall_seconds, locale, 3)}
            <small>
              {typeof data.presentation.wall_seconds === "number" ? "s" : ""}
            </small>
          </strong>
          <p>
            {data.presentation.wall_source &&
            data.presentation.wall_source !== "未记录"
              ? data.presentation.wall_source
              : t("missing")}
          </p>
        </div>
        <div>
          <Activity />
          <span>{t("controlSteps")}</span>
          <strong>
            {format(data.report?.runtime_result?.length, locale, 0)}
            <small>steps</small>
          </strong>
          <p>
            {t("simulationTime")}{" "}
            {format(
              model.eventWindow(data.graph.nodes.map((node) => node.id))?.[1],
              locale,
            )}{" "}
            s
          </p>
        </div>
        <div>
          <Layers3 />
          <span>{t("semanticCalls")}</span>
          <strong>
            {data.graph.nodes.length}
            <small>{t("calls")}</small>
          </strong>
          <p>
            {data.graph.task_groups.length} {t("stages")} /{" "}
            {model.envRows.length} {t("env")}
          </p>
        </div>
        <div>
          <ShieldCheck />
          <span>{t("goalMetric")}</span>
          <strong
            className={
              model.strict === "satisfied"
                ? "good"
                : model.strict === "failed"
                  ? "bad"
                  : "unknown"
            }
          >
            {passed} / {goals.length}
            <small>{t("satisfied")}</small>
          </strong>
          <p>
            {t(
              model.strict === "satisfied"
                ? "fullSpec"
                : model.strict === "failed"
                  ? "strictFail"
                  : "strictUnknown",
            )}
          </p>
        </div>
      </div>
      <main
        className={
          "task-workspace " +
          (inspectorOpen ? "" : "inspector-hidden") +
          (sidebarOpen ? " sidebar-open" : "")
        }
      >
        <nav className="task-sidebar" aria-label={t("stage")}>
          <div className="sidebar-heading">
            <Layers3 size={14} />
            {t("stages")}
            <span>{data.graph.task_groups.length}</span>
          </div>
          {data.graph.task_groups.map((group, index) => (
            <button
              key={group.id}
              className={
                "stage-nav " +
                (selected === "group:" + group.id ? "active" : "")
              }
              onClick={() => onSelect("group:" + group.id)}
            >
              <span className="stage-number">
                {String(index + 1).padStart(2, "0")}
              </span>
              <span>
                <strong title={model.groupLabel(group)}>
                  {model.groupLabel(group)}
                </strong>
                <small>
                  {group.task_type} · {group.node_ids.length} {t("calls")}
                </small>
              </span>
              <span className={"status-dot " + model.groupStatus(group)} />
            </button>
          ))}
          <div className="sidebar-heading asset-heading">
            <Box size={14} />
            {t("assets")}
            <span>{model.assets.size}</span>
          </div>
          {[...model.assets.values()].map((value) => (
            <div key={value.uid} className="asset-row">
              <button
                className="asset-thumb"
                title={model.name(value.uid)}
                onClick={() => setAsset(value)}
              >
                {safeImage(value.cutout) || safeImage(value.render) ? (
                  <img
                    src={safeImage(value.cutout) || safeImage(value.render)}
                    alt={model.name(value.uid)}
                  />
                ) : (
                  <Box size={18} />
                )}
              </button>
              <button onClick={() => onSelect("asset:" + value.uid)}>
                <strong>{model.name(value.uid)}</strong>
                <small title={value.uid}>{value.uid}</small>
              </button>
            </div>
          ))}
          {!model.assets.size && (
            <p className="muted empty-assets">{t("noAssets")}</p>
          )}
        </nav>
        <section className="task-center">
          <div className="graph-top">
            <div>
              <h2>{t("dependencies")}</h2>
              <span>
                {data.graph.task_groups.length} {t("stages")} ·{" "}
                {data.graph.nodes.length} {t("calls")}
              </span>
            </div>
            <div className="graph-top-actions">
              <button
                className="icon-button mobile-assets"
                aria-label={t("assets")}
                aria-expanded={sidebarOpen}
                onClick={() => {
                  setSidebarOpen(!sidebarOpen);
                  setInspectorOpen(false);
                }}
              >
                <Box size={16} />
              </button>
              <button
                className="icon-button"
                id="export-svg"
                title={t("exportSvg")}
                aria-label={t("exportSvg")}
                onClick={exportSvg}
              >
                <ArrowDownToLine size={16} />
              </button>
              <button
                className="icon-button"
                id="export-json"
                title={t("exportJson")}
                aria-label={t("exportJson")}
                onClick={() =>
                  download(
                    JSON.stringify(data, null, 2),
                    "semantic_graph_data.json",
                    "application/json",
                  )
                }
              >
                <FileJson size={16} />
              </button>
              <button
                className="icon-button"
                aria-label={t("inspector")}
                onClick={() => setInspectorOpen(!inspectorOpen)}
              >
                <PanelRightClose size={16} />
              </button>
            </div>
          </div>
          <div className="graph-toolbar">
            <div className="segmented">
              {(["groups", "calls"] as const).map((value) => (
                <button
                  key={value}
                  id={"mode-" + value}
                  aria-pressed={mode === value}
                  onClick={() => {
                    if (value !== mode) {
                      setViewportReady(false);
                      setMode(value);
                    }
                  }}
                >
                  {t(value === "groups" ? "stages" : "calls")}
                </button>
              ))}
            </div>
            <label className="search-box">
              <Search size={14} />
              <input
                id="search"
                placeholder={t("search")}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
            </label>
            <select
              id="environment"
              aria-label={t("environment")}
              value={environment}
              disabled={!data.report}
              onChange={(event) => setEnvironment(event.target.value)}
            >
              <option value="all">{t("envAll")}</option>
              {data.report?.environments.map((row) => (
                <option key={row.env_id} value={String(row.env_id)}>
                  {t("env")} {row.env_id}
                </option>
              ))}
            </select>
            <select
              id="status-filter"
              aria-label={t("statusFilter")}
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
            >
              <option value="all">{t("allStates")}</option>
              {(["succeeded", "failed", "unknown", "unavailable"] as const).map(
                (value) => (
                  <option key={value} value={value}>
                    {t(value)}
                  </option>
                ),
              )}
            </select>
          </div>
          <div className="task-canvas" aria-busy={!viewportReady}>
            <ReactFlow
              nodes={nodes}
              onNodesChange={onNodesChange}
              deleteKeyCode={null}
              edges={edges}
              nodeTypes={nodeTypes}
              nodesDraggable={false}
              nodesConnectable={false}
              elementsSelectable={false}
              panOnDrag={viewportReady}
              zoomOnScroll={viewportReady}
              zoomOnPinch={viewportReady}
              selectionOnDrag={false}
              zoomOnDoubleClick={false}
              minZoom={0.12}
              maxZoom={3}
              fitView
              onPaneClick={() => {}}
              proOptions={{ hideAttribution: false }}
            >
              <Background
                color={theme === "light" ? "#cdd5db" : "#3a4653"}
                gap={22}
                size={1}
              />
              <Viewport
                layout={layout}
                mode={mode}
                revision={viewportRevision}
                onReady={setViewportReady}
              />
              <MiniMap
                pannable={viewportReady}
                zoomable={viewportReady}
                nodeColor={(node) =>
                  node.type === "lane"
                    ? "transparent"
                    : good((node.data.row as ViewRow | undefined)?.status || "")
                      ? "#5d9b83"
                      : "#8b9aad"
                }
              />
              <Panel position="bottom-center">
                <fieldset
                  className="viewport-controls"
                  disabled={!viewportReady}
                >
                  <GraphViewportControls
                    labels={{
                      zoomOut: t("zoomOut"),
                      actualSize:
                        locale === "zh"
                          ? "按 100% 阅读"
                          : "Read at 100 percent",
                      zoomIn: t("zoomIn"),
                      fit: t("fit"),
                    }}
                  />
                </fieldset>
              </Panel>
              <Panel position="top-right">
                <button
                  className="reset-button"
                  id="reset"
                  onClick={reset}
                  disabled={!viewportReady}
                >
                  {t("reset")}
                </button>
              </Panel>
            </ReactFlow>
          </div>
          <div className="graph-caption">
            <span>
              <i />
              {t("dependency")}
            </span>
            <small>{t("orderNote")}</small>
            <span>
              {layout.rows.filter((row) => !row.faded).length} /{" "}
              {layout.rows.length}
            </span>
          </div>
          <section className="task-goals">
            <div className="section-title">
              <h2>
                <ShieldCheck size={15} />
                {t("finalGoals")}
              </h2>
              <span>
                {passed} / {goals.length} {t("satisfied")}
              </span>
            </div>
            <div className="goal-cards">
              {goals.map((goal) => {
                const status = model.goalStatus(goal);
                const Icon = good(status)
                  ? CheckCircle2
                  : status === "failed"
                    ? CircleX
                    : CircleAlert;
                return (
                  <button
                    key={goal.id}
                    className={
                      "goal-card " +
                      (selected === "goal:" + goal.id ? "active" : "")
                    }
                    onClick={() => onSelect("goal:" + goal.id)}
                  >
                    <Icon
                      className={
                        good(status)
                          ? "good"
                          : status === "failed"
                            ? "bad"
                            : "unknown"
                      }
                      size={16}
                    />
                    <span>
                      <strong>{model.goalLabel(goal)}</strong>
                      <small>{goal.id}</small>
                    </span>
                    <Badge status={model.goalStatus(goal)}>
                      {t(model.goalStatus(goal))}
                    </Badge>
                  </button>
                );
              })}
              {!goals.length && <p className="muted">{t("noSpec")}</p>}
            </div>
          </section>
        </section>
        {inspectorOpen && (
          <aside id="inspector" className="task-inspector">
            <div className="inspector-heading">
              <span>
                <Code2 size={14} />
                {t("inspector")}
              </span>
              <button
                className="icon-button"
                aria-label={t("close")}
                onClick={() => setInspectorOpen(false)}
              >
                <X size={15} />
              </button>
            </div>
            <div id="detail" className="inspector-body">
              <Inspector
                key={selected}
                data={data}
                model={model}
                selected={selected}
                locale={locale}
                onSelect={onSelect}
                onAsset={setAsset}
              />
              {safeVideo && (
                <section className="recording">
                  <h3>
                    {t("recording")}
                    <a
                      href={safeVideo}
                      target="_blank"
                      rel="noopener"
                      aria-label={t("openVideo")}
                      title={t("openVideo")}
                    >
                      <ArrowUpRight size={14} />
                    </a>
                  </h3>
                  <video
                    src={safeVideo}
                    controls
                    playsInline
                    preload="metadata"
                  />
                  <p className="muted">
                    {t("recordingCaption")} / {video}
                  </p>
                </section>
              )}
            </div>
          </aside>
        )}
      </main>
      <footer className="task-footer">
        <span>GRAPH {data.identities.graph_digest.slice(0, 12)}</span>
        <button onClick={() => onSelect("overview")}>
          {t("identity")}
          <ArrowUpRight size={11} />
        </button>
        <span>{t("unchanged")}</span>
      </footer>
      {asset && (
        <AssetDialog
          asset={asset}
          model={model}
          onClose={() => setAsset(null)}
        />
      )}
    </div>
  );
}
