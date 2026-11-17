// Copyright (c) 2021-2026 DexForce Technology Co., Ltd. Licensed under Apache-2.0.
// Drawing data is bounded; the downloadable NPZ retains the original sample grid.
const $ = (id) => document.getElementById(id);
const colors = { original: "#fb8696", native: "#f2b664", patched: "#66d5bc", proposed: "#b6a1ff" };
const fields = {
  joints: "fit-joints", robot_profile: "fit-profile", qpos_semantics: "fit-semantics",
  scene: "fit-scene", object_id: "fit-object", participant: "fit-participant",
  time_offset: "fit-offset", qpos_key: "fit-qpos-key", timestamps_key: "fit-timestamps-key",
  joint_dt: "fit-joint-dt", env_index: "fit-env-index", device: "fit-device",
};

async function jsonRequest(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
  return data;
}

function nearest(values, value) {
  let low = 0, high = values.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (values[mid] < value) low = mid + 1; else high = mid;
  }
  if (!low) return 0;
  if (low === values.length) return low - 1;
  return value - values[low - 1] < values[low] - value ? low - 1 : low;
}

function surface(canvas) {
  const { width, height } = canvas.getBoundingClientRect();
  if (!width || !height) return null;
  const ratio = window.devicePixelRatio || 1;
  const w = Math.round(width * ratio), h = Math.round(height * ratio);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

export function createFittingPanel({ context, saveAnnotations, seekTime, playRange }) {
  const state = { job: null, data: null, timer: null, loading: false, starting: false, time: 0, hover: null, seeking: false };
  const viewer = { job: null, pending: null, sending: false, timer: null, nextAt: 0 };
  const scene = { state: null, timer: null, starting: false };
  const sceneInputs = () => ({ scene: $("fit-scene").value.trim(), joints: $("fit-joints").value.trim() });

  function showSceneStatus() {
    $("fit-load-scene").disabled = scene.starting || scene.state?.phase === "running";
    $("fit-scene-status").textContent = scene.state?.error || scene.state?.message || "加载场景后准备碰撞网格。";
  }

  async function prepareScene() {
    if (scene.starting || scene.state?.phase === "running") return;
    scene.starting = true;
    showSceneStatus();
    clearTimeout(scene.timer);
    try {
      scene.state = await jsonRequest("/api/fitting/scene", {
        method: "POST", headers: { "Content-Type": "application/json", "X-Annotation-Token": context().token },
        body: JSON.stringify(sceneInputs()),
      });
      if (scene.state.phase === "running") scene.timer = setTimeout(() => void pollScene(), 600);
    } catch (error) {
      scene.state = { phase: "error", error: error.message };
    } finally { scene.starting = false; showSceneStatus(); }
  }

  async function pollScene() {
    clearTimeout(scene.timer);
    try {
      scene.state = await jsonRequest("/api/fitting/scene");
      showSceneStatus();
      if (scene.state.phase === "running") scene.timer = setTimeout(() => void pollScene(), 600);
      else if (JSON.stringify(scene.state.inputs) !== JSON.stringify(sceneInputs())) void prepareScene();
    } catch (error) {
      $("fit-scene-status").textContent = `无法读取场景准备状态：${error.message}`;
      scene.timer = setTimeout(() => void pollScene(), 1500);
    }
  }
  const series = () => state.data?.report.accepted ? ["original", "native", "patched"] : ["original", "native", "patched", "proposed"];
  const stale = () => !!state.job && (state.job.stale || context().dirty || state.job.revision !== context().revision);

  function notify(text, kind = "") {
    $("fit-message").textContent = text;
    $("fit-message").className = `fit-message ${kind}`;
  }

  function selectionChanged() {
    const current = context();
    $("fit-selection").textContent = `完整轨迹 · ${current.frameCount} 帧 · ${current.segments.length} 段已标注`;
    $("fit-selection-hint").textContent = current.draft ? "请先提交或取消正在编辑的区间。" : "按时间顺序处理全部标注；未标注区间作为 DemoSkill 保留。拟合核心之外的过渡采样保持原始值。";
    $("fit-run").disabled = !!current.draft || state.starting || state.job?.phase === "running";
    $("fit-run").textContent = state.starting || state.job?.phase === "running" ? "正在拟合完整轨迹…" : "拟合并回填完整轨迹";
    if (!state.job) return;
    const outdated = stale();
    $("fit-play-segment").disabled = outdated || state.job.phase !== "done";
    $("fit-play-all").disabled = outdated || state.job.phase !== "done";
    if (outdated) clearViewer();
    const labels = { running: "拟合中", error: "拟合失败", done: state.job.accepted ? "通过校验" : "未通过校验" };
    $("fit-state").textContent = outdated ? "标注已变化" : labels[state.job.phase] || "未开始";
    $("fit-state").className = `status-pill ${outdated ? "dirty" : state.job.phase === "done" && state.job.accepted ? "saved" : state.job.phase === "error" ? "error" : ""}`;
    const link = $("fit-download");
    const taskLink = $("fit-download-task");
    const taskReady = !outdated && state.job.phase === "done" && state.job.run_task_available;
    taskLink.setAttribute("aria-disabled", String(!taskReady));
    if (taskReady) taskLink.href = `/api/fitting/jobs/${state.job.job_id}/run-task`;
    else taskLink.removeAttribute("href");
    $("fit-export-status").textContent = outdated ? "" : state.job.run_task_error ? `运行包导出失败：${state.job.run_task_error}。NPZ 拟合结果仍可下载。` : taskReady ? "运行包包含配置、模型和原生技能参数；仅 DemoSkill 保存原始 DOF × n_frame tensor。解压后按 README 执行 run-task。" : "";
    link.setAttribute("aria-disabled", String(outdated || state.job.phase !== "done"));
    if (outdated) {
      link.removeAttribute("href");
      notify("标注已变化，当前结果属于先前修订。请重新拟合后下载。", "warning");
    } else if (state.job.phase === "done") link.href = `/api/fitting/jobs/${state.job.job_id}/bundle`;
  }

  function readInputs() {
    const result = Object.fromEntries(Object.entries(fields).map(([key, id]) => [key, $(id).value.trim()]));
    for (const key of ["time_offset", "joint_dt", "env_index"]) result[key] = result[key] === "" ? null : Number(result[key]);
    result.phase_hints = $("fit-phase-hints").value.trim() ? JSON.parse($("fit-phase-hints").value) : null;
    result.fit_config = $("fit-config").value.trim() ? JSON.parse($("fit-config").value) : {};
    if (!result.joints || !result.robot_profile) throw new Error("请填写关节记录路径和机器人配置路径。");
    if (!result.qpos_semantics || result.time_offset === null) throw new Error("请显式选择数据语义，并填写视频零时刻对应的关节时间。");
    return result;
  }

  async function start() {
    if (state.starting || state.job?.phase === "running") return;
    state.starting = true;
    selectionChanged();
    try {
      const inputs = readInputs();
      await saveAnnotations();
      const current = context();
      const job = await jsonRequest("/api/fitting/jobs", {
        method: "POST", headers: { "Content-Type": "application/json", "X-Annotation-Token": current.token },
        body: JSON.stringify({ scope: "episode", expected_revision: current.revision, inputs }),
      });
      state.job = { ...job, revision: current.revision };
      clearViewer();
      state.data = null;
      $("fit-result").hidden = true;
      notify(job.message);
      clearTimeout(state.timer);
      void poll();
    } catch (error) { notify(error.message, "error"); }
    finally { state.starting = false; selectionChanged(); }
  }

  async function poll(initial = false) {
    clearTimeout(state.timer);
    try {
      const job = await jsonRequest("/api/fitting");
      if (initial) {
        const defaults = job.inputs || job.defaults;
        for (const [key, id] of Object.entries(fields)) $(id).value = defaults[key] ?? (key === "participant" ? "auto" : "");
        $("fit-phase-hints").value = defaults.phase_hints ? JSON.stringify(defaults.phase_hints, null, 2) : "";
        $("fit-config").value = Object.keys(defaults.fit_config || {}).length ? JSON.stringify(defaults.fit_config, null, 2) : "";
        if (defaults.scene || defaults.joints) void prepareScene();
      }
      if (!job.job_id) return;
      state.job = { ...job, revision: job.annotation_revision };
      notify(job.error || job.message, job.error ? "error" : job.phase === "done" && !job.accepted ? "warning" : "");
      selectionChanged();
      if (job.phase === "running") state.timer = setTimeout(() => void poll(), 600);
      else if (job.phase === "done" && !stale() && !state.data && !state.loading) {
        state.loading = true;
        try {
          const data = await jsonRequest(`/api/fitting/jobs/${job.job_id}/preview`);
          if (state.job.job_id === job.job_id) {
            state.data = data;
            showResult();
          }
        } finally { state.loading = false; }
      }
    } catch (error) {
      notify(`无法读取拟合任务：${error.message}`, "error");
      // Retain the running job guard; polling failure is not solver completion.
      if (state.job?.phase === "running") state.timer = setTimeout(() => void poll(), 1500);
    }
  }

  function showResult() {
    const { report } = state.data;
    const selection = state.job.selection;
    const core = selection.core_interval || selection;
    $("fit-result-title").textContent = report.scope === "episode" ? `完整轨迹 · ${report.atomic_skill_count} 段原子技能 · ${report.demo_skill_count} 段 DemoSkill` : `${selection.label} · ${report.skill_id} · 拟合 [${core.start_frame}, ${core.end_frame}]`;
    const parts = Object.values(report.selected_participants || {}).map((p) => `${p.motion} / ${p.grasp}`).join("，");
    const graspRotation = report.parameters.grasp_rotation_error_rad;
    $("fit-binding").textContent = `${parts ? `参与部件：${parts}` : ""}${report.scene ? ` · 目标物体：${report.scene.object_id} · 有效抓取候选：${report.scene.candidate_count}` : ""}${Number.isFinite(graspRotation) ? ` · 抓取旋转差：${(graspRotation * 180 / Math.PI).toFixed(1)}°` : ""}`;
    $("fit-result").hidden = false;
    $("fit-inputs").open = false;
    $("fit-proposed-legend").hidden = report.accepted;
    $("fit-joint").replaceChildren(...state.data.joint_names.map((name, index) => {
      const option = document.createElement("option"); option.value = index; option.textContent = name; return option;
    }));
    const tcp = Math.max(0, ...Object.values(report.tcp_errors).map((value) => value.native_tcp_rmse_m));
    const metrics = [
      [report.accepted ? "通过" : "未通过", "独立校验"],
      [report.native_joint_rmse.toExponential(2), "原生关节 RMSE · 原始关节单位"],
      [`${(tcp * 1000).toFixed(3)} mm`, "最大参与臂 TCP RMSE"],
      [`${report.stop - report.start} 个采样`, "关节采样 · 原时钟"],
    ];
    $("fit-metrics").replaceChildren(...metrics.map(([value, label]) => {
      const node = document.createElement("div"); node.className = "fit-metric";
      const strong = document.createElement("strong"); strong.textContent = value;
      node.append(strong, document.createTextNode(label)); return node;
    }));
    $("fit-parameters").replaceChildren();
    for (const [name, value] of Object.entries(report.parameters)) {
      if (typeof value !== "number") continue;
      const label = document.createElement("dt"), number = document.createElement("dd");
      label.textContent = name;
      number.textContent = `${Number(value.toPrecision(6))}${/distance|height/.test(name) ? " m" : /angle/.test(name) ? " rad" : ""}`;
      $("fit-parameters").append(label, number);
    }
    $("fit-report").textContent = JSON.stringify(report, null, 2);
    if (report.scope === "episode") {
      const summary = (report.fits || []).map((fit) => {
        const rotation = fit.parameters.grasp_rotation_error_rad;
        const interval = fit.video_interval;
        return `${fit.annotation_segment?.label || fit.skill_id}${interval ? ` [${interval.start_frame}, ${interval.end_frame}] 帧` : ""}${fit.scene ? ` · ${fit.scene.object_id}` : ""}${Number.isFinite(rotation) ? ` · 抓取旋转差 ${(rotation * 180 / Math.PI).toFixed(1)}°` : ""}`;
      });
      $("fit-binding").textContent = [parts ? `参与部件：${parts}` : "", ...summary].filter(Boolean).join("；");
    }
    if (report.failures.length) notify(`${state.job.message}\n${report.failures.join("\n")}`, "warning");
    if (state.data.total_joints > state.data.joint_names.length) notify("仅预览前 128 个关节；完整 NPZ 包含所有关节。");
    clearViewer();
    if (state.data.viewer.phase === "ready") {
      viewer.job = state.job.job_id;
      const url = new URL(window.location.href);
      url.port = String(state.data.viewer.port);
      url.pathname = "/"; url.search = ""; url.hash = "";
      $("fit-viewer").src = url.href;
      $("fit-viewer").hidden = false;
      syncViewer();
    } else $("fit-viewer-status").textContent = state.data.viewer.error || "Viser 模型未就绪。";
    selectionChanged(); draw();
  }

  function clearViewer() {
    viewer.job = null; viewer.pending = null;
    clearTimeout(viewer.timer);
    viewer.timer = null;
    $("fit-viewer").hidden = true;
    $("fit-viewer").removeAttribute("src");
    $("fit-viewer-status").textContent = "";
  }

  function syncViewer() {
    if (!viewer.job || stale()) return;
    viewer.pending = { job: viewer.job, time: state.time };
    scheduleViewer();
  }

  function scheduleViewer() {
    if (viewer.sending || viewer.timer !== null || !viewer.pending) return;
    viewer.timer = setTimeout(() => {
      viewer.timer = null;
      void pumpViewer();
    }, Math.max(0, viewer.nextAt - performance.now()));
  }

  async function pumpViewer() {
    if (viewer.sending || !viewer.pending) return;
    viewer.sending = true;
    viewer.nextAt = performance.now() + 34;
    const task = viewer.pending;
    viewer.pending = null;
    try {
      const result = await jsonRequest(`/api/fitting/jobs/${task.job}/viewer`, {
        method: "POST", headers: { "Content-Type": "application/json", "X-Annotation-Token": context().token },
        body: JSON.stringify({ time_seconds: task.time }),
      });
      if (viewer.job === task.job) {
        if (result.phase !== "ready") throw new Error(result.error || "Viser 模型未就绪。");
        $("fit-viewer-status").textContent = `与视频同步 · ${task.time.toFixed(3)} s${state.data.report.accepted ? "" : " · 未通过校验的区间保留原始轨迹"}`;
      }
    } catch (error) {
      if (viewer.job === task.job) {
        $("fit-viewer-status").textContent = `模型同步失败：${error.message}`;
        viewer.pending = null;
      }
    } finally {
      viewer.sending = false;
      // Only the most recent displayed time is retained, at most 30 updates/s.
      scheduleViewer();
    }
  }

  function chart(canvas, errors = false) {
    const s = surface(canvas);
    if (!s) return;
    const { ctx, width, height } = s, data = state.data, joint = Number($("fit-joint").value);
    const left = 62, top = 21, right = width - 14, bottom = height - 27;
    const times = data.timestamps, t0 = times[0], span = Math.max(1e-9, times.at(-1) - t0);
    const x = (time) => left + (time - t0) / span * (right - left);
    const names = errors ? series().filter((name) => name !== "original") : series();
    const values = {};
    let low = errors ? 0 : Infinity, high = errors ? 0 : -Infinity;
    for (const name of names) {
      values[name] = data[`${name}_qpos`].map((q, i) => q[joint] - (errors ? data.original_qpos[i][joint] : 0));
      for (const value of values[name]) { low = Math.min(low, value); high = Math.max(high, value); }
    }
    let padding = Math.max((high - low) * 0.12, errors ? 1e-7 : 1e-4);
    low -= padding; high += padding;
    const y = (value) => bottom - (value - low) / (high - low) * (bottom - top);
    ctx.font = "10px ui-monospace,monospace"; ctx.fillStyle = "#8496b1";
    ctx.fillText(errors ? "相对原始的误差" : "关节位置 · 原始关节单位", left, 13);
    for (let i = 0; i <= (errors ? 2 : 4); i++) {
      const value = low + (high - low) * i / (errors ? 2 : 4), py = y(value);
      ctx.strokeStyle = "#253248"; ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py); ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(Math.abs(value) < 0.001 && value !== 0 ? value.toExponential(1) : value.toFixed(3), left - 6, py + 3);
    }
    for (let i = 0; i <= 4; i++) {
      const time = t0 + span * i / 4;
      ctx.textAlign = "center"; ctx.fillText(`${(time - data.report.time_offset).toFixed(2)}s`, x(time), height - 9);
    }
    ctx.save(); ctx.beginPath(); ctx.rect(left, top, right - left, bottom - top); ctx.clip();
    const spans = data.report.timeline?.filter((s) => s.kind === "atomic_skill") || [{ start: data.report.start, stop: data.report.stop }];
    ctx.fillStyle = "#66d5bc12";
    for (const interval of spans) {
      const a = nearest(data.indices, interval.fit_start ?? interval.start), b = nearest(data.indices, (interval.fit_stop ?? interval.stop) - 1);
      ctx.fillRect(x(times[a]), top, Math.max(1, x(times[b]) - x(times[a])), bottom - top);
    }
    for (const name of names) {
      ctx.strokeStyle = colors[name]; ctx.lineWidth = name === "original" ? 2.8 : 1.5;
      ctx.setLineDash(name === "native" ? [5, 3] : name === "proposed" ? [2, 3] : []);
      ctx.beginPath(); values[name].forEach((v, i) => i ? ctx.lineTo(x(times[i]), y(v)) : ctx.moveTo(x(times[i]), y(v))); ctx.stroke();
    }
    ctx.setLineDash([]);
    const time = state.hover ?? state.time + data.report.time_offset;
    if (time >= t0 && time <= times.at(-1)) {
      ctx.strokeStyle = "#d6e2f0aa"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x(time), top); ctx.lineTo(x(time), bottom); ctx.stroke();
    }
    ctx.restore(); ctx.textAlign = "left";
  }

  function draw() {
    if (!state.data) return;
    chart($("fit-joint-chart")); chart($("fit-error-chart"), true);
    const data = state.data, joint = Number($("fit-joint").value), index = nearest(data.timestamps, state.hover ?? state.time + data.report.time_offset);
    $("fit-cursor").textContent = `预览采样 · 视频 ${(data.timestamps[index] - data.report.time_offset).toFixed(3)}s · ${data.joint_names[joint]} · 原始 ${data.original_qpos[index][joint].toFixed(6)} · 技能 ${data.native_qpos[index][joint].toFixed(6)} · 回填 ${data.patched_qpos[index][joint].toFixed(6)}`;
  }

  for (const id of ["fit-joint-chart", "fit-error-chart"]) {
    const canvas = $(id);
    const timeAt = (event) => {
      const bounds = canvas.getBoundingClientRect(), data = state.data;
      const fraction = Math.max(0, Math.min(1, (event.clientX - bounds.left - 62) / (bounds.width - 76)));
      return data.timestamps[0] + fraction * (data.timestamps.at(-1) - data.timestamps[0]);
    };
    canvas.addEventListener("pointermove", (event) => { if (state.data) { state.hover = timeAt(event); draw(); } });
    canvas.addEventListener("pointerleave", () => { state.hover = null; draw(); });
    canvas.addEventListener("click", async (event) => {
      if (!state.data || state.seeking) return;
      state.seeking = true;
      try { await seekTime(Math.max(0, timeAt(event) - state.data.report.time_offset)); }
      catch (error) { notify(error.message, "error"); }
      finally { state.seeking = false; }
    });
  }
  $("fit-joint").onchange = draw;
  $("fit-run").onclick = () => void start();
  $("fit-load-scene").onclick = () => void prepareScene();
  $("fit-scene").addEventListener("change", () => void prepareScene());
  $("fit-joints").addEventListener("change", () => void prepareScene());
  $("fit-play-segment").onclick = () => {
    if (!state.job?.selection || stale()) return;
    playRange(state.job.selection.start_frame, state.job.selection.end_frame);
  };
  $("fit-play-all").onclick = () => { if (!stale()) playRange(); };
  new ResizeObserver(draw).observe($("fit-result"));
  return { initialize: () => void poll(true), selectionChanged, displayTime: (time) => { state.time = time; draw(); syncViewer(); } };
}
