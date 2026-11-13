// Copyright (c) 2021-2026 DexForce Technology Co., Ltd. Licensed under Apache-2.0.
// Drawing data is bounded; the downloadable NPZ retains the original sample grid.
import { createRobotPreview } from "./robot_preview.js";
const $ = (id) => document.getElementById(id);
const colors = { original: "#fb8696", native: "#f2b664", patched: "#66d5bc", proposed: "#b6a1ff" };
const fields = {
  joints: "fit-joints", robot_profile: "fit-profile", qpos_semantics: "fit-semantics",
  time_offset: "fit-offset", qpos_key: "fit-qpos-key", timestamps_key: "fit-timestamps-key",
  joint_dt: "fit-joint-dt", env_index: "fit-env-index", device: "fit-device",
};
// Standard mode removes these overrides and uses the fitter's canonical defaults.
const relaxedTolerances = {
  max_joint_rmse: 0.5, max_tcp_rmse_m: 0.1, max_rotation_rmse_rad: 1.0,
  max_contact_error_m: 0.1, max_contact_rotation_rad: 1.0,
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

export function createFittingPanel({ context, saveAnnotations, seekTime, seekFrame }) {
  const robotPreview = createRobotPreview({ context, seekFrame });
  const state = { job: null, data: null, timer: null, loading: false, starting: false, time: 0, hover: null, seeking: false };
  const camera = { yaw: -0.8, pitch: 0.5, zoom: 1, center: [0, 0, 0], radius: 1 };
  const series = () => state.data?.report.accepted ? ["original", "native", "patched"] : ["original", "native", "patched", "proposed"];
  const stale = () => !!state.job && (state.job.stale || context().dirty || state.job.revision !== context().revision);
  const key = (value) => value.toLowerCase().replace(/[_\s-]/g, "");
  const selectedSkill = () => context().skills.find((skill) => context().selection && [key(skill.id), key(skill.name)].includes(key(context().selection.label)));

  function notify(text, kind = "") {
    $("fit-message").textContent = text;
    $("fit-message").className = `fit-message ${kind}`;
  }

  function toleranceHint() {
    const mode = $("fit-tolerance").value;
    $("fit-tolerance-hint").textContent = (mode === "relaxed" ? "TCP / 阶段位置容差 10 cm，姿态容差 1 rad。" : mode === "standard" ? "使用标准误差阈值。" : "使用下方 JSON 中的误差阈值。") + "更改后重新拟合生效；关节限制独立校验。";
  }

  function fitConfig() {
    const config = $("fit-config").value.trim() ? JSON.parse($("fit-config").value) : {};
    if (!config || typeof config !== "object" || Array.isArray(config)) throw new Error("拟合配置需要 JSON 对象。");
    return config;
  }

  function syncTolerance() {
    try {
      const config = fitConfig(), entries = Object.entries(relaxedTolerances);
      $("fit-tolerance").value = entries.every(([key, value]) => config[key] === value) ? "relaxed" : entries.every(([key]) => config[key] === undefined) ? "standard" : "custom";
    } catch { $("fit-tolerance").value = "custom"; }
    toleranceHint();
  }

  function applyTolerance() {
    const mode = $("fit-tolerance").value;
    if (mode === "custom") {
      $("fit-inputs").open = true; $("fit-config").closest("details").open = true;
      $("fit-config").focus(); toleranceHint(); return;
    }
    try {
      const config = fitConfig();
      for (const [key, value] of Object.entries(relaxedTolerances)) {
        if (mode === "relaxed") config[key] = value;
        else delete config[key];
      }
      $("fit-config").value = Object.keys(config).length ? JSON.stringify(config, null, 2) : "";
      syncTolerance();
    } catch (error) { syncTolerance(); notify(error.message, "error"); }
  }

  function selectionChanged() {
    const current = context();
    const skill = selectedSkill();
    const core = current.selection?.core_interval;
    $("fit-selection").textContent = current.selection ? `${current.selection.label} · 标注 [${current.selection.start_frame}, ${current.selection.end_frame}] 帧${core ? ` · 核心 [${core.start_frame}, ${core.end_frame}]` : ""}` : "请在右侧选择一段已标注区间";
    const demo = skill?.id === "execute_demo_skill";
    $("fit-selection-hint").textContent = demo ? "DemoSkill 不拟合原子技能，也不执行轨迹回填；保留该区间的原始轨迹。" : skill && !skill.can_fit ? `${skill.name} 可保存为示教标注；当前未提供该技能的参数重建。` : core ? "仅拟合核心区间，保留标注内前后过渡动作的原始采样。" : "标注包含额外过渡动作时，可在编辑区间中单独设置拟合核心范围。";
    $("fit-skill").disabled = !!skill;
    if (skill?.can_fit) $("fit-skill").value = skill.id;
    if (skill && !skill.can_fit) $("fit-skill").value = "";
    if (!skill && !$("fit-skill").value) $("fit-skill").value = "pick_up";
    $("fit-run").disabled = !current.selection || !!current.draft || (!!skill && !skill.can_fit) || state.starting || state.job?.phase === "running";
    $("fit-run").textContent = state.starting || state.job?.phase === "running" ? "正在拟合…" : demo ? "DemoSkill · 保留原始轨迹" : "拟合并回填选中区间";
    $("fit-tolerance").disabled = (!!skill && !skill.can_fit) || state.starting || state.job?.phase === "running";
    if (!state.job) return;
    const outdated = stale();
    const labels = { running: "拟合中", error: "拟合失败", done: state.job.accepted ? "通过校验" : "未通过校验" };
    $("fit-state").textContent = outdated ? "标注已变化" : labels[state.job.phase] || "未开始";
    $("fit-state").className = `status-pill ${outdated ? "dirty" : state.job.phase === "done" && state.job.accepted ? "saved" : state.job.phase === "error" ? "error" : ""}`;
    const link = $("fit-download");
    link.setAttribute("aria-disabled", String(outdated || state.job.phase !== "done"));
    if (outdated) {
      robotPreview.invalidate();
      link.removeAttribute("href");
      notify("标注已变化，当前结果属于先前修订。请重新拟合后下载。", "warning");
    } else if (state.job.phase === "done") link.href = `/api/fitting/jobs/${state.job.job_id}/bundle`;
  }

  function readInputs() {
    const result = Object.fromEntries(Object.entries(fields).map(([key, id]) => [key, $(id).value.trim()]));
    for (const key of ["time_offset", "joint_dt", "env_index"]) result[key] = result[key] === "" ? null : Number(result[key]);
    result.phase_hints = $("fit-phase-hints").value.trim() ? JSON.parse($("fit-phase-hints").value) : null;
    result.fit_config = fitConfig();
    if (!result.joints || !result.robot_profile) throw new Error("请填写关节记录路径和机器人配置路径。");
    if (!result.qpos_semantics || result.time_offset === null) throw new Error("请显式选择数据语义，并填写视频零时刻对应的关节时间。");
    return result;
  }

  async function start() {
    if (state.starting || state.job?.phase === "running") return;
    state.starting = true;
    selectionChanged();
    try {
      if (selectedSkill() && !selectedSkill().can_fit) throw new Error("当前技能支持标注，尚不支持参数重建。");
      const inputs = readInputs();
      const selection = structuredClone(context().selection);
      await saveAnnotations();
      const current = context();
      if (!selection || JSON.stringify(selection) !== JSON.stringify(current.selection)) throw new Error("选区已变化，请重新开始拟合。");
      const job = await jsonRequest("/api/fitting/jobs", {
        method: "POST", headers: { "Content-Type": "application/json", "X-Annotation-Token": current.token },
        body: JSON.stringify({ segment_id: selection.id, expected_revision: current.revision, skill_id: $("fit-skill").value, inputs }),
      });
      state.job = { ...job, revision: current.revision };
      state.data = null;
      robotPreview.reset(null, false);
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
        for (const [key, id] of Object.entries(fields)) $(id).value = defaults[key] ?? "";
        $("fit-phase-hints").value = defaults.phase_hints ? JSON.stringify(defaults.phase_hints, null, 2) : "";
        $("fit-config").value = Object.keys(defaults.fit_config || {}).length ? JSON.stringify(defaults.fit_config, null, 2) : "";
        if (!job.job_id && Object.keys(relaxedTolerances).every((key) => defaults.fit_config?.[key] === undefined)) {
          $("fit-tolerance").value = "relaxed"; applyTolerance();
        } else syncTolerance();
        if (job.skill_id) $("fit-skill").value = job.skill_id;
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
    robotPreview.reset(state.job.job_id, report.accepted, selection);
    const boundary = report.boundary_connection;
    $("fit-boundary").hidden = !boundary;
    $("fit-boundary").textContent = boundary ? `首尾各 ${boundary.fixed_samples_per_end} 个采样保持原值 · 边界过渡目标 ${boundary.requested_duration_s} s（可用长度受技能阶段约束）` : "";
    const core = selection.core_interval || selection;
    $("fit-result-title").textContent = `${selection.label} · ${report.skill_id} · 拟合 [${core.start_frame}, ${core.end_frame}]`;
    $("fit-result").hidden = false;
    $("fit-inputs").open = false;
    $("fit-proposed-legend").hidden = report.accepted;
    $("fit-joint").replaceChildren(...state.data.joint_names.map((name, index) => {
      const option = document.createElement("option"); option.value = index; option.textContent = name; return option;
    }));
    const tcp = Math.max(...Object.values(report.tcp_errors).map((value) => value.native_tcp_rmse_m));
    const metrics = [
      [report.accepted ? "通过" : "未通过", "独立校验"],
      [report.native_joint_rmse.toExponential(2), "原生关节 RMSE · 原始关节单位"],
      [`${(tcp * 1000).toFixed(3)} mm`, "最大参与臂 TCP RMSE"],
      [`${report.stop - report.start} 个采样`, "选区关节采样 · 原时钟"],
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
    if (report.failures.length) notify(`${state.job.message}\n${report.failures.join("\n")}`, "warning");
    if (state.data.total_joints > state.data.joint_names.length) notify("仅预览前 128 个关节；完整 NPZ 包含所有关节。");
    const lower = [Infinity, Infinity, Infinity], upper = [-Infinity, -Infinity, -Infinity];
    for (const frame of state.data.original_links) for (const arm of frame) for (const point of arm) for (let axis = 0; axis < 3; axis++) {
      lower[axis] = Math.min(lower[axis], point[axis]); upper[axis] = Math.max(upper[axis], point[axis]);
    }
    camera.center = lower.map((value, axis) => (value + upper[axis]) / 2);
    camera.radius = Math.max(0.05, Math.hypot(...upper.map((value, axis) => value - lower[axis])) / 2);
    camera.yaw = -0.8; camera.pitch = 0.5; camera.zoom = 1;
    selectionChanged(); draw();
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
    const a = data.indices.indexOf(data.report.start), b = data.indices.indexOf(data.report.stop - 1);
    ctx.fillStyle = "#66d5bc0c"; ctx.fillRect(x(times[a]), top, Math.max(1, x(times[b]) - x(times[a])), bottom - top);
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

  function geometry() {
    const s = surface($("fit-geometry"));
    if (!s) return;
    const { ctx, width, height } = s, data = state.data;
    const scale = Math.min(width, height) * 0.38 / camera.radius * camera.zoom;
    const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw), cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
    function project(point) {
      const [x, y, z] = point.map((value, axis) => value - camera.center[axis]);
      return [width / 2 + (cy * x - sy * y) * scale, height / 2 - (sp * (sy * x + cy * y) + cp * z) * scale];
    }
    function line(points, color, thickness, dash = []) {
      ctx.strokeStyle = color; ctx.lineWidth = thickness; ctx.setLineDash(dash); ctx.beginPath();
      points.forEach((point, i) => { const [x, y] = project(point); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.stroke(); ctx.setLineDash([]);
    }
    const radius = camera.radius;
    for (let i = -4; i <= 4; i++) {
      const v = i * radius / 4;
      line([[camera.center[0] - radius, camera.center[1] + v, 0], [camera.center[0] + radius, camera.center[1] + v, 0]], "#27354a", 0.6);
      line([[camera.center[0] + v, camera.center[1] - radius, 0], [camera.center[0] + v, camera.center[1] + radius, 0]], "#27354a", 0.6);
    }
    const multiplier = Number($("fit-exaggeration").value);
    const index = nearest(data.geometry_times, state.time + data.report.time_offset);
    function exaggerate(point, original) { return point.map((value, axis) => original[axis] + (value - original[axis]) * multiplier); }
    for (const name of series()) {
      const arms = data[`${name}_tcp`][0].length;
      for (let arm = 0; arm < arms; arm++) {
        const points = data[`${name}_tcp`].map((value, i) => exaggerate(value[arm], data.original_tcp[i][arm]));
        line(points, `${colors[name]}99`, name === "original" ? 2.5 : 1.5, name === "native" ? [5, 3] : []);
        const skeleton = data[`${name}_links`][index][arm].map((point, node) => exaggerate(point, data.original_links[index][arm][node]));
        line(skeleton, colors[name], name === "original" ? 4 : 2, name === "native" ? [5, 3] : []);
        const [px, py] = project(exaggerate(data[`${name}_tcp`][index][arm], data.original_tcp[index][arm]));
        ctx.fillStyle = colors[name]; ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
      }
    }
    ctx.font = "10px ui-monospace,monospace";
    ["X", "Y", "Z"].forEach((name, axis) => {
      const origin = [camera.center[0] - radius * 0.8, camera.center[1] - radius * 0.8, 0];
      const end = [...origin]; end[axis] += radius * 0.35;
      line([origin, end], ["#dc7e8a", "#7aba9d", "#80a4e1"][axis], 1.4);
      const [x, y] = project(end); ctx.fillStyle = "#b4c5dc"; ctx.fillText(name, x + 3, y);
    });
    ctx.fillStyle = "#8496b1";
    ctx.fillText(`TCP · 视频 ${state.time.toFixed(3)}s · ${multiplier}×`, 12, 18);
  }

  function draw() {
    if (!state.data) return;
    chart($("fit-joint-chart")); chart($("fit-error-chart"), true); geometry();
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
  const canvas = $("fit-geometry");
  let drag = null;
  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault(); drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (drag?.id !== event.pointerId) return;
    camera.yaw += (event.clientX - drag.x) * 0.008;
    camera.pitch = Math.max(-1.5, Math.min(1.5, camera.pitch + (event.clientY - drag.y) * 0.008));
    drag.x = event.clientX; drag.y = event.clientY; draw();
  });
  const release = (event) => { if (drag?.id === event.pointerId) { drag = null; if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId); } };
  canvas.addEventListener("pointerup", release); canvas.addEventListener("pointercancel", release); canvas.addEventListener("lostpointercapture", release);
  canvas.addEventListener("wheel", (event) => { event.preventDefault(); camera.zoom = Math.max(0.2, Math.min(6, camera.zoom * Math.exp(-event.deltaY * 0.001))); draw(); }, { passive: false });
  $("fit-reset-view").onclick = () => { camera.yaw = -0.8; camera.pitch = 0.5; camera.zoom = 1; draw(); };
  $("fit-joint").onchange = draw; $("fit-exaggeration").onchange = draw; $("fit-run").onclick = () => void start();
  $("fit-tolerance").onchange = applyTolerance;
  $("fit-config").addEventListener("input", syncTolerance);
  new ResizeObserver(draw).observe($("fit-result"));
  return { initialize: () => void poll(true), selectionChanged, displayFrame: (index, time, image) => { state.time = time; robotPreview.displayFrame(index, time, image); draw(); } };
}
