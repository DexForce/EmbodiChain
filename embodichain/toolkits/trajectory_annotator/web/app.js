// Copyright (c) 2021-2026 DexForce Technology Co., Ltd. Licensed under Apache-2.0.
// Completed atomic skill intervals are immutable snapshots; draft and displayed-frame state
// stay separate so neither an in-flight seek nor autosave can move an endpoint.
import { createFittingPanel } from "/static/fitting.js";
const $ = (id) => document.getElementById(id);
let fittingPanel = null;
const state = {
  project: null, skillIntervals: [], selected: null, draft: null, lastLabel: "",
  displayed: null, target: 0, delay: 1 / 30, playing: false, loading: false,
  serial: 0, pending: null, inFlight: false, timer: null, imageUrl: null,
  speed: 1, zoom: 1, displayTimes: [], undo: [], redo: [],
  version: 0, savedVersion: 0, persisted: false, saving: false,
  saveTimer: null, saveError: "", conflict: false, token: "", autosave: 1000,
};
const clone = (value) => structuredClone(value);
const maximum = () => state.project.document.source.frame_count - 1;
const selected = () => state.skillIntervals.find((skillInterval) => skillInterval.id === state.selected);
const dirty = () => state.version !== state.savedVersion;

function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}

async function checked(response) {
  if (response.ok) return response;
  let detail = `${response.status} ${response.statusText}`;
  try {
    const body = await response.json();
    detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
  } catch { /* Keep the HTTP error if the server did not send JSON. */ }
  const error = new Error(detail);
  error.status = response.status;
  throw error;
}

function timecode(seconds) {
  const total = Math.round(seconds * 1000);
  const hours = Math.floor(total / 3600000);
  const minutes = Math.floor(total / 60000) % 60;
  const rest = (Math.floor(total / 1000) % 60).toString().padStart(2, "0");
  return `${hours ? `${hours.toString().padStart(2, "0")}:` : ""}${minutes.toString().padStart(2, "0")}:${rest}.${(total % 1000).toString().padStart(3, "0")}`;
}

function renderTransport() {
  $("play").textContent = state.playing ? "Ⅱ 暂停" : "▶ 播放";
  $("play").setAttribute("aria-label", state.playing ? "暂停" : "播放");
  $("playback-state").textContent = state.playing ? "播放中" : "已暂停";
  $("loading-frame").hidden = !state.loading;
  $("mark-start").disabled = state.displayed === null || (state.loading && !state.playing);
  $("mark-end").disabled = $("mark-start").disabled;
}

function pause(freeze = true) {
  state.playing = false;
  clearTimeout(state.timer);
  if (freeze) {
    state.serial += 1;
    state.pending = null;
    state.loading = false;
    state.target = state.displayed ?? 0;
  }
  renderTransport();
}

function requestFrame(index, playback = false) {
  if (!state.project) return;
  if (!playback) pause(false);
  if (!Number.isInteger(index) || index < 0 || index > maximum()) {
    message(`帧号必须是 0–${maximum()} 之间的整数。`, true);
    return;
  }
  state.serial += 1;
  state.target = index;
  state.pending = { index, serial: state.serial };
  state.loading = true;
  renderTransport();
  void pumpFrames();
}

async function pumpFrames() {
  if (state.inFlight) return;
  state.inFlight = true;
  try {
    while (state.pending) {
      const task = state.pending;
      state.pending = null;
      const started = performance.now();
      let url = null;
      try {
        const response = await checked(await fetch(`/api/frames/${task.index}?request_id=${task.serial}`));
        if (task.serial !== state.serial) continue;
        const index = Number(response.headers.get("X-Frame-Index"));
        const requestId = Number(response.headers.get("X-Request-ID"));
        if (index !== task.index || requestId !== task.serial) throw new Error("返回的画面与请求帧号不一致。请重新定位。");
        const blob = await response.blob();
        if (task.serial !== state.serial) continue;
        url = URL.createObjectURL(blob);
        const image = new Image();
        image.src = url;
        await image.decode();
        if (task.serial !== state.serial) continue;
        image.id = "video-frame";
        image.alt = `视频第 ${index} 帧`;
        $("video-frame").replaceWith(image);
        if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
        state.imageUrl = url;
        url = null;
        state.displayed = index;
        state.delay = Number(response.headers.get("X-Frame-Delay"));
        state.loading = false;
        $("frame-current").textContent = index;
        $("frame-time").textContent = timecode(Number(response.headers.get("X-Frame-Time")));
        fittingPanel?.displayTime(Number(response.headers.get("X-Frame-Time")));
        $("scrubber").value = index;
        // Do not change a number the user is still typing.
        if (document.activeElement !== $("goto-frame")) $("goto-frame").value = index;
        $("playhead").hidden = false;
        $("playhead").style.left = `${frameCenterPercent(index)}%`;
        $("timeline").setAttribute("aria-valuenow", String(index));
        $("timeline").setAttribute("aria-valuetext", `第 ${index} 帧`);
        renderTransport();
        if (state.playing) {
          state.displayTimes.push(performance.now());
          state.displayTimes = state.displayTimes.slice(-20);
          if (state.displayTimes.length > 1) {
            const fps = (state.displayTimes.length - 1) * 1000 / (state.displayTimes.at(-1) - state.displayTimes[0]);
            $("actual-fps").textContent = `实际 ${fps.toFixed(1)} fps`;
          }
          if (index === maximum()) pause(false);
          else {
            const wait = Math.max(0, state.delay * 1000 / state.speed - (performance.now() - started));
            state.timer = setTimeout(() => {
              if (state.playing) requestFrame(state.displayed + 1, true);
            }, wait);
          }
        }
      } catch (error) {
        if (task.serial === state.serial) {
          pause();
          message(`读取画面失败：${error.message}`, true);
        }
      } finally {
        if (url) URL.revokeObjectURL(url);
      }
    }
  } finally {
    state.inFlight = false;
  }
}

function step(delta) {
  if (!state.project) return;
  const base = state.playing ? state.displayed : (state.loading ? state.target : state.displayed);
  requestFrame(Math.max(0, Math.min(maximum(), (base ?? 0) + delta)));
}

function play() {
  if (!state.project || state.displayed === null) return;
  if (state.playing) { pause(); return; }
  pause();
  state.playing = true;
  state.displayTimes = [];
  requestFrame(state.displayed === maximum() ? 0 : state.displayed + 1, true);
  renderTransport();
}

function skillIntervalError(draft) {
  if (!draft) return "";
  if (!draft.label.trim()) return "请输入原子技能名称。";
  if (!Number.isInteger(draft.start_frame) || !Number.isInteger(draft.end_frame)) return "请设置整数起点和终点。";
  if (!(0 <= draft.start_frame && draft.start_frame <= draft.end_frame && draft.end_frame <= maximum())) return `起止帧需满足 0 ≤ 起点 ≤ 终点 ≤ ${maximum()}。`;
  const conflict = state.skillIntervals.find((skillInterval) => skillInterval.id !== draft.id && draft.start_frame <= skillInterval.end_frame && skillInterval.start_frame <= draft.end_frame);
  return conflict ? `与“${conflict.label}” [${conflict.start_frame}, ${conflict.end_frame}] 重叠。` : "";
}

function renderDraft(syncInputs = true) {
  const draft = state.draft;
  $("editor-title").textContent = draft?.id ? "编辑原子技能区间" : "新建原子技能区间";
  $("draft-status").textContent = draft ? "草稿未提交" : "在起点按 I 开始";
  $("start-frame").disabled = !draft;
  $("end-frame").disabled = !draft;
  $("note").disabled = !draft;
  $("cancel-draft").disabled = !draft;
  if (syncInputs) {
    $("label").value = draft?.label ?? state.lastLabel;
    $("start-frame").value = draft?.start_frame ?? "";
    $("end-frame").value = draft?.end_frame ?? "";
    $("note").value = draft?.note ?? "";
  }
  const error = skillIntervalError(draft);
  $("draft-error").textContent = error;
  $("commit").disabled = !draft || !!error;
  const length = draft && Number.isInteger(draft.start_frame) && Number.isInteger(draft.end_frame) && draft.end_frame >= draft.start_frame ? draft.end_frame - draft.start_frame + 1 : null;
  $("skill-interval-length").textContent = length === null ? "— 帧" : `${length} 帧`;
  $("draft-range").hidden = !draft || !Number.isInteger(draft.start_frame);
  if (!$("draft-range").hidden) {
    positionSkillInterval($("draft-range"), draft.start_frame, draft.start_frame + Math.max(1, length ?? 1) - 1);
    $("draft-range").firstElementChild.textContent = draft.label || "草稿";
  }
  renderSelection();
  renderSave();
}

function mark(endpoint) {
  if (state.displayed === null || (state.loading && !state.playing)) return;
  pause();
  if (!state.draft) {
    if (endpoint === "end_frame") { message("请先按 I 设置起点。", true); return; }
    state.draft = { id: null, label: $("label").value || state.lastLabel, start_frame: null, end_frame: null, note: "" };
  }
  state.draft[endpoint] = state.displayed;
  renderDraft();
  message(`已将第 ${state.displayed} 帧设为${endpoint === "start_frame" ? "起点" : "终点"}。`);
}

function historySnapshot() { return { skillIntervals: clone(state.skillIntervals), selected: state.selected }; }
function beforeMutation() {
  state.undo.push(historySnapshot());
  state.undo = state.undo.slice(-100);
  state.redo = [];
}
function afterMutation() {
  state.skillIntervals.sort((a, b) => a.start_frame - b.start_frame || a.end_frame - b.end_frame);
  state.version += 1;
  state.saveError = "";
  renderSkillIntervals();
  renderDraft();
  clearTimeout(state.saveTimer);
  if (!state.conflict) state.saveTimer = setTimeout(() => void save(), state.autosave);
}

function commit() {
  const draft = state.draft;
  if (!draft || skillIntervalError(draft)) return;
  beforeMutation();
  const skillInterval = { ...draft, id: draft.id || `skill_${crypto.randomUUID()}`, label: draft.label.trim() };
  state.skillIntervals = state.skillIntervals.filter((item) => item.id !== skillInterval.id);
  state.skillIntervals.push(skillInterval);
  state.selected = skillInterval.id;
  state.lastLabel = skillInterval.label;
  state.draft = null;
  afterMutation();
  $("video-stage").focus({ preventScroll: true });
  message(`已提交“${skillInterval.label}” [${skillInterval.start_frame}, ${skillInterval.end_frame}]。`);
}

function edit() {
  const skillInterval = selected();
  if (!skillInterval) return;
  if (state.draft) { message("请先提交或取消当前草稿。", true); return; }
  pause();
  state.draft = clone(skillInterval);
  renderDraft();
  message("编辑中；提交后替换原标注，Esc 取消。 ");
}
function cancelDraft() {
  state.draft = null;
  renderDraft();
  $("video-stage").focus({ preventScroll: true });
}
function removeSelected() {
  if (state.draft) { message("请先提交或取消当前草稿。", true); return; }
  const skillInterval = selected();
  if (!skillInterval) return;
  beforeMutation();
  state.skillIntervals = state.skillIntervals.filter((item) => item.id !== skillInterval.id);
  state.selected = null;
  afterMutation();
  message(`已删除“${skillInterval.label}”，可撤销。`);
}
function undo(redo = false) {
  if (state.draft) { message("请先提交或取消当前草稿。", true); return; }
  const source = redo ? state.redo : state.undo;
  const destination = redo ? state.undo : state.redo;
  if (!source.length) return;
  destination.push(historySnapshot());
  const snapshot = source.pop();
  state.skillIntervals = snapshot.skillIntervals;
  state.selected = snapshot.selected;
  afterMutation();
  message(redo ? "已重做。" : "已撤销。 ");
}

function selectSkillInterval(id) {
  state.selected = id;
  renderSkillIntervals();
}
function renderSelection() {
  const skillInterval = selected();
  $("selection-info").textContent = skillInterval ? `${skillInterval.label} · [${skillInterval.start_frame}, ${skillInterval.end_frame}]${skillInterval.note ? ` · ${skillInterval.note}` : ""}` : "选择一段标注以定位或编辑";
  $("to-start").disabled = !skillInterval;
  $("to-end").disabled = !skillInterval;
  $("edit").disabled = !skillInterval || !!state.draft;
  $("delete").disabled = !skillInterval || !!state.draft;
  $("undo").disabled = !state.undo.length || !!state.draft;
  $("redo").disabled = !state.redo.length || !!state.draft;
  fittingPanel?.selectionChanged();
}
function renderSkillIntervals() {
  $("skill-interval-list").replaceChildren();
  $("timeline-skill-intervals").replaceChildren();
  $("label-suggestions").replaceChildren();
  state.skillIntervals.forEach((skillInterval, index) => {
    const row = document.createElement("tr");
    row.className = `annotation-row${state.selected === skillInterval.id ? " selected" : ""}`;
    row.tabIndex = 0;
    row.setAttribute("aria-label", `${skillInterval.label}，${skillInterval.start_frame} 至 ${skillInterval.end_frame} 帧`);
    const label = row.insertCell();
    const number = document.createElement("span");
    number.className = "row-number";
    number.textContent = String(index + 1).padStart(2, "0");
    label.append(number, document.createTextNode(skillInterval.label));
    label.title = skillInterval.label;
    for (const endpoint of ["start_frame", "end_frame"]) {
      const button = document.createElement("button");
      button.textContent = skillInterval[endpoint];
      button.title = `跳转到第 ${skillInterval[endpoint]} 帧`;
      button.addEventListener("click", (event) => {
        event.stopPropagation(); selectSkillInterval(skillInterval.id); requestFrame(skillInterval[endpoint]);
      });
      row.insertCell().append(button);
    }
    row.insertCell().textContent = skillInterval.end_frame - skillInterval.start_frame + 1;
    row.addEventListener("click", () => selectSkillInterval(skillInterval.id));
    row.addEventListener("keydown", (event) => {
      if (!state.draft && (event.key === " " || event.key === "Enter")) { event.preventDefault(); event.stopPropagation(); selectSkillInterval(skillInterval.id); }
    });
    $("skill-interval-list").append(row);
    const block = document.createElement("div");
    block.className = `timeline-skill-interval${state.selected === skillInterval.id ? " selected" : ""}`;
    block.dataset.skillId = skillInterval.id;
    block.setAttribute("role", "img");
    block.setAttribute("aria-label", `${skillInterval.label}，${skillInterval.start_frame} 至 ${skillInterval.end_frame} 帧`);
    positionSkillInterval(block, skillInterval.start_frame, skillInterval.end_frame);
    const caption = document.createElement("span");
    caption.className = "skill-interval-caption";
    caption.textContent = skillInterval.label;
    block.append(caption);
    block.title = `${skillInterval.label} [${skillInterval.start_frame}, ${skillInterval.end_frame}]`;
    $("timeline-skill-intervals").append(block);
  });
  [...new Set(state.skillIntervals.map((skillInterval) => skillInterval.label))].forEach((label) => {
    const option = document.createElement("option"); option.value = label; $("label-suggestions").append(option);
  });
  $("empty-list").hidden = state.skillIntervals.length > 0;
  $("skill-interval-count").textContent = state.skillIntervals.length;
  const count = state.skillIntervals.reduce((total, skillInterval) => total + skillInterval.end_frame - skillInterval.start_frame + 1, 0);
  const percent = count / (maximum() + 1) * 100;
  $("coverage-count").textContent = `${count.toLocaleString()} 帧已标注`;
  $("coverage-percent").textContent = `${percent.toFixed(1)}%`;
  $("coverage-bar").style.width = `${percent}%`;
  renderSelection();
}

function renderSave() {
  let text = state.persisted ? "已保存" : "尚未保存";
  let kind = state.persisted ? "saved" : "";
  if (dirty()) { text = "未保存"; kind = "dirty"; }
  if (state.saving) { text = "保存中…"; kind = "dirty"; }
  if (state.saveError) { text = "保存失败"; kind = "error"; }
  if (state.draft) text += " · 有草稿";
  $("save-status").textContent = text;
  $("save-status").className = `status-pill ${kind}`;
  $("save-status").title = state.saveError || "只保存已提交的区间，草稿需先提交。";
}
async function save() {
  if (!state.project) return;
  clearTimeout(state.saveTimer);
  if (state.saving) return;
  if (!dirty() && state.persisted) {
    message(state.draft ? "已完成标注均已保存；当前草稿尚未提交。" : "已完成标注均已保存。 ");
    return;
  }
  state.saving = true;
  state.saveError = "";
  renderSave();
  try {
    do {
      const version = state.version;
      const skillIntervals = clone(state.skillIntervals);
      const response = await checked(await fetch("/api/annotations", {
        method: "PUT", headers: { "Content-Type": "application/json", "X-Annotation-Token": state.token },
        body: JSON.stringify({ segments: skillIntervals, expected_revision: state.project.document.revision }),
      }));
      const document = await response.json();
      state.project.document.revision = document.revision;
      state.savedVersion = version;
      state.persisted = true;
      message(state.draft ? "已保存完成标注；当前草稿尚未提交。" : `已保存 ${skillIntervals.length} 段标注。`);
    } while (dirty());
  } catch (error) {
    state.saveError = error.message;
    if (error.status === 409) state.conflict = true;
    message(`保存失败：${error.message} 可使用“导出 JSON”保留当前标注。`, true);
  } finally {
    state.saving = false;
    renderSave();
    fittingPanel?.selectionChanged();
  }
}
function exportJson() {
  if (!state.project) return;
  const document = { ...state.project.document, segments: clone(state.skillIntervals) };
  const blob = new Blob([`${JSON.stringify(document, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = window.document.createElement("a");
  link.href = url;
  link.download = state.project.annotation_path.split(/[\\/]/).at(-1);
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  message(state.draft ? "已导出完成标注；草稿未包含在导出中。" : "已导出当前标注副本。 ");
}

function frameCenterPercent(index) {
  return (index + 0.5) / (maximum() + 1) * 100;
}
function positionSkillInterval(element, start, end) {
  const first = Math.max(0, Math.min(maximum(), start));
  const last = Math.max(first, Math.min(maximum(), end));
  element.style.left = `${first / (maximum() + 1) * 100}%`;
  element.style.width = `${(last - first + 1) / (maximum() + 1) * 100}%`;
}

function zoom(factor) {
  state.zoom = Math.max(1, Math.min(32, factor));
  $("timeline").style.width = `${state.zoom * 100}%`;
  $("zoom-value").textContent = `${state.zoom}×`;
  $("zoom-out").disabled = state.zoom === 1;
  $("zoom-in").disabled = state.zoom === 32;
}
function bindControls() {
  $("play").onclick = play;
  $("previous").onclick = () => step(-1);
  $("next").onclick = () => step(1);
  $("mark-start").onclick = () => mark("start_frame");
  $("mark-end").onclick = () => mark("end_frame");
  $("commit").onclick = commit;
  $("cancel-draft").onclick = cancelDraft;
  $("edit").onclick = edit;
  $("delete").onclick = removeSelected;
  $("undo").onclick = () => undo();
  $("redo").onclick = () => undo(true);
  $("save").onclick = () => void save();
  $("export").onclick = exportJson;
  $("to-start").onclick = () => selected() && requestFrame(selected().start_frame);
  $("to-end").onclick = () => selected() && requestFrame(selected().end_frame);
  $("speed").onchange = () => { state.speed = Number($("speed").value); };
  $("scrubber").oninput = () => requestFrame(Number($("scrubber").value));
  $("goto-frame").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.isComposing) {
      event.preventDefault();
      requestFrame($("goto-frame").value === "" ? NaN : Number($("goto-frame").value));
      $("video-stage").focus({ preventScroll: true });
    }
  });
  for (const [id, key] of [["label", "label"], ["note", "note"], ["start-frame", "start_frame"], ["end-frame", "end_frame"]]) {
    $(id).addEventListener("input", () => {
      if (state.draft) state.draft[key] = key.endsWith("_frame") ? ($(id).value === "" ? null : Number($(id).value)) : $(id).value;
      else if (key === "label") state.lastLabel = $(id).value;
      renderDraft(false);
    });
    $(id).addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.isComposing) { event.preventDefault(); $("video-stage").focus({ preventScroll: true }); }
    });
  }
  $("zoom-in").onclick = () => zoom(state.zoom * 2);
  $("zoom-out").onclick = () => zoom(state.zoom / 2);
  $("zoom-reset").onclick = () => zoom(1);
  const timeline = $("timeline");
  let timelinePointer = null;
  const seekTimeline = (event) => {
    const bounds = timeline.getBoundingClientRect();
    const index = Math.floor((event.clientX - bounds.left) / bounds.width * (maximum() + 1));
    requestFrame(Math.max(0, Math.min(maximum(), index)));
  };
  timeline.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || timelinePointer !== null) return;
    event.preventDefault();
    timelinePointer = event.pointerId;
    timeline.focus({ preventScroll: true });
    timeline.setPointerCapture(event.pointerId);
    const interval = event.target.closest("[data-skill-id]");
    if (interval) selectSkillInterval(interval.dataset.skillId);
    seekTimeline(event);
  });
  timeline.addEventListener("pointermove", (event) => {
    if (event.pointerId === timelinePointer) seekTimeline(event);
  });
  const stopTimeline = (event) => {
    if (event.pointerId !== timelinePointer) return;
    timelinePointer = null;
    if (timeline.hasPointerCapture(event.pointerId)) timeline.releasePointerCapture(event.pointerId);
  };
  timeline.addEventListener("pointerup", stopTimeline);
  timeline.addEventListener("pointercancel", stopTimeline);
  timeline.addEventListener("lostpointercapture", stopTimeline);
  window.addEventListener("keydown", keyboard);
  window.addEventListener("blur", () => pause());
  window.addEventListener("beforeunload", (event) => {
    if (dirty() || state.draft) { event.preventDefault(); event.returnValue = ""; }
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) pause(); });
}

function keyboard(event) {
  if (!state.project || event.isComposing) return;
  if ($("help-dialog").open) {
    if (event.key === "?") { event.preventDefault(); $("help-dialog").close(); }
    return;
  }
  const key = event.key.toLowerCase();
  const modifier = event.ctrlKey || event.metaKey;
  if (modifier && key === "s") { event.preventDefault(); if (!event.repeat) void save(); return; }
  if (event.target.closest("textarea, select, [contenteditable='true'], input:not([type='range'])")) return;
  if (modifier) {
    if (key === "z") { event.preventDefault(); if (!event.repeat) undo(event.shiftKey); }
    return;
  }
  if (event.altKey) return;
  if (event.repeat && !["arrowleft", "arrowright"].includes(key)) { event.preventDefault(); return; }
  const actions = {
    " ": play,
    arrowleft: () => step(event.shiftKey ? -10 : -1),
    arrowright: () => step(event.shiftKey ? 10 : 1),
    home: () => requestFrame(0), end: () => requestFrame(maximum()),
    g: () => { $("goto-frame").focus(); $("goto-frame").select(); },
    l: () => $("label").focus(), i: () => mark("start_frame"), o: () => mark("end_frame"),
    enter: commit, escape: cancelDraft, e: edit,
    delete: removeSelected, backspace: removeSelected,
    "?": () => { pause(); $("help-dialog").showModal(); },
  };
  if (actions[key]) { event.preventDefault(); actions[key](); }
}

$("help").onclick = () => { pause(); $("help-dialog").showModal(); };
$("close-help").onclick = () => $("help-dialog").close();
$("cancel-init").onclick = async () => {
  $("cancel-init").disabled = true;
  try {
    await checked(await fetch("/api/cancel", { method: "POST", headers: { "X-Annotation-Token": state.token } }));
  } catch (error) { message(error.message, true); }
};

async function initialize() {
  try {
    const status = await (await checked(await fetch("/api/status"))).json();
    state.token = status.session_token;
    if (status.phase === "error" || status.phase === "cancelled" || status.phase === "closed") {
      $("init-title").textContent = status.phase === "error" ? "无法打开标注项目" : "初始化已取消";
      $("init-message").textContent = status.error || "关闭此页后，可从命令行重新打开图像序列。";
      $("init-progress").hidden = true;
      $("cancel-init").hidden = true;
      message(status.error || "请从命令行重新打开图像序列。", true);
      return;
    }
    if (status.phase !== "ready") {
      $("init-title").textContent = status.phase === "hashing" ? "正在读取图像来源" : "正在建立帧索引";
      $("init-message").textContent = status.phase === "indexing" ? `已扫描 ${status.completed.toLocaleString()} 帧；完成后可准确逐帧标注。` : "正在校验图像来源，请稍候。";
      if (status.total > 0) { $("init-progress").max = status.total; $("init-progress").value = status.completed; }
      else $("init-progress").removeAttribute("value");
      setTimeout(() => void initialize(), 350);
      return;
    }
    const project = await (await checked(await fetch("/api/project"))).json();
    state.project = project;
    state.token = project.session_token;
    state.autosave = project.autosave_ms;
    state.skillIntervals = clone(project.document.segments);
    state.persisted = project.persisted;
    const source = project.document.source;
    $("video-name").textContent = project.source_name;
    const fps = source.fps ? `${source.fps.toFixed(2)} fps` : "";
    const timing = source.timing === "fixed_fps" ? " · 时间按 FPS 推算" : " · 视频 PTS";
    $("video-meta").textContent = `${source.width} × ${source.height} · ${source.frame_count.toLocaleString()} 帧${fps ? ` · ${fps}` : ""}${timing}`;
    $("annotation-path").textContent = project.annotation_path;
    $("frame-last").textContent = maximum();
    for (const id of ["scrubber", "goto-frame", "start-frame", "end-frame"]) $(id).max = maximum();
    $("save").disabled = false;
    $("export").disabled = false;
    $("initialization").hidden = true;
    $("workspace").hidden = false;
    document.title = `${project.source_name} · 原子技能区间标注`;
    $("timeline").setAttribute("aria-valuemax", String(maximum()));
    const tickFrames = [...new Set(Array.from({ length: 6 }, (_, index) => Math.round(index / 5 * maximum())))];
    $("timeline-ticks").replaceChildren(...tickFrames.map((frame) => {
      const tick = document.createElement("span");
      tick.className = "timeline-tick";
      tick.style.left = `${frameCenterPercent(frame)}%`;
      const caption = document.createElement("span");
      caption.textContent = frame;
      tick.append(caption);
      return tick;
    }));
    fittingPanel = createFittingPanel({
      context: () => ({ selection: selected(), revision: state.project.document.revision, dirty: dirty(), draft: state.draft, token: state.token }),
      saveAnnotations: async () => {
        if (state.draft) throw new Error("请先提交或取消当前标注草稿。");
        while (state.saving) await new Promise((resolve) => setTimeout(resolve, 50));
        await save();
        if (dirty() || state.saveError || state.conflict) throw new Error("标注尚未成功保存，请处理保存错误后重新拟合。");
      },
      seekTime: async (seconds) => {
        pause();
        const serial = state.serial;
        const response = await checked(await fetch(`/api/frame-index?time_seconds=${encodeURIComponent(seconds)}`));
        const result = await response.json();
        if (serial === state.serial) requestFrame(result.index);
      },
    });
    fittingPanel.initialize();
    bindControls(); zoom(1); renderSkillIntervals(); renderDraft(); requestFrame(0);
    message("已就绪。定位起点按 I，终点按 O，填写名称后按 Enter 提交。 ");
    $("video-stage").focus({ preventScroll: true });
  } catch (error) {
    $("init-title").textContent = "连接标注服务失败";
    $("init-message").textContent = `${error.message}。确认命令行中的服务仍在运行后刷新页面。`;
    $("init-progress").hidden = true;
    message(error.message, true);
  }
}
void initialize();
