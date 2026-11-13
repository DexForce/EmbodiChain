// Copyright (c) 2021-2026 DexForce Technology Co., Ltd. Licensed under Apache-2.0.
// One in-flight cursor request and one latest time, independent of video decoding.
export function createRobotPreview({ context, seekFrame }) {
  const $ = (id) => document.getElementById(id);
  const frame = $("fit-robot-frame"), button = $("fit-robot-load"), message = $("fit-robot-message"), link = $("fit-robot-link");
  const scrubber = $("fit-robot-scrubber"), scope = $("fit-robot-scope"), imageCanvas = $("fit-robot-image");
  let selection = null, current = null;
  let job = null, generation = 0, ready = false, busy = false, time = 0, pending = null, sent = null, timer = null, polling = null, controller = null;
  const endpoint = () => `/api/fitting/jobs/${job}/robot-preview`;

  function bounds() {
    if (scope.value === "all" || !selection) return [0, context().frameCount - 1];
    const interval = scope.value === "core" && selection.core_interval ? selection.core_interval : selection;
    return [interval.start_frame, interval.end_frame];
  }

  function cursor() {
    const [first, last] = bounds(), value = Math.max(first, Math.min(last, current ?? first));
    scrubber.value = value;
    scrubber.setAttribute("aria-valuetext", `视频第 ${value} 帧`);
    $("fit-robot-current").textContent = current === null ? "等待视频帧" : `第 ${current} 帧 · ${time.toFixed(3)} s`;
    $("fit-robot-position").textContent = current === null ? "" : current < first || current > last ? "当前帧在查看范围外" : `${current - first + 1} / ${last - first + 1} 帧`;
    $("fit-robot-previous").disabled = !job || value <= first;
    $("fit-robot-next").disabled = !job || value >= last;
  }

  function range() {
    const [first, last] = bounds();
    scrubber.min = first; scrubber.max = last; scrubber.disabled = !job; scope.disabled = !job;
    const coreOption = scope.querySelector('option[value="core"]');
    coreOption.hidden = coreOption.disabled = !selection?.core_interval;
    $("fit-robot-range").textContent = `${scope.value === "all" || !selection ? "视频" : selection.label} · ${first}–${last} · 共 ${last - first + 1} 帧`;
    cursor();
  }

  function seek(index) {
    if (!job) return;
    const [first, last] = bounds(), value = Math.max(first, Math.min(last, index));
    scrubber.value = value;
    scrubber.setAttribute("aria-valuetext", `定位到视频第 ${value} 帧`);
    // The main decoder owns exact frame indices and discards stale responses.
    // Labels, the image and both robot series advance only after that frame decodes.
    seekFrame(value);
  }

  async function request(url, value, signal) {
    const response = await fetch(url, value === undefined ? { signal } : {
      method: "POST", headers: { "Content-Type": "application/json", "X-Annotation-Token": context().token },
      body: JSON.stringify({ time_seconds: value }), signal,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "无法连接模型预览。");
    return data;
  }

  function clear() {
    generation++; clearTimeout(timer); clearTimeout(polling); controller?.abort(); controller = null;
    ready = false; busy = false; pending = null; sent = null;
    frame.hidden = true; frame.removeAttribute("src"); link.hidden = true; link.removeAttribute("href");
    $("fit-robot-placeholder").hidden = false;
  }

  async function flush() {
    if (!ready || busy || pending === null || pending === sent) return;
    const version = generation, value = pending;
    pending = null; busy = true;
    try {
      await request(endpoint(), value, controller.signal);
      if (version === generation) sent = value;
    } catch (error) {
      if (version === generation && error.name !== "AbortError") { ready = false; message.textContent = error.message; }
    } finally {
      if (version === generation) {
        busy = false;
        // Scrubbing replaces pending time; requests never accumulate.
        if (pending !== null) timer = setTimeout(() => void flush(), 33);
      }
    }
  }

  async function load() {
    if (!job || button.disabled) return;
    const version = generation;
    button.disabled = true; controller = new AbortController();
    message.textContent = "正在加载机器人模型…";
    try {
      let status = await request(endpoint(), time, controller.signal);
      const poll = async () => {
        if (version !== generation) return;
        if (status.phase === "error") throw new Error(status.error);
        if (status.phase === "ready") {
          const url = new URL(location.href);
          url.port = String(status.port); url.pathname = "/"; url.search = ""; url.hash = "";
          frame.src = url.href; frame.hidden = false; link.href = url.href; link.hidden = false;
          $("fit-robot-placeholder").hidden = true;
          ready = true; pending = time;
          message.textContent = "实体为原始机械臂，青色虚影为所选轨迹。可在模型控件中切换候选、回填或原生技能。";
          void flush();
          return;
        }
        polling = setTimeout(async () => {
          try { status = await request(endpoint(), undefined, controller.signal); await poll(); }
          catch (error) { if (version === generation && error.name !== "AbortError") { message.textContent = `模型预览不可用：${error.message}`; button.disabled = false; } }
        }, 300);
      };
      await poll();
    } catch (error) {
      if (version === generation && error.name !== "AbortError") { message.textContent = `模型预览不可用：${error.message}`; button.disabled = false; }
    }
  }

  button.onclick = () => void load();
  scrubber.oninput = () => seek(Number(scrubber.value));
  scrubber.addEventListener("keydown", (event) => {
    // Native range keys stay within the selected interval, including its endpoints.
    if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"].includes(event.key)) event.stopPropagation();
  });
  scope.onchange = () => { range(); seek(Number(scrubber.value)); };
  $("fit-robot-previous").onclick = () => seek(Number(scrubber.value) - 1);
  $("fit-robot-next").onclick = () => seek(Number(scrubber.value) + 1);
  window.addEventListener("pagehide", clear);
  return {
    reset(jobId, accepted, interval = null) {
      clear(); job = jobId; selection = interval; button.disabled = !job; scope.value = "annotation";
      range();
      if (job && (current === null || current < Number(scrubber.min) || current > Number(scrubber.max))) seek(Number(scrubber.value));
      message.textContent = accepted ? "原始机械臂实体 + 连续回填虚影，与视频同步。" : "原始机械臂实体 + 未通过校验的候选虚影，与视频同步。";
    },
    invalidate() { clear(); job = null; button.disabled = true; range(); message.textContent = "标注已变化，请重新拟合后加载机器人模型。"; },
    displayFrame(index, value, image) {
      current = index; time = value;
      // Copy the already decoded bitmap; no second fetch, decoder or blob lifetime.
      if (imageCanvas.width !== image.naturalWidth || imageCanvas.height !== image.naturalHeight) {
        imageCanvas.width = image.naturalWidth; imageCanvas.height = image.naturalHeight;
      }
      imageCanvas.getContext("2d").drawImage(image, 0, 0);
      imageCanvas.setAttribute("aria-label", `视频第 ${index} 帧`);
      cursor();
      if (ready) { pending = value; if (!busy) void flush(); }
    },
  };
}
