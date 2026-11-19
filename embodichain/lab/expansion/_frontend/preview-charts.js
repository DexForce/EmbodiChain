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

const deviationCharts = (() => {
  const left = 48, right = 12, top = 8, bottom = 24;
  const plots = [];
  let data, playbackFrame = 0, hoverFrame = null, failed = false;
  let number, tcpNote, jointNote;
  const description = document.getElementById('deviation-description');
  const readout = document.getElementById('deviation-time');

  function updateReadout() {
    if (!data || !plots.length) return;
    const frame = hoverFrame ?? playbackFrame;
    const time = data.time_s[frame];
    const fraction = time / data.time_s.at(-1);
    readout.textContent = previewText('deviationTime', {
      time: number.format(time),
      count: data.count[frame],
      total: data.episodes,
    });
    for (const plot of plots) {
      const value = plot.values[frame];
      const format = v => `${number.format(v)} ${plot.unit}`;
      if (value === null) {
        plot.value.textContent = previewText('deviationNoSample');
      } else if (plot.range) {
        plot.value.textContent = previewText('deviationRange', {
          min: format(plot.range.min[frame]),
          mean: format(value),
          max: format(plot.range.max[frame]),
        });
      } else {
        plot.value.textContent = format(value);
      }
      plot.cursor.style.left = `calc(${fraction * 100}% + ${left - fraction * (left + right)}px)`;
    }
  }

  function nearestFrame(time) {
    let lo = 0, hi = data.time_s.length - 1;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (data.time_s[mid] < time) lo = mid + 1;
      else hi = mid;
    }
    if (lo > 0 && time - data.time_s[lo - 1] < data.time_s[lo] - time) return lo - 1;
    return lo;
  }

  function tickStep(span) {
    const target = span / 4;
    const magnitude = 10 ** Math.floor(Math.log10(target));
    return magnitude * [1, 2, 2.5, 5, 10].find(step => step * magnitude >= target);
  }

  function drawBand(ctx, range, color, x, y) {
    // All episodes start at zero. Only the tail can have no samples.
    const last = range.max.findLastIndex(value => value !== null);
    if (last < 0) return;
    ctx.beginPath();
    ctx.moveTo(x(0), y(range.max[0]));
    for (let i = 1; i <= last; i++) ctx.lineTo(x(i), y(range.max[i]));
    for (let i = last; i >= 0; i--) ctx.lineTo(x(i), y(range.min[i]));
    ctx.closePath();
    ctx.fillStyle = `${color}40`;
    ctx.fill();
    ctx.strokeStyle = `${color}99`;
    ctx.stroke();
  }

  function draw(plot) {
    const canvas = plot.canvas;
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (width <= left + right || height <= top + bottom) return;
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext('2d');
    ctx.scale(scale, scale);
    const w = width - left - right, h = height - top - bottom;
    const duration = data.time_s.at(-1);
    ctx.font = '11px system-ui, sans-serif';
    ctx.lineWidth = 1;
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = '#384451';
    ctx.fillStyle = '#adbdcb';
    ctx.textAlign = 'right';
    for (let i = 0; i <= plot.ticks; i++) {
      const y = top + h * (1 - i / plot.ticks);
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(width - right, y);
      ctx.stroke();
      ctx.fillText(number.format(i * plot.step), left - 5, y, left - 6);
    }
    const seconds = tickStep(duration);
    for (let i = 0; i <= Math.floor(duration / seconds); i++) {
      const fraction = i * seconds / duration;
      ctx.textAlign = fraction > 0.95 ? 'right' : 'center';
      ctx.fillText(`${number.format(i * seconds)} s`, left + w * fraction, height - 9);
    }
    const x = frame => left + w * data.time_s[frame] / duration;
    const y = value => top + h * (1 - value / plot.max);
    if (plot.range) drawBand(ctx, plot.range, plot.color, x, y);
    ctx.strokeStyle = plot.color; ctx.lineWidth = 2;
    ctx.beginPath();
    let previous = false;
    for (let i = 0; i < plot.values.length; i++) {
      const value = plot.values[i];
      if (value === null) {
        previous = false;
        continue;
      }
      const px = x(i), py = y(value);
      if (previous) ctx.lineTo(px, py);
      else {
        ctx.moveTo(px, py);
        ctx.fillStyle = plot.color;
        ctx.fillRect(px - 1, py - 1, 2, 2);
      }
      previous = true;
    }
    ctx.stroke();
  }

  function addCurve(panel, key, titleKey, unit, color) {
    const values = data.mean[key];
    if (!values) return;
    const range = data.range?.[key];
    const figure = document.createElement('figure');
    figure.className = 'curve'; figure.style.setProperty('--line', color);
    const caption = document.createElement('figcaption');
    const label = document.createElement('span');
    const value = document.createElement('output');
    caption.append(label, value);
    const area = document.createElement('div');
    area.className = 'curve-plot';
    const canvas = document.createElement('canvas');
    canvas.setAttribute('role', 'img');
    const cursor = document.createElement('i');
    area.append(canvas, cursor); figure.append(caption, area); panel.append(figure);
    const peak = (range?.max ?? values).reduce((max, v) => v === null ? max : Math.max(max, v), 0);
    const step = tickStep(peak || 1);
    const ticks = Math.ceil((peak || 1) / step);
    const plot = {values, value, range, titleKey, label, unit, color, canvas, cursor, step, ticks, max: step * ticks};
    plots.push(plot);
    new ResizeObserver(() => draw(plot)).observe(canvas);
    canvas.addEventListener('pointermove', event => {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left - left) / (rect.width - left - right)));
      hoverFrame = nearestFrame(ratio * data.time_s.at(-1)); updateReadout();
    });
    canvas.addEventListener('pointerleave', () => { hoverFrame = null; updateReadout(); });
  }

  function renderLanguage() {
    number = new Intl.NumberFormat(document.documentElement.lang, {maximumSignificantDigits: 4});
    if (failed || !data?.episodes) {
      const key = failed ? 'deviationsFailed' : data ? 'deviationsEmpty' : 'deviationsLoading';
      description.textContent = previewText(key);
      return;
    }
    description.textContent = previewText('deviationsHelp', {
      reference: previewText(data.explicit_reference ? 'reference' : 'firstReference'),
      count: data.episodes,
    });
    if (Object.keys(data.range ?? {}).length) description.append(' ', previewText('deviationsBand'));
    if (tcpNote) tcpNote.textContent = previewText('tcpUnavailable');
    if (jointNote) {
      const separator = document.documentElement.lang === 'en' ? ', ' : '、';
      const names = Object.values(data.arm_joint_groups).flat();
      jointNote.textContent = names.length
        ? previewText('jointNames', {names: names.join(separator)})
        : previewText('jointsUnavailable');
    }
    for (const plot of plots) {
      const values = {title: previewText(plot.titleKey), unit: plot.unit};
      plot.label.textContent = previewText('curveLabel', values);
      plot.canvas.setAttribute('aria-label', previewText(plot.range ? 'curveWithBand' : 'curveMean', values));
      draw(plot);
    }
    updateReadout();
  }

  async function load() {
    try {
      const response = await fetch('/deviations');
      if (!response.ok) throw Error('Could not read deviations');
      data = await response.json();
      if (data.episodes) {
        document.getElementById('deviation-panels').hidden = false;
        const tcp = document.getElementById('tcp-deviations');
        addCurve(tcp, 'translation_mm', 'translationDistance', 'mm', '#80c7ff');
        addCurve(tcp, 'rotation_deg', 'rotationAngle', '°', '#e6af77');
        if (!data.mean.translation_mm) {
          tcpNote = document.createElement('p');
          tcp.append(tcpNote);
        }
        const joints = document.getElementById('joint-deviations');
        addCurve(joints, 'qpos_deg', 'revoluteDeviation', '°', '#85d5b8');
        addCurve(joints, 'qpos_mm', 'prismaticDeviation', 'mm', '#c9a9f2');
        jointNote = document.createElement('p');
        joints.append(jointNote);
      }
    } catch {
      failed = true;
    }
    renderLanguage();
  }

  document.addEventListener('preview-language-change', renderLanguage);
  renderLanguage();
  load();
  return {setFrame(frame) {
    if (frame !== playbackFrame) { playbackFrame = frame; updateReadout(); }
  }};
})();
