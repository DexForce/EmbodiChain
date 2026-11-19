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

(() => {
  const element = name => document.getElementById('heatmap-' + name);
  const canvas = element('plot');
  const ctx = canvas.getContext('2d');
  const objectSelect = element('object');
  const peak = counts => Math.max(...counts.map(row => Math.max(...row)));
  let data, object, view, xAxis, yAxis, colors, area, cell = null, failed = false;

  function axisScale(edges) {
    const bins = edges.length - 1;
    const step = (edges[bins] - edges[0]) / bins;
    const digits = Math.min(20, Math.max(0, 1 - Math.floor(Math.log10(step))));
    const number = new Intl.NumberFormat(document.documentElement.lang, {
      maximumFractionDigits: digits,
      useGrouping: false,
    });
    const ticks = [0, 0.25, 0.5, 0.75, 1].map(t => number.format(edges[Math.round(t * bins)]));
    const meters = (a, b) => previewText('meters', {low: number.format(a), high: number.format(b)});
    return {step, ticks, meters};
  }

  function colorScale(max) {
    const stops = [[70, 111, 172], [47, 178, 166], [245, 215, 98]];
    const scale = ['#202c38'];
    for (let n = 1; n <= max; n++) {
      const t = max === 1 ? 2 : 2 * (n - 1) / (max - 1);
      const i = Math.min(1, Math.floor(t));
      const rgb = stops[i].map((v, c) => Math.round(v + (stops[i + 1][c] - v) * (t - i)));
      scale.push(`rgb(${rgb.join(',')})`);
    }
    return scale;
  }

  function draw() {
    if (!view || !canvas.clientWidth) return;
    const width = canvas.clientWidth, height = canvas.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.font = '11px system-ui, sans-serif';
    const labelWidth = axis => Math.max(...axis.ticks.map(label => ctx.measureText(label).width));
    const xLabelWidth = labelWidth(xAxis);
    const left = Math.ceil(Math.max(48, labelWidth(yAxis) + 9, xLabelWidth / 2 + 2));
    const right = Math.ceil(Math.max(18, xLabelWidth / 2 + 2));
    const top = 22, size = Math.min(width - left - right, height - top - 40);
    if (size <= 0) {
      area = null;
      return;
    }
    const n = view.counts.length, step = size / n;
    area = {left, top, size};
    ctx.fillStyle = colors[0];
    ctx.fillRect(left, top, size, size);
    for (let row = 0; row < n; row++) {
      view.counts[row].forEach((count, col) => {
        if (!count) return;
        ctx.fillStyle = colors[count];
        ctx.fillRect(left + col * step, top + (n - 1 - row) * step, step, step);
      });
    }

    ctx.fillStyle = '#b6c5d3';
    ctx.strokeStyle = '#94b2c340';
    ctx.lineWidth = 1;
    const stride = [1, 2, 4].find(skip => size * skip / 4 >= xLabelWidth + 6) ?? 4;
    for (let i = 0; i <= 4; i += stride) {
      const fraction = i / 4;
      const x = left + size * fraction, y = top + size * (1 - fraction);
      ctx.beginPath();
      ctx.moveTo(x, top); ctx.lineTo(x, top + size);
      ctx.moveTo(left, y); ctx.lineTo(left + size, y);
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.fillText(xAxis.ticks[i], x, top + size + 16);
      ctx.textAlign = 'right';
      ctx.fillText(yAxis.ticks[i], left - 7, y + 4);
    }
    ctx.fillStyle = '#d4dfe9';
    ctx.textAlign = 'left';
    ctx.fillText('Y (m)', left, top - 9);
    ctx.textAlign = 'center';
    ctx.fillText('X (m)', left + size / 2, top + size + 33);
    if (cell) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.strokeRect(left + cell.col * step, top + (n - 1 - cell.row) * step, step, step);
    }
  }

  function reportCell() {
    const output = element('cell');
    if (!cell) {
      output.textContent = previewText('heatmapInspect');
      return;
    }
    const {row, col} = cell;
    const x = object.x_edges_m, y = object.y_edges_m;
    output.textContent = previewText('heatmapCell', {
      count: view.counts[row][col],
      total: object.episodes,
      x: xAxis.meters(x[col], x[col + 1]),
      y: yAxis.meters(y[row], y[row + 1]),
    });
  }

  function renderLanguage() {
    const description = element('description');
    if (failed || !object) {
      const key = failed ? 'heatmapFailed' : data ? 'heatmapEmpty' : 'heatmapLoading';
      description.textContent = previewText(key);
      return;
    }
    xAxis = axisScale(object.x_edges_m);
    yAxis = axisScale(object.y_edges_m);
    description.textContent = previewText('heatmapHelp', {count: object.episodes, object: object.uid});
    const [low, high] = view.bounds_m;
    const x = xAxis.meters(low[0], high[0]);
    const y = yAxis.meters(low[1], high[1]);
    const millimeters = new Intl.NumberFormat(document.documentElement.lang, {maximumSignificantDigits: 4});
    element('x').textContent = x;
    element('y').textContent = y;
    element('step').textContent = `${millimeters.format(xAxis.step * 1000)} × ${millimeters.format(yAxis.step * 1000)} mm`;
    const max = peak(view.counts);
    element('peak').textContent = previewText('heatmapPeak', {max, total: object.episodes});
    canvas.setAttribute('aria-label', previewText('heatmapAlt', {object: object.uid, x, y, max}));
    reportCell();
    draw();
  }

  function selectObject() {
    object = data.objects[objectSelect.selectedIndex];
    view = object.views.start;
    cell = null;
    const max = peak(view.counts);
    colors = colorScale(max);
    element('scale-max').textContent = max;
    element('scale').replaceChildren(...colors.map(color => {
      const swatch = document.createElement('i');
      swatch.style.backgroundColor = color;
      return swatch;
    }));
    renderLanguage();
  }

  function inspect(event) {
    if (!area) return;
    const rect = canvas.getBoundingClientRect(), n = view.counts.length;
    const col = Math.floor((event.clientX - rect.left - area.left) / area.size * n);
    const row = n - 1 - Math.floor((event.clientY - rect.top - area.top) / area.size * n);
    const inside = row >= 0 && row < n && col >= 0 && col < n;
    if (inside ? cell?.row === row && cell.col === col : !cell) return;
    cell = inside ? {row, col} : null;
    reportCell();
    draw();
  }

  objectSelect.addEventListener('change', selectObject);
  canvas.addEventListener('pointermove', inspect);
  canvas.addEventListener('pointerdown', inspect);
  canvas.addEventListener('pointerleave', event => {
    if (event.pointerType !== 'mouse' || !cell) return;
    cell = null;
    reportCell();
    draw();
  });
  new ResizeObserver(draw).observe(canvas);

  async function load() {
    try {
      const response = await fetch('/object-heatmaps');
      if (!response.ok) throw Error('Could not read starting positions');
      data = await response.json();
      if (data.objects.length) {
        for (const item of data.objects) objectSelect.add(new Option(item.uid, item.uid));
        element('panel').hidden = false;
        selectObject();
        return;
      }
    } catch {
      failed = true;
    }
    renderLanguage();
  }

  document.addEventListener('preview-language-change', renderLanguage);
  renderLanguage();
  load();
})();
