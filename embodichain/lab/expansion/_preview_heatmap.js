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
  const number = new Intl.NumberFormat('zh-CN', {maximumFractionDigits: 4});
  const meters = (a, b) => `${number.format(a)} ～ ${number.format(b)} m`;
  const peak = counts => Math.max(...counts.map(row => Math.max(...row)));
  let data, object, view, colors, area, cell = null;

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
    const left = 48, top = 22, size = Math.min(width - left - 18, height - top - 40);
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

    ctx.font = '11px system-ui, sans-serif';
    ctx.fillStyle = '#b6c5d3';
    ctx.strokeStyle = '#94b2c340';
    ctx.lineWidth = 1;
    const ticks = width < 260 ? 2 : 4;
    for (let i = 0; i <= ticks; i++) {
      const fraction = i / ticks, edge = Math.round(fraction * n);
      const x = left + size * fraction, y = top + size * (1 - fraction);
      ctx.beginPath();
      ctx.moveTo(x, top); ctx.lineTo(x, top + size);
      ctx.moveTo(left, y); ctx.lineTo(left + size, y);
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.fillText(number.format(object.x_edges_m[edge]), x, top + size + 16);
      ctx.textAlign = 'right';
      ctx.fillText(number.format(object.y_edges_m[edge]), left - 7, y + 4);
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
      output.textContent = '指向或轻点网格，查看该位置的坐标和轨迹数。';
      return;
    }
    const {row, col} = cell;
    const x = object.x_edges_m, y = object.y_edges_m;
    output.textContent = `起点 ${view.counts[row][col]} / ${object.episodes} 条轨迹 · X ${meters(x[col], x[col + 1])} · Y ${meters(y[row], y[row + 1])}`;
  }

  function selectObject() {
    object = data.objects[objectSelect.selectedIndex];
    view = object.views.start;
    cell = null;
    element('description').textContent = `${object.episodes} 条 object_pose 扩增轨迹记录了 ${object.uid} 的起点。每条轨迹按首帧实测 XY 坐标计入一个网格。`;
    const [low, high] = view.bounds_m;
    element('x').textContent = meters(low[0], high[0]);
    element('y').textContent = meters(low[1], high[1]);
    element('step').textContent = number.format(object.x_edges_m[1] - object.x_edges_m[0]) + ' m';
    const max = peak(view.counts);
    element('peak').textContent = `${max} / ${object.episodes} 条`;
    colors = colorScale(max);
    element('scale-max').textContent = max;
    element('scale').replaceChildren(...colors.map(color => {
      const swatch = document.createElement('i');
      swatch.style.backgroundColor = color;
      return swatch;
    }));
    canvas.setAttribute('aria-label', `${object.uid} 扩增起点 · X ${meters(low[0], high[0])} · Y ${meters(low[1], high[1])} · 单格最多 ${max} 条轨迹`);
    reportCell();
    draw();
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
    const description = element('description');
    try {
      const response = await fetch('/object-heatmaps');
      if (!response.ok) throw Error('heatmaps');
      data = await response.json();
      if (!data.objects.length) {
        description.textContent = '当前输入中没有记录 object_pose 扩增物体的起点。';
        return;
      }
      for (const item of data.objects) objectSelect.add(new Option(item.uid, item.uid));
      element('panel').hidden = false;
      selectObject();
    } catch {
      description.textContent = '扩增起点加载失败，请刷新页面重试。';
    }
  }
  load();
})();
