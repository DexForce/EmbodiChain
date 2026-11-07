/*
----------------------------------------------------------------------------
Copyright (c) 2021-2026 DexForce Technology Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
----------------------------------------------------------------------------
*/
(function (root) {
  'use strict';
  const physicsFields = { mass: 'mass', friction: 'friction', com: 'com_pose', stiffness: 'joint_stiffness', damping: 'joint_damping' };
  const physical = episode => episode.kind === 'physical';
  function recipe(episode) {
    const value = episode.metadata.expansion_recipe || episode.metadata.recipe;
    return value && typeof value === 'object' ? value : {};
  }
  function operators(episode) {
    const selected = (episode.metadata.expansion || []).flatMap(record =>
      (record.trajectory_variants || []).filter(v => v.variant_ordinal === record.trajectory_selected).map(v => v.spatial_operator));
    return [...selected, ...(episode.metadata.real2sim_lineage?.calls || []).map(call => call.operator)];
  }
  function nominal(episode) {
    const values = operators(episode);
    return values.length > 0 && values.every(value => value === 'none' || value === 'nominal');
  }
  function baselineId(c) {
    return c.source === 'ur5' ? (c.backend === 'newton' ? 'newton-baseline' : 'ur5-baseline') : 'carton-baseline';
  }
  function referenceFor(c, episode, episodes) {
    if (!episode) return episodes.find(e => e.case === baselineId(c) && physical(e));
    if (episode.parent) return episodes.find(e => e.id === episode.parent);
    const same = episodes.filter(e => e.case === c.id && physical(e));
    if (c.tags.includes('affordance')) {
      const reference = same.find(e => recipe(e).motion_recipe === 0);
      if (reference && reference.id !== episode.id) return reference;
    }
    if (c.tags.includes('motion') && c.source === 'ur5' && !c.tags.includes('combined')) {
      const reference = same.find(nominal);
      if (reference && reference.id !== episode.id) return reference;
    }
    return episodes.find(e => e.case === baselineId(c) && physical(e)) || same[0];
  }
  function flatten(value, prefix = '') {
    if (typeof value === 'number' && Number.isFinite(value)) return [[prefix, value]];
    if (Array.isArray(value)) return value.flatMap((v, i) => flatten(v, prefix + '[' + i + ']'));
    return [];
  }
  function parameterChanges(c, left, right) {
    if (!left || !right) return [];
    const suffix = c.tags.map(t => physicsFields[t]).find(Boolean);
    const a = left.metadata.physics_parameters || {}, b = right.metadata.physics_parameters || {};
    const names = right.metadata.joint_names || [];
    const result = [];
    for (const key of Object.keys(b)) {
      if (suffix && !key.endsWith('.' + suffix)) continue;
      if (!suffix && !/\.(mass|friction|com_pose|joint_stiffness|joint_damping)$/.test(key)) continue;
      const original = new Map(flatten(a[key]));
      for (const [index, value] of flatten(b[key])) {
        const before = original.get(index);
        if (before === undefined || Math.abs(value - before) <= Math.max(1e-7, Math.abs(before) * 1e-6)) continue;
        const axis = Number(index.match(/\[(\d+)\]$/)?.[1]);
        if (key.endsWith('.com_pose') && (!Number.isInteger(axis) || axis > 2)) continue;
        let label = key + index, unit = '';
        if (key.endsWith('.com_pose')) { label = key + index.replace(/\[\d+\]$/, '') + '.' + 'xyz'[axis]; unit = 'mm'; }
        if (/\.joint_(stiffness|damping)$/.test(key)) label = key + ' / ' + (names[Number(index.slice(1, -1))] || index);
        if (key.endsWith('.mass')) unit = 'kg';
        const scale = unit === 'mm' ? 1000 : 1;
        result.push({ label, before: before * scale, after: value * scale, delta: (value - before) * scale,
          percent: before === 0 ? null : 100 * (value - before) / Math.abs(before), unit });
      }
    }
    return result;
  }
  function selectPair(c, episodes) {
    const candidates = episodes.filter(e => e.case === c.id);
    const baseline = referenceFor(c, undefined, episodes);
    let ranked = candidates.filter(physical), reason = '同来源、同后端的基准对照';
    if (c.tags.includes('rerender')) {
      ranked = candidates.filter(e => e.kind === 'visual');
      const order = { combined: 4, light: 3, camera: 2, material: 1 };
      ranked.sort((a, b) => (order[b.profile] ?? -1) - (order[a.profile] ?? -1));
      reason = '视觉子样本与其精确物理父样本对照；3D 轨迹相同是预期结果';
    } else if (c.tags.includes('affordance')) {
      ranked.sort((a, b) => (recipe(b).motion_recipe || 0) - (recipe(a).motion_recipe || 0));
      reason = '同一作业内的不同抓取分支对照';
    } else if (c.tags.includes('motion') && c.source === 'ur5' && !c.tags.includes('combined')) {
      const requested = c.tags.find(t => ['joint_residual', 'via_points'].includes(t));
      ranked.sort((a, b) => Number(operators(b).includes(requested)) - Number(operators(a).includes(requested)));
      reason = '默认选择实际执行了扩增算子的样本；左侧使用同作业内的名义配方';
    } else if (c.tags.includes('physics') && baseline) {
      const score = e => parameterChanges(c, baseline, e).reduce((n, v) => Math.max(n, Math.abs(v.percent ?? v.delta)), 0);
      ranked.sort((a, b) => score(b) - score(a));
      reason = '默认展示已提交样本中参数变化较明显的一条，不代表总体分布';
    } else if (c.tags.includes('scene') && baseline) {
      const score = e => Object.entries(e.initial_poses || {}).reduce((n, [uid, pose]) => {
        const base = baseline.initial_poses?.[uid];
        return base ? n + Math.hypot(...pose.slice(0, 3).map((v, i) => v - base[i])) : n;
      }, 0);
      ranked.sort((a, b) => score(b) - score(a));
      reason = '默认展示已提交样本中场景位移较明显的一条';
    }
    const right = ranked[0] || candidates[0];
    if (c.tags.includes('baseline')) reason='这是未扩增基准，左右可以是同一条轨迹；请在左侧选择具体扩增能力。';
    return { left: referenceFor(c, right, episodes), right, reason };
  }
  function episodeLabel(episode) {
    if (episode.kind === 'visual') return ({material:'材质',light:'灯光',camera:'相机',combined:'视觉组合'}[episode.profile] || episode.profile) + ' · 观测变体';
    const ops = [...new Set(operators(episode))].filter(op => op !== 'none');
    const native = (episode.metadata.expansion || []).find(r => r.recipe_index != null);
    if (episode.case?.endsWith('-affordance')) return '抓取配方分支 '+(recipe(episode).motion_recipe ?? native?.recipe_index ?? '—');
    if (nominal(episode)) return '名义配方 · 实测';
    return (ops.join(' / ') || (native ? '配方 ' + native.recipe_index : '实测轨迹'));
  }
  function segments(replay) {
    const duration = replay.times.at(-1);
    const ranges = replay.segments || [];
    const steps = replay.action_count || (ranges.length ? ranges.at(-1).end_step : 0);
    return steps > 0 ? ranges.map((s, i) => ({ start: s.start_step / steps * duration, end: s.end_step / steps * duration,
      name: s.name || s.instruction || String(i) })) : [];
  }
  function phaseCoordinate(replay, time) {
    const ranges = segments(replay);
    if (!ranges.length) return time / (replay.times.at(-1) || 1);
    let i = ranges.findIndex(s => time <= s.end);
    if (i < 0) i = ranges.length - 1;
    const s = ranges[i];
    return i + Math.max(0, Math.min(1, (time - s.start) / (s.end - s.start || 1)));
  }
  function timeAtPhase(replay, coordinate) {
    const ranges = segments(replay);
    if (!ranges.length) return coordinate * replay.times.at(-1);
    const i = Math.max(0, Math.min(ranges.length - 1, Math.floor(coordinate)));
    const s = ranges[i];
    return s.start + Math.max(0, Math.min(1, coordinate - i)) * (s.end - s.start);
  }
  function interpolate(times, values, time) {
    if (time <= times[0]) return values[0].slice();
    if (time >= times.at(-1)) return values.at(-1).slice();
    let lo = 0, hi = times.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (times[mid] <= time) lo = mid; else hi = mid; }
    const w = (time - times[lo]) / (times[hi] - times[lo]);
    return values[lo].map((v, i) => v + w * (values[hi][i] - v));
  }
  function jointUnit(replay, index) {
    const kind=replay.joint_types?.[index];
    if (kind==='revolute'||kind==='continuous') return { key:'rad',label:'rad',scale:1 };
    if (kind==='prismatic') return { key:'m',label:'mm',scale:1000 };
    return { key:null,label:'单位未记录',scale:1 };
  }
  function analyze(left, right, alignment = 'phase') {
    if (!left || !right || left.joint_names.join('\0') !== right.joint_names.join('\0')) return null;
    if(left.joint_names.some((_,i)=>jointUnit(left,i).key!==jointUnit(right,i).key))return null;
    const l = segments(left), r = segments(right);
    const phase = alignment === 'phase' && l.length > 0 && l.length === r.length && l.every((s, i) => s.name === r[i].name);
    const times = right.times.filter(t => phase || t <= left.times.at(-1));
    const differences = [], tcpDifference = [], leftTimes = [], energy = Array(left.joint_names.length).fill(0);
    times.forEach((time, i) => {
      const otherTime = phase ? timeAtPhase(left, phaseCoordinate(right, time)) : time;
      const q = interpolate(right.times, right.qpos, time), reference = interpolate(left.times, left.qpos, otherTime);
      const delta = q.map((v, j) => v - reference[j]);
      delta.forEach((v, j) => { energy[j] += v * v; });
      differences.push(delta); leftTimes.push(otherTime);
      const tcp = interpolate(right.times, right.tcp, time), other = interpolate(left.times, left.tcp, otherTime);
      tcpDifference.push(Math.hypot(...tcp.map((v, j) => v - other[j])) * 1000);
    });
    const angular=energy.map((_,i)=>i).filter(i=>jointUnit(right,i).key==='rad');
    const linear=energy.map((_,i)=>i).filter(i=>jointUnit(right,i).key==='m');
    const eligible=angular.length?angular:linear.length?linear:[0];
    const joint=eligible.reduce((a,b)=>energy[a]>=energy[b]?a:b);
    const angularPeak=angular.length?Math.max(0,...differences.flatMap(row=>angular.map(i=>Math.abs(row[i])))):null;
    const linearPeak=linear.length?Math.max(0,...differences.flatMap(row=>linear.map(i=>Math.abs(row[i])))):null;
    const peak=angularPeak??linearPeak??Math.max(0,...differences.map(row=>Math.abs(row[joint])));
    const peakIndex=differences.reduce((best,row,i)=>Math.abs(row[joint])>Math.abs(differences[best]?.[joint]||0)?i:best,0);
    return { times, leftTimes, differences, tcpDifference, joint, peak, peakIndex, peakUnit:jointUnit(right,joint).key, angularPeak, linearPeakMm:linearPeak===null?null:linearPeak*1000,
      peakTime: times[peakIndex] || 0, rms: Math.sqrt(energy[joint] / Math.max(times.length, 1)),
      alignment: phase ? 'phase' : 'time', maxTcpMm: Math.max(0, ...tcpDifference) };
  }
  const api = { recipe, operators, nominal, baselineId, referenceFor, parameterChanges, selectPair, episodeLabel, segments, phaseCoordinate, timeAtPhase, interpolate, jointUnit, analyze };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ShowcaseComparison = api;
})(typeof window === 'undefined' ? globalThis : window);
