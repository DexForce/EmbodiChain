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

const previewText = (() => {
  // Keep labels and runtime messages together. Recorded IDs are passed as values.
  const messages = {
    title: ['EmbodiChain · 扩增结果回放', 'EmbodiChain · Expansion replay'],
    subtitle: ['完整模型虚影 · 已确认的真实扩增结果', 'Full-model overlays of confirmed expansion recordings'],
    language: ['语言', 'Language'],
    disconnected: ['渲染服务未连接', 'Renderer disconnected'],
    connected: ['● 原生视频 · WebRTC', '● Native video · WebRTC'],
    connecting: ['正在连接 WebRTC 视频…', 'Connecting WebRTC video…'],
    reconnecting: ['WebRTC 视频连接失败，正在重试…', 'WebRTC video unavailable. Reconnecting…'],
    controlFailed: ['控制未成功应用', 'Could not apply replay control'],
    play: ['播放', 'Play'],
    pause: ['暂停', 'Pause'],
    reset: ['回到起点', 'Back to start'],
    alignment: ['对齐方式', 'Alignment'],
    alignPhase: ['按动作阶段对齐', 'Align by motion phase'],
    alignTime: ['按真实时间对齐', 'Align by recorded time'],
    camera: ['观察视角', 'Camera view'],
    cameraCustom: ['自由视角', 'Free view'],
    cameraOrbit: ['斜视', 'Orbit view'],
    cameraTop: ['俯视', 'Top view'],
    cameraSide: ['侧视', 'Side view'],
    cameraClose: ['抓取区域近景', 'Grasp close-up'],
    spaceKey: ['空格', 'Space'],
    controlsHelp: ['播放 / 暂停 · 拖动画面旋转 · 滚轮缩放', 'Play / pause · Drag to rotate · Scroll to zoom'],
    frameAlt: ['EmbodiChain 原生场景渲染画面', 'Native EmbodiChain scene rendering'],
    loadingModel: ['正在加载完整机器人模型…', 'Loading the full robot model…'],
    progress: ['回放进度', 'Playback progress'],
    coverageTitle: ['采样覆盖 · 包含未通过点', 'Sampling coverage, including unqualified points'],
    coverageDetails: ['边界与样本明细', 'Bounds and sample details'],
    coverageCounts: ['计划 {planned} · 已尝试 {attempted} · 已确认 {confirmed}。{result}',
      'Planned {planned} · Attempted {attempted} · Confirmed {confirmed}. {result}'],
    coverageComplete: ['全部计划样本已确认', 'All planned samples confirmed.'],
    coverageIncomplete: ['包含未通过、回退或未执行点', 'Includes rejected, fallback or unrun points.'],
    coverageScope: ['有限参数采样，不代表连续空间或全部边界组合穷举。',
      'This finite sample covers selected parameter points.'],
    coverageRecord: ['#{index} {role} · {status} · {confirmed}/{planned} 确认',
      '#{index} {role} · {status} · {confirmed}/{planned} confirmed'],
    filters: ['分布筛选 · 三维联动', 'Filter 3D overlays'],
    layoutGroup: ['参数组', 'Layout group'],
    allSpaces: ['全部空间组', 'All layout groups'],
    affordance: ['抓取分支', 'Affordance'],
    allAffordances: ['全部抓取分支', 'All grasp branches'],
    trajectory: ['轨迹', 'Trajectory'],
    allVariants: ['全部变体', 'All variants'],
    matching: [' 条符合筛选', ' matching'],
    matchingTotal: [' / {total} 条符合筛选', ' / {total} matching'],
    unrecorded: ['未记录', 'Unrecorded'],
    candidateInfo: ['{space} · 抓取 {aff} · 变体 {variant}', '{space} · Grasp {aff} · Variant {variant}'],
    layers: ['场景图层', 'Scene layers'],
    showGhosts: ['显示全部匹配虚影', 'Show all matching overlays'],
    showPaths: ['末端轨迹线', 'End-effector paths'],
    showTargets: ['抓取目标坐标轴', 'Grasp target axes'],
    showPositions: ['初始空间位置', 'Initial positions'],
    layersHelp: ['实色模型始终播放固定对照。点击轨迹按钮可分别显示或隐藏对应的虚影。',
      'The solid model follows the reference. Click a trajectory button to toggle its overlay.'],
    execution: ['真实执行记录', 'Recorded execution'],
    waitingResults: ['等待渲染结果…', 'Waiting for the first rendered frame…'],
    reference: ['原始轨迹', 'Original trajectory'],
    firstReference: ['首条对照记录', 'First recording'],
    firstRecording: ['首条记录 {label}（固定对照）', 'First recording {label} (fixed reference)'],
    clock: ['{time} s · 记录帧 {frame} / {last}', '{time} s · Recorded frame {frame} / {last}'],
    caption: ['{reference} · {count} 条扩增虚影', '{reference} · Overlays shown: {count}'],
    executionDetail: ['{reference} · 已通过验收\n抓取候选 {candidate} · 物理推进 {steps} 步\n{episode}',
      '{reference} · Accepted\nGrasp candidate: {candidate} · Physics steps: {steps}\n{episode}'],
    deviations: ['扩增轨迹偏差', 'Expansion trajectory deviations'],
    deviationsLoading: ['正在读取统计曲线…', 'Loading deviation curves…'],
    deviationsEmpty: ['当前只有对照轨迹。加入扩增轨迹即可显示偏差曲线。',
      'Only the reference is available. Add expanded trajectories to display deviation curves.'],
    deviationsFailed: ['偏差曲线加载失败，请刷新页面重试。', 'Could not load deviation curves. Refresh to try again.'],
    deviationsHelp: ['相对{reference}，按实测时间对齐全部 {count} 条扩增轨迹。每个时刻只计仍有记录的轨迹。',
      'All {count} expanded trajectories are aligned to the {reference} by recorded time. Each time uses trajectories with data at that point.'],
    deviationsBand: ['色带覆盖最小值至最大值，实线表示均值。', 'The band spans the minimum to maximum. The line shows the mean.'],
    deviationTime: ['{time} s · {count} / {total} 条轨迹', '{time} s · Trajectories: {count} / {total}'],
    deviationRange: ['最小 {min} · 均值 {mean} · 最大 {max}', 'Min {min} · Mean {mean} · Max {max}'],
    deviationNoSample: ['无样本', 'No samples'],
    tcpDeviations: ['TCP 位姿偏差', 'TCP pose deviation'],
    jointDeviations: ['机械臂最大关节偏差', 'Maximum arm joint deviation'],
    translationDistance: ['平移距离', 'Translation distance'],
    rotationAngle: ['相对旋转角', 'Relative rotation angle'],
    revoluteDeviation: ['转动关节最大绝对偏差', 'Maximum absolute revolute-joint deviation'],
    prismaticDeviation: ['移动关节最大绝对偏差', 'Maximum absolute prismatic-joint deviation'],
    curveLabel: ['{title}（{unit}）', '{title} ({unit})'],
    curveWithBand: ['{title}随实测时间的均值曲线及最小值至最大值色带，单位 {unit}',
      '{title} over recorded time, with mean and minimum-to-maximum range, in {unit}'],
    curveMean: ['{title}随实测时间的均值曲线，单位 {unit}', '{title} mean over recorded time, in {unit}'],
    tcpUnavailable: ['记录中缺少所选机械臂的完整 TCP 标定。', 'The selected arm has incomplete TCP calibration in the recordings.'],
    jointsUnavailable: ['记录中缺少可识别的机械臂关节组。', 'The recordings do not identify an arm joint group.'],
    jointNames: ['统计关节 · {names}', 'Included joints · {names}'],
    heatmapTitle: ['object_pose 物体起点分布', 'object_pose starting positions'],
    heatmapLoading: ['正在读取扩增起点…', 'Loading starting positions…'],
    heatmapEmpty: ['当前输入中没有记录 object_pose 扩增物体的起点。',
      'The inputs contain no recorded object_pose starting positions.'],
    heatmapFailed: ['扩增起点加载失败，请刷新页面重试。', 'Could not load starting positions. Refresh to try again.'],
    heatmapObject: ['扩增物体', 'Expanded object'],
    heatmapX: ['实测 X 范围', 'Recorded X range'],
    heatmapY: ['实测 Y 范围', 'Recorded Y range'],
    heatmapCellSize: ['单格尺寸（X × Y）', 'Cell size (X × Y)'],
    heatmapMostVisited: ['单格最多轨迹', 'Most trajectories in one cell'],
    heatmapLegend: ['起点落入该格的轨迹数', 'Trajectories starting in each cell'],
    heatmapInspect: ['指向或轻点网格，查看该位置的坐标和轨迹数。', 'Hover or tap a cell to see its coordinates and count.'],
    heatmapCell: ['起点 {count} / {total} 条轨迹 · X {x} · Y {y}', 'Starts: {count} / {total} trajectories · X {x} · Y {y}'],
    heatmapHelp: ['{count} 条 object_pose 扩增轨迹记录了 {object} 的起点。X、Y 轴按各自的首帧实测范围缩放。',
      'The starting positions of {object} come from {count} object_pose trajectories. Each axis fits its recorded first-frame range.'],
    heatmapPeak: ['{max} / {total} 条', '{max} / {total}'],
    heatmapAlt: ['{object} 扩增起点 · X {x} · Y {y} · 单格最多 {max} 条轨迹',
      '{object} starting positions · X {x} · Y {y} · Most trajectories in one cell: {max}'],
    meters: ['{low} ～ {high} m', '{low} to {high} m'],
    footer: ['EmbodiChain SimulationManager / RobotMarkers / VisualizationMarkers → DexSim Hybrid 原生渲染。仅回放，物理推进 0 步。机器人与物体使用逐帧记录；轨迹线每 4 帧取点并连接。视角、灯光与透明材质仅用于验收展示。',
      'Native DexSim Hybrid rendering through EmbodiChain. Replay advances physics by zero steps. Robot and object states come from recorded frames. Path lines connect every fourth frame. Camera, lighting and transparency are for display.'],
  };
  const language = document.getElementById('language');
  function t(key, values = {}) {
    const message = messages[key][language.value === 'en' ? 1 : 0];
    return message.replace(/\{(\w+)\}/g, (_, name) => values[name]);
  }
  function applyLanguage() {
    document.documentElement.lang = language.value;
    document.title = t('title');
    for (const node of document.querySelectorAll('[data-text]')) {
      node.textContent = t(node.dataset.text);
    }
    for (const node of document.querySelectorAll('[data-label]')) {
      node.setAttribute('aria-label', t(node.dataset.label));
    }
    document.dispatchEvent(new Event('preview-language-change'));
  }
  language.addEventListener('change', applyLanguage);
  applyLanguage();
  return t;
})();
