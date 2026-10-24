/* Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 * https://www.apache.org/licenses/LICENSE-2.0
 */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const data = JSON.parse($('graph-data').textContent);
  const graph = data.graph, report = data.report, spec = data.task_spec;
  const p = data.presentation, coverage = data.validation?.coverage;
  const localized = p.localizations || {};
  const assetOrder = new Map();
  for (const node of graph.nodes) for (const uid of [node.call.object,node.call.arguments?.object,node.call.arguments?.reference]) if (typeof uid === 'string' && !assetOrder.has(uid)) assetOrder.set(uid,assetOrder.size);
  const assetRows = (Array.isArray(p.assets) ? p.assets.filter(asset => asset && typeof asset.uid === 'string') : []).sort((a,b) => (assetOrder.get(a.uid) ?? Infinity)-(assetOrder.get(b.uid) ?? Infinity));
  const calls = new Map(graph.nodes.map(n => [n.id, n]));
  const groups = new Map(graph.task_groups.map(g => [g.id, g]));
  const assets = new Map(assetRows.map(a => [a.uid, a]));
  const NS = 'http://www.w3.org/2000/svg';
  const words = {
    graph:['任务语义图','Task graph'], archive:['历史运行','Archived run'], language:['语言','Language'], identity:['运行身份与原始数据','Run identity and original data'], instructionLabel:['任务指令','Task instruction'],
    staticGraph:['静态任务图','Static task graph'], runRecord:['运行记录','Run record'], originalInstruction:['原始指令（未经翻译）','Original instruction (verbatim)'], displayTranslation:['展示译文 · 不参与执行与验收','Display translation · not used for execution or acceptance'], noTranslation:['未提供可用译文，显示原始指令','Translation unavailable; showing the original instruction'], noSpec:['未提供 TaskSpec','TaskSpec not provided'], noAssets:['未提供资产元数据','Asset metadata not provided'], emptyGraph:['当前图没有节点','This graph has no nodes'], summary:['运行摘要','Run summary'], assets:['关联资产','Assets'], dependencies:['执行依赖','Execution dependencies'],
    graphView:['图视图','Graph view'], stages:['阶段','Stages'], calls:['调用','Calls'], search:['搜索节点 / 资产','Search nodes / assets'], environment:['执行环境','Environment'],
    statusFilter:['证据状态','Evidence status'], allStates:['全部状态','All states'], passed:['通过','Passed'], failed:['失败','Failed'], unknown:['未验证','Unverified'],
    zoomOut:['缩小','Zoom out'], zoomIn:['放大','Zoom in'], fit:['适应画布','Fit graph'], reset:['重置视图','Reset view'], exportSvg:['导出当前 SVG 图','Export current SVG'], exportJson:['导出原始数据','Export original data'],
    dependencyGraph:['控制通道依赖图','Control-channel dependency graph'], dependency:['依赖关系','Dependency'], selection:['当前选择','Selected'], orderNote:['依赖顺序，不是等比例时间轴','Dependency order, not a time-proportional axis'],
    finalGoals:['终态目标','Final goals'], inspector:['证据详情','Evidence inspector'], close:['关闭','Close'], recording:['运行记录','Run recording'], openVideo:['打开原始视频','Open original video'],
    recordingCaption:['已记录仿真视频 · 非独立成功证明','Recorded simulation video · not an independent success certificate'], provenance:['来源回执','Provenance'], unchanged:['原始图与验收数据保持不变','Original graph and acceptance data unchanged'],
    render:['GLB 渲染','GLB render'], cutout:['源图裁切','Source cutout'], completed:['程序完成','Program complete'], strictPass:['严格验收通过','Strict acceptance passed'], strictFail:['严格验收未通过','Strict acceptance failed'],
    strictUnknown:['严格证据不足','Strict evidence incomplete'], missing:['未记录','Not recorded'], unavailable:['证据不可用','Evidence unavailable'], satisfied:['已满足','Satisfied'], succeeded:['通过','Passed'], not_run:['未运行','Not run'],
    wallTime:['运行墙钟用时','Wall-clock duration'], controlSteps:['控制步数','Control steps'], semanticCalls:['语义调用','Semantic calls'], goalMetric:['TaskSpec · 终态目标','TaskSpec · final goals'],
    wallScope:['CLI 全程，含发布、不含 HTML 刷新','Full CLI run, including publication; excluding HTML refresh'], cliScope:['CLI 用时，截止最终图展示处理前','CLI wall time before final graph presentation'], workflowScope:['工作流至发布，不含图展示处理','Workflow through publication, excluding graph presentation'], simulationTime:['仿真事件末时刻','Final simulation event'], fullSpec:['完整规范评估已满足','Full specification satisfied'],
    envAll:['全部环境','All environments'], env:['环境','Environment'], mode:['验收模式','Acceptance mode'], objects:['个对象','objects'], assetSource:['源图 / 生成阶段渲染','source / generation-stage renders'],
    missingImage:['未提供可用预览','No preview available'], assetMissing:['资产图片不可用','Asset image unavailable'], left:['左臂','Left arm'], right:['右臂','Right arm'], auto:['未绑定','Unbound'],
    stagePassed:['阶段验收','Stage acceptance'], callEvidence:['调用证据','Call evidence'], finalEvidence:['关联终态目标','Related final goals'], programEvidence:['程序执行','Program execution'],
    stage:['阶段','Stage'], call:['语义调用','Semantic call'], callList:['调用序列','Call sequence'], channel:['控制通道','Control channel'], events:['运行事件','Runtime events'], eventSpan:['仿真事件区间','Simulation event interval'],
    eventScope:['事件区间来自仿真时钟，不代表墙钟规划耗时','Event intervals use simulation time, not wall-clock planning duration'], directDeps:['直接依赖','Direct dependencies'], rawStage:['原始阶段与调用','Original stage and calls'],
    rawCall:['原始调用证据','Original call evidence'], rawNode:['原始图节点','Original graph node'], rawSpec:['原始 TaskSpec','Original TaskSpec'], rawGraph:['原始 Graph','Original graph'], rawReport:['完整执行报告','Full execution report'],
    rawCoverage:['静态覆盖与绑定','Static coverage and bindings'], rawGoal:['原始终态条件','Original final-goal contract'], observedGoal:['当前环境观测','Observed evidence by environment'], rawAsset:['原始资产元数据','Original asset metadata'],
    groupId:['来源阶段','Source stages'], predicate:['谓词','Predicate'], assetId:['对象 ID','Object ID'], originalName:['原始名称','Original name'], sourceFile:['源资产文件','Source asset file'],
    sourceRender:['生成阶段 GLB 调试渲染，不代表最终网格或物理验收。来源：','Generation-stage GLB debug render, not final geometry or physical acceptance. Source: '],
    sourceCutout:['源图片对象裁切，并非 GLB 实时渲染。来源：','Object cutout from the source image, not a live GLB render. Source: '],
    graphIdentity:['图身份','Graph identity'], assemblyIdentity:['装配身份','Integration identity'], reason:['原始判据原因','Original evaluator reason'], associatedGoals:['关联目标','Related goals'],
    place:['放置','Place'], pick:['抓取','Pick'], handover:['交接','Hand over'], pour:['倾倒','Pour'], upright:['对齐姿态','Align orientation'], twist:['旋拧','Twist'], press:['按压','Press'], pressPrepare:['按压准备','Prepare press'], park:['机械臂归位','Park arm'], move:['持物移动','Move held object'], joint:['关节操作','Joint operation'], coordinatedAction:['双臂操作','Coordinated manipulation'],
    on:['置于','on'], inside:['放入','inside'], unbound:['未显式绑定','Not explicitly bound'], coordinated:['双臂协同','Both arms'], mixedChannel:['协同 / AUTO','Coord. / AUTO'], semanticGoal:['任务语义目标','Semantic goal'], fade:['项已淡化','items dimmed'], sourceOnly:['原始字段保持原文','Source fields remain verbatim'], description:['说明','Description'], rejected:['已拒绝','Rejected'], aborted:['已中止','Aborted']
  };
  let locale = 'zh';
  try { const saved = localStorage.getItem('gensim.graph.language'); if (saved === 'en') locale = saved; } catch {}
  const t = key => words[key]?.[locale === 'zh' ? 0 : 1] ?? key;
  const name = uid => localized.assets?.[uid]?.name?.[locale] || assets.get(uid)?.name || uid || '—';
  const stateName = state => t(state === 'unknown' ? 'unknown' : state);
  let mode = 'groups', selected = graph.task_groups.length ? 'group:' + graph.task_groups[0].id : 'overview';
  let view = {x:0,y:0,scale:1}, positions = new Map(), bounds = {width:1000,height:350}, dragging = null, modalAsset = null, modalMode = 'render';
  const el = (tag, text = '', cls = '') => { const node = document.createElement(tag); node.textContent = text; node.className = cls; return node; };
  const svgEl = (tag, attrs = {}, parent) => { const node = document.createElementNS(NS, tag); for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value)); parent?.append(node); return node; };
  const icon = (key, parent, cls = 'icon') => { const node = svgEl('svg', {class:cls,viewBox:'0 0 24 24','aria-hidden':true}); svgEl('use', {href:'#i-' + key}, node); parent?.append(node); return node; };
  const text = (parent, value, x, y, cls) => { const node = svgEl('text', {x,y,class:cls}, parent); node.textContent = value; return node; };
  const number = (value, digits = 1) => Number.isFinite(value) ? value.toLocaleString(locale === 'zh' ? 'zh-CN' : 'en-US', {maximumFractionDigits:digits}) : t('missing');
  const good = status => status === 'succeeded' || status === 'satisfied';
  const statusClass = status => good(status) ? 'ok' : status === 'failed' ? 'fail-text' : 'warn-text';
  const statusIcon = status => good(status) ? 'CircleCheck' : status === 'failed' ? 'CircleX' : 'CircleAlert';
  const aggregate = (values, pass = 'succeeded') => values.includes('failed') ? 'failed' : values.length && values.every(value => value === pass) ? pass : values.includes('unavailable') ? 'unavailable' : 'unknown';
  const envRows = () => (report?.environments || []).filter(row => $('environment').value === 'all' || String(row.env_id) === $('environment').value);
  function callStatus(node) {
    const segment = data.call_evidence[node.id], ids = segment?.metadata?.runtime?.env_ids;
    if (!Array.isArray(ids) || new Set(ids).size !== ids.length || !Array.isArray(segment.successes) || !Array.isArray(segment.active) || ids.length !== segment.successes.length || ids.length !== segment.active.length || ids.some(id => !Number.isInteger(id) || !report.environments.some(row => row.env_id === id)) || segment.active.some(value => typeof value !== 'boolean')) return 'unknown';
    const states = envRows().map(row => { const index = ids.indexOf(row.env_id); return index < 0 || segment.active[index] !== true ? null : segment.successes[index]; });
    return states.some(value => value === false) ? 'failed' : states.length && states.every(value => value === true) ? 'succeeded' : 'unknown';
  }
  const groupStatus = group => aggregate(envRows().map(row => row.semantic_success?.[group.id] === true ? 'succeeded' : row.semantic_success?.[group.id] === false ? 'failed' : 'unknown'));
  const goalRows = goal => (data.task_spec_evaluation?.environments || []).filter(row => $('environment').value === 'all' || String(row.env_id) === $('environment').value).map(row => row.goal.find(value => value.predicate_id === goal.id));
  const goalStatus = goal => aggregate(goalRows(goal).map(row => row?.status || 'unavailable'), 'satisfied');
  const subject = node => node.call.object || node.call.arguments?.object;
  const subjects = group => [...new Set(group.node_ids.map(id => subject(calls.get(id))).filter(Boolean))];
  function lane(node) { const resources = Object.values(node.call.resources || {}); return resources.includes('left') && resources.includes('right') ? 'coordinated' : resources.includes('left') ? 'left' : resources.includes('right') ? 'right' : 'auto'; }
  function groupLane(group) { const nodes = group.node_ids.map(id => calls.get(id)), primary = nodes.filter(node => node.role === 'primary'), lanes = new Set((primary.length ? primary : nodes).map(lane)); return lanes.has('coordinated') || lanes.has('left') && lanes.has('right') ? 'coordinated' : lanes.has('auto') ? 'auto' : lanes.has('left') ? 'left' : lanes.has('right') ? 'right' : 'auto'; }
  const events = id => { const values = data.call_evidence[id]?.metadata?.runtime?.events; return Array.isArray(values) ? values : []; };
  function eventWindow(ids) { const values = ids.flatMap(id => events(id).map(event => event.timestamp)).filter(Number.isFinite); return values.length ? [Math.min(...values), Math.max(...values)] : null; }
  function steps(ids) { const values = ids.map(id => data.call_evidence[id]).filter(Boolean).map(segment => Number.isInteger(segment.start_step) && Number.isInteger(segment.end_step) ? segment.end_step - segment.start_step : null); return values.length === ids.length && values.every(Number.isFinite) ? values.reduce((a,b) => a + b, 0) : null; }
  function callLabel(node) { const id = node.call.call_id || node.call.kind; if (id === 'pick' || id.startsWith('gen_sim.pick.')) return t('pick'); if (id.includes('place')) return t('place'); if (id.includes('park')) return t('park'); if (id === 'hand_over') return t('handover'); if (id.includes('move_held')) return t('move'); if (id.includes('pour')) return t('pour'); if (id.startsWith('gen_sim.press_prepare.')) return t('pressPrepare'); if (id.includes('press')) return t('press'); if (id.includes('twist')) return t('twist'); if (id.includes('align')) return t('upright'); if (id.includes('coordinated')) return t('coordinatedAction'); return id; }
  function groupLabel(group) { const action = {E1:'place',E2:'upright',E3:'pour',E4:'handover',E5:'coordinatedAction',E6:'joint',E8:'twist',E9:'press'}[group.task_type]; const uid = subjects(group)[0], object = uid ? name(uid) : ''; return action ? t(action) + (locale === 'en' ? ' ' + object.toLocaleLowerCase('en-US') : object) : group.task_type; }
  function goalLabel(goal) { const ids = coverage?.goals.find(value => value.id === goal.id)?.groups || [], group = ids.length === 1 ? groups.get(ids[0]) : null; const relation = goal.relation || goal.arguments?.relation; if (group && ['on','inside'].includes(relation)) { const node = group.node_ids.map(id => calls.get(id)).find(node => node.call.arguments?.reference && node.call.arguments.relation === relation); if (node) return name(subject(node)) + (locale === 'en' ? ' ' : '') + t(relation) + (locale === 'en' ? ' ' : '') + name(node.call.arguments.reference); } return (goal.predicate === 'semantic_goal' ? t('semanticGoal') : goal.predicate) + ' / ' + goal.id; }
  function fields(parent, values) { const list = el('dl'); for (const [key, value] of values) list.append(el('dt', key), el('dd', typeof value === 'string' ? value : JSON.stringify(value))); parent.append(list); }
  function raw(parent, label, value) { const details = el('details'); details.append(el('summary', label)); details.addEventListener('toggle', () => { if (details.open && !details.querySelector('pre')) details.append(el('pre', JSON.stringify(value, null, 2))); }); parent.append(details); }
  function statusPill(parent, status, label, neutral = false) { const node = el('span', '', 'status' + (neutral ? ' neutral' : status === 'failed' ? ' fail' : good(status) ? '' : ' warn')); icon(statusIcon(status), node); node.append(document.createTextNode(label)); parent.append(node); }
  function evidence(parent, label, status) { const row = el('div', '', 'evidence-row'), value = el('span', stateName(status), statusClass(status)); icon(statusIcon(status), value); row.append(el('span', label), value); parent.append(row); }
  function heading(parent, label, ico) { const node = el('h3'); if (ico) icon(ico, node); node.append(document.createTextNode(label)); parent.append(node); }
  const safeImage = value => typeof value === 'string' && /^data:image\/(png|jpeg|webp);base64,/.test(value) ? value : null;
  function showAsset(parent, uid) { const asset = assets.get(uid), image = safeImage(asset?.render) || safeImage(asset?.cutout); if (!asset) return; if (image) { const button = el('button', '', 'media-button'), img = el('img', '', 'asset-preview'); button.title = name(uid); button.setAttribute('aria-label', button.title); img.src = image; img.alt = name(uid); button.append(img); button.onclick = () => openAsset(asset); parent.append(button); } else parent.append(el('p', t('assetMissing'), 'caption')); }
  function openAsset(asset) { modalAsset = asset; modalMode = asset.render ? 'render' : 'cutout'; updateModal(); $('asset-dialog').showModal(); }
  function updateModal() { if (!modalAsset) return; $('modal-title').textContent = name(modalAsset.uid) + ' / ' + modalAsset.uid; const src = safeImage(modalAsset[modalMode]); $('modal-image').hidden = !src; $('modal-image').src = src || ''; $('modal-image').alt = name(modalAsset.uid); $('modal-source').textContent = src ? t(modalMode === 'render' ? 'sourceRender' : 'sourceCutout') + (modalAsset[modalMode + '_source'] || t('missing')) : t('assetMissing'); for (const key of ['render','cutout']) { $('asset-' + key).disabled = !safeImage(modalAsset[key]); $('asset-' + key).setAttribute('aria-pressed', String(key === modalMode)); } }
  function renderAssets() { $('assets').replaceChildren(); for (const asset of assetRows) { const uid = asset.uid, button = el('button', '', 'asset'), image = safeImage(asset.cutout) || safeImage(asset.render); if (image) { const img = el('img'); img.src = image; img.alt = name(uid); button.append(img); } else { const placeholder = el('span', '', 'asset-placeholder'); icon('Box', placeholder); button.append(placeholder); } const label = el('span'); label.append(el('strong', name(uid)), el('small', uid, 'mono')); if (!image) label.append(el('small', t('missingImage'), 'missing')); button.append(label); button.title = name(uid) + ' / ' + uid; button.onclick = () => select('asset:' + uid); $('assets').append(button); } $('asset-count').textContent = assetRows.length ? assetRows.length + ' ' + t('objects') : t('noAssets'); }
  function renderMetrics() {
    $('metrics').replaceChildren();
    const add = (label, ico, value, unit, note, warn = false) => { const node = el('div', '', 'metric' + (warn ? ' warn' : '')), head = el('div', '', 'metric-label'); icon(ico, head); head.append(document.createTextNode(label)); node.append(head, el('strong', value, 'metric-value'), el('span', unit, 'metric-unit'), el('div', note, 'metric-note')); $('metrics').append(node); };
    const goals = spec?.goal || [], satisfied = goals.filter(goal => goalStatus(goal) === 'satisfied').length;
    const rows = (data.task_spec_evaluation?.environments || []).filter(row => $('environment').value === 'all' || String(row.env_id) === $('environment').value);
    const complete = Boolean(spec) && rows.length > 0 && rows.every(row => row.status === 'satisfied' && row.certificate_ready === true), failed = rows.some(row => row.status === 'failed');
    const scopeKey = {'CLI wall time through publication, excluding HTML refresh':'wallScope','CLI wall time before final graph presentation':'cliScope','Workflow wall time through publication, excluding graph presentation':'workflowScope'}[p.wall_source];
    const wallScope = scopeKey ? t(scopeKey) : p.wall_source;
    add(t('wallTime'), 'Timer', number(p.wall_seconds, 3), p.wall_seconds === null ? '' : 's', p.wall_seconds === null ? t('missing') : wallScope || t('missing'));
    add(t('controlSteps'), 'Cpu', number(report?.runtime_result?.length, 0), 'steps', t('simulationTime') + ' ' + number(eventWindow(graph.nodes.map(node => node.id))?.[1]) + ' s');
    add(t('semanticCalls'), 'Layers', String(graph.nodes.length), t('calls'), graph.task_groups.length + ' ' + t('stages') + ' / ' + envRows().length + ' ' + t('env'));
    add(t('goalMetric'), 'ShieldCheck', spec ? satisfied + ' / ' + goals.length : t('missing'), spec ? t('satisfied') : '', !spec ? t('noSpec') : complete ? t('fullSpec') : failed ? t('strictFail') : t('strictUnknown'), !complete);
    $('run-status').replaceChildren(); statusPill($('run-status'), complete ? 'satisfied' : failed ? 'failed' : 'unavailable', t(complete ? 'strictPass' : failed ? 'strictFail' : 'strictUnknown')); statusPill($('run-status'), report?.status || 'unknown', report?.status === 'succeeded' ? t('completed') : stateName(report?.status || 'unknown'), true);
  }
  function renderGoals() { const goals = spec?.goal || []; $('goal-summary').textContent = spec ? goals.filter(goal => goalStatus(goal) === 'satisfied').length + ' / ' + goals.length + ' ' + t('satisfied') : t('noSpec'); $('goal-list').replaceChildren(); for (const goal of goals) { const state = goalStatus(goal), button = el('button', '', 'goal' + (good(state) ? '' : state === 'failed' ? ' fail' : ' warn')); icon(statusIcon(state), button); const label = el('span'); label.append(el('strong', goalLabel(goal)), el('small', goal.id + ' / ' + (coverage?.goals.find(value => value.id === goal.id)?.groups || []).join(', ')), el('span', stateName(state), 'goal-state')); button.append(label); button.onclick = () => select('goal:' + goal.id); $('goal-list').append(button); } }
  function callRows(parent, group) { for (const id of group.node_ids) { const node = calls.get(id), button = el('button', '', 'call-row'), span = eventWindow([id]); icon(statusIcon(callStatus(node)), button, 'icon ' + statusClass(callStatus(node))); button.append(el('span', callLabel(node)), el('small', span ? number(span[1] - span[0]) + ' s' : t('missing'))); button.title = id; button.onclick = () => select('call:' + id); parent.append(button); } }
  function select(key, open = true) { selected = key; draw(); renderDetail(); if (open) $('inspector').classList.add('open'); }
  function renderDetail() {
    const parent = $('detail'); parent.replaceChildren();
    if (selected === 'overview') { parent.append(el('span', 'RUN / IDENTITY', 'eyebrow'), el('h2', graph.task_id), el('p', t('sourceOnly'), 'caption')); fields(parent, [['Run', p.run_id], [t('mode'), p.acceptance_mode], [t('graphIdentity'), data.identities.graph_digest], [t('assemblyIdentity'), graph.integration_fingerprint]]); raw(parent, t('rawSpec'), spec); raw(parent, t('rawGraph'), graph); raw(parent, t('rawReport'), report); raw(parent, t('rawCoverage'), data.validation); return; }
    if (selected.startsWith('asset:')) { const asset = assets.get(selected.slice(6)); parent.append(el('span', 'ASSET', 'eyebrow'), el('h2', name(asset.uid)), el('p', asset.uid, 'mono subid')); showAsset(parent, asset.uid); fields(parent, [[t('originalName'), asset.name], [t('description'), localized.assets?.[asset.uid]?.description?.[locale] || asset.description || '—'], [t('sourceFile'), asset.glb_source], ['SHA256', asset.glb_sha256 || t('missing')]]); raw(parent, t('rawAsset'), Object.fromEntries(Object.entries(asset).filter(([key]) => !['cutout','render'].includes(key)))); return; }
    if (selected.startsWith('goal:')) { const goal = spec.goal.find(value => value.id === selected.slice(5)); parent.append(el('span', 'TASKSPEC', 'eyebrow'), el('h2', goalLabel(goal)), el('p', goal.id, 'mono subid')); statusPill(parent, goalStatus(goal), stateName(goalStatus(goal))); fields(parent, [[t('predicate'), goal.predicate], [t('groupId'), coverage?.goals.find(value => value.id === goal.id)?.groups || []]]); raw(parent, t('rawGoal'), goal); raw(parent, t('observedGoal'), goalRows(goal)); return; }
    const isGroup = selected.startsWith('group:'), node = isGroup ? null : calls.get(selected.slice(5)), group = isGroup ? groups.get(selected.slice(6)) : groups.get(node.task_instance_id), ids = isGroup ? group.node_ids : [node.id], state = isGroup ? groupStatus(group) : callStatus(node), control = isGroup ? groupLane(group) : lane(node);
    parent.append(el('span', (isGroup ? 'STAGE' : 'CALL') + ' / ' + group.task_type, 'eyebrow'), el('h2', isGroup ? groupLabel(group) : callLabel(node)), el('p', isGroup ? group.id : node.id, 'mono subid')); statusPill(parent, state, t(isGroup ? 'stagePassed' : 'callEvidence') + ' · ' + stateName(state));
    const related = (spec?.goal || []).filter(goal => coverage?.goals.find(value => value.id === goal.id)?.groups.includes(group.id));
    const stack = el('div', '', 'evidence-stack'); evidence(stack, t('programEvidence'), report?.status === 'succeeded' ? 'succeeded' : report?.status === 'failed' ? 'failed' : 'unknown'); evidence(stack, t('stagePassed'), groupStatus(group)); if (related.length) evidence(stack, t('finalEvidence'), aggregate(related.map(goalStatus), 'satisfied')); parent.append(stack);
    const span = eventWindow(ids); fields(parent, [[t('channel'), control === 'coordinated' ? t('coordinated') : control === 'auto' ? t('unbound') : t(control)], [t('controlSteps'), number(steps(ids), 0)], [t('eventSpan'), span ? number(span[0]) + ' – ' + number(span[1]) + ' s' : t('missing')], [t('directDeps'), isGroup ? group.depends_on : node.depends_on]]);
    if (isGroup) { heading(parent, t('callList') + ' / ' + ids.length, 'ListTree'); callRows(parent, group); raw(parent, t('rawStage'), {group, calls:ids.map(id => calls.get(id))}); } else { heading(parent, t('events'), 'Clock3'); for (const event of events(node.id).filter(event => ['action_planned','trajectory_segment_entered','phase_effect_gate_satisfied','action_completed','action_failed'].includes(event.kind))) { const row = el('div', number(event.timestamp) + ' s · ' + (event.segment_name || event.kind), 'event'); row.append(el('small', event.kind)); parent.append(row); } raw(parent, t('rawCall'), data.call_evidence[node.id]); raw(parent, t('rawNode'), node); }
  }
  function layout() {
    positions.clear(); const rows = mode === 'groups' ? graph.task_groups : graph.nodes, levels = new Map(), occupied = new Map(), counts = {left:0,auto:0,right:0};
    const levelFor = id => { if (levels.has(id)) return levels.get(id); const item = mode === 'groups' ? groups.get(id) : calls.get(id), value = Math.max(0, ...item.depends_on.map(dep => levelFor(dep) + 1)); levels.set(id, value); return value; };
    const slots = {left:1,auto:1,right:1};
    for (const row of rows) { const control = mode === 'groups' ? groupLane(row) : lane(row), track = control === 'coordinated' ? 'auto' : control, level = levelFor(row.id), index = occupied.get(level + ':' + track) || 0; occupied.set(level + ':' + track, index + 1); counts[track]++; slots[track] = Math.max(slots[track], index + 1); positions.set((mode === 'groups' ? 'group:' : 'call:') + row.id, {x:96 + level * 172,y:index * 126,w:152,h:110,track,control}); }
    const lanes = []; let y = 40;
    for (const track of ['left','auto','right']) if (counts[track]) { lanes.push({track,y,height:134 + (slots[track] - 1) * 126}); for (const pos of positions.values()) if (pos.track === track) pos.y += y; y += 148 + (slots[track] - 1) * 126; }
    bounds = {width:Math.max(600, ...[...positions.values()].map(pos => pos.x + pos.w + 25)),height:y + 5}; return lanes;
  }
  function matches(item, state) { const search = $('search').value.trim().toLocaleLowerCase(); const label = mode === 'groups' ? groupLabel(item) : callLabel(item), source = mode === 'groups' ? {group:item,calls:item.node_ids.map(id => calls.get(id))} : item; return (!search || (JSON.stringify(source) + label + (mode === 'groups' ? subjects(item).map(name).join(' ') : name(subject(item)))).toLocaleLowerCase().includes(search)) && ($('status-filter').value === 'all' || $('status-filter').value === state); }
  function fitText(node, maxWidth) { let size = 13; while (node.getComputedTextLength() > maxWidth && size > 10) node.style.fontSize = --size + 'px'; if (node.getComputedTextLength() > maxWidth) { const original = node.textContent; let letters = Array.from(original); while (node.getComputedTextLength() > maxWidth && letters.length) { letters.pop(); node.textContent = letters.join('') + '…'; } const tooltip = svgEl('title', {}, node.parentNode); tooltip.textContent = original; } }
  function draw() {
    const lanes = layout(); for (const key of ['lanes','edges','nodes']) $(key).replaceChildren();
    for (const {track,y,height} of lanes) { const controls = new Set([...positions.values()].filter(pos => pos.track === track).map(pos => pos.control)); const label = track !== 'auto' ? track : controls.size === 1 && controls.has('coordinated') ? 'coordinated' : controls.has('coordinated') ? 'mixedChannel' : 'auto'; svgEl('rect', {x:12,y:y-12,width:bounds.width-24,height,rx:3,class:'lane-band ' + track}, $('lanes')); svgEl('line', {x1:85,y1:y+55,x2:bounds.width-15,y2:y+55,class:'lane-line'}, $('lanes')); const laneText = text($('lanes'), t(label), 23, y+46, 'lane-title'); fitText(laneText,65); text($('lanes'), track === 'auto' ? 'AUTO / COORD.' : track.toUpperCase(), 23, y+64, 'lane-sub'); }
    const rows = mode === 'groups' ? graph.task_groups : graph.nodes, prefix = mode === 'groups' ? 'group:' : 'call:';
    if (!rows.length) text($('nodes'),t('emptyGraph'),24,35,'node-sub');
    for (const row of rows) for (const id of row.depends_on) { const a = positions.get(prefix + id), b = positions.get(prefix + row.id); if (!a || !b) continue; const x1 = a.x+a.w, y1 = a.y+a.h/2, x2 = b.x, y2 = b.y+b.h/2, mid = (x1+x2)/2; svgEl('path', {d:`M ${x1} ${y1} C ${mid} ${y1} ${mid} ${y2} ${x2-4} ${y2}`,class:'edge','marker-end':'url(#arrow)','data-from':prefix+id,'data-to':prefix+row.id}, $('edges')); }
    let faded = 0;
    for (const [index,row] of rows.entries()) { const key = prefix + row.id, pos = positions.get(key), isGroup = mode === 'groups', state = isGroup ? groupStatus(row) : callStatus(row), related = !selected.startsWith('goal:') || (coverage?.goals.find(goal => goal.id === selected.slice(5))?.groups || []).includes(isGroup ? row.id : row.task_instance_id), match = matches(row,state) && related, title = isGroup ? groupLabel(row) : callLabel(row); if (!match) faded++;
      const node = svgEl('g', {transform:`translate(${pos.x},${pos.y})`,class:'graph-node' + (selected === key ? ' selected' : '') + (match ? '' : ' faded'),tabindex:0,role:'button','aria-label':row.id + ' ' + title,'data-key':key}, $('nodes'));
      svgEl('rect', {width:pos.w,height:pos.h,rx:4,class:'node-box'}, node); if (selected === key) svgEl('rect', {x:0,y:12,width:3,height:pos.h-24,class:'selected-accent'}, node);
      text(node, isGroup ? row.task_type + ' / ' + String(index+1).padStart(2,'0') : row.id.split('__').at(-1), 12, 21, 'node-index');
      const uid = isGroup ? subjects(row)[0] : subject(row), image = safeImage(assets.get(uid)?.cutout); if (image) svgEl('image', {href:image,x:pos.w-36,y:6,width:27,height:27,preserveAspectRatio:'xMidYMid meet'}, node);
      const titleNode = text(node, title, 12, 48, 'node-title'); fitText(titleNode,pos.w-24);
      const ids = isGroup ? row.node_ids : [row.id]; text(node, isGroup ? ids.length + ' ' + t('calls') + ' / ' + number(steps(ids),0) + ' steps' : uid || '—', 12, 69, 'node-sub');
      svgEl('line', {x1:12,y1:81,x2:pos.w-12,y2:81,class:'node-rule'}, node); const statusNode = text(node, (good(state) ? '✓ ' : state === 'failed' ? '× ' : '○ ') + stateName(state), 12, 99, 'node-state' + (good(state) ? '' : state === 'failed' ? ' fail' : ' warn')); statusNode.setAttribute('aria-label',stateName(state));
      const tooltip = svgEl('title', {}, node); tooltip.textContent = title + ' / ' + row.id;
      node.addEventListener('click', () => select(key)); node.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(key); } });
    }
    $('visibility').textContent = faded ? faded + ' ' + t('fade') : graph.task_groups.length + ' ' + t('stages') + ' / ' + graph.nodes.length + ' ' + t('calls'); transform();
  }
  function transform() { $('world').setAttribute('transform', `translate(${view.x},${view.y}) scale(${view.scale})`); $('zoom').textContent = Math.round(view.scale * 100) + '%'; }
  function fit(readable = false) { const box = $('canvas').getBoundingClientRect(), topAligned = readable && (innerWidth <= 800 || mode === 'calls'); view.scale = Math.min(1.1,(box.width-20)/bounds.width,(box.height-20)/bounds.height); if (topAligned) view.scale = Math.max(.9,view.scale); view.x = topAligned ? 8 : (box.width-bounds.width*view.scale)/2; view.y = topAligned ? 20 : (box.height-bounds.height*view.scale)/2; transform(); }
  function zoom(factor, x, y) { const box = $('canvas').getBoundingClientRect(), cx = x ?? box.width/2, cy = y ?? box.height/2, next = Math.max(.12,Math.min(3,view.scale*factor)); view.x = cx-(cx-view.x)*next/view.scale; view.y = cy-(cy-view.y)*next/view.scale; view.scale = next; transform(); }
  function setMode(next) { mode = next; for (const key of ['groups','calls']) $('mode-'+key).setAttribute('aria-pressed', String(key === next)); draw(); fit(true); }
  function setLocale(next) { locale = next; try { localStorage.setItem('gensim.graph.language',locale); } catch {} render(); }
  function render() {
    document.documentElement.lang = locale === 'zh' ? 'zh-CN' : 'en';
    document.title = graph.task_id + ' · ' + t('graph');
    document.querySelectorAll('[data-i18n]').forEach(node => node.textContent = t(node.dataset.i18n));
    document.querySelectorAll('[data-title]').forEach(node => { node.title = t(node.dataset.title); node.setAttribute('aria-label',node.title); });
    document.querySelectorAll('[data-aria]').forEach(node => node.setAttribute('aria-label',t(node.dataset.aria)));
    document.querySelectorAll('[data-placeholder]').forEach(node => node.placeholder = t(node.dataset.placeholder));
    document.querySelectorAll('[data-language]').forEach(node => node.setAttribute('aria-pressed',String(node.dataset.language === locale)));
    $('breadcrumb-task').textContent = graph.task_id;
    $('task-title').textContent = localized.title?.[locale] || graph.task_id;
    $('archive-label').textContent = t(report ? 'runRecord' : 'staticGraph');
    const translation = localized.instruction?.[locale];
    $('instruction').textContent = translation || graph.instruction;
    $('translation-status').textContent = t(translation ? 'displayTranslation' : 'noTranslation');
    $('original-instruction-text').textContent = graph.instruction;
    $('original-instruction').hidden = !translation;
    $('run-meta').replaceChildren();
    for (const [ico,value] of [['Cpu',p.robot],['Clock3',p.created_at],['GitBranch',p.run_id],['SlidersHorizontal',(p.acceptance_mode && p.acceptance_mode !== '未记录') ? t('mode') + ': ' + p.acceptance_mode : '']]) {
      if (!value || value === '未记录') continue;
      const span = el('span'); icon(ico,span); span.append(document.createTextNode(String(value).replace('T',' '))); $('run-meta').append(span);
    }
    const env = $('environment').value || 'all'; $('environment').replaceChildren();
    const all = el('option',t('envAll')); all.value = 'all'; $('environment').append(all);
    for (const row of report?.environments || []) { const option = el('option',t('env') + ' ' + row.env_id); option.value = String(row.env_id); $('environment').append(option); }
    $('environment').value = env; $('environment').disabled = !report;
    $('identity').textContent = 'GRAPH ' + data.identities.graph_digest.slice(0,12) + ' / RUN ' + (p.run_id || t('missing'));
    renderAssets(); renderMetrics(); renderGoals(); draw(); renderDetail(); renderMedia();
    if ($('asset-dialog').open) updateModal();
  }
  function renderMedia() {
    const href = p.video_href, safe = typeof href === 'string' && /^videos\/(?:env_\d+\/)?[^/\\:]+\.mp4$/.test(href) && !href.endsWith('.partial.mp4');
    $('recording').hidden = !safe;
    if (!safe) return;
    const encoded = href.split('/').map(encodeURIComponent).join('/');
    if ($('video').getAttribute('src') !== encoded) { $('video').src = encoded; $('video-link').href = encoded; }
    $('recording-caption').textContent = t('recordingCaption') + ' / ' + href;
  }
  function download(value, filename, type) { const url = URL.createObjectURL(new Blob([value],{type})), link = el('a'); link.href = url; link.download = filename; link.click(); setTimeout(() => URL.revokeObjectURL(url),1000); }
  document.querySelectorAll('[data-language]').forEach(node => node.onclick = () => setLocale(node.dataset.language));
  $('environment').onchange = () => { renderMetrics(); renderGoals(); draw(); renderDetail(); }; $('search').oninput = draw; $('status-filter').onchange = draw;
  $('overview').onclick = $('provenance').onclick = () => select('overview'); $('close-detail').onclick = () => $('inspector').classList.remove('open'); $('mode-groups').onclick = () => setMode('groups'); $('mode-calls').onclick = () => setMode('calls');
  $('zoom-in').onclick = () => zoom(1.2); $('zoom-out').onclick = () => zoom(1/1.2); $('fit').onclick = () => fit(); $('reset').onclick = () => { $('search').value = ''; $('status-filter').value = 'all'; selected = graph.task_groups.length ? 'group:' + graph.task_groups[0].id : 'overview'; setMode('groups'); renderDetail(); };
  $('export-json').onclick = () => download(JSON.stringify(data,null,2),'semantic_graph_data.json','application/json');
  $('export-svg').onclick = () => {
    const clone = $('canvas').cloneNode(true);
    clone.setAttribute('width',bounds.width); clone.setAttribute('height',bounds.height); clone.setAttribute('viewBox',`0 0 ${bounds.width} ${bounds.height}`);
    clone.style.fontFamily = getComputedStyle($('canvas')).fontFamily;
    clone.querySelector('#world').removeAttribute('transform');
    clone.prepend(document.querySelector('body>svg>defs').cloneNode(true));
    const style = svgEl('style'); style.textContent = document.querySelector('style').textContent; clone.prepend(style);
    const notice = svgEl('metadata'); notice.textContent = JSON.parse($('font-license-data').textContent);
    const comments = document.createTreeWalker(document,NodeFilter.SHOW_COMMENT);
    while (comments.nextNode()) if (comments.currentNode.data.includes('Lucide')) notice.textContent += '\n' + comments.currentNode.data;
    clone.prepend(notice);
    download(new XMLSerializer().serializeToString(clone),'semantic_graph.svg','image/svg+xml');
  };
  $('canvas').addEventListener('wheel',event => { event.preventDefault(); const box = $('canvas').getBoundingClientRect(); zoom(event.deltaY < 0 ? 1.1 : 1/1.1,event.clientX-box.left,event.clientY-box.top); },{passive:false});
  $('canvas').addEventListener('pointerdown',event => { if (event.target.closest('[data-key]')) return; dragging = {x:event.clientX,y:event.clientY,vx:view.x,vy:view.y}; $('canvas').setPointerCapture(event.pointerId); $('canvas').classList.add('dragging'); });
  $('canvas').addEventListener('pointermove',event => { if (!dragging) return; view.x = dragging.vx+event.clientX-dragging.x; view.y = dragging.vy+event.clientY-dragging.y; transform(); });
  const release = event => { dragging = null; $('canvas').classList.remove('dragging'); if ($('canvas').hasPointerCapture(event.pointerId)) $('canvas').releasePointerCapture(event.pointerId); }; $('canvas').addEventListener('pointerup',release); $('canvas').addEventListener('pointercancel',release);
  $('modal-close').onclick = () => $('asset-dialog').close(); for (const key of ['render','cutout']) $('asset-'+key).onclick = () => { modalMode = key; updateModal(); };
  $('asset-dialog').addEventListener('click', event => { if (event.target === $('asset-dialog')) { const box = event.target.getBoundingClientRect(); if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) event.target.close(); } });
  render(); fit(true); new ResizeObserver(() => fit(true)).observe($('canvas'));
  document.fonts.ready.then(() => draw());
  const instructionWidth = matchMedia('(min-width:801px)');
  $('instruction-expand').open = instructionWidth.matches;
  instructionWidth.addEventListener('change',event => $('instruction-expand').open = event.matches);
  document.addEventListener('keydown', event => { if (event.key === 'Escape') $('inspector').classList.remove('open'); });
  window.__graphViewer = {data,positions,view,counts:() => ({calls:calls.size,groups:groups.size,edges:graph.nodes.reduce((sum,node) => sum+node.depends_on.length,0)}),locale:() => locale,mode:() => mode,selected:() => selected,setLocale,select,groupStatus,callStatus,goalStatus,fit};
})();
