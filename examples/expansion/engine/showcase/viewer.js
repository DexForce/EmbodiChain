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
'use strict';
const data=JSON.parse(document.getElementById('showcase-data').textContent),$=id=>document.getElementById(id);
const labels={collected:'已提交 · 待验收',passed:'已通过',failed:'失败',not_qualified:'未资格验证',not_run:'未运行',unsupported:'不支持',separate_world:'独立 World',not_integrated:'未接入'};
const comparison=window.ShowcaseComparison;
let pairAnalysis=null,comparisonReason='',wantedReplays=new Set();
const models=new Map(),replays=new Map(),pending=new Map();
window.showcaseRegister=(kind,key,value)=>{(kind==='model'?models:replays).set(key,value)};
function load(kind,key,path){const cache=kind==='model'?models:replays;if(cache.has(key))return Promise.resolve(cache.get(key));if(pending.has(path))return pending.get(path);const promise=new Promise((resolve,reject)=>{const script=document.createElement('script');script.src=path;script.onload=()=>{script.remove();pending.delete(path);cache.has(key)?resolve(cache.get(key)):reject(Error('Missing replay payload'));if(kind==='episode'&&!wantedReplays.has(key))cache.delete(key)};script.onerror=()=>{script.remove();pending.delete(path);reject(Error('Cannot load '+path))};document.head.append(script)});pending.set(path,promise);return promise}
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n}
function table(container,heads,rows){const t=el('table'),thead=el('thead'),tr=el('tr');heads.forEach(h=>tr.append(el('th',h)));thead.append(tr);t.append(thead);const body=el('tbody');for(const row of rows){const r=el('tr');for(const v of row){const cell=el('td');if(v instanceof Node)cell.append(v);else cell.textContent=v??'—';r.append(cell)}body.append(r)}t.append(body);$(container).replaceChildren(t)}
function pill(status){return el('span',labels[status]||status,'pill '+status)}
const totals=data.cases.reduce((a,c)=>{for(const k of ['attempts','measured_successes','committed','observation_variants'])a[k]+=(c.summary[k]||0);return a},{attempts:0,measured_successes:0,committed:0,observation_variants:0});
$('attempts').textContent=totals.attempts;$('successes').textContent=totals.measured_successes;$('committed').textContent=totals.committed;$('children').textContent=totals.observation_variants;
$('runStatus').textContent=(data.mode==='full'?'完整展示':'快速预览')+' · '+(data.acceptance.acceptance_passed?'必验项通过':data.acceptance.report_complete?'含未通过项':'验收尚未完成');
for(const b of document.querySelectorAll('nav button'))b.onclick=()=>{document.querySelectorAll('nav button').forEach(v=>v.classList.toggle('active',v===b));document.querySelectorAll('.tab').forEach(v=>v.classList.toggle('hidden',v.id!==b.dataset.tab))};
table('matrixTable',['案例','来源 / 后端','能力','物理提交 / 目标','视觉子样本','结果'],data.cases.map(c=>[c.title,(c.source==='ur5'?'规划生成':'真实示范')+' / '+c.backend,c.tags.join(' · '),(c.summary.committed||0)+' / '+c.target,c.summary.observation_variants||0,pill(c.status)]));
for(const b of data.boundaries){const n=el('div',undefined,'notice');n.append(pill(b.status),document.createTextNode(' '+b.name+'：'+b.reason));$('boundaries').append(n)}
table('acceptanceTable',['验收项','要求','结果','证据'],data.acceptance.checks.map(c=>[c.id,c.required?'必验':'诊断',pill(c.status),c.reason||c.evidence?.error|| (c.verified_parents!==undefined?`${c.verified_parents} 个父样本 / ${c.verified_children} 个视觉子样本`:'见原始验收报告')]));
table('coverageTable',['案例','几何桶','时间桶','配方物理条件','实测参数指纹','视觉子样本','来源组'],data.cases.map(c=>{const v=c.coverage_report||{};return [c.title,v.geometry?.buckets??'—',v.geometry?.timing_buckets??'—',v.physics?.requested_conditions??'—',v.physics?.condition_fingerprints??'—',v.visual?.variants??'—',v.sources?.ancestor_groups?Object.keys(v.sources.ancestor_groups).length:'—']}));
const gib=x=>x===undefined||x===null?'未采样':(x/2**30).toFixed(2)+' GiB';
table('performanceTable',['案例','实测成功 / 已执行','累计作业时间','确认物理 episode/s','主存峰值','显存采样峰值'],data.cases.map(c=>{const seconds=c.summary.wall_time_lower_bound_s;return [c.title,`${c.summary.measured_successes||0} / ${c.summary.rollout_attempted||0}`,seconds===undefined?'—':seconds.toFixed(1)+' s'+(c.summary.wall_time_uncertain?'（下界）':''),seconds&&!c.summary.wall_time_uncertain?((c.summary.committed||0)/seconds).toFixed(4):'—',gib(c.telemetry?.sampled_peak_rss_bytes||c.runtime?.peak_rss_bytes),gib(c.telemetry?.sampled_peak_gpu_bytes)]}));
let selectedCase=null,sides=[null,null],playing=false,elapsed=0,last=0,selectionVersion=0;
const byId=new Map(data.episodes.map(e=>[e.id,e]));
for(const [source,title]of [['ur5','规划生成 · UR5'],['carton','真实示范驱动 · Franka']]){const h=el('h3',title);$('caseList').append(h);for(const c of data.cases.filter(c=>c.source===source)){const b=el('button',c.title,'case');b.dataset.case=c.id;b.append(el('small',`${c.backend} · ${labels[c.status]||c.status} · ${c.summary.committed||0} 已提交`));b.onclick=()=>chooseCase(c.id);$('caseList').append(b)}}
function fillSelect(id, episodes, selected) {
  const select = $(id); select.replaceChildren();
  for (const episode of episodes) {
    const backend = data.cases.find(c => c.id === episode.case)?.backend || '';
    const option = el('option', `${backend} · ${comparison.episodeLabel(episode)} · ${episode.id.slice(-7)}`);
    option.value = episode.id; select.append(option);
  }
  if (selected) select.value = selected;
  select.disabled = !episodes.length;
}
async function chooseCase(id) {
  selectedCase = data.cases.find(c => c.id === id);
  if (!selectedCase) { $('caseTitle').textContent='输入准备尚未完成'; return; }
  document.querySelectorAll('.case').forEach(b => b.classList.toggle('active', b.dataset.case === id));
  $('caseTitle').textContent=selectedCase.title;
  $('caseSubtitle').textContent=`${selectedCase.source==='ur5'?'规划生成的专家轨迹':'真实示范驱动的专家轨迹'} · ${selectedCase.backend} · B=${selectedCase.batch_size}`;
  $('caseStatus').textContent=labels[selectedCase.status]||selectedCase.status;
  $('caseStatus').className='pill '+selectedCase.status;
  const pair=comparison.selectPair(selectedCase,data.episodes);
  comparisonReason=pair.reason;
  fillSelect('rightSelect',data.episodes.filter(e=>e.case===id),pair.right?.id);
  setReferenceOptions(pair.right,pair.left);
  $('alignment').value=selectedCase.tags.includes('motion')&&!selectedCase.tags.includes('retiming')?'phase':'time';
  $('runtime').textContent=JSON.stringify({runtime:selectedCase.runtime,telemetry:selectedCase.telemetry,attempts:selectedCase.attempts,error:selectedCase.error},null,2);
  table('attemptTable',['尝试','状态','真实动作数','原因 / 证据'],selectedCase.attempts.map(a=>{
    const evidence=el('a',(a.details||[]).join('；')||a.reason||'查看记录'); evidence.href=a.evidence;evidence.target='_blank';return [a.sequence,a.status,a.steps??0,evidence];
  }));
  await choosePair(true);
}
function setReferenceOptions(right, preferred) {
  const reference=preferred||comparison.referenceFor(selectedCase,right,data.episodes);
  let options;
  if (right?.parent) options=reference?[reference]:[];
  else options=data.episodes.filter(e=>e.kind==='physical'&&(e.case===comparison.baselineId?.(selectedCase)||e.case===reference?.case));
  if (!options.length&&reference) options=[reference];
  fillSelect('leftSelect',options,reference?.id);
}
$('leftSelect').onchange=()=>{comparisonReason='手动选择的基准对照';choosePair(true)};
$('rightSelect').onchange=()=>{
  const episode=byId.get($('rightSelect').value);
  setReferenceOptions(episode);
  comparisonReason=episode?.parent?'视觉子样本与其精确物理父样本对照；3D 轨迹不变':'已按来源与后端选择基准；可手动切换样本';
  choosePair(true);
};
async function choosePair(autoFocus=false) {
  const version=++selectionVersion;playing=false;elapsed=0;pairAnalysis=null;
  $('play').textContent='播放';$('error').classList.add('hidden');sides=[null,null];
  wantedReplays=new Set(['left','right'].map(name=>byId.get($(name+'Select').value)).filter(Boolean).map(e=>e.replay_id||e.id));
  $('effectMetrics').replaceChildren();$('parameterTable').replaceChildren();$('lineageDetails').textContent='读取当前样本…';$('parameters').textContent='';
  $('effectHint').textContent='正在读取当前对照样本…';$('sourceVideo').replaceChildren();$('physicsProbe').classList.add('hidden');
  $('pixelToggle').checked=false;$('pixelDifference').classList.add('hidden');
  for(const name of ['curves','deltaCurves','pixelCanvas']){const c=$(name);c.getContext('2d').clearRect(0,0,c.width,c.height)}
  $('deltaCaption').textContent='';$('deltaJoint').textContent='';viewers.forEach(v=>v.replay=null);
  $('comparisonNote').textContent=comparisonReason;
  for(const name of ['left','right']){
    $(name+'Empty').textContent='读取当前样本…';$(name+'Empty').classList.remove('hidden');
    for(const field of ['Rgb','Depth']){
      const video=$(name+field);video.pause();video.onloadedmetadata=null;video.removeAttribute('src');video.load();video.style.display='none';
      video.parentElement.querySelector('.no-media')?.remove();video.parentElement.append(el('div','读取当前样本…','no-media'));
    }
  }
  try {
    await Promise.all(['left','right'].map(async(name,index)=>{
      const episode=byId.get($(name+'Select').value);
      if (!episode) {
        $(name+'Empty').textContent='此案例没有已确认的实测轨迹';$(name+'Empty').classList.remove('hidden');
        for(const field of ['Rgb','Depth']) {const video=$(name+field);video.pause();video.removeAttribute('src');video.load();video.style.display='none';video.parentElement.querySelector('.no-media')?.remove();video.parentElement.append(el('div','无已确认观测','no-media'));}
        viewers[index].replay=null;return;
      }
      $(name+'Empty').textContent='读取冻结的三维与媒体证据';$(name+'Empty').classList.remove('hidden');
      const key=episode.replay_id||episode.id;
      const [model,replay]=await Promise.all([load('model',episode.model,episode.model_path),load('episode',key,episode.replay)]);
      if(version!==selectionVersion)return;
      replay.action_count=episode.actions;replay.joint_types=episode.joint_types||replay.joint_types;
      sides[index]={episode,replay};viewers[index].set(model,replay,episode.model);
      if(viewers[index].gl)$(name+'Empty').classList.add('hidden');else $(name+'Empty').textContent='WebGL 2 未启用；RGB-D 和证据仍可查看';
      for(const field of ['Rgb','Depth']) {
        const video=$(name+field),src=episode.media?.[field.toLowerCase()];video.parentElement.querySelector('.no-media')?.remove();
        if(src){video.style.display='block';video.onloadedmetadata=sync;video.src=src;}
        else {video.removeAttribute('src');video.load();video.style.display='none';video.parentElement.append(el('div','纯状态作业 · 未采集图像','no-media'));}
      }
    }));
    if(version!==selectionVersion)return;
    const right=sides[1]?.episode,left=sides[0]?.episode;
    $('comparisonNote').textContent=comparisonReason;
    document.querySelector('#lab .card').classList.toggle('comparison-visual',right?.kind==='visual');
    $('pixelToggleLabel').classList.toggle('hidden',right?.kind!=='visual');
    $('pixelToggle').checked=right?.kind==='visual';
    $('pixelDifference').classList.toggle('hidden',!$('pixelToggle').checked);
    const changes=comparison.parameterChanges(selectedCase,left,right);
    $('parameters').textContent=right?JSON.stringify({physical_changes:changes,sampled:comparison.recipe(right),capture_profile:right.metadata.visual_profile||'规范基线',render_profile:right.profile||null,visual_changes:right.child_evidence?.profile||null},null,2):selectedCase.error||'无合格样本';
    $('lineageDetails').textContent=right?JSON.stringify({episode_id:right.id,physical_parent:right.parent||null,recipe_id:right.recipe_id,commit_id:right.commit_id,attempt:right.attempt,artifact:right.artifact,metadata:right.metadata,child_evidence:right.child_evidence},null,2):'未生成样本';
    const keep=new Set(sides.filter(Boolean).map(s=>s.episode.replay_id||s.episode.id));for(const k of replays.keys())if(!keep.has(k))replays.delete(k);
    $('jointChoice').replaceChildren();(sides[1]||sides[0])?.replay.joint_names.forEach((name,index)=>{const o=el('option',name);o.value=index;$('jointChoice').append(o)});
    if(sides.every(Boolean)) {
      pairAnalysis=comparison.analyze(sides[0].replay,sides[1].replay,$('alignment').value);
      if(pairAnalysis)$('jointChoice').value=String(pairAnalysis.joint);
      const drive=selectedCase.tags.includes('stiffness')?'joint_stiffness_scale':selectedCase.tags.includes('damping')?'joint_damping_scale':null;
      const target=drive?Object.keys(comparison.recipe(right).physics?.[drive]||{})[0]:null;
      const targetIndex=target?sides[1].replay.joint_names.indexOf(target):-1;
      if(targetIndex>=0)$('jointChoice').value=String(targetIndex);
      const mean=[0,1].map(k=>sides.reduce((n,side)=>n+side.replay.tcp.reduce((a,p)=>a+p[k],0)/side.replay.tcp.length,0)/2);
      viewers.forEach(v=>{v.target=[mean[0]*.5,mean[1]*.5,.3]});
      viewers[1].setReference(sides[0].replay.tcp);
    }
    renderEffect(changes);sourceVideo();setAlignmentHint();
    $('mediaZoom').value=right?.kind==='visual'?'1.5':'1';applyMediaZoom();
    if(autoFocus&&!selectedCase.tags.includes('baseline'))focusDifference();
    drawCurves();sync();
  } catch(error) {if(version!==selectionVersion)return;$('effectHint').textContent='当前样本加载失败，未显示上一对样本的测量值。';$('error').textContent=error.stack||error.message;$('error').classList.remove('hidden');}
}
function sourceVideo(){const container=$('sourceVideo');container.replaceChildren();const side=sides[1];if(!side?.replay.source_videos)return;const note=el('p','DROID 原始视角 · CC BY 4.0 · '+side.replay.source_id);container.append(note);const row=el('div');row.style.cssText='display:flex;gap:12px;margin:12px 0';for(const item of side.replay.source_videos){const figure=el('figure');figure.style.cssText='margin:0;flex:1;min-width:0';const video=el('video');video.className='source-video';video.onloadedmetadata=sync;video.src=item.path;video.muted=true;video.preload='metadata';video.style.cssText='width:100%;max-height:250px;background:#101d2c';figure.append(video,el('figcaption',item.label));row.append(figure)}container.append(row,el('p','原始视频通过已记录的时间映射同步；入口连接无源帧时不假造对应关系。'));}
function duration(){return Math.max(...sides.filter(Boolean).map(s=>s.replay.times.at(-1)),0)}
function mappedTime(side){if(!side)return 0;if($('alignment').value==='time')return Math.min(elapsed,side.replay.times.at(-1));const progress=duration()?elapsed/duration():0,segments=side.replay.segments||[];if(!segments.length)return progress*side.replay.times.at(-1);const scaled=Math.min(progress*(segments.length),segments.length-1e-9),index=Math.floor(scaled),segment=segments[index];const totalSteps=side.episode.actions,dt=side.replay.times.at(-1)/totalSteps;return (segment.start_step+(segment.end_step-segment.start_step)*(scaled-index))*dt}
function sync(){for(let i=0;i<2;i++){const side=sides[i];if(!side)continue;const mapped=mappedTime(side);let frame=0;while(frame+1<side.replay.times.length&&side.replay.times[frame+1]<=mapped)frame++;const time=side.replay.times[frame];viewers[i].time=time;$((i?'right':'left')+'Frame').textContent='实测 t='+time.toFixed(3)+' s · 拖动旋转 / 滚轮缩放';for(const field of ['Rgb','Depth']){const video=$((i?'right':'left')+field);if(video.src&&video.readyState>=1&&Math.abs(video.currentTime-time)>.07)video.currentTime=Math.min(time,Math.max(0,(video.duration||0)-.001))}}const source=sides[1];if(source?.replay.video_seconds){const t=mappedTime(source);let index=0;while(index+1<source.replay.times.length&&source.replay.times[index+1]<=t)index++;const encoded=source.replay.video_seconds[index];for(const video of document.querySelectorAll('.source-video')){video.style.opacity=encoded===null?'.35':'1';if(encoded!==null&&video.readyState>=1&&Math.abs(video.currentTime-encoded)>.04)video.currentTime=encoded;}}
$('timeline').value=duration()?elapsed/duration()*1000:0;$('clock').textContent=$('alignment').value==='phase'?`阶段 ${(duration()?100*elapsed/duration():0).toFixed(1)}%`:`${elapsed.toFixed(2)} / ${duration().toFixed(2)} s`;}
$('play').onclick=()=>{if(!sides.some(Boolean))return;if(elapsed>=duration())elapsed=0;playing=!playing;$('play').textContent=playing?'暂停':'播放'};$('reset').onclick=()=>{elapsed=0;playing=false;$('play').textContent='播放';sync()};$('timeline').oninput=()=>{elapsed=+$('timeline').value/1000*duration();sync()};$('alignment').onchange=()=>{$('alignmentHint').textContent=$('alignment').value==='phase'?'阶段对齐按各自已记录的 segment 起止时间归一化，仅用于比较，不改变原数据时间戳。':'真实时间同步；不同物理执行可能有不同终止时间，视觉父子共享原始时钟。';sync()};
$('jointChoice').onchange=()=>{drawCurves();renderEffect(comparison.parameterChanges(selectedCase,sides[0]?.episode,sides[1]?.episode))};
function drawCurves() {
  const joint=+$('jointChoice').value||0;
  const unit=comparison.jointUnit((sides[1]||sides[0])?.replay||{},joint);
  const canvas=$('curves'),ctx=canvas.getContext('2d'),w=canvas.width=650,h=canvas.height=160;
  ctx.clearRect(0,0,w,h);ctx.strokeStyle='#e1e7ee';ctx.beginPath();ctx.moveTo(45,10);ctx.lineTo(45,h-25);ctx.lineTo(w-10,h-25);ctx.stroke();
  let min=Infinity,max=-Infinity;
  for(const side of sides.filter(Boolean))for(const q of side.replay.qpos){min=Math.min(min,q[joint]*unit.scale);max=Math.max(max,q[joint]*unit.scale)}
  if(!Number.isFinite(min)){ $('deltaCurves').getContext('2d').clearRect(0,0,650,160);return; }
  const span=Math.max(1e-6,max-min),d=Math.max(duration(),.001);
  sides.forEach((side,index)=>{if(!side)return;ctx.strokeStyle=index?'#15826c':'#3864d9';ctx.lineWidth=2;ctx.beginPath();side.replay.times.forEach((time,i)=>{const x=45+time/d*(w-60),y=12+(max-side.replay.qpos[i][joint]*unit.scale)/span*(h-42);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()});
  ctx.fillStyle='#6c7b89';ctx.font='11px system-ui';ctx.fillText(max.toPrecision(3)+' '+unit.label,0,10);ctx.fillText(min.toPrecision(3),0,h-25);ctx.fillText('0 s',40,h-5);ctx.fillText(d.toFixed(2)+' s',w-60,h-5);
  const delta=$('deltaCurves'),dc=delta.getContext('2d');delta.width=w;delta.height=h;dc.clearRect(0,0,w,h);
  $('deltaJoint').textContent=(sides[1]||sides[0]).replay.joint_names[joint];
  if(!pairAnalysis)return;
  const values=pairAnalysis.differences.map(v=>v[joint]*unit.scale);const extent=Math.max(1e-7,...values.map(Math.abs));
  dc.strokeStyle='#d5dfe9';dc.beginPath();dc.moveTo(45,h/2);dc.lineTo(w-10,h/2);dc.stroke();
  dc.strokeStyle='#8957ba';dc.lineWidth=2;dc.beginPath();
  const end=pairAnalysis.times.at(-1)||1;
  values.forEach((value,i)=>{const x=45+pairAnalysis.times[i]/end*(w-60),y=h/2-value/extent*(h/2-22);i?dc.lineTo(x,y):dc.moveTo(x,y)});dc.stroke();
  dc.fillStyle='#6c7b89';dc.font='11px system-ui';dc.fillText(extent.toPrecision(3)+' '+unit.label,0,12);dc.fillText('-'+extent.toPrecision(3),0,h-10);dc.fillText(end.toFixed(2)+' s',w-60,h-5);
  $('deltaCaption').textContent=`${pairAnalysis.alignment==='phase'?'按对应任务阶段对齐':'按共同的真实时间对齐'}，参考值线性插值。纵轴按实际差值自适应，单位 ${unit.label}；包含执行差异，不等同于纯算法残差。`;
}
const sub=(a,b)=>a.map((v,i)=>v-b[i]),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm=a=>{const n=Math.hypot(...a)||1;return a.map(v=>v/n)},identity=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
function mul(a,b){const o=Array(16).fill(0);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)o[c*4+r]+=a[k*4+r]*b[c*4+k];return o}
class ReplayViewer{
constructor(canvas){this.canvas=canvas;this.time=0;this.angle=-1.1;this.elevation=.5;this.distance=1.8;this.target=[0,0,.3];this.buffers=[];const gl=this.gl=canvas.getContext('webgl2',{antialias:true,preserveDrawingBuffer:true});if(!gl)return;const vs=`#version 300 es
in vec3 position;in vec3 normal;uniform mat4 vp;uniform mat4 model;out vec3 n;void main(){n=mat3(model)*normal;gl_Position=vp*model*vec4(position,1.0);}`;
const fs=`#version 300 es
precision highp float;in vec3 n;uniform vec3 color;uniform float unlit;out vec4 outColor;void main(){float l=.50+.44*max(0.,dot(normalize(n),normalize(vec3(-.3,-.5,1.))));outColor=vec4(color*mix(l,1.,unlit),1.);}`;
const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s};this.program=gl.createProgram();gl.attachShader(this.program,shader(gl.VERTEX_SHADER,vs));gl.attachShader(this.program,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));this.loc=Object.fromEntries(['vp','model','color','unlit'].map(k=>[k,gl.getUniformLocation(this.program,k)]));gl.enable(gl.DEPTH_TEST);gl.clearColor(.93,.95,.97,1);let drag=null;canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>drag=null;canvas.onpointercancel=()=>drag=null;canvas.onpointermove=e=>{if(!drag)return;this.angle-=(e.clientX-drag[0])*.007;this.elevation=Math.max(.06,Math.min(1.4,this.elevation+(e.clientY-drag[1])*.007));drag=[e.clientX,e.clientY];linkCameras(this)};canvas.addEventListener('wheel',e=>{e.preventDefault();this.distance=Math.max(.6,Math.min(6,this.distance*Math.exp(e.deltaY*.001)));linkCameras(this)},{passive:false})}
buffer(mesh,lines=false){const gl=this.gl,vao=gl.createVertexArray(),buffers=[];gl.bindVertexArray(vao);for(const[name,values]of [['position',mesh.vertices],['normal',mesh.normals]]){const b=gl.createBuffer();buffers.push(b);gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(values),gl.STATIC_DRAW);const at=gl.getAttribLocation(this.program,name);gl.enableVertexAttribArray(at);gl.vertexAttribPointer(at,3,gl.FLOAT,false,0,0)}const b=gl.createBuffer();buffers.push(b);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,b);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint32Array(mesh.faces),gl.STATIC_DRAW);const result={vao,buffers,count:mesh.faces.length,mode:lines?gl.LINES:gl.TRIANGLES};this.buffers.push(result);return result}
line(points){return this.buffer({vertices:points.flat(),normals:points.flatMap(()=>[0,0,1]),faces:points.map((_,i)=>i)},true)}
set(model,replay,key){if(!this.gl)return;const gl=this.gl;for(const b of this.buffers){gl.deleteVertexArray(b.vao);b.buffers.forEach(v=>gl.deleteBuffer(v))}this.buffers=[];this.reference=null;this.model=model;this.replay=replay;this.geometry=model.meshes.map(m=>this.buffer(m));const grid=[];for(let i=-10;i<=10;i++){const v=i*.1;grid.push([v,-1,-.003],[v,1,-.003],[-1,v,-.003],[1,v,-.003])}this.grid=this.line(grid);const path=[];for(let i=1;i<replay.tcp.length;i++)path.push(replay.tcp[i-1],replay.tcp[i]);this.path=this.line(path);if(replay.tcp.length){const mean=[0,1,2].map(k=>replay.tcp.reduce((s,p)=>s+p[k],0)/replay.tcp.length);this.target=[mean[0]*.5,mean[1]*.5,.3]}this.time=0}
setReference(points){if(!this.gl)return;const pairs=[];for(let i=1;i<points.length;i+=2)pairs.push(points[i-1],points[i]);this.reference=this.line(pairs)}

camera(aspect){const t=this.target,eye=[t[0]+this.distance*Math.cos(this.elevation)*Math.cos(this.angle),t[1]+this.distance*Math.cos(this.elevation)*Math.sin(this.angle),t[2]+this.distance*Math.sin(this.elevation)],z=norm(sub(eye,t)),x=norm(cross([0,0,1],z)),y=cross(z,x),v=[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1],f=1/Math.tan(.7/2),n=.02,far=50,p=[f/aspect,0,0,0,0,f,0,0,0,0,(far+n)/(n-far),-1,0,0,2*far*n/(n-far),0];return mul(p,v)}
draw(b,model,color,unlit=0){const gl=this.gl;gl.bindVertexArray(b.vao);gl.uniformMatrix4fv(this.loc.model,false,model);gl.uniform3fv(this.loc.color,color);gl.uniform1f(this.loc.unlit,unlit);gl.drawElements(b.mode,b.count,gl.UNSIGNED_INT,0)}
render(){const gl=this.gl;if(!gl||!this.replay)return;const rect=this.canvas.getBoundingClientRect();if(!rect.width||!rect.height)return;const ratio=Math.min(devicePixelRatio||1,2),w=Math.round(rect.width*ratio),h=Math.round(rect.height*ratio);if(this.canvas.width!==w||this.canvas.height!==h){this.canvas.width=w;this.canvas.height=h}gl.viewport(0,0,w,h);gl.useProgram(this.program);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.uniformMatrix4fv(this.loc.vp,false,this.camera(w/h));this.draw(this.grid,identity,[.80,.84,.87],1);if(this.reference&&$('overlayPath').checked)this.draw(this.reference,identity,[.22,.39,.85],1);this.draw(this.path,identity,this.roleColor||[.17,.58,.48],1);let index=0;while(index+1<this.replay.times.length&&this.replay.times[index+1]<=this.time)index++;for(let i=0;i<this.geometry.length;i++){const m=this.model.meshes[i];this.draw(this.geometry[i],this.replay.matrices[index][i],m.kind==='robot'?[.60,.68,.78]:m.uid==='cube'||m.uid==='carton'?[.93,.64,.23]:[.68,.72,.74])}}
}
const viewers=[new ReplayViewer($('leftWorld')),new ReplayViewer($('rightWorld'))];viewers[0].roleColor=[.22,.39,.85];
function frame(now){const dt=last?Math.min((now-last)/1000,.1):0;last=now;if(playing){elapsed=Math.min(duration(),elapsed+dt*+$('speed').value);if(elapsed>=duration()){playing=false;$('play').textContent='播放'}sync()}viewers.forEach(v=>v.render());requestAnimationFrame(frame)}requestAnimationFrame(frame);
function linkCameras(sender) { if(!$('linkedViews').checked)return;for(const viewer of viewers)if(viewer!==sender){viewer.angle=sender.angle;viewer.elevation=sender.elevation;viewer.distance=sender.distance;viewer.target=sender.target.slice()} }
function metric(title,value) {const box=el('div');box.append(el('small',title),el('strong',value));return box;}
function number(value) {return Number.isFinite(value)?Number(value.toPrecision(5)).toString():'—';}
function renderEffect(changes) {
  const left=sides[0]?.episode,right=sides[1]?.episode,metrics=$('effectMetrics');metrics.replaceChildren();
  if(!left||!right){$('effectHint').textContent='无合格样本，不能展示成功扩增。';return;}
  const timing=right.duration-left.duration;
  metrics.append(metric('实际时长：基准 → 选中',`${left.duration.toFixed(2)} → ${right.duration.toFixed(2)} s`));
  if(right.kind==='visual') {
    const evidence=data.acceptance.checks.find(c=>c.id===selectedCase.id)?.visual_checks?.find(v=>('observation_'+v.variant_id)===right.id);
    metrics.append(metric('物理轨迹 / 动作','同一个父 episode'),metric('相机标定',JSON.stringify(left.media.calibration)===JSON.stringify(right.media.calibration)?'相同':'已变化'));
    $('effectHint').textContent='纯视觉扩增不会改变 3D 运动。请对照更大的 RGB-D 画面和差异图；滚动放大或双击画面可查看细节。'+(evidence?' 原始 RGB 差异已通过独立验收。':'');
    const profile=right.child_evidence?.profile||{};
    table('parameterTable',['视觉配置','实际请求'],Object.entries(profile).map(([key,value])=>[key,JSON.stringify(value)]));
  } else {
    const joint=+$('jointChoice').value||0,unit=comparison.jointUnit(sides[1].replay,joint);
    const peak=pairAnalysis?Math.max(0,...pairAnalysis.differences.map(v=>Math.abs(v[joint]))):0;
    metrics.append(metric('选中关节采样差（最大）',unit.key==='rad'?number(peak*180/Math.PI)+'°':unit.key==='m'?number(peak*1000)+' mm':'单位未记录'),metric('回放采样 TCP 距离（最大）',pairAnalysis?number(pairAnalysis.maxTcpMm)+' mm':'—'));
    $('effectHint').textContent=selectedCase.tags.includes('physics')?'物理参数扰动可能很小；位置控制下轨迹可近似重合。下面列出实际读回差异，不用放大后的运动冒充物理效果。':selectedCase.tags.includes('retiming')?`实际时长变化 ${timing>=0?'+':''}${timing.toFixed(2)} s。retiming 只改允许的自由阶段，并非整段按固定倍数伸缩。`:'默认显示实际扩增样本，优先选择转动关节中差异较明显者；移动关节以毫米显示。下方差值曲线可查看肉眼难分辨的小扰动。';
    const rows=changes.map(v=>[v.label,number(v.before)+' '+v.unit,number(v.after)+' '+v.unit,(v.delta>=0?'+':'')+number(v.delta)+' '+v.unit+(v.percent===null?'':` (${v.percent>=0?'+':''}${v.percent.toFixed(3)}%)`)]);
    if(selectedCase.tags.includes('scene'))for(const [uid,pose]of Object.entries(right.initial_poses||{})){const base=left.initial_poses?.[uid];if(!base)continue;for(let i=0;i<3;i++){const delta=(pose[i]-base[i])*1000;if(Math.abs(delta)>.01)rows.push([uid+' 初始 '+ 'XYZ'[i],number(base[i]*1000)+' mm',number(pose[i]*1000)+' mm',(delta>=0?'+':'')+number(delta)+' mm'])}}
    table('parameterTable',['参数 / 状态','基准','选中','差值'],rows.length?rows:[['物理参数','—','—','无显著读回差异；运动差异见曲线']]);
  }
  if(left.id===right.id)$('effectHint').textContent='当前左右是同一条实测轨迹，差值为零。请在左侧选择扩增能力，或选择不同的样本。';
  if(selectedCase.tags.includes('batch')){
    const identity=data.acceptance.checks.find(c=>c.id==='batch-recipe-identity');
    metrics.replaceChildren(metric('同一 World 的容量',`B=${selectedCase.batch_size}`),metric('确认提交 / 目标',`${selectedCase.summary.committed||0} / ${selectedCase.target}`),metric('跨 B 的配方身份',identity?.status==='passed'?`${identity.shared_ordinals.length} 个一致`:'待验证'));
    $('effectHint').textContent='批量规模不是运动增幅。同一个逻辑配方在不同 B 下应保持一致；本项展示真实并行、尾批与身份验证，而不是让动作看起来更大。';
  }
  const probe=$('physicsProbe');probe.classList.add('hidden');
  if(selectedCase.tags.includes('mass')||selectedCase.tags.includes('friction')){
    const p=data.acceptance.checks.find(c=>c.id==='probe-'+selectedCase.backend)?.evidence;
    if(p?.ok){probe.classList.remove('hidden');probe.textContent=selectedCase.tags.includes('mass')?`独立受控实验（不属于当前抓放轨迹）：质量 ${p.mass.map(number).join(' / ')} kg，在相同冲量下速度 ${p.velocity_after_equal_impulse.map(number).join(' / ')} m/s，动量一致性检查通过。`:`独立受控实验（不属于当前抓放轨迹）：摩擦 ${p.friction.join(' / ')}，相同初速度后的滑动速度 ${p.sliding_velocity.map(number).join(' / ')} m/s。`}
  }
  if(!probe.classList.contains('hidden')){
    const evidence=data.acceptance.checks.find(c=>c.id==='probe-'+selectedCase.backend)?.evidence;
    const values=selectedCase.tags.includes('mass')?evidence.velocity_after_equal_impulse:evidence.sliding_velocity;
    const labels=selectedCase.tags.includes('mass')?evidence.mass.map(v=>number(v)+' kg'):evidence.friction.map(v=>'摩擦 '+v);
    const maximum=Math.max(...values.map(Math.abs),1e-9);const bars=el('div');bars.style.marginTop='10px';
    values.forEach((v,i)=>{const row=el('div');row.style.cssText='display:grid;grid-template-columns:95px 1fr 85px;gap:7px;align-items:center;margin:5px 0';const track=el('div');track.style.cssText='height:9px;background:#dde7ed;border-radius:4px';const bar=el('div');bar.style.cssText=`height:9px;width:${100*Math.abs(v)/maximum}%;background:#3864d9;border-radius:4px`;track.append(bar);row.append(el('span',labels[i]),track,el('span',number(v)+' m/s'));bars.append(row)});probe.append(bars);
  }
}
function setAlignmentHint(){ $('alignmentHint').textContent=$('alignment').value==='phase'?'阶段对齐只用于比较，各侧显示其实际采样时间；原数据和运动幅度均未改变。':'真实时间同步；不同执行的结束时间可以不同。视觉父子使用同一物理时钟。'; }
function focusDifference() {
  const right=sides[1]?.episode;
  if(!right)return;
  let target=pairAnalysis?.peakTime||0;
  if(pairAnalysis){const joint=+$('jointChoice').value||0;const delta=pairAnalysis.differences.map(v=>Math.abs(v[joint]));target=pairAnalysis.times[delta.indexOf(Math.max(...delta))]||0;}
  if(right.kind==='visual'){
    const evidence=data.acceptance.checks.find(c=>c.id===right.case)?.visual_checks?.find(v=>'observation_'+v.variant_id===right.id);
    const values=evidence?.mean_absolute_rgb_differences||[];
    const index=values.length?values.indexOf(Math.max(...values)):1;
    target=index===0?0:index===1?Math.floor(right.actions/2)/right.actions*right.duration:right.duration;
  }
  if($('alignment').value==='phase'&&sides[1]){
    const ranges=comparison.segments(sides[1].replay);elapsed=duration()*comparison.phaseCoordinate(sides[1].replay,target)/(ranges.length||1);
  }else elapsed=target;
  playing=false;$('play').textContent='播放';sync();
}
$('focusDifference').onclick=focusDifference;
$('alignment').onchange=()=>{pairAnalysis=sides.every(Boolean)?comparison.analyze(sides[0].replay,sides[1].replay,$('alignment').value):null;setAlignmentHint();renderEffect(comparison.parameterChanges(selectedCase,sides[0]?.episode,sides[1]?.episode));drawCurves();sync()};
function applyMediaZoom(){const zoom=+$('mediaZoom').value;$('zoomLabel').textContent=zoom.toFixed(1)+'×';document.querySelectorAll('.media video').forEach(v=>v.style.transform=`scale(${zoom})`)}
$('mediaZoom').oninput=applyMediaZoom;
$('pixelToggle').onchange=()=>{$('pixelDifference').classList.toggle('hidden',!$('pixelToggle').checked);drawPixelDifference()};
const pixelInputs=[document.createElement('canvas'),document.createElement('canvas')];let pixelAccessBlocked=false;
function drawPixelDifference(){
  if(!$('pixelToggle').checked||sides[1]?.episode.kind!=='visual'||pixelAccessBlocked)return;
  const left=$('leftRgb'),right=$('rightRgb'),canvas=$('pixelCanvas'),ctx=canvas.getContext('2d');
  if(left.readyState<2||right.readyState<2||left.seeking||right.seeking||Math.abs(left.currentTime-right.currentTime)>.02){ctx.clearRect(0,0,canvas.width,canvas.height);$('pixelStatus').textContent='正在同步两侧视频；同步前不显示差异';return;}
  const w=left.videoWidth,h=left.videoHeight;if(w!==right.videoWidth||h!==right.videoHeight||!w)return;
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
  try {
  const input=pixelInputs.map((c,i)=>{if(c.width!==w||c.height!==h){c.width=w;c.height=h;}const context=c.getContext('2d',{willReadFrequently:true});context.drawImage(i?right:left,0,0);return context.getImageData(0,0,w,h).data});
  const output=ctx.createImageData(w,h),gain=+$('pixelGain').value;
  for(let i=0;i<output.data.length;i+=4){const delta=Math.max(Math.abs(input[0][i]-input[1][i]),Math.abs(input[0][i+1]-input[1][i+1]),Math.abs(input[0][i+2]-input[1][i+2]));output.data[i]=output.data[i+1]=output.data[i+2]=Math.min(255,delta*gain);output.data[i+3]=255;}
  ctx.putImageData(output,0,0);$('pixelStatus').textContent=`预览帧 t=${right.currentTime.toFixed(3)} s · 显示增益 ${gain}× · 含渲染与编码差异`;
  }catch(error){ctx.clearRect(0,0,canvas.width,canvas.height);pixelAccessBlocked=true;$('pixelStatus').textContent=error.name==='SecurityError'?'浏览器禁止读取 file:// 视频像素。RGB-D 对照仍可使用；通过本地 HTTP 预览可启用差异图。':'差异图读取失败：'+error.message;}
}
setInterval(()=>{if(!document.hidden&&!$('lab').classList.contains('hidden'))drawPixelDifference()},200);

chooseCase(data.cases.find(c=>c.id==='default-rerender-controls'&&data.episodes.some(e=>e.case===c.id))?.id||data.cases.find(c=>data.episodes.some(e=>e.case===c.id))?.id||data.cases[0]?.id);

for(const figure of document.querySelectorAll('.media figure'))figure.ondblclick=()=>figure.requestFullscreen?.();
