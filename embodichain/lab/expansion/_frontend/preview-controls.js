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

let status = null, busy = false, dragging = false, filterKey = null, sending = false;
const pending = [];
const filterKeys = ['space', 'aff', 'variant'];
const selectKeys = [...filterKeys, 'align'];
const layerKeys = ['paths', 'targets', 'positions'];
const $ = id => document.getElementById(id);
const viewport = $('viewport');
let videoConnectionFailed = false;
let video;
const recordedLabel = value => value === 'unrecorded' ? previewText('unrecorded') : value;
let connection = 'connecting';
function setConnection(state) {
  connection = state;
  $('connection').textContent = previewText(state);
  $('connection').classList.toggle('error', state === 'controlFailed' || state === 'disconnected');
}
async function connectVideo() {
  while (!window.RTC_SceneLeaving) {
    try {
      const description = await RTC_ReadSceneDescription();
      if (description.state === 'error' || !description.room || !description.sources?.length) {
        throw Error('Native video source unavailable');
      }
      const sources = description.sources;
      await window.RTC_Run({
        Room: description.room, Role: 'offerer', Video: true, View: true,
        TestPattern: false, Timeout: 30, Input: false,
        InputStreams: sources.map(() => false), Cameras: sources.length,
        FramesPerSecond: description.fps, BitrateBps: description.bitrate,
        StreamNames: sources.map(source => source.name),
        StreamSizes: sources.map(source => [source.width, source.height]),
      });
    } catch {
      // The page shows a localized reconnecting status below.
    }
    videoConnectionFailed = true;
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}

function updateVideo() {
  const peer = window.RTC_ViewPeer;
  const next = peer?.Videos[0]?.Element;
  if (next && next !== video) {
    video?.remove();
    video = next;
    video.controls = false;
    video.dataset.label = 'frameAlt';
    video.setAttribute('aria-label', previewText('frameAlt'));
    viewport.insertBefore(video, $('caption'));
  }
  if (peer?.ViewReady && !peer.Stopped && video?.readyState >= 2) {
    videoConnectionFailed = false;
    return 'connected';
  }
  return videoConnectionFailed ? 'reconnecting' : 'connecting';
}

function isDiscrete(update){return 'toggle_playback' in update || 'toggle_ghost' in update || 'ghosts' in update}
async function control(update) {
  const last = pending.at(-1);
  // Coalesce continuous controls, preserving individual clicks and their filters.
  if (last && !isDiscrete(last) && !isDiscrete(update)) Object.assign(last, update);
  else pending.push(update);
  if (sending) return;
  sending = true;
  try {
    while (pending.length) {
      const peer = window.RTC_ViewPeer;
      if (!peer?.ViewReady || peer.Stopped) throw Error('WebRTC viewer is not ready');
      // The native listener validates and strips these room credentials.
      const body = {...pending.shift(), room: peer.Config.Room, token: peer.Token, after: peer.After};
      const response = await fetch('/api/command', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
      if (!response.ok) throw Error('Preview control request failed');
    }
  } catch {
    pending.length = 0;
    setConnection('controlFailed');
  } finally {
    sending = false;
  }
}
for (const key of selectKeys) $(key).onchange = () => control({[key]: $(key).value});
for (const key of ['ghosts', ...layerKeys]) $(key).onchange = () => control({[key]: $(key).checked});
$('play').onclick=()=>control({toggle_playback:true});
$('reset').onclick=()=>control({progress:0,playing:false});
document.addEventListener('keydown',event=>{
  if((event.code!=='Space' && event.key!==' ') || event.defaultPrevented || event.isComposing ||
     event.altKey || event.ctrlKey || event.metaKey || event.shiftKey)return;
  if(event.target.isContentEditable || event.target.closest?.('select, textarea, input:not([type="range"]), summary'))return;
  event.preventDefault();
  if(!event.repeat)$('play').click();
});
$('progress').onpointerdown=()=>dragging=true;$('progress').oninput=()=>control({progress:+$('progress').value/1000,playing:false});window.addEventListener('pointerup',()=>dragging=false);
const cameras={orbit:{yaw:-2.15,elevation:.55,distance:1.65},top:{yaw:-1.57,elevation:1.48,distance:1.5},front:{yaw:-1.57,elevation:.2,distance:1.5},close:{yaw:-2.15,elevation:.7,distance:.85}};
$('camera').onchange=()=>{if(cameras[$('camera').value])control(cameras[$('camera').value])};
let pointer = null;
viewport.onpointerdown = event => {
  if (!status) return;
  pointer = {x: event.clientX, y: event.clientY, yaw: status.yaw, elevation: status.elevation};
  viewport.setPointerCapture(event.pointerId);
};
viewport.onpointermove = event => {
  if (pointer) control({
    yaw: Math.max(-20, Math.min(20, pointer.yaw - (event.clientX - pointer.x) * .008)),
    elevation: Math.max(.1, Math.min(1.5, pointer.elevation + (event.clientY - pointer.y) * .006)),
  });
};
viewport.onpointerup = viewport.onpointercancel = () => { pointer = null; };
viewport.addEventListener('wheel', event => {
  event.preventDefault();
  if (status) control({distance: Math.max(.2, Math.min(20, status.distance + event.deltaY * .001))});
}, {passive: false});
function updateCandidates(s){
  const list=$('candidates');
  const key=s.visible.join(',');
  if(key!==filterKey){
    filterKey=key;
    const buttons=s.clips.filter(c=>s.visible.includes(c.index)).map(c=>{
      const button=document.createElement('button');
      button.type='button';
      button.dataset.index=c.index;
      button.textContent=c.label;
      const details=document.createElement('small');
      details.textContent = previewText('candidateInfo', {
        space: c.space,
        aff: recordedLabel(c.aff),
        variant: recordedLabel(c.variant),
      });
      button.append(details);
      button.onclick=()=>control({toggle_ghost:c.index});
      return button;
    });
    list.replaceChildren(...buttons);
  }
  const shown=new Set(s.visible_ghosts);
  for(const button of list.children){
    button.setAttribute('aria-pressed', String(shown.has(Number(button.dataset.index))));
  }
  const all=$('ghosts');
  all.disabled=s.count===0;
  all.checked=s.count>0 && shown.size===s.count;
  all.indeterminate=shown.size>0 && !all.checked;
}
function renderCoverage(coverage) {
  $('coverage-panel').hidden = false;
  $('coverage-summary').textContent = previewText('coverageCounts', {
    planned: coverage.planned_attempts,
    attempted: coverage.records.reduce((count, record) => count + record.attempts, 0),
    confirmed: coverage.records.reduce((count, record) => count + record.confirmed, 0),
    result: previewText(coverage.coverage_complete ? 'coverageComplete' : 'coverageIncomplete'),
  });
  const axes = coverage.axes.map(axis => `${axis.name}: [${axis.lower}, ${axis.upper}]`);
  const records = coverage.records.map(record => {
    const summary = previewText('coverageRecord', {
      ...record,
      planned: record.planned ?? coverage.recipes_per_point,
    });
    const values = Object.entries(record.values).map(([name, value]) => `${name}=${Number(value).toPrecision(4)}`);
    return `${summary}\n${values.join(', ')}`;
  });
  $('coverage-details').textContent = [previewText('coverageScope'), axes.join('\n'), records.join('\n\n')].join('\n\n');
}

function renderReplayText() {
  setConnection(connection);
  $('play').textContent = previewText(status?.playing ? 'pause' : 'play');
  if (!status) {
    $('caption').textContent = previewText('loadingModel');
    $('detail').textContent = previewText('waitingResults');
    return;
  }

  const reference = status.has_reference
    ? previewText('reference')
    : previewText('firstRecording', {label: status.label});
  $('total').textContent = previewText('matchingTotal', {total: status.total});
  $('clock').textContent = previewText('clock', {
    time: status.time.toFixed(2),
    frame: status.source_frame,
    last: status.source_frames - 1,
  });
  $('phase').textContent = recordedLabel(status.phase);
  $('caption').textContent = previewText('caption', {reference, count: status.visible_ghosts.length});
  $('detail').textContent = previewText('executionDetail', {
    reference,
    candidate: recordedLabel(status.candidate),
    steps: status.physics_steps,
    episode: status.episode,
  });
}

function populateFilters(s) {
  for (const key of filterKeys) {
    for (const value of s.options[key]) {
      const option = new Option(recordedLabel(value), value);
      if (value === 'unrecorded') option.dataset.text = 'unrecorded';
      $(key).append(option);
    }
  }
  if (!s.options.align.includes('phase')) $('align').querySelector('[value="phase"]').remove();
}

async function poll() {
  if (busy || document.hidden) return;
  busy = true;
  try {
    const response = await fetch('/api/state', {cache: 'no-store'});
    if (!response.ok) throw Error('Could not read replay state');
    const s = await response.json();
    if (s.frame === undefined) return;
    if (!status) {
      populateFilters(s);
      if (s.coverage) renderCoverage(s.coverage);
    }
    status = s;
    if (!sending) {
      for (const key of selectKeys) {
        if (document.activeElement !== $(key)) $(key).value = s[key];
      }
      for (const key of layerKeys) $(key).checked = s[key];
      if (document.activeElement !== $('camera')) {
        $('camera').value = Object.keys(cameras).find(key =>
          Object.keys(cameras[key]).every(axis => Math.abs(s[axis] - cameras[key][axis]) < .001)
        ) || 'custom';
      }
    }
    connection = updateVideo();
    renderReplayText();
    $('count').textContent = s.count;
    if (!dragging) $('progress').value = Math.round(s.progress * 1000);
    updateCandidates(s);
    deviationCharts.setFrame(s.source_frame);
  } catch {
    setConnection('disconnected');
  } finally {
    busy = false;
  }
}

document.addEventListener('preview-language-change', () => {
  renderReplayText();
  if (status) {
    filterKey = null;
    updateCandidates(status);
    if (status.coverage) renderCoverage(status.coverage);
  }
});
renderReplayText();
connectVideo();
setInterval(poll, 100);
poll();
