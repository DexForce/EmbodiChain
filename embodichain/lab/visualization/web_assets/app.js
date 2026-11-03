'use strict';
const $ = id => document.getElementById(id);
let state, stopped = false, busy = false;
const showError = message => { $('error').textContent = message || ''; $('error').hidden = !message; };
function canControl() {
    const peer = window.RTC_ViewPeer;
    return !!state && peer?.ViewReady && !peer.Stopped && !peer.StopRequested && peer.InputOwner !== 'local';
}
function update(next) {
    state = next;
    $('pause').checked = next.paused;
    $('gizmo').checked = next.gizmo;
    $('scene').textContent = JSON.stringify({run: next.run_id, revision: next.revision, assets: next.assets}, null, 2);
    if (next.error) showError(next.error);
}
async function command(fields) {
    if (!canControl()) return;
    busy = true;
    try {
        const peer = window.RTC_ViewPeer;
        const response = await fetch('/api/command', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({...fields, run_id: state.run_id, revision: state.revision,
                room: peer.Config.Room, token: peer.Token, after: peer.After}),
            signal: AbortSignal.timeout(5000)
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error);
        update(result); showError('');
    } catch (error) { showError(error.message); } finally { busy = false; }
}
$('reset').onclick = () => command({action: 'reset_view'});
$('render').onclick = () => command({action: 'render'});
$('pause').onchange = () => command({action: 'pause', enabled: $('pause').checked});
$('gizmo').onchange = () => command({action: 'gizmo', enabled: $('gizmo').checked});
$('reconnect').onclick = () => location.reload();
$('RTC_Stop').addEventListener('click', () => { stopped = true; });
$('save').onclick = () => {
    const video = $('main-view').querySelector('video');
    if (!video?.videoWidth || !state?.frames) { showError('Wait for a rendered frame.'); return; }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(blob => {
        if (!blob) return;
        const link = document.createElement('a'); link.href = URL.createObjectURL(blob);
        link.download = `embodichain-${state.run_id}.png`; link.click();
        setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    });
};
async function connect() {
    while (!stopped && !window.RTC_SceneLeaving) {
        try {
            const description = await RTC_ReadSceneDescription();
            if (description.state === 'error') throw new Error(description.error);
            if (!description.room || !description.sources?.length) throw new Error('Waiting for scene cameras…');
            const sources = description.sources;
            await window.RTC_Run({Room: description.room, Role: 'offerer', Video: true, View: true,
                TestPattern: false, Timeout: 30, Input: true,
                InputStreams: sources.map(source => source.input === true), Cameras: sources.length,
                FramesPerSecond: description.fps, BitrateBps: description.bitrate,
                StreamNames: sources.map(source => source.name),
                StreamSizes: sources.map(source => [source.width, source.height])});
        } catch (error) { showError(error.message); }
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
}
async function poll() {
    if (window.RTC_SceneLeaving) return;
    try {
        if (!busy) {
            const response = await fetch('/api/state', {cache: 'no-store', signal: AbortSignal.timeout(5000)});
            if (!response.ok) throw new Error(`Scene status HTTP ${response.status}`);
            update(await response.json());
        }
        const peer = window.RTC_ViewPeer;
        for (const stream of peer?.Videos || []) {
            if (!stream.Element || stream.Element.dataset.mounted || !state) continue;
            const main = peer.Config.StreamNames[stream.Index] === state.camera_name;
            stream.Element.dataset.mounted = 'true';
            const container = $(main ? 'main-view' : 'sensor-views');
            if (main) {
                for (const child of [...container.children]) if (child.id !== 'waiting') child.remove();
            }
            container.append(stream.Card);
        }
        const ready = peer?.ViewReady && !peer.Stopped && !stopped;
        $('waiting').hidden = !!(ready && state?.frames);
        $('waiting').textContent = stopped ? 'Viewing stopped' : ready ? 'Waiting for a rendered frame…' : 'Connecting video…';
        for (const id of ['reset', 'render', 'pause']) $(id).disabled = !canControl() || busy;
        $('gizmo').disabled = !canControl() || busy || !state?.allow_commands;
    } catch (error) { showError(error.message); }
    setTimeout(poll, 500);
}
connect(); poll();
