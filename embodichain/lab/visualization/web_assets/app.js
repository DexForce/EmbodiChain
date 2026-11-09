'use strict';

const $ = id => document.getElementById(id);
let state;
let summaryKey;

function showError(message) {
    $('error').textContent = message || '';
    $('error').hidden = !message;
}

function update(next) {
    state = next;
    document.title = next.title;
    const summary = next.summary;
    const key = JSON.stringify(summary);
    if (key !== summaryKey) {
        summaryKey = key;
        $('summary-title').textContent = summary?.title || 'Initialization information';
        $('summary-disabled').hidden = !!summary;
        $('initialization-summary').hidden = !summary;
        const body = $('initialization-summary').querySelector('tbody');
        body.replaceChildren();
        let previous;
        for (const [section, setting, value] of summary?.rows || []) {
            const row = document.createElement('tr');
            if (section !== previous) row.classList.add('section-start');
            for (const text of [section === previous ? '' : section, setting, value]) {
                const cell = document.createElement('td');
                cell.textContent = text;
                row.append(cell);
            }
            previous = section;
            body.append(row);
        }
    }
    showError(next.error);
}

async function readState() {
    const response = await fetch('/api/state', {
        cache: 'no-store', signal: AbortSignal.timeout(5000)
    });
    if (!response.ok) throw new Error('Information unavailable (HTTP ' + response.status + ')');
    update(await response.json());
}

async function connect() {
    while (!window.RTC_SceneLeaving) {
        try {
            const description = await RTC_ReadSceneDescription();
            if (description.state === 'error') throw new Error(description.error);
            if (!description.room || !description.sources?.length) throw new Error('Waiting for scene cameras…');
            await readState();
            const sources = description.sources;
            await window.RTC_Run({
                Room: description.room, Role: 'offerer', Video: true, View: true,
                TestPattern: false, Timeout: 30, Input: false,
                InputStreams: sources.map(() => false), Cameras: sources.length,
                FramesPerSecond: description.fps, BitrateBps: description.bitrate,
                StreamNames: sources.map(source => source.name),
                StreamSizes: sources.map(source => [source.width, source.height])
            });
        } catch (error) {
            showError(error.message);
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
}

async function poll() {
    if (window.RTC_SceneLeaving) return;
    try {
        await readState();
        const peer = window.RTC_ViewPeer;
        for (const stream of peer?.Videos || []) {
            if (!stream.Element || stream.Element.dataset.mounted || !state) continue;
            const main = peer.Config.StreamNames[stream.Index] === state.camera_name;
            stream.Element.dataset.mounted = 'true';
            stream.Element.tabIndex = -1;
            const container = $(main ? 'main-view' : 'sensor-views');
            if (main) {
                for (const child of [...container.children]) {
                    if (child.id !== 'waiting') child.remove();
                }
            }
            container.append(stream.Card);
            if (!main) $('task-cameras').hidden = false;
        }
        const ready = peer?.ViewReady && !peer.Stopped;
        $('waiting').hidden = !!(ready && state?.frames);
        $('waiting').textContent = ready ? 'Waiting for a rendered frame…' : 'Connecting video…';
    } catch (error) {
        showError(error.message);
        $('waiting').hidden = false;
        $('waiting').textContent = 'Preview unavailable. Reconnecting…';
    }
    setTimeout(poll, 1000);
}

connect();
poll();
