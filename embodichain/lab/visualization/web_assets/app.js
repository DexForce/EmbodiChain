'use strict';
const $ = id => document.getElementById(id);
let state, stopped = false, busy = false;
const showError = message => { $('error').textContent = message || ''; $('error').hidden = !message; };
function canControl() {
    const peer = window.RTC_ViewPeer;
    return !!state && peer?.ViewReady && !peer.Stopped && !peer.StopRequested && peer.InputOwner !== 'local';
}
function update(next) {
    if (state && (state.run_id!==next.run_id || state.revision!==next.revision)) {
        dragging=null;targetPending=null;heldKeys.clear();
    }
    state = next;
    $('pause').checked = next.paused;
    $('gizmo').checked = next.gizmo;
    $('scene').textContent = JSON.stringify({run: next.run_id, revision: next.revision, assets: next.assets}, null, 2);
    if (next.error) showError(next.error);
    updateControls();
}
async function command(fields) {
    if (!canControl()) return;
    busy = true;
    try {
        const peer = window.RTC_ViewPeer;
        const response = await fetch('/api/command', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({run_id: state.run_id, revision: state.revision, ...fields,
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
                for (const child of [...container.children]) if (!['waiting','targets'].includes(child.id)) child.remove();
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

// Shared simulator controls use creator-thread commands; the native video is read-only.
let dragging = null, targetPending = null, rotationMode = false;
const heldKeys = new Set();
const codeOf = event => RTC_InputScancode(event.code);
document.addEventListener('keydown', event => {
    if (event.target.tagName !== 'VIDEO' || !canControl() || !state.allow_commands) return;
    const key = codeOf(event);
    if (key) { heldKeys.add(key); command({action:'key', key, down:true}); }
}, true);
document.addEventListener('keyup', event => {
    const key = codeOf(event);
    if (heldKeys.delete(key)) command({action:'key', key, down:false});
}, true);
function releaseKeys() {
    for (const key of heldKeys) command({action:'key', key, down:false});
    heldKeys.clear();
}
window.addEventListener('blur', releaseKeys);
document.addEventListener('visibilitychange', () => { if (document.hidden) releaseKeys(); });
setInterval(() => {
    if (!busy && heldKeys.size) for (const key of heldKeys) command({action:'key', key, down:true});
    if (!busy && targetPending) {
        const value=targetPending; targetPending=null; command(value);
    }
}, 100);
$('rotate').onclick = () => {
    rotationMode=!rotationMode;
    $('rotate').textContent=rotationMode?'Mode: rotate':'Mode: translate';
};
function projection(point) {
    const m=state.camera_pose,k=state.camera_intrinsic;
    const delta=point.map((v,i)=>v-m[i][3]);
    const c=[0,1,2].map(i=>delta.reduce((sum,v,j)=>sum+m[j][i]*v,0));
    if(c[2]>=-.01)return null;
    return [k[0][0]*c[0]/-c[2]+k[0][2],-k[1][1]*c[1]/-c[2]+k[1][2]];
}
function svgNode(name, attributes) {
    const node=document.createElementNS('http://www.w3.org/2000/svg',name);
    for(const [key,value] of Object.entries(attributes))node.setAttribute(key,String(value));
    return node;
}
function updateControls() {
    if(!state)return;
    document.title=state.title;
    const video=$('main-view').querySelector('video'),svg=$('targets');
    svg.replaceChildren();
    if(video && state.camera_intrinsic && canControl() && state.allow_commands) {
        const r=video.getBoundingClientRect(),parent=$('main-view').getBoundingClientRect();
        Object.assign(svg.style,{left:`${r.left-parent.left}px`,top:`${r.top-parent.top}px`,width:`${r.width}px`,height:`${r.height}px`});
        svg.setAttribute('viewBox',`0 0 ${video.videoWidth} ${video.videoHeight}`);
        for(const gizmo of state.gizmos || []) {
            const origin=gizmo.pose.slice(0,3).map(row=>row[3]),a=projection(origin);
            if(!a)continue;
            const label=svgNode('text',{x:a[0]+8,y:a[1]-12,fill:'white','font-size':14});
            label.textContent=gizmo.id;svg.append(label);
            for(let axis=0;axis<3;axis++) {
                const end=[...origin];end[axis]+=.18;
                const b=projection(end);if(!b)continue;
                const line=svgNode('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:['#ff6969','#69ef96','#639fff'][axis],'stroke-width':8,'pointer-events':'stroke'});
                line.style.cursor='grab';
                line.addEventListener('pointerdown',event=>{
                    event.preventDefault();event.stopPropagation();
                    const rect=svg.getBoundingClientRect();
                    dragging={run_id:state.run_id,revision:state.revision,id:gizmo.id,pose:structuredClone(gizmo.pose),axis,dx:b[0]-a[0],dy:b[1]-a[1],x:event.clientX,y:event.clientY,sx:video.videoWidth/rect.width,sy:video.videoHeight/rect.height};
                    svg.setPointerCapture(event.pointerId);
                });
                svg.append(line);
            }
        }
    }
    const revision=`${state.run_id}:${state.revision}:${(state.joints||[]).map(j=>j.control_id).join(",")}`;
    if($('joints').dataset.revision!==revision) {
        $('joints').dataset.revision=revision;$('joints').replaceChildren();
        for(const joint of state.joints || []) {
            const label=document.createElement('label'),input=document.createElement('input');
            input.type='range';input.min=joint.lower??-Math.PI;input.max=joint.upper??Math.PI;
            input.step=joint.step;input.value=joint.initial_value;
            input.disabled=!state.allow_commands;
            input.onchange=()=>command({action:'joint',id:joint.control_id,value:Number(input.value)});
            label.append(document.createTextNode(joint.joint_name),input);$('joints').append(label);
        }
    }
    $('replay-controls').hidden=!state.replay?.visible;
    $('seek').disabled=!canControl() || !state.allow_commands;
    if(state.replay?.visible && document.activeElement!==$('seek')) {
        $('seek').max=state.replay.max_step;$('seek').value=state.replay.step;
    }
}
$('seek').oninput=()=>command({action:'seek',step:Number($('seek').value)});
$('targets').addEventListener('pointermove',event=>{
    if(!dragging)return;
    const d=dragging,dx=(event.clientX-d.x)*d.sx,dy=(event.clientY-d.y)*d.sy;
    const pose=structuredClone(d.pose),den=d.dx*d.dx+d.dy*d.dy;
    if(den<1)return;
    const delta=(dx*d.dx+dy*d.dy)/den;
    if(rotationMode) {
        const angle=delta*.5,c=Math.cos(angle),s=Math.sin(angle),axis=d.axis;
        const r=[[c,0,0],[0,c,0],[0,0,c]];r[axis][axis]=1;
        const i=(axis+1)%3,j=(axis+2)%3;r[i][j]=-s;r[j][i]=s;
        for(let row=0;row<3;row++)for(let col=0;col<3;col++)pose[row][col]=r[row].reduce((sum,v,k)=>sum+v*d.pose[k][col],0);
    } else pose[d.axis][3]+=delta*.18;
    targetPending={action:'target',run_id:d.run_id,revision:d.revision,id:d.id,pose};
});
function endTargetDrag(){dragging=null;}
$('targets').addEventListener('pointerup',endTargetDrag);
$('targets').addEventListener('pointercancel',()=>{dragging=null;targetPending=null;});
$('targets').addEventListener('lostpointercapture',endTargetDrag);

$('main-view').addEventListener('pointerdown', event=>{
    if(event.button===2 && state?.pan_enabled===false){event.preventDefault();event.stopImmediatePropagation();}
},true);
