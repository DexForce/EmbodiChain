# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Offline interactive joint-domain slices showing observed evidence only."""

from __future__ import annotations

from pathlib import Path
import json

from .search import FeasibilityAtlas

__all__ = ["write_feasibility_report"]


def write_feasibility_report(atlas: FeasibilityAtlas, path: str | Path) -> None:
    """Write a standalone SVG/HTML slice inspector without external resources.

    Args:
        atlas: Exact joint parameter samples and frozen region statistics.
        path: Destination HTML path.

    Points denote observed executions. Region outlines denote proposal bounds,
    never a certificate for unsampled points. Other coordinates are explicitly
    filtered by the user's normalized condition and slice tolerance.
    """
    data = {
        "axes": [
            f"{v.name}/{axis}"
            for v in atlas.request.variables
            for axis in ("dx", "dy", "yaw")
        ],
        "bounds": [
            [a, b] for v in atlas.request.variables for a, b in zip(v.lower, v.upper)
        ],
        "fixed": [
            a == b for v in atlas.request.variables for a, b in zip(v.lower, v.upper)
        ],
        "points": [
            {
                "p": list(o.candidate.parameters),
                "mode": o.candidate.mode_id,
                "ok": o.evaluation.accepted,
                "holdout": o.region_id is not None,
                "id": o.candidate.candidate_id,
                "checks": [
                    {"stage": c.stage, "status": c.status, "reason": c.reason}
                    for c in o.evaluation.checks
                ],
            }
            for o in atlas.observations
        ],
        "regions": [
            {
                "mode": r.mode_id,
                "lo": r.lower,
                "hi": r.upper,
                "n": r.validation_completed,
                "s": r.validation_successes,
                "ci": r.confidence_interval,
                "complete": r.validation_complete,
            }
            for r in atlas.regions
        ],
        "task": atlas.request.task_id,
    }
    payload = (
        json.dumps(data, allow_nan=False)
        .replace("<", r"\u003c")
        .replace("&", r"\u0026")
    )
    document = r"""<!doctype html><meta charset="utf-8"><title>Scene / Robot Feasibility</title>
<style>body{font:15px system-ui;max-width:1100px;margin:32px auto;color:#17212e;background:#f5f7fa}main{background:white;padding:24px;border-radius:12px}label{display:inline-block;margin:8px}svg{width:100%;background:#fbfcfe;border:1px solid #d8dee7}pre{white-space:pre-wrap}input{width:72px}h1{font-size:24px}.note{color:#596579}</style>
<main><h1>Scene / Robot Feasibility</h1><p id="task"></p><p class="note">Observed samples only. Outlined regions are conditional proposal domains. All coordinates are normalized to their declared bounds. Other dimensions are fixed by the slice controls; this is not a projection claiming continuous feasibility.</p>
<label>X <select id="x"></select></label><label>Y <select id="y"></select></label><label>Mode <select id="mode"></select></label><label>Slice tolerance <input id="tol" type="number" value="0.15" min="0" max="1" step="0.01"></label><div id="conditions"></div><svg id="plot" viewBox="0 0 800 520"></svg><p id="summary"></p><pre id="detail">Select a sample to inspect its stage evidence.</pre></main>
<script type="application/json" id="data">PAYLOAD</script><script>
const d=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg';
$('task').textContent=d.task;
for(const id of ['x','y'])d.axes.forEach((a,i)=>{let o=document.createElement('option');o.value=i;o.textContent=a+(d.fixed[i]?' (fixed)':'')+' ['+d.bounds[i].join(', ')+']';$(id).append(o)});
$('y').value=Math.min(1,d.axes.length-1);
for(const mode of new Set(d.points.map(p=>p.mode))){let o=document.createElement('option');o.value=mode;o.textContent=mode;$('mode').append(o)}
d.axes.forEach((a,i)=>{let l=document.createElement('label');l.textContent=a+' = ';let v=document.createElement('input');v.id='c'+i;v.type='number';v.min=0;v.max=1;v.step=.01;v.value=d.fixed[i]?0:.5;v.disabled=d.fixed[i];l.append(v);$('conditions').append(l);v.oninput=draw});
function node(tag,attrs){let n=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);$('plot').append(n);return n}
function draw(){const x=+$('x').value,y=+$('y').value,tol=+$('tol').value,mode=$('mode').value;$('plot').replaceChildren();if(!d.axes.length){$('summary').textContent='Fixed domain: inspect atlas.json for exact candidate evidence.';return}const px=v=>40+720*v,py=v=>480-440*v;node('path',{d:'M40 40V480H760',stroke:'#657184',fill:'none'});let included=0;
function matches(p){return p.every((v,i)=>i===x||i===y||Math.abs(v-+$('c'+i).value)<=tol)}
for(const r of d.regions){if(r.mode!==mode||!r.lo.every((v,i)=>i===x||i===y||(v<=+$('c'+i).value&&r.hi[i]>=+$('c'+i).value)))continue;node('rect',{x:px(r.lo[x]),y:py(r.hi[y]),width:720*(r.hi[x]-r.lo[x]),height:440*(r.hi[y]-r.lo[y]),fill:'none',stroke:'#8b96a8','stroke-dasharray':'5 4'})}
for(const p of d.points){if(p.mode!==mode||!matches(p.p))continue;included++;let n=node('circle',{cx:px(p.p[x]),cy:py(p.p[y]),r:p.holdout?5:3,fill:p.ok?'#137d54':'#c44848',opacity:.8});let title=document.createElementNS(ns,'title');title.textContent=p.id;n.append(title);n.onclick=()=>{$('detail').textContent=JSON.stringify({...p,physical_offsets:p.p.map((v,i)=>d.bounds[i][0]+v*(d.bounds[i][1]-d.bounds[i][0]))},null,2)}}
$('summary').textContent=included+' observed samples in this conditional slice. Green: measured accepted; red: rejected or unverified. Large points: independent validation. '+JSON.stringify(d.regions.filter(r=>r.mode===mode).map(r=>({successes:r.s,trials:r.n,interval95:r.ci,complete:r.complete})));}
for(const id of ['x','y','mode','tol'])$(id).onchange=draw;draw();
</script>"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document.replace("PAYLOAD", payload), encoding="utf-8")
