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
const assert = require('node:assert/strict');
const C = require(process.argv[2]);
const episode = (id, extra={}) => ({ id,case:'motion',kind:'physical',metadata:{},...extra });
const nominal=episode('nominal',{metadata:{expansion:[{trajectory_selected:0,trajectory_variants:[{variant_ordinal:0,spatial_operator:'none'}]}]}});
const augmented=episode('augmented',{metadata:{expansion:[{trajectory_selected:1,trajectory_variants:[{variant_ordinal:0,spatial_operator:'none'},{variant_ordinal:1,spatial_operator:'via_points'}]}]}});
const baseline=episode('baseline',{case:'ur5-baseline'});
const motion={id:'motion',source:'ur5',backend:'default',tags:['motion','via_points']};
assert.equal(C.selectPair(motion,[nominal,augmented,baseline]).right.id,'augmented');
assert.equal(C.selectPair(motion,[nominal,augmented,baseline]).left.id,'nominal');
const parent=episode('parent',{case:'visual'});
const child=episode('child',{case:'visual',kind:'visual',profile:'camera',parent:'parent'});
const visual={id:'visual',source:'ur5',backend:'default',tags:['visual','rerender']};
assert.equal(C.selectPair(visual,[parent,child,baseline]).right.id,'child');
assert.equal(C.selectPair(visual,[parent,child,baseline]).left.id,'parent');
assert.equal(C.referenceFor({source:'ur5',backend:'newton',tags:[]},episode('x'),[baseline,episode('newton',{case:'newton-baseline'})]).id,'newton');
const baseMass=episode('mass-base',{case:'ur5-baseline',metadata:{physics_parameters:{'cube.mass':1,'cube.com_pose':[0,0,0,0,0,0,1]}}});
const massA=episode('near',{case:'mass',metadata:{physics_parameters:{'cube.mass':1.001}}});
const massB=episode('far',{case:'mass',metadata:{physics_parameters:{'cube.mass':.98}}});
const massCase={id:'mass',source:'ur5',backend:'default',tags:['physics','mass']};
assert.equal(C.selectPair(massCase,[baseMass,massA,massB]).right.id,'far');
const changes=C.parameterChanges(massCase,baseMass,massB);
assert.ok(Math.abs(changes[0].percent+2)<1e-10);
const com=episode('com',{metadata:{physics_parameters:{'cube.com_pose':[.001,0,0,0,0,0,1]}}});
assert.equal(C.parameterChanges({tags:['com']},baseMass,com)[0].delta,1);
assert.equal(C.parameterChanges({tags:['com']},baseMass,com)[0].unit,'mm');
function replay(times,qpos) {return {times,qpos,tcp:qpos.map(q=>[q[1],0,0]),joint_names:['still','changed'],joint_types:['revolute','revolute'],action_count:times.length-1,segments:[{name:'move',start_step:0,end_step:times.length-1}]};}
const left=replay([0,1,2],[[0,0],[0,1],[0,2]]);
const tiny=replay([0,1,2],[[0,0],[0,1.00001],[0,2]]);
const result=C.analyze(left,tiny,'time');
assert.equal(result.joint,1);
assert.ok(Math.abs(result.peak-.00001)<1e-12);
assert.ok(Math.abs(result.maxTcpMm-.01)<1e-9);
const slower=replay([0,2,4],[[0,0],[0,1],[0,2]]);
assert.equal(C.analyze(left,slower,'phase').peak,0);
assert.equal(C.analyze(left,slower,'time').peak,1);
assert.equal(C.analyze(left,slower,'time').times.at(-1),2); // no fabricated held tail
console.log('Comparison selection, provenance, units and alignment checks passed.');

const mixedA={...left,joint_types:['revolute','prismatic'],qpos:[[0,0],[0,0],[0,0]]};
const mixedB={...left,joint_types:['revolute','prismatic'],qpos:[[0,0],[.01,.1],[0,0]]};
const mixed=C.analyze(mixedA,mixedB,'time');
assert.equal(mixed.joint,0);assert.equal(mixed.peakUnit,'rad');assert.equal(mixed.peak,.01);assert.equal(mixed.linearPeakMm,100);
assert.equal(C.jointUnit(mixedB,1).label,'mm');
assert.equal(C.jointUnit({...mixedB,joint_types:[]},1).key,null);
