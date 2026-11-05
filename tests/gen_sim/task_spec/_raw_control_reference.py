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


"""Explicit fixture regeneration from the pinned public physical producer.

Run in a separate process with ``--producer-root`` pointing to PR #1057's exact
checkout. The producer imports no normative/Task Engine code and constructs no
native SimulationManager/Robot. Numerical tests supply native-shaped inputs;
these packets do not qualify physical behavior.
"""

from __future__ import annotations

import ast, gzip, hashlib, importlib.abc, inspect, json, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import argparse

parser = argparse.ArgumentParser(
    description="Regenerate frozen packets from the exact public #1057 producer without evaluating TaskSpec."
)
parser.add_argument("--producer-root", type=Path, required=True)
parser.add_argument(
    "--output-directory", type=Path, default=Path(__file__).parent / "data"
)
args = parser.parse_args()
producer = args.producer_root.resolve()
head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=producer, text=True
).strip()
assert head == "0948a69f20e91d7442226cd41aee77c4c9b2f74f", head
sys.path.insert(0, str(producer))
sys.path.insert(0, str(producer / "tests/gym/envs/task_program"))


class NoNormative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "embodichain.gen_sim" or fullname.startswith(
            "embodichain.gen_sim."
        ):
            raise AssertionError(
                "Producer export must not evaluate TaskSpec or import Task Engine"
            )


sys.meta_path.insert(0, NoNormative())
from embodichain.lab.sim.sim_manager import SimulationManager
from embodichain.lab.sim.objects import Robot


def no_native(cls, *args, **kwargs):
    raise AssertionError("Native construction excluded")


SimulationManager.__new__ = no_native
Robot.__new__ = no_native
import torch
import test_simulation_policies as fixture
from embodichain.lab.task_program.integrations.invocation_policy import (
    task_program_fingerprint,
)
from embodichain.lab.gym.envs.episode_context import EpisodeContext
from embodichain.lab.gym.envs.task_program.bridge import (
    TaskProgramDemoBridge,
    BufferedGymCommandSink,
    RuntimeCommandFrameEncoder,
)
from embodichain.lab.task_program.integrations.simulation.control_observations import (
    SimulationControlObservationCollector,
)

cases = []


def packet(kind, collector, program, segments, selectors, raw, label):
    return {
        "case": label,
        "kind": kind,
        "program": program,
        "entity_id": collector.route_payload["binding"]["object_id"],
        "episode_ids": list(collector.episode_ids),
        "control_identity": collector.identity,
        "policy_occurrences": selectors,
        "operation": {
            "route_payload": collector.route_payload,
            "terminal_segment_id": selectors[-1]["segment_name"],
            "terminal_source_path": selectors[-1]["source_path"],
            "preset": selectors[-1]["preset"],
            "required_execution": [
                {
                    "segment_id": s["name"],
                    "source_path": s["metadata"]["program_segment_source_path"],
                }
                for s in segments
            ],
        },
        "evidence": {
            "segments": segments,
            "episode_ids": list(collector.episode_ids),
            "control_acquisition": raw,
            "boundary_frames": raw["boundary_frames"],
        },
    }


for empty in (False, True):
    system, collector, program, segments, selectors, raw = (
        fixture._raw_press_observation(empty_initial=empty)
    )
    cases.append(
        packet(
            "press",
            collector,
            program,
            segments,
            selectors,
            raw,
            "press_empty_initial" if empty else "press",
        )
    )
    system.port.close_runtime_resources()
# The published unit fixture's opaque episode literal is replaced only at its
# constructor input, matching the actual EpisodeContext row used by the native
# collector. No production implementation or returned observation is modified.
text = inspect.getsource(fixture._twist_control_case)
assert text.count('"episode"') == 2
text = text.replace('"episode"', '"0"')
exec(
    compile(text, "<published fixture with actual native episode input>", "exec"),
    fixture.__dict__,
)
program, compiled = fixture._full_control_program(("ready", "chunk", "park"))
captured = []


def initialize(system):
    runtime = SimpleNamespace(
        result=None,
        status=None,
        start=lambda *a, **k: None,
        step=lambda: None,
        cancel=lambda reason: None,
        adopt_verified_task_state=lambda state: None,
    )
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(
            SimpleNamespace(current_qpos=lambda ids: system.robot.get_qpos())
        ),
        system.clock,
    )
    bridge = TaskProgramDemoBridge(
        compiled,
        runtime,
        sink,
        system.clock,
        post_policy_port=system.port,
        isolate_policy_sessions=False,
    )
    owner = EpisodeContext(1)
    owner._begin_episode([0])
    owner._finish_initialization([0])
    env = SimpleNamespace(
        sim=system.port._simulation, robot=system.robot, episode_context=owner
    )
    captured.append(SimulationControlObservationCollector(env, bridge, "twist0"))


system = fixture._twist_control_case(
    program_fingerprint=task_program_fingerprint(compiled),
    compiled=compiled,
    initialize=initialize,
    retain=True,
)
collector = captured[0]
segments = []
selectors = []
for segment in compiled.iter_segments():
    policy = segment.post_policies[0]
    scope = system.port.control_policy_scope(policy, segment=segment)
    facts = system.port.control_policy_measurements(policy, segment=segment)
    selectors.append(
        {
            "segment_id": segment.segment_id,
            "segment_name": segment.name,
            "source_path": list(segment.source_path),
            "policy_index": 0,
            "preset": policy.cfg.preset,
            "phase": scope["phase"],
            "terminal": scope["terminal"],
        }
    )
    segments.append(
        {
            "name": segment.name,
            "active": [True],
            "successes": [True],
            "end_steps": [facts.hold_end_step],
            "metadata": {
                "task_program_id": compiled.program_id,
                "program_segment_id": segment.segment_id,
                "program_segment_source_path": list(segment.source_path),
                "runtime": {"env_ids": [0]},
                "post_policies": [
                    {
                        "kind": "wait_stable",
                        "policy_index": 0,
                        "source_path": list(policy.source_path),
                        "result_mask": [True],
                        "result": system.port.post_policy_metadata(
                            policy, segment=segment
                        ),
                    }
                ],
            },
        }
    )
system.operation.close()
system.art.get_qpos = lambda: (_ for _ in ()).throw(
    AssertionError("Retained facts cannot read live-later qpos")
)
raw = collector.capture(selectors, terminal_policy=selectors[-1])
cases.append(packet("twist", collector, program, segments, selectors, raw, "twist"))
system.port.close_runtime_resources()
encoded = json.dumps(
    {"cases": cases}, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode()
root = args.output_directory.resolve()
root.mkdir(parents=True, exist_ok=True)
(root / "raw_control_public1057.json.gz").write_bytes(gzip.compress(encoded, mtime=0))
provenance = {
    "producer_pr": 1057,
    "producer_head": head,
    "source": "actual public ControlOperationRuntime/SimulationSegmentPolicyPort/SimulationControlObservationCollector APIs; source numeric tests provide native-shaped inputs",
    "cases": [x["case"] for x in cases],
    "sha256_uncompressed": hashlib.sha256(encoded).hexdigest(),
    "uncompressed_bytes": len(encoded),
    "compressed_bytes": (root / "raw_control_public1057.json.gz").stat().st_size,
    "native_sim_robot_construction": False,
    "normative_or_task_engine_import": False,
    "physical_qualification": False,
    "twist_fixture_input": "opaque episode literal set to actual initialized native episode 0 before calling unchanged real producer; no returned facts modified",
}
(root / "raw_control_public1057_provenance.json").write_text(
    json.dumps(provenance, indent=2) + "\n"
)
print(json.dumps(provenance))
