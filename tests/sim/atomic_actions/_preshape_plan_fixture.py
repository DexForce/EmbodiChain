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

"""Complete Slide recovery inputs using actual sampled grasp rows and URDF hand FK."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import json, hashlib, random
import numpy as np
import torch
from preshape_hand_support import preshape_fixture
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    JointPositionCommand,
    Slide,
    SlideOptions,
    SlideGoal,
    SlideAffordance,
    ObjectSemantics,
    RecoveryPolicy,
)
from embodichain.lab.sim.atomic_actions.articulation_continuation import (
    SlidePlanningRecovery,
)
from embodichain.lab.sim.utility.preshape_geometry import (
    capture_preshape_hand_clearance,
)
from ._articulation_plan_fixture import articulation_plan_fixture, command_digest
from ._ordinary_pick_fixture import OrdinaryCandidates


def preshape_plan_fixture(directory, mode, *, source_wrapper=None):
    _, generator, context, original, calls = articulation_plan_fixture(
        directory, "exhausted" if mode == "rebuild_failed" else "nominal"
    )
    captured = json.loads(
        (Path(__file__).parents[2] / "toolkits/data/preshape_grasp_b0.json").read_text()
    )
    case = captured["cases"][0]
    frozen_pose = torch.tensor(captured["pose"])
    poses, widths, costs = (torch.tensor(value) for value in case["raw_candidates"][0])
    local = torch.linalg.inv(frozen_pose[0]) @ poses
    raw_calls = []

    class Candidates(OrdinaryCandidates):
        def get_grasp_candidates(self, **kwargs):
            raw_calls.append(
                {
                    "approach": kwargs["approach_direction"].tolist(),
                    "pose": kwargs["obj_poses"].tolist(),
                    "draws": [
                        random.random(),
                        float(np.random.rand()),
                        float(torch.rand(())),
                    ],
                }
            )
            return [
                (pose @ local, widths.clone(), costs.clone())
                for pose in kwargs["obj_poses"]
            ]

    sampler = Candidates()
    hand, cfg = preshape_fixture(directory)
    if mode == "aperture_exhausted":
        cfg.open_qpos = (0.04, -0.04)
    if mode in ("nominal", "table_only"):
        cfg.open_qpos = (0.1, -0.1)
    table_z = 0.3 if mode in ("table_rejected", "table_only") else 0.0
    measurements = []

    def capture(fraction):
        return capture_preshape_hand_clearance(
            hand,
            cfg=cfg,
            expected_qpos=hand.qpos,
            table_z=table_z,
            preshape_fraction=fraction,
        )

    nominal = capture(0.85)

    class Provider:
        robot = generator.robot

        def nominal(self, request, current):
            return nominal

        def measure(self, request, current, fraction):
            measurements.append(fraction)
            return capture(fraction)

    provider = Provider()
    if source_wrapper is not None:
        sampler = source_wrapper(
            sampler, nominal, lambda fraction: provider.measure(None, None, fraction)
        )
    engine = AtomicActionEngine(
        generator,
        load_builtins=False,
        control_profiles={
            "hand": ControlPartCommandProfile(
                commands={
                    "open": JointPositionCommand(torch.tensor(cfg.open_qpos)),
                    "grasp": JointPositionCommand(torch.tensor(cfg.closed_qpos)),
                }
            )
        },
        grasp_pose_generators={"hand": sampler},
        slide_hand_clearance_provider=provider if source_wrapper is None else None,
    )
    engine.register(Slide())
    affordance = SlideAffordance(
        mesh_vertices=torch.tensor(captured["vertices"]),
        mesh_triangles=torch.tensor(captured["triangles"]),
        translation_axis=torch.tensor([0.0, -1.0, 0.0]),
        joint_name="rail",
        joint_limits=(0.0, 0.2),
    )
    goal = replace(
        original.goal, semantics=ObjectSemantics(affordance, {}, entity_id="handle")
    )
    invocation = engine.make_invocation(
        "slide",
        goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=original.motion_policy,
        skill_options=replace(
            original.skill_options, qualify_hand_clearance=source_wrapper is None
        ),
        recovery_policy=RecoveryPolicy(
            max_replans=0,
            max_action_retries=0,
            planning_recovery=SlidePlanningRecovery(
                preshape_aperture_recovery=source_wrapper is None
            ),
        ),
        invocation_id="source-slide",
    )
    return SimpleNamespace(
        engine=engine,
        generator=generator,
        context=context,
        invocation=invocation,
        calls=calls,
        raw_calls=raw_calls,
        measurements=measurements,
        sampler=sampler,
        hand=hand,
    )


def preshape_plan_record(fixture, plan):
    def digest(value):
        return hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    trajectory = plan.joint_trajectory
    return {
        "success": plan.plan_success.tolist(),
        "commands": command_digest(plan),
        "positions": (
            None if trajectory is None else digest(trajectory.positions.tolist())
        ),
        "dt": None if trajectory is None else digest(trajectory.dt.tolist()),
        "segments": [[value.name, value.start, value.stop] for value in plan.segments],
        "metadata": dict(plan.diagnostics.metadata),
        "message": list(plan.diagnostics.messages),
        "failure": (
            None if plan.diagnostics.failure is None else plan.diagnostics.failure.code
        ),
        "raw_candidates": fixture.raw_calls,
        "measurements": fixture.measurements,
        "motion_calls": [
            {
                **{key: value for key, value in call.items() if key != "targets"},
                "targets_sha256": digest(call["targets"]),
            }
            for call in fixture.calls
        ],
        "next_random": [
            random.random(),
            float(np.random.rand()),
            float(torch.rand(())),
        ],
    }
