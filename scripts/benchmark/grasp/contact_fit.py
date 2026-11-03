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

"""Compare sampled contact fitting with the pinned B0-F1 implementation.

Run from a source checkout with the reference commit available. Only the three
SHA-verified reference definitions are executed; no Task Engine is imported.
This measures CPU tensor fitting with an analytic box oracle, not native robot
execution, Open3D queries, transfer latency or GPU performance.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import torch
from embodichain.toolkits.graspkit.pg_grasp.contact_fit import (
    SampledHandGeometry,
    HandContactChecker,
)

__all__ = ["run_benchmark"]
REFERENCE = "05da44bd8f9c0fdace1a6c793d73d927daf10e72"
REFERENCE_SHA256 = "0d35ccbcf2800e22f3d3d4459b4e43917d500357989cd1b39edb06bb7c6d0be4"


class _Box:
    def query_batch_points(
        self, points: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = points.abs() - points.new_tensor([0.03, 0.1, 0.1])
        distance = torch.linalg.vector_norm(delta.clamp_min(0), dim=-1)
        distance += delta.amax(-1).clamp_max(0)
        return distance <= 0, distance


def _reference() -> ModuleType:
    root = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(
        [
            "git",
            "show",
            REFERENCE
            + ":embodichain/gen_sim/task_engine/_task_program/coordinated_grasp.py",
        ],
        cwd=root,
        text=True,
    )
    if hashlib.sha256(source.encode()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Unexpected reference source; refusing to execute it.")
    names = {"HandGeometry", "_inward_faces", "HandCollisionChecker"}
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    module = ModuleType("_pinned_contact_reference")
    sys.modules[module.__name__] = module
    module.__dict__.update(
        torch=torch,
        dataclass=dataclass,
        Any=Any,
        _CONTACT_TOLERANCE=0.0005,
        _CONTACT_NORMAL_COSINE=0.5,
        logger=SimpleNamespace(log_info=lambda *args: None),
    )
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), "pinned_contact_reference", "exec"
        ),
        module.__dict__,
    )
    return module


def _samples() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gaps = torch.tensor([0.12, 0.08, 0.04, 0.01])
    pads = []
    for gap in gaps:
        pads.append(
            torch.stack(
                [
                    torch.tensor(
                        [
                            [sign * (float(gap) / 2 + depth), y, z]
                            for depth, y, z in product(
                                (0.0, 0.005), (-0.01, 0.01), (-0.02, 0.02)
                            )
                        ]
                    )
                    for sign in (-1, 1)
                ]
            )
        )
    body = torch.tensor([[[0.0, 0.0, 0.5]]]).expand(4, 4096, -1).clone()
    return body, torch.stack(pads), gaps


def run_benchmark(*, repeats: int = 30, warmup: int = 5) -> dict[str, object]:
    """Measure paired CPU fit calls using identical inputs and source settings.

    Args:
        repeats: Timed repetitions per implementation and candidate population.
        warmup: Untimed calls per implementation/shape.

    Returns:
        Hardware/software/scope metadata, exact output checks and median latency.
    """
    if repeats < 1 or warmup < 1:
        raise ValueError("repeats and warmup must be positive")
    reference = _reference()
    body, pads, gaps = _samples()
    old = reference.HandCollisionChecker(
        reference.HandGeometry(body, pads, gaps), _Box()
    )
    new = HandContactChecker(
        SampledHandGeometry(body, pads, gaps),
        _Box(),
        contact_tolerance=0.0005,
        minimum_normal_cosine=0.5,
    )
    rows = []
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            for count in (1, 32, 256):
                poses = torch.eye(4).repeat(count, 1, 1)
                poses[:, 0, 3] = 0.005
                widths = torch.full((count,), 0.04)
                object_pose = torch.eye(4)
                operations = [
                    lambda checker=checker: checker.fit_contacts(
                        object_pose, poses, widths
                    )
                    for checker in (old, new)
                ]
                expected, actual = operations[0](), operations[1]()
                assert all(torch.equal(a, b) for a, b in zip(expected, actual))
                for op in operations:
                    for _ in range(warmup):
                        op()
                times: list[list[float]] = [[], []]
                for i in range(repeats):
                    for index in ((0, 1) if i % 2 == 0 else (1, 0)):
                        start = time.perf_counter()
                        result = operations[index]()
                        times[index].append((time.perf_counter() - start) * 1000)
                        del result
                medians = [statistics.median(values) for values in times]
                rows.append(
                    dict(
                        candidates=count,
                        body_points=4096,
                        pad_points=8,
                        exact_output=True,
                        source_ms=medians[0],
                        shared_ms=medians[1],
                        ratio=medians[1] / medians[0],
                        samples_ms=times,
                    )
                )
    finally:
        torch.set_num_threads(prior_threads)
    return dict(
        reference=REFERENCE,
        reference_sha256=REFERENCE_SHA256,
        platform=platform.platform(),
        processor=platform.processor(),
        torch=torch.__version__,
        device="cpu",
        dtype="float32",
        threads=1,
        grad_enabled=False,
        scope="completed CPU fit calls; analytic query; geometry construction and imports excluded",
        warmup=warmup,
        repeats=repeats,
        rows=rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            [
                {k: v for k, v in row.items() if k != "samples_ms"}
                for row in result["rows"]
            ],
            indent=2,
        )
    )
