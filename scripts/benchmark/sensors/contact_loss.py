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

"""Measure contact-loss correctness and cost against trusted local Git revisions.

Run with ``python -m scripts.benchmark.sensors.contact_loss --device cuda:0
--output /tmp/contact-loss.json``. Compares original main, the initial loss fix,
and the working tree. Real sensor/history/Warp and evidence acquisition execute
against fixed SDK buffers; native physics, rendering and policy inference are
excluded. Rotating paired trials report all raw timings and exact output checks.
Reference classes execute trusted Git source; unchanged original kernel/history
implementations are verified before comparing the added fused entry point.
"""

from __future__ import annotations
import ast
import subprocess
import json
import time
import statistics
import gc
import hashlib
import platform
from pathlib import Path
import torch
import warp as wp
from scripts.benchmark.sensors.contact_default import (
    reference_class,
    build,
    synchronize,
)
from embodichain.lab.sim.sensors.contact import sampling as sampling_module
from embodichain.lab.sim.evidence import PhysicalEvidenceRequest

from embodichain.lab.sim.sensors.contact import ContactSensor


def sampler(ref: str) -> type:
    """Load only the acquisition class from a trusted local reference."""
    source = subprocess.check_output(
        ["git", "show", ref + ":embodichain/lab/sim/sensors/contact/sampling.py"],
        text=True,
    )
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "ContactEvidenceSampler"
    )
    ns = dict(vars(sampling_module))
    ns["__name__"] = "bench_sampler"
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "reference_sampler", "exec"), ns)
    return ns["ContactEvidenceSampler"]


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-ref", default="7cb0c97c24598769d5efbfd226e0617f433f7caf"
    )
    parser.add_argument(
        "--initial-fix-ref", default="db7515d3b8174415e6831d23bf0837144db3122f"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 64, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    if min(*args.envs, args.warmup, args.repeats, args.steps) < 1:
        parser.error("Sizes must be positive.")
    device = torch.device(args.device)
    out = args.output
    torch.set_num_threads(1)
    torch.manual_seed(0)
    wp.init()
    BaseSensor, base_revision, base_hash = reference_class(args.reference_ref)
    FixedSensor, fix_revision, fix_hash = reference_class(args.initial_fix_ref)
    warmup, repeats, steps = args.warmup, args.repeats, args.steps
    rows = []
    for envs in args.envs:
        cases = [
            build(cls, envs, 64, device)
            for cls in (BaseSensor, FixedSensor, ContactSensor)
        ]
        samplers = [
            cls(case.sensor)
            for cls, case in zip(
                (
                    sampler(args.reference_ref),
                    sampler(args.initial_fix_ref),
                    sampling_module.ContactEvidenceSampler,
                ),
                cases,
            )
        ]
        requests = [PhysicalEvidenceRequest(1.0, 0, torch.arange(envs)) for _ in cases]
        funcs = {
            "control_step_4_substeps": [c.control_step for c in cases],
            "loss_read": [
                lambda: cases[0].sensor.dropped_contacts,
                lambda: max(
                    cases[1].sensor.dropped_contacts,
                    cases[1].sensor.latest_sample_dropped_contacts,
                ),
                lambda: cases[2].sensor.evidence_dropped_contacts,
            ],
            "evidence_refresh": [],
        }
        for c, s, r in zip(cases, samplers, requests):

            def refresh(c=c, s=s, r=r):
                c.owner.physics_step_index += 1
                return s.sample(r)

            funcs["evidence_refresh"].append(refresh)
        for scope, functions in funcs.items():
            for _ in range(warmup):
                for fn in functions:
                    fn()
            samples = [[], [], []]
            gc.disable()
            for trial in range(repeats):
                for i in [(trial + j) % 3 for j in range(3)]:
                    synchronize(device)
                    start = time.perf_counter()
                    for _ in range(steps):
                        functions[i]()
                    synchronize(device)
                    samples[i].append((time.perf_counter() - start) * 1e6 / steps)
            gc.enable()
            medians = [statistics.median(x) for x in samples]
            rows.append(
                dict(
                    envs=envs,
                    scope=scope,
                    median_us=dict(zip(("main", "initial_fix", "optimized"), medians)),
                    samples_us=samples,
                )
            )
            print(rows[-1]["envs"], scope, rows[-1]["median_us"], flush=True)
        results = [fn() for fn in funcs["evidence_refresh"]]
        for name, batch in results[0].frame.batches.items():
            for r in results[1:]:
                torch.testing.assert_close(
                    batch.values, r.frame.batches[name].values, atol=0, rtol=0
                )
                assert torch.equal(batch.valid, r.frame.batches[name].valid)
        for c in cases:
            c.sensor.begin_control_step()
            c.source.buffer.dropped_count = 3
            c.source.buffer.device_counts[1] = 3
        invalid = [fn() for fn in funcs["evidence_refresh"]]
        assert all(b.valid.all() for b in invalid[0].frame.batches.values())
        for r in invalid[1:]:
            assert all(not b.valid.any() for b in r.frame.batches.values())
        for c in cases[1:]:
            c.sensor.reset([0])
            assert c.sensor.latest_sample_dropped_contacts == 3
            c.sensor.reset()
            assert c.sensor.latest_sample_dropped_contacts == 0
        del cases, samplers
        gc.collect()
    report = dict(
        scope="Real sensor/history/Warp and evidence sampler, fixed SDK buffers; excludes native physics/rendering",
        base=base_revision,
        initial_fix=fix_revision,
        reference_sensor_hashes=[base_hash, fix_hash],
        candidate_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        candidate_files_sha256={
            str(p.resolve().relative_to(Path.cwd())): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in [
                Path(sampling_module.__file__),
                Path(
                    sampling_module.ContactSensor.__module__.replace(".", "/") + ".py"
                ),
                Path("embodichain/lab/sim/sensors/contact/_warp/scatter.py"),
            ]
        },
        visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cpu=next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        torch=torch.__version__,
        warp=wp.__version__,
        python=platform.python_version(),
        threads=1,
        contacts_per_env=4,
        capacity=64,
        warmup=warmup,
        repeats=repeats,
        steps=steps,
        sync="Warp+Torch device synchronization at window boundaries",
        exact_zero_loss_outputs=True,
        overflow_rejection_verified=True,
        rows=rows,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
