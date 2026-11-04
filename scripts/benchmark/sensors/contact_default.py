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

"""Compare observer-disabled contact sensing with a pinned local main revision.

Run with ``--reference-ref origin/main --device cuda:0 --envs 1 256 2048``.
The SDK-shaped source supplies fixed contact buffers; real sensor/history code,
Warp kernels, internal CUDA graphs and stream ordering execute. Native physics,
rendering and policy inference are excluded, so this is not a training FPS test.
Only trusted local Git revisions may be used as executable references.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
from types import SimpleNamespace

import torch
import warp as wp
from dexsim.scene import ContactActorInfo, ContactBuffer, ContactQueryCapabilities
from embodichain.lab.sim.sensors.contact import (
    ArticulationContactFilterCfg,
    ContactSensor,
    ContactSensorCfg,
    ContactHistory,
)
from embodichain.lab.sim.sensors.contact import sensor as sensor_module


class ContactSource:
    """Fixed four-foot contact fixture with native SDK buffers and actor metadata."""

    def __init__(self, environments: int, capacity: int, device: torch.device) -> None:
        self.fetches = 0
        self.capabilities = ContactQueryCapabilities(True, True, True)
        self.selected_actor_ids = tuple(range(1, environments * 4 + 1))
        self.actors = tuple(
            ContactActorInfo(
                row * 4 + foot + 1,
                f"arena_{row}/robot",
                f"foot_{foot}",
                f"arena_{row}",
                row,
            )
            for row in range(environments)
            for foot in range(4)
        )
        self.by_id = {actor.actor_id: actor for actor in self.actors}
        self.buffer = ContactBuffer.allocate(environments * capacity, str(device))
        count = environments * 4
        self.buffer.count = count
        self.buffer.data.zero_()
        self.buffer.data[:count, 5] = 1.0
        self.buffer.data[:count, 9] = 0.2
        self.buffer.actor_ids[:count, 0] = torch.arange(
            1, count + 1, device=device, dtype=torch.int32
        )
        self.buffer.actor_ids[:count, 1] = -1
        self.buffer.env_ids[:count] = torch.arange(
            environments, device=device, dtype=torch.int32
        ).repeat_interleave(4)
        self.buffer.device_counts.zero_()
        self.buffer.device_counts[0] = count

    def actor_info(self, actor_id: int) -> ContactActorInfo:
        return self.by_id[actor_id]

    def fetch(self) -> ContactBuffer:
        self.fetches += 1
        return self.buffer

    def fetch_async(self) -> ContactBuffer:
        self.fetches += 1
        return self.buffer


@dataclass
class Case:
    sensor: ContactSensor
    history: ContactHistory
    owner: SimpleNamespace
    source: ContactSource

    def control_step(self) -> None:
        # Match the host's selection and four-substep acquisition path.
        assert self.sensor.requires_substep_update
        self.sensor.begin_control_step()
        for _ in range(4):
            self.owner.physics_step_index += 1
            self.sensor.update_physics_step(0.0025)
        self.sensor.get_data()


def reference_class(ref: str) -> tuple[type[ContactSensor], str, str]:
    revision = subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
    for name in ("contact/sensor.py", "contact_sensor.py"):
        path = "embodichain/lab/sim/sensors/" + name
        result = subprocess.run(
            ["git", "show", revision + ":" + path], capture_output=True, text=True
        )
        if result.returncode == 0:
            nodes = ast.parse(result.stdout).body
            cls = next(
                (
                    n
                    for n in nodes
                    if isinstance(n, ast.ClassDef) and n.name == "ContactSensor"
                ),
                None,
            )
            if cls is not None:
                break
    else:
        raise ValueError("Reference revision has no ContactSensor implementation.")
    # Do not silently benchmark a main sensor against changed candidate kernels.
    shared_files = (
        ("contact/history.py", "contact_history.py"),
        ("contact/_warp/history.py", "_warp/contact_history.py"),
        ("contact/_warp/scatter.py", "_warp/contact.py"),
        ("base_sensor.py", "base_sensor.py"),
    )
    sensor_root = Path(sensor_module.__file__).parent.parent
    for current_name, legacy_name in shared_files:
        for reference_name in dict.fromkeys((current_name, legacy_name)):
            shared = subprocess.run(
                [
                    "git",
                    "show",
                    revision + ":embodichain/lab/sim/sensors/" + reference_name,
                ],
                capture_output=True,
                text=True,
            )
            if shared.returncode == 0:
                break
        if shared.returncode:
            raise ValueError(f"Missing reference dependency: {current_name}")
        before = shared.stdout.replace(
            "from embodichain.lab.sim.sensors._warp.contact_history import",
            "from ._warp.history import",
        )
        previous_tree = ast.parse(before)
        current_tree = ast.parse((sensor_root / current_name).read_text())
        if current_name == "contact/_warp/scatter.py" and not any(
            isinstance(node, ast.FunctionDef)
            and node.name == "scatter_contact_rows_with_loss"
            for node in previous_tree.body
        ):
            # The candidate adds a fused entry point. The reference still calls
            # the original kernel, whose complete implementation must match.
            current_tree.body = [
                node
                for node in current_tree.body
                if not (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "scatter_contact_rows_with_loss"
                )
            ]
        if ast.dump(previous_tree, include_attributes=False) != ast.dump(
            current_tree,
            include_attributes=False,
        ):
            raise ValueError(f"Shared dependency differs from main: {current_name}")
    cls.name = "ReferenceContactSensor"
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    namespace = dict(vars(sensor_module))
    from embodichain.lab.sim.sensors.contact._warp.scatter import scatter_contact_rows

    namespace["scatter_contact_rows"] = scatter_contact_rows
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return (
        namespace["ReferenceContactSensor"],
        revision,
        hashlib.sha256(result.stdout.encode()).hexdigest(),
    )


def build(
    cls: type[ContactSensor], environments: int, capacity: int, device: torch.device
) -> Case:
    source = ContactSource(environments, capacity, device)
    owner = SimpleNamespace(
        num_envs=environments,
        physics_step_index=0,
        arena_offsets=torch.zeros(environments, 3, device=device),
        _spawn_scene=SimpleNamespace(
            handles=lambda uid: tuple(
                SimpleNamespace(
                    path=f"arena_{row}/robot",
                    get_link_names=lambda: [f"foot_{i}" for i in range(4)],
                )
                for row in range(environments)
            )
        ),
        spawn_result=SimpleNamespace(
            create_contact_query=lambda *args, **kwargs: source
        ),
    )
    sensor = cls(
        ContactSensorCfg(
            uid="contacts",
            articulation_cfg_list=[
                ArticulationContactFilterCfg(
                    articulation_uid="robot",
                    link_name_list=[f"foot_{i}" for i in range(4)],
                )
            ],
            max_contacts_per_env=capacity,
            filter_need_both_actor=False,
        ),
        device,
        owner=owner,
    )
    history = sensor.create_history(
        "feet",
        torch.arange(1, environments * 4 + 1, device=device).reshape(environments, 4),
        include_unknown_counterpart=True,
    )
    return Case(sensor, history, owner, source)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        wp.synchronize_device(str(device))
        torch.cuda.synchronize(device)


def measure(case: Case, device: torch.device, steps: int) -> float:
    synchronize(device)
    started = time.perf_counter()
    for _ in range(steps):
        case.control_step()
    synchronize(device)
    return (time.perf_counter() - started) * 1000 / steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ref", default="origin/main")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.capacity < 4
        or min(args.envs) < 1
        or min(args.warmup, args.repeats, args.steps) < 1
    ):
        parser.error("Positive sizes and capacity >= 4 are required.")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    wp.init()
    device = torch.device(args.device)
    reference, revision, source_hash = reference_class(args.reference_ref)
    rows = []
    for count in args.envs:
        before, after = build(reference, count, args.capacity, device), build(
            ContactSensor, count, args.capacity, device
        )
        for _ in range(args.warmup):
            before.control_step()
            after.control_step()
        samples = [[], []]
        enabled = gc.isenabled()
        gc.disable()
        try:
            for trial in range(args.repeats):
                for index in ((0, 1) if trial % 2 == 0 else (1, 0)):
                    samples[index].append(
                        measure((before, after)[index], device, args.steps)
                    )
        finally:
            if enabled:
                gc.enable()
        for field in (
            "contact",
            "found",
            "first_contact",
            "contact_count",
            "force",
            "peak_force",
            "current_air_time",
            "last_air_time",
        ):
            torch.testing.assert_close(
                getattr(before.history, field),
                getattr(after.history, field),
                rtol=0,
                atol=0,
            )
        for field, value in before.sensor.get_data().items():
            torch.testing.assert_close(
                value, after.sensor.get_data()[field], rtol=0, atol=0
            )
        expected_fetches = (args.warmup + args.repeats * args.steps) * 4
        assert before.source.fetches == after.source.fetches == expected_fetches
        medians = [statistics.median(values) for values in samples]
        rows.append(
            {
                "environments": count,
                "main_ms_per_control_step": medians[0],
                "candidate_ms_per_control_step": medians[1],
                "candidate_over_main": medians[1] / medians[0],
                "main_samples_ms": samples[0],
                "candidate_samples_ms": samples[1],
                "exact_outputs": True,
                "native_fetches_per_implementation": expected_fetches,
            }
        )
        del before, after
        gc.collect()
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    report = {
        "scope": "Real ContactSensor + ContactHistory and Warp kernels over fixed SDK buffers; no physics/rendering/policy inference",
        "reference_commit": revision,
        "reference_source_sha256": source_hash,
        "candidate_source_sha256": hashlib.sha256(
            Path(sensor_module.__file__).read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "warp": wp.__version__,
        "cpu": cpu,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": "float32",
        "torch_threads": 1,
        "seed": 0,
        "contacts_per_env": 4,
        "capacity_per_env": args.capacity,
        "substeps": 4,
        "warmup_control_steps": args.warmup,
        "repeats": args.repeats,
        "control_steps_per_repeat": args.steps,
        "synchronization": (
            "owning Warp device and Torch CUDA device before/after each repeated window"
            if device.type == "cuda"
            else "synchronous CPU"
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": [
                    {k: v for k, v in row.items() if not k.endswith("samples_ms")}
                    for row in rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
