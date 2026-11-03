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

"""One isolated native process per operation; all outcomes remain on disk."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from ._cases import dump, ProcessMetrics, ensure_prepared

__all__: list[str] = []


def install_host_observer() -> None:
    """Observe public host boundaries without owning another scheduling loop."""
    import embodichain.lab.expansion.host as native

    original = native.GymExpansionHost

    class ObservedHost(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            try:
                catalogue = {
                    "schema_version": 1,
                    "sampled_only": True,
                    "physical_episode_credit": 0,
                    "recipes": [
                        {"recipe_id": recipe.recipe_id, **recipe.to_dict()}
                        for recipe in (self.recipe(i) for i in range(33))
                    ],
                }
                catalogue = json.loads(json.dumps(catalogue, allow_nan=False))
                path = self.root / "recipe-catalog.json"
                if path.exists() and json.loads(path.read_text()) != catalogue:
                    raise ValueError("logical recipe catalogue changed during recovery")
                dump(path, catalogue)
            except BaseException:
                self.close()
                raise

        def prepare_batch(self, requests, session):
            items = super().prepare_batch(requests, session)
            self.witness_path = (
                self.root
                / "cohorts"
                / (self.journal.run_id + "-" + str(requests[0].sequence) + ".json")
            )
            self.witness = {
                "capacity": self.capacity,
                "state": "prepared",
                "recipes": [r.recipe.recipe_id for r in requests],
                "prepared_rows": [item.env_id for item in items],
            }
            dump(self.witness_path, self.witness)
            return items

        def execute_batch(self, assignments, *, on_started):
            started = []
            slots = {a.item.spec.identity: a.env_id for a in assignments}

            def observed(identity):
                on_started(identity)
                started.append(slots[identity])

            result = super().execute_batch(assignments, on_started=observed)
            self.witness.update(
                state="executed",
                started_rows=started,
                env_ids=[a.env_id for a in assignments],
                lengths=[
                    (
                        len(e.actions)
                        if e is not None
                        else self.prepared[a.env_id][4].get("steps", 0)
                    )
                    for a, e in zip(assignments, result)
                ],
                episode_ids=[e.episode_id if e is not None else None for e in result],
            )
            dump(self.witness_path, self.witness)
            return result

    native.GymExpansionHost = ObservedHost


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "operation",
        choices=(
            "prepare",
            "atlas",
            "collect",
            "replay",
            "audit",
            "probe",
            "faults",
            "mixed",
        ),
    )
    p.add_argument("root", type=Path)
    p.add_argument("--job", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--backend", default="default")
    p.add_argument("--fault", choices=("after_commit", "render"))
    args = p.parse_args()
    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    metrics = ProcessMetrics(
        args.root, args.operation + "-" + (args.job.stem if args.job else args.backend)
    )
    result = {"operation": args.operation, "ok": False}
    code = 0
    try:
        if args.operation == "prepare":
            from embodichain.data import get_data_path
            from examples.expansion.trajectory.real2sim_carton.prepare import (
                prepare,
                acquire,
            )

            destination = args.root / "inputs/carton"
            from embodichain.lab.expansion.configured import (
                preflight_expansion_job,
                ExpansionRunCfg,
            )

            def validate():
                acquire(Path(get_data_path("Real2SimCarton/raw")))
                qualification = json.loads(
                    (destination / "qualification.json").read_text()
                )
                if not qualification.get("admitted"):
                    raise ValueError("prepared source did not qualify")
                preflight_expansion_job(
                    ExpansionRunCfg(
                        deployment=str(destination / "task.measured.yaml"),
                        source_profile=str(
                            destination / "profile.measured.single.yaml"
                        ),
                        output_dir=str(args.root / "qualification"),
                    )
                )
                return qualification

            ensure_prepared(
                destination,
                lambda: prepare(Path(get_data_path("Real2SimCarton/raw")), destination),
                validate,
            )
            result["ok"] = True
        elif args.operation == "atlas":
            from embodichain.lab.task_program.integrations.feasibility import (
                run_feasibility,
            )
            from ._cases import ENGINE

            source = (
                args.job or ENGINE.parent / "scene_robot_feasibility/search.ur5.yaml"
            )
            destination = (
                args.root
                / "inputs"
                / (
                    "atlas-" + args.job.stem.removeprefix("search.")
                    if args.job
                    else "atlas"
                )
            )
            if not (destination / "atlas.json").exists():
                run_feasibility(source, destination)
            from embodichain.lab.expansion.scene import JointSceneSource

            atlas = JointSceneSource(destination / "atlas.json")
            result.update(ok=True, candidates=len(atlas.candidates))
        elif args.operation == "collect":
            from embodichain.lab.expansion import ExpansionEngine, load_expansion_job

            if args.fault:
                from ._diagnostics import install_fault

                install_fault(args.fault, args.root)
            install_host_observer()
            engine = ExpansionEngine.from_job(load_expansion_job(args.job))
            result["preflight"] = {
                k: v
                for k, v in engine.preflight().items()
                if k != "resolved_deployment"
            }
            result["result"] = engine.resume() if args.resume else engine.run()
            result["ok"] = bool(result["result"]["complete"])
            code = 0 if result["ok"] else 2
        elif args.operation == "replay":
            from ._replay import export_replays

            result.update(export_replays(args.root, args.job))
        elif args.operation == "audit":
            from ._report import audit

            result.update(audit(args.root))
        elif args.operation == "probe":
            from ._diagnostics import physics_probe

            result.update(physics_probe(args.backend))
        elif args.operation == "mixed":
            from ._diagnostics import mixed_row_probe

            result.update(mixed_row_probe(args.root, args.job))
        elif args.operation == "faults":
            from ._diagnostics import fault_checks

            result.update(fault_checks(args.root))
    except BaseException as e:
        traceback.print_exc()
        result.update(error=str(e), exception=type(e).__name__)
        code = 1
    target = (
        args.root
        / "results"
        / (
            (
                ("replay." if args.operation == "replay" else "")
                + args.job.stem
                + ".json"
            )
            if args.job
            else args.operation
            + ("-" + args.backend if args.operation == "probe" else "")
            + ".json"
        )
    )
    result["telemetry"] = metrics.close()
    dump(target, result)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
