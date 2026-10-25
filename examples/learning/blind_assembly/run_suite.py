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

"""Prepare, run or evaluate the native blind-assembly reproduction matrix.

Run with ``embodichain run-example learning/blind_assembly/run_suite.py -- --help``.
Preparing a suite does not train models. Formal configs retain their exact-model
qualification gate; selecting ``--nominal`` explicitly requests development runs.
OOD axes require separate held-out geometry configs and remain pending otherwise.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from embodichain.utils.config_paths import resolve_config_path
from embodichain.utils.utility import load_config

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TASKS = ("peg_insertion", "gear_meshing", "nut_threading")
_VARIANTS = ("full", "no_history", "no_contact")


def _working_directory(suite_directory: Path) -> Path:
    """Use the source checkout or a writable suite directory after installation."""
    return _REPO_ROOT if (_REPO_ROOT / "pyproject.toml").is_file() else suite_directory


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def _prepare(args: argparse.Namespace, path: Path) -> dict:
    if path.exists():
        raise FileExistsError(
            f"Suite already exists: {path}; choose another --suite-dir."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    configuration_dir = path.parent / "configs"
    configuration_dir.mkdir()
    runs = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for task in args.tasks:
        for profile in ("study", "fullrange"):
            template_reference = (
                Path("extensions/embodichain_tasks/configs/tasks/manipulation/assembly")
                / task
                / "agents"
                / f"ppo.{profile}{'.nominal' if args.nominal else ''}.yaml"
            )
            source_template = _REPO_ROOT / template_reference
            template_path = (
                source_template
                if source_template.is_file()
                else resolve_config_path(template_reference).resolve()
            )
            template = load_config(template_path)
            for variant in args.variants:
                for seed in args.seeds:
                    config = deepcopy(template)
                    trainer = config["trainer"]
                    rollout_len = int(
                        trainer.get("buffer_size", trainer.get("rollout_steps", 32))
                    )
                    name = (
                        f"blind_assembly_{stamp}_{task}_{profile}_{variant}_seed{seed}"
                    )
                    trainer.update(
                        {
                            "exp_name": name,
                            "seed": seed,
                            "device": args.device,
                            "num_envs": args.num_envs,
                            "iterations": args.max_updates,
                            "enable_eval": True,
                            "eval_seed": seed + 10000,
                            "num_eval_episodes": args.validation_episodes,
                            "eval_freq": args.eval_every_updates
                            * args.num_envs
                            * rollout_len,
                            "save_frequency_updates": args.eval_every_updates,
                            "best_eval_metric": "eval/success_rate",
                            "best_eval_mode": "max",
                        }
                    )
                    source_config = Path(trainer["gym_config"])
                    repository_config = _REPO_ROOT / source_config
                    trainer["gym_config"] = str(
                        repository_config.resolve()
                        if not source_config.is_absolute()
                        and repository_config.exists()
                        else resolve_config_path(
                            source_config, base_dir=template_path.parent
                        ).resolve()
                    )
                    if variant == "no_history":
                        config["policy"].setdefault("cfg", {})["use_history"] = False
                    if variant == "no_contact":
                        config["policy"].setdefault("cfg", {})["contact_points"] = 0
                        config["algorithm"]["cfg"]["auxiliary_coef"] = 0.0
                    configuration = (
                        configuration_dir
                        / f"{task}_{profile}_{variant}_seed{seed}.yaml"
                    )
                    configuration.write_text(yaml.safe_dump(config, sort_keys=False))
                    runs.append(
                        {
                            "task": task,
                            "profile": profile,
                            "variant": variant,
                            "seed": seed,
                            "name": name,
                            "config": str(configuration.resolve()),
                            "template": str(template_path),
                            "status": "pending",
                            "run_directory": None,
                            "evaluation": {
                                "in_distribution": {
                                    "episodes": args.id_episodes,
                                    "seed": seed + 20000,
                                    "status": "pending",
                                },
                                "out_of_distribution": {
                                    axis: {
                                        "episodes": args.ood_episodes,
                                        "seed": seed + 30000,
                                        "status": "requires_held_out_config",
                                    }
                                    for axis in ("size", "aspect", "shape")
                                },
                            },
                        }
                    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "nominal_development" if args.nominal else "exact_model_required"
        ),
        "formal_environment_count": 12288,
        "same_configuration_reproduction": args.num_envs == 12288 and not args.nominal,
        "max_updates": args.max_updates,
        "runs": runs,
        "notes": [
            "Validation chooses best checkpoint; held-out testing uses that checkpoint.",
            "OOD axes need separately generated held-out geometry; train geometry is never substituted.",
            "Full experiment includes all profiles, seeds and variants; a prepared manifest is not a result.",
        ],
    }
    _write_manifest(path, manifest)
    return manifest


def _run_command(command: list[str], log: Path) -> None:
    environment = os.environ.copy()
    if (_REPO_ROOT / "pyproject.toml").is_file():
        # Editable entry points can otherwise import a different checkout's
        # extension modules even though the subprocess works in this repo.
        roots = [
            _REPO_ROOT / "extensions",
            _REPO_ROOT / "extensions/embodichain_tasks",
            _REPO_ROOT,
        ]
        inherited = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(root) for root in roots if root.is_dir()]
            + ([inherited] if inherited else [])
        )
    with log.open("w") as stream:
        subprocess.run(
            command,
            cwd=_working_directory(log.parent),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            env=environment,
        )


def _train(manifest: dict, path: Path) -> None:
    outputs = _working_directory(path.parent) / "outputs"
    for index, run in enumerate(manifest["runs"]):
        if run["status"] == "complete":
            continue
        run["status"] = "running"
        _write_manifest(path, manifest)
        try:
            previous_outputs = set(outputs.glob(f"{run['name']}_*"))
            _run_command(
                [
                    sys.executable,
                    "-m",
                    "embodichain",
                    "train-rl",
                    "--config",
                    run["config"],
                ],
                path.parent / f"train_{index}.log",
            )
            candidates = sorted(
                set(outputs.glob(f"{run['name']}_*")) - previous_outputs
            )
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Expected one native training output for {run['name']}."
                )
            run["run_directory"] = str(candidates[0].resolve())
            run["status"] = "complete"
        except Exception as exc:
            run["status"] = "failed"
            run["error"] = str(exc)
            raise
        finally:
            _write_manifest(path, manifest)


def _evaluate(manifest: dict, path: Path, args: argparse.Namespace) -> None:
    from embodichain.learning.rl.policy_evaluation.manifest import RunManifest

    for index, run in enumerate(manifest["runs"]):
        if run["status"] != "complete":
            raise RuntimeError(f"Train {run['name']} before evaluation.")
        evaluation = run["evaluation"]["in_distribution"]
        if evaluation["status"] == "complete":
            continue
        saved_run = RunManifest.load(run["run_directory"], check_files=False)
        best_checkpoint = saved_run.checkpoints.get("best")
        if best_checkpoint is None or not best_checkpoint.is_file():
            raise RuntimeError(
                f"No validation-selected best checkpoint exists for {run['name']}; "
                "the reproduction suite cannot substitute the latest checkpoint."
            )
        output = path.parent / f"evaluation_{index}"
        _run_command(
            [
                sys.executable,
                "-m",
                "embodichain",
                "eval-policy",
                run["run_directory"],
                "--checkpoint",
                str(best_checkpoint),
                "--episodes",
                str(evaluation["episodes"]),
                "--seed",
                str(evaluation["seed"]),
                "--num-envs",
                str(args.eval_envs),
                "--renderer",
                "no-render",
                "--output",
                str(output.resolve()),
            ],
            path.parent / f"evaluate_{index}.log",
        )
        evaluation["status"] = "complete"
        evaluation["output"] = str(output.resolve())
        _write_manifest(path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("prepare", "train", "evaluate"), default="prepare"
    )
    parser.add_argument("--suite-dir", default="outputs/assembly_suite")
    parser.add_argument("--nominal", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=_TASKS, default=list(_TASKS))
    parser.add_argument(
        "--variants", nargs="+", choices=_VARIANTS, default=list(_VARIANTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-envs", type=int, default=12288)
    parser.add_argument("--max-updates", type=int, default=5000)
    parser.add_argument("--eval-every-updates", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=256)
    parser.add_argument("--id-episodes", type=int, default=8192)
    parser.add_argument("--ood-episodes", type=int, default=2048)
    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (
        min(
            args.num_envs,
            args.max_updates,
            args.eval_every_updates,
            args.validation_episodes,
            args.id_episodes,
            args.ood_episodes,
            args.eval_envs,
        )
        <= 0
    ):
        parser.error("Counts and update periods must be positive.")
    path = Path(args.suite_dir).expanduser().resolve() / "manifest.json"
    if args.phase == "prepare":
        manifest = _prepare(args, path)
    else:
        manifest = json.loads(path.read_text())
        if args.phase == "train":
            _train(manifest, path)
        else:
            _evaluate(manifest, path, args)
    print(f"Suite manifest: {path} ({len(manifest['runs'])} runs)")


if __name__ == "__main__":
    main()
