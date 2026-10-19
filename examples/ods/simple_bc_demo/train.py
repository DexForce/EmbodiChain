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

"""Train and compare ChunkBC with live ODS or resident LeRobot data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[3]
    sys.path[:0] = [
        str(repository),
        str(repository / "extensions"),
        str(repository / "extensions/embodichain_tasks"),
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument("--train-device", default="cuda:0")
    p.add_argument("--output", type=Path, default=Path("/tmp/ods-pick-place-showcase"))
    p.add_argument(
        "--mode",
        choices=(
            "online",
            "frozen",
            "offline-32",
            "offline-full",
        ),
        default="online",
    )
    p.add_argument(
        "--policy",
        choices=("chunk-pose",),
        default="chunk-pose",
    )
    p.add_argument("--pose-ik", choices=("dls",), default="dls")
    p.add_argument("--prediction-horizon", type=int, default=64)
    p.add_argument("--action-horizon", type=int, default=16)
    p.add_argument("--observation-noise", type=float, default=0.01)
    p.add_argument(
        "--stage",
        choices=("showcase", "overfit-one", "overfit-four"),
        default="showcase",
    )
    p.add_argument(
        "--offline-data",
        type=Path,
        help="Showcase LeRobot dataset directory",
    )
    p.add_argument("--offline-storage", choices=("lerobot",), default="lerobot")
    p.add_argument("--offline-memory-limit-mib", type=int, default=512)
    p.add_argument("--offline-episodes", type=int, default=None)
    p.add_argument("--offline-sample-budget", type=int, default=None)
    p.add_argument("--offline-sample-hard-limit", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--qualification-scenes", type=int, default=40)
    p.add_argument(
        "--qualification-report",
        type=Path,
        help="Reuse a matching 40-scene qualification with source/config checks",
    )
    p.add_argument("--dev-scenes", type=int, default=20)
    p.add_argument("--final-test-scenes", type=int, default=100)
    p.add_argument("--wall-time-budget", type=float, default=None)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--updates-per-round", type=int, default=5000)
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--producer-seed", type=int, default=None)
    p.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Optional independent CPU sampling RNG; omitted uses the global stream",
    )
    p.add_argument("--producer-envs", type=int, default=None)
    p.add_argument("--dev-seed-base", "--dev-seed-start", type=int, default=20000)
    p.add_argument("--test-seed-base", type=int, default=None)
    p.add_argument("--max-generated-episodes", type=int, default=None)
    p.add_argument("--usage-statistics", action="store_true")
    p.add_argument(
        "--auto-config-input",
        type=Path,
        help="Two measured throughputs, reuse profile and runtime/binding metadata JSON",
    )
    p.add_argument(
        "--defer-evaluation",
        action="store_true",
        help="Save checkpoints while training; evaluate only after producer shutdown",
    )
    p.add_argument(
        "--deferred-evaluation-schedule",
        choices=("updates", "updates_and_generations"),
        default="updates_and_generations",
    )

    p.add_argument("--resume", type=Path)
    p.add_argument("--viser-port", type=int, default=8080)
    p.add_argument("--tensorboard-port", type=int, default=6006)
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--qualification-only", action="store_true")
    args = p.parse_args()
    if args.offline_memory_limit_mib < 1:
        p.error("--offline-memory-limit-mib must be positive")
    if args.auto_config_input is not None:
        from examples.ods.simple_bc_demo.autoconfiguration import (
            apply_configuration,
        )

        try:
            apply_configuration(args)
        except (TypeError, ValueError, OSError) as error:
            p.error(str(error))
    if args.batch_size is None:
        args.batch_size = 256
    if args.buffer_size is None:
        args.buffer_size = 32
    if args.producer_envs is None:
        args.producer_envs = 1
    if args.smoke:
        args.rounds, args.updates_per_round, args.buffer_size, args.batch_size = (
            2,
            5,
            2,
            4,
        )
        args.qualification_scenes, args.dev_scenes, args.final_test_scenes = 2, 2, 2
    if args.stage != "showcase":
        args.mode = "frozen"
        args.buffer_size = 1 if args.stage == "overfit-one" else 4
        args.rounds = 1
        args.final_test_scenes = 0
    if (
        args.eval_every < 1
        or args.qualification_scenes < 1
        or args.dev_scenes < 1
        or args.final_test_scenes < 0
    ):
        p.error("Evaluation counts must be positive (final test may be disabled)")
    if args.wall_time_budget is not None and (
        not np.isfinite(args.wall_time_budget) or args.wall_time_budget <= 0
    ):
        p.error("Wall-time budget must be finite and positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("Learning rate must be finite and positive")
    if not 1 <= args.action_horizon <= args.prediction_horizon:
        p.error("Action horizon must be within the positive prediction horizon")
    if not np.isfinite(args.observation_noise) or args.observation_noise < 0:
        p.error("Observation noise must be finite and nonnegative")
    if min(args.rounds, args.updates_per_round, args.buffer_size, args.batch_size) < 1:
        p.error("Training counts must be positive")
    if args.producer_seed is None:
        args.producer_seed = args.seed
    if args.test_seed_base is None:
        args.test_seed_base = 90000 if args.smoke else 50000
    if args.producer_envs < 1 or args.buffer_size % args.producer_envs:
        p.error("Producer environments must divide the positive cache capacity")
    cold_offline = (
        args.mode in {"offline-full", "offline-32"} and args.offline_data is None
    )
    if args.offline_episodes is not None:
        if (
            not cold_offline
            or args.offline_episodes < args.buffer_size
            or args.offline_episodes % args.buffer_size
        ):
            p.error(
                "Offline collection count must be a cache multiple in a cold offline mode"
            )
        if args.mode == "offline-32" and args.offline_episodes != args.buffer_size:
            p.error("offline-32 collects exactly the initial cache")
        if args.max_generated_episodes not in {None, args.offline_episodes}:
            p.error("Offline collection count conflicts with max-generated-episodes")
        args.max_generated_episodes = args.offline_episodes
    if (
        args.offline_sample_budget is not None
        or args.offline_sample_hard_limit is not None
    ):
        if args.resume is not None:
            p.error(
                "Protected offline resume requires saved quota state and is not supported"
            )
        if (
            args.offline_sample_budget is None
            or args.offline_sample_hard_limit is None
            or not 0 < args.offline_sample_budget <= args.offline_sample_hard_limit
            or not (cold_offline or args.offline_data is not None)
        ):
            p.error(
                "Offline quotas require positive budget <= hard limit and a resident offline corpus"
            )
    if args.max_generated_episodes is not None and (
        args.max_generated_episodes < args.buffer_size
        or args.max_generated_episodes % args.buffer_size
    ):
        p.error("Accepted episode budget must be a positive multiple of cache capacity")
    if args.defer_evaluation and args.stage != "showcase":
        p.error("Deferred evaluation is only supported for the showcase stage")
    dev = set(range(args.dev_seed_base, args.dev_seed_base + args.dev_scenes))
    test = set(range(args.test_seed_base, args.test_seed_base + args.final_test_scenes))
    if dev & test:
        p.error("Development and final-test seeds must be disjoint")
    return args


def main() -> None:
    from examples.ods.simple_bc_demo.trainer import run

    run(_parse_args())


if __name__ == "__main__":
    main()
