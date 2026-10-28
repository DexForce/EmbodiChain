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

"""Prepare RigidFormer assets, run an engine demo, evaluate and export weights."""

from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys
from ._prepare import FOLDERS, download, prepare, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dataset", choices=tuple(FOLDERS), default="movi_b")
    p.add_argument("--cache", type=Path, default=Path("outputs/rigidformer/assets"))
    p.add_argument("--output", type=Path, default=Path("outputs/rigidformer/data"))
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--assets", choices=("data", "weights", "all"), default="data")
    p.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Debug subset only; zero prepares the full frozen corpus",
    )
    p = sub.add_parser("demo", help="Compare live Newton dynamics with RigidFormer")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/rigidformer/assets/model/rigidformer_movisphere.pth"),
    )
    p.add_argument(
        "--data-root", type=Path, default=Path("outputs/rigidformer/data/MOVi-Sphere")
    )
    p.add_argument("--output", type=Path, default=Path("outputs/rigidformer/demo"))
    p.add_argument("--scene", help="Prepared test scene; default selects the first")
    p.add_argument(
        "--object-ids", type=int, nargs="+", help="Two or three template object IDs"
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attention-backend", choices=("sdpa", "flash"), default="sdpa")
    p.add_argument(
        "--headless", action="store_true", help="Record without opening a window"
    )
    p.add_argument("--frames", type=int, default=51)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument(
        "--fps", type=int, default=12, help="Window playback and video frame rate"
    )
    p.add_argument(
        "--hold-seconds", type=float, default=3.0, help="Final window hold duration"
    )
    p = sub.add_parser("export")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", choices=("ema", "raw"), default="ema")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attention-backend", choices=("sdpa", "flash"), default="sdpa")
    p.add_argument("--step-sizes", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--num-frames", type=int, default=480)
    p = sub.add_parser("compare")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Pinned author checkout's RigidFormer directory",
    )
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--attention-backend", choices=("sdpa", "flash"), default="flash")
    p.add_argument("--updates", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("check-recovery")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stop-after", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        weights = None
        if args.local_files_only:
            raw = args.cache / "dataset"
            if args.assets in ("weights", "all"):
                from ._prepare import WEIGHTS, WEIGHT_SHA256

                weights = args.cache / "model" / WEIGHTS[args.dataset]
                if (
                    not weights.is_file()
                    or sha256(weights) != WEIGHT_SHA256[args.dataset]
                ):
                    raise ValueError(
                        "A verified official checkpoint is required in the local cache"
                    )
        else:
            raw, weights = download(args.cache, args.dataset, assets=args.assets)
        root = None
        if args.assets in ("data", "all"):
            root = prepare(
                raw, args.output, args.dataset, limit_per_split=args.limit_per_split
            )
        print(
            json.dumps(
                {
                    "data_root": str(root) if root else None,
                    "checkpoint": str(weights) if weights else None,
                    "scope": "full" if args.limit_per_split == 0 else "debug-subset",
                },
                indent=2,
            )
        )
    elif args.command == "demo":
        from ._demo import run_demo

        run_demo(args)
    elif args.command == "export":
        from embodichain_models.dynamics.rigidformer.checkpoint import (
            export_rigidformer_checkpoint,
        )

        print(
            export_rigidformer_checkpoint(
                args.checkpoint, args.output, weights=args.weights
            )
        )
    elif args.command == "check-recovery":
        from ._support._recovery import check_recovery

        check_recovery(args)
    elif args.command == "compare":
        from ._support._compare import compare

        compare(args)
    else:
        evaluate(args)


def evaluate(args: argparse.Namespace) -> None:
    from embodichain_models.dynamics.rigidformer import load_rigidformer_checkpoint
    from embodichain_models.dynamics.rigidformer.data import _verified_manifest
    from ._support._evidence import evidence
    from ._support._evaluator import Evaluator

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    try:
        root = args.data_root.resolve()
        manifest = _verified_manifest(root)
        model = load_rigidformer_checkpoint(
            args.checkpoint,
            device=args.device,
            attention_backend=args.attention_backend,
        )
        config = deepcopy(model.reference_config)
        split = json.loads((root / "splits.json").read_text())
        key = "val_datasets" if args.split == "validation" else "test_datasets"
        scenes = split[key]
        if not scenes:
            raise ValueError(f"No {args.split} scenes in the prepared corpus")
        selected = {
            **split,
            "val_datasets": [],
            "test_datasets": scenes,
            "evaluated_partition": args.split,
        }
        splitfile = args.output / "evaluation-splits.json"
        splitfile.write_text(json.dumps(selected, sort_keys=True))
        config["data"].update(
            dataset_path=str(root),
            split_file=str(splitfile.resolve()),
            object_scale_file=str(root / "object_scales.json"),
            eval_step_sizes=args.step_sizes,
        )
        config["training"]["metrics_test"].update(
            enable=True,
            initial_eval=True,
            print_every_n_epochs=1,
            eval_num_frames=args.num_frames,
            eval_warmup_frames=4,
        )
        # A standalone, fixed-checkpoint evaluator: no training or checkpoint selection.
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(args.output / "tensorboard"))
        trainer = SimpleNamespace(
            model=model.network,
            config=config,
            device=args.device,
            writer=writer,
            log_dir=args.output,
        )
        try:
            metrics = Evaluator(trainer).evaluate_rollout_rmse(0)
        finally:
            writer.close()
        # Author aggregates include NumPy scalars; persist portable JSON numbers.
        metrics = json.loads(
            json.dumps(metrics, default=lambda value: value.item(), allow_nan=False)
        )
        record = {
            **evidence(args.checkpoint, root, args.attention_backend),
            "author_revision": __import__(
                "embodichain_models.dynamics.rigidformer",
                fromlist=["RIGIDFORMER_REVISION"],
            ).RIGIDFORMER_REVISION,
            "checkpoint_sha256": sha256(args.checkpoint),
            "data_manifest_sha256": sha256(root / "manifest.json"),
            "split": args.split,
            "num_scenes": len(scenes),
            "num_frames": args.num_frames,
            "step_sizes": args.step_sizes,
            "attention_backend": args.attention_backend,
            "scope": manifest["scope"],
            "metrics": metrics,
        }
        (args.output / "summary.json").write_text(
            json.dumps(record, sort_keys=True, indent=2, allow_nan=False)
        )
        print(json.dumps(record, indent=2, allow_nan=False))
    except BaseException as exc:
        (args.output / "failure.json").write_text(
            json.dumps({"type": type(exc).__name__, "message": str(exc)})
        )
        raise


if __name__ == "__main__":
    main()
