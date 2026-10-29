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

"""Execute a frozen author implementation independently of the native model."""

from __future__ import annotations
import argparse
from copy import deepcopy
import importlib
from importlib.resources import files
import json
from pathlib import Path
import sys
import types
import random
import numpy as np
import torch
from .._prepare import sha256, WEIGHT_SHA256
from embodichain_models.dynamics.rigidformer import (
    load_rigidformer_checkpoint,
    RIGIDFORMER_REVISION,
)
from embodichain_models.dynamics.rigidformer.data import MOViSequenceSource
from embodichain_models.dynamics.rigidformer.supervised import (
    RigidFormerObjective,
    rigidformer_batch,
)
from embodichain_models.dynamics.rigidformer.sl_recipe import _optimizer, _EpochSchedule
from embodichain_models.dynamics.rigidformer.ema import RigidFormerEMA
from embodichain.learning.sl import SupervisedLearner
from ._evidence import evidence


def _assert_tree(actual, expected) -> float:
    """Assert complete numerical/state agreement and return its largest error."""
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        return float((actual - expected).abs().max()) if actual.numel() else 0.0
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        return max(
            (_assert_tree(v, expected[k]) for k, v in actual.items()), default=0.0
        )
    if isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        return max((_assert_tree(a, b) for a, b in zip(actual, expected)), default=0.0)
    if isinstance(actual, float):
        assert actual == expected
        return 0.0
    assert actual == expected
    return 0.0


def _optimizer_state(optimizer, model) -> dict:
    """Compare optimizer groups and moments by parameter name, not object ID."""
    names = {id(p): n.removeprefix("network.") for n, p in model.named_parameters()}
    return {
        "groups": [
            {
                **{k: v for k, v in group.items() if k != "params"},
                "parameters": [names[id(p)] for p in group["params"]],
            }
            for group in optimizer.param_groups
        ],
        "moments": {names[id(p)]: state for p, state in optimizer.state.items()},
    }


def compare(args: argparse.Namespace) -> None:
    if args.updates < 1:
        raise ValueError("updates must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = {
        **evidence(args.checkpoint, args.data_root, args.attention_backend),
        "qualification": "numerical-parity-only",
        "author_revision": RIGIDFORMER_REVISION,
        "attention_backend": args.attention_backend,
        "updates": args.updates,
        "seed": args.seed,
        "comparison_steps_per_epoch": 1,
    }
    try:
        manifest = json.loads(
            files("embodichain_models.dynamics.rigidformer")
            .joinpath("REFERENCE_SOURCES.json")
            .read_text()
        )
        for path, expected in manifest["files"].items():
            if sha256(args.reference / path) != expected:
                raise ValueError(
                    f"Reference source differs from pinned author revision: {path}"
                )
        report["reference_source_sha256"] = manifest["files"]
        if args.attention_backend == "sdpa":
            # Explicit compatibility profile, recorded separately from real Flash-Attention.
            # The author's model/trainer source is not rewritten.
            module = types.ModuleType("flash_attn")

            def sdpa(q, k, v, dropout_p=0.0, softmax_scale=None, **kwargs):
                return torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    dropout_p=dropout_p,
                    scale=softmax_scale,
                ).transpose(1, 2)

            module.flash_attn_func = sdpa
            sys.modules["flash_attn"] = module
            report["reference_attention_profile"] = "explicit-torch-sdpa-compatibility"
        else:
            import flash_attn

            report["reference_attention_profile"] = (
                "flash-attn-" + flash_attn.__version__
            )
        sys.path.insert(0, str(args.reference.resolve()))
        original = importlib.import_module("src.train")
        native = load_rigidformer_checkpoint(
            args.checkpoint, device="cuda", attention_backend=args.attention_backend
        )
        cfg = deepcopy(native.reference_config)
        cfg["paths"] = {
            "log_dir": str(args.output / "reference-logs"),
            "checkpoint_dir": str(args.output / "reference-checkpoints"),
        }
        cfg["data"].update(
            dataset_path=str(args.data_root),
            split_file=str(args.data_root / "splits.json"),
            object_scale_file=str(args.data_root / "object_scales.json"),
        )
        cfg["training"]["checkpoint"]["save_video"] = False
        cfg["training"]["compile"] = {"enable": False}
        source = MOViSequenceSource(
            args.data_root,
            args.data_root / "splits.json",
            args.data_root / "object_scales.json",
            data_config=cfg["data"],
        )
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        raw = source.collate([source.dataset[0]])
        report["input_scenes"] = raw["dataset_name"]
        report["input_frame_ids"] = raw["frame_ids"].tolist()
        raw = {k: v.to("cuda") for k, v in raw.items() if isinstance(v, torch.Tensor)}
        reference = original.CUT3RDynamicsModel(deepcopy(cfg)).cuda().eval()
        reference.load_state_dict(native.network.state_dict(), strict=True)
        with torch.no_grad():
            expected = reference(raw)
            got = native(raw)
        prediction_errors = {
            k: float((got[k] - v).abs().max())
            for k, v in expected.items()
            if isinstance(v, torch.Tensor)
        }
        for key in prediction_errors:
            torch.testing.assert_close(got[key], expected[key], atol=1e-5, rtol=1e-4)
        warm = {
            k: (v[:, :4] if k in ("obj_positions", "valid_vtx_mask") else v)
            for k, v in raw.items()
        }
        trajectory_ref = reference.rollout(
            warm, num_steps=12, warmup_frames=4, return_extras=False
        )
        trajectory_native = native.rollout(warm, num_frames=12, warmup_frames=4)
        torch.testing.assert_close(
            trajectory_native["obj_positions"],
            trajectory_ref["obj_positions"],
            atol=1e-5,
            rtol=1e-4,
        )
        report["prediction_max_abs"] = prediction_errors
        report["rollout_max_abs"] = float(
            (trajectory_native["obj_positions"] - trajectory_ref["obj_positions"])
            .abs()
            .max()
        )
        # The real, unmodified author Trainer computes its own losses, optimizer
        # groups, gradient clipping and EMA; the native side uses shared SL.
        reference_trainer = original.Trainer(
            deepcopy(cfg), reference, [raw], [], device=torch.device("cuda")
        )
        native_optimizer = _optimizer(native, cfg["training"]["optimizer"])
        schedule = _EpochSchedule(
            native_optimizer,
            1,
            cfg["training"]["scheduler"],
            int(cfg["training"]["max_epochs"]),
        )
        average = RigidFormerEMA(native, float(cfg["training"]["ema"]["decay"]))
        if reference_trainer.ema is None:
            raise ValueError("The comparison requires the released EMA-enabled recipe")
        _assert_tree(
            _optimizer_state(native_optimizer, native),
            _optimizer_state(reference_trainer.optimizer, reference),
        )
        from embodichain_models.dynamics.rigidformer._native.data.gpu_augmentation import (
            GPUAugmentation,
        )

        objective = RigidFormerObjective(
            {
                name: cfg["training"]["loss_weights"][name]
                for name in (
                    "position_before_icp",
                    "position_after_icp",
                    "acceleration_before_icp",
                    "acceleration_after_icp",
                )
            },
            time_step_scale=cfg["training"]["loss_weights"].get(
                "if_time_step_scale", True
            ),
            augmentation=GPUAugmentation(cfg, torch.device("cuda")),
        )
        learner = SupervisedLearner(
            native,
            objective,
            native_optimizer,
            max_grad_norm=cfg["training"]["gradient_clip"],
            amp_dtype=torch.bfloat16,
        )
        records = []
        for update in range(args.updates):
            cpu = torch.get_rng_state()
            cuda = torch.cuda.get_rng_state_all()
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            reference_trainer.epoch = update
            expected_loss = reference_trainer.train_epoch()["loss"]
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state_all(cuda)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            metrics = learner.update(rigidformer_batch(raw))
            errors, gradient_errors = [], []
            for name, parameter in native.network.named_parameters():
                target = dict(reference.named_parameters())[name]
                torch.testing.assert_close(parameter, target, atol=1e-5, rtol=1e-4)
                if parameter.grad is not None:
                    torch.testing.assert_close(
                        parameter.grad, target.grad, atol=1e-5, rtol=1e-4
                    )
                    gradient_errors.append(
                        float((parameter.grad - target.grad).abs().max())
                    )
                else:
                    assert target.grad is None
                errors.append(float((parameter - target).abs().max()))
            average.on_update(learner)
            buffer_error = _assert_tree(
                dict(native.network.named_buffers()), dict(reference.named_buffers())
            )
            ema_error = _assert_tree(
                {k.removeprefix("network."): v for k, v in average.shadow.items()},
                reference_trainer.ema.shadow,
            )
            assert average.num_updates == reference_trainer.ema.num_updates
            reference_trainer.scheduler.step()
            schedule.on_update(learner)
            optimizer_error = _assert_tree(
                _optimizer_state(native_optimizer, native),
                _optimizer_state(reference_trainer.optimizer, reference),
            )
            scheduler_error = _assert_tree(
                schedule.scheduler.state_dict(),
                reference_trainer.scheduler.state_dict(),
            )
            if abs(metrics["loss"] - expected_loss) > 1e-5 + 1e-4 * abs(expected_loss):
                raise AssertionError("Paired training loss differs")
            records.append(
                {
                    "update": update + 1,
                    "reference_loss": expected_loss,
                    "sl_loss": metrics["loss"],
                    "parameter_max_abs": max(errors),
                    "gradient_max_abs": max(gradient_errors, default=0.0),
                    "buffer_max_abs": buffer_error,
                    "optimizer_max_abs": optimizer_error,
                    "ema_max_abs": ema_error,
                    "scheduler_max_abs": scheduler_error,
                    "learning_rates": [
                        group["lr"] for group in native_optimizer.param_groups
                    ],
                }
            )
        report["training"] = records
        reference_trainer.writer.close()
        report["checkpoint_sha256"] = sha256(args.checkpoint)
        report["checkpoint_origin"] = (
            "official-release"
            if report["checkpoint_sha256"] in WEIGHT_SHA256.values()
            else "other-explicit-weights"
        )
        report["data_manifest_sha256"] = sha256(args.data_root / "manifest.json")
        report["passed"] = True
        (args.output / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        )
        print(json.dumps(report, indent=2, allow_nan=False))
    except BaseException as exc:
        report.update(passed=False, error_type=type(exc).__name__, error=str(exc))
        (args.output / "failure.json").write_text(
            json.dumps(report, indent=2, sort_keys=True)
        )
        raise
