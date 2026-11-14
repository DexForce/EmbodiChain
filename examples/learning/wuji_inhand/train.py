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
"""Train a fresh 20-DOF Wuji teacher using native physics and EC APG."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from _dependency import _activate

__all__ = ["main"]


def main() -> None:
    """Run online physical collection, world-model fitting and APG updates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-replay", type=Path)
    args = parser.parse_args()
    if args.resume_replay and not args.resume:
        raise ValueError("A replay receipt requires its paired model checkpoint")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh run directory")
    output.mkdir(parents=True)
    _activate(args.upstream.resolve(), output / "learner_dependency")
    import numpy as np
    import torch
    from _native_checkpoint import _restore_learner, _runtime_source
    from _resource_learner import _ResourceAPGLearner
    from _task import _WujiTask
    from native_full import _save, _wait_memory
    from omegaconf import OmegaConf
    from sharpa_rl_unilab.training.configuration import configure_threads
    from sharpa_rl_unilab.training.policy import make_models
    from unilab.base.torch_env import TorchEnvState

    cfg = OmegaConf.load(args.reference_config)
    if args.resume:
        snapshot = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(snapshot["config"])
        if snapshot["native_identity"].get("physical_dof") != 20:
            raise ValueError("Wuji resume requires an actual20-DOF checkpoint")
        if int(cfg.algo.seed) != args.seed:
            raise ValueError("Resume with the saved training seed")
        previous = json.loads((args.resume.parent / "experiment.json").read_text())
        for name in ["_physics.py", "_asset.py", "_task.py", "_dependency.py"]:
            digest = hashlib.sha256(
                (Path(__file__).parent / name).read_bytes()
            ).hexdigest()
            if digest != previous["source_hashes"][name]:
                raise ValueError(f"Physical source changed since training: {name}")
        if args.gain != previous["gain"]:
            raise ValueError("Resume must preserve native controller gains")
        inputs = json.loads(
            (args.resume.parent / "runtime_inputs_manifest.json").read_text()
        )
        for name, digest in inputs["assets"].items():
            if hashlib.sha256((args.assets / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Physical asset changed since training: {name}")
        for name, digest in inputs["core"].items():
            current = _runtime_source(name).read_bytes()
            if hashlib.sha256(current).hexdigest() != digest:
                raise ValueError(f"Runtime source changed since training: {name}")
    else:
        cfg.algo.seed = args.seed
        cfg.algo.buffer_steps = 2048000 // args.num_envs
        cfg.algo.warmup_steps = 131072 // args.num_envs
        cfg.algo.wm_hidden = [2048, 2048, 2048]
        cfg.algo.wm_batch = 2048
        cfg.algo.wm_batches = 10
        cfg.algo.pi_batch = 1024
        cfg.algo.horizon = 32
        cfg.algo.compile = False
        cfg.algo.replay_device = "cpu"
        cfg.algo.activation_checkpoint = "world_model_mlp_nonreentrant"
        if args.smoke:
            cfg.algo.buffer_steps = 64
            cfg.algo.warmup_steps = 16
            cfg.algo.wm_hidden = [64, 64]
            cfg.algo.wm_batch = 8
            cfg.algo.wm_batches = 1
            cfg.algo.pi_batch = 8
            cfg.algo.horizon = 4
            cfg.algo.wm_only_iters = 2
            cfg.algo.critic_epochs = 1
            cfg.algo.critic_minibatches = 1
            cfg.algo.use_amp = False
    cfg.algo.algo = "embodichain_apg"
    cfg.algo.num_envs = args.num_envs
    cfg.algo.max_iterations = args.iterations
    cfg.training.num_envs = args.num_envs
    cfg.training.device = "cuda:0"
    configure_threads(cfg)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    (output / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    source_hashes = {}
    (output / "sources").mkdir()
    for name in ["_physics.py", "_asset.py", "_task.py", "_dependency.py", "train.py"]:
        p = Path(__file__).parent / name
        source_hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        shutil.copy2(p, output / "sources" / name)
    experiment = {
        "hand": "Wuji Hand V1 right",
        "native_environment": True,
        "source_hashes": source_hashes,
        "assets": str(args.assets.resolve()),
        "gain": args.gain,
        "resource_profile": "reference_host20",
        "smoke": args.smoke,
        "seed": args.seed,
        "parameter_randomization": False,
        "starting_actor": (
            "native resume"
            if args.resume
            else "fresh random initialization; no source checkpoint loaded"
        ),
        "scope": "Palm-up Wuji v2; eight sizes; physically screened thumb/two-opposing-finger grasp and controller preloads; contact-gated reward and loss-of-grasp termination; axial phase randomization; fixed mass/friction/PD/vertical object gravity. Fresh training, no v1 cache or policy reuse. Not source Sharpa DR parity.",
    }
    (output / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    runtime_files = [
        "embodichain/learning/rl/algo/apg.py",
        "embodichain/learning/rl/experimental/_host_replay.py",
        "embodichain/lab/sim/objects/articulation.py",
        "embodichain/lab/sim/physics/_cylinder_variants.py",
        "embodichain/lab/sim/sensors/contact/_geometry_contact_sensor.py",
        "embodichain/lab/sim/sensors/contact/_warp/geometry_contact.py",
        "examples/learning/sharpa_inhand/_apg_learner.py",
        "examples/learning/sharpa_inhand/_resource_learner.py",
        "examples/learning/sharpa_inhand/_native_checkpoint.py",
        "examples/learning/sharpa_inhand/native_full.py",
    ]
    assets = args.assets.resolve()
    inputs = {
        "timing": "Frozen before simulator construction and training",
        "asset_root": str(assets),
        "assets": {
            str(p.relative_to(assets)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(assets.rglob("*"))
            if p.is_file()
        },
        "core": {
            name: hashlib.sha256(_runtime_source(name).read_bytes()).hexdigest()
            for name in runtime_files
        },
    }
    (output / "runtime_inputs_manifest.json").write_text(
        json.dumps(inputs, indent=2) + "\n"
    )

    class GuardedTask(_WujiTask):
        def step(self, action: torch.Tensor) -> TorchEnvState:
            _wait_memory("cuda:0", 4.0)
            return super().step(action)

    _wait_memory("cuda:0", 6.0)
    actor, critic, _ = make_models(cfg, "cuda:0")
    initial = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    torch.save(
        initial, output / ("starting_actor.pt" if args.resume else "initial_actor.pt")
    )
    task = None
    started = time.monotonic()
    try:
        task = GuardedTask(
            cfg,
            args.assets.resolve(),
            output / "physical",
            num_envs=args.num_envs,
            seed=args.seed,
            gain=args.gain,
        )
        assert task.s.shape == (args.num_envs, 83) and task.xi.shape == (
            args.num_envs,
            51,
        )
        learner = _ResourceAPGLearner(
            task, cfg, actor, critic, task.mdp, task.capture, value_bootstrap=True
        )
        start = 0
        restored = False
        if args.resume:
            restored = _restore_learner(
                learner,
                snapshot,
                args.resume.resolve(),
                args.resume_replay.resolve() if args.resume_replay else None,
                output,
            )
            learner.collected = int(snapshot["collected"])
            learner.received = int(snapshot["received"])
            learner.collection_credit = float(snapshot.get("collection_credit", 0.0))
            start = int(snapshot["iteration"])
            torch.save(
                {k: v.detach().cpu() for k, v in actor.state_dict().items()},
                output / "starting_actor.pt",
            )
        if not restored:
            for offset in range(0, int(cfg.algo.warmup_steps), 16):
                learner.collect(min(16, int(cfg.algo.warmup_steps) - offset))
        with (output / "metrics.jsonl").open("a") as metrics:
            for iteration in range(start, args.iterations):
                _wait_memory("cuda:0", 4.0)
                values = learner.collect(5)
                _wait_memory("cuda:0", 4.0)
                for _ in range(int(cfg.algo.wm_batches)):
                    loss, error = learner.wm_update()
                values.update(
                    {
                        "loss/world_model": float(loss),
                        "loss/world_model_last": float(error),
                    }
                )
                if iteration >= int(cfg.algo.wm_only_iters):
                    _wait_memory("cuda:0", 4.0)
                    values.update(learner.policy_update())
                values.update(learner.episodes.metrics())
                values.update(
                    iteration=iteration + 1,
                    collected=learner.collected,
                    wall_seconds=time.monotonic() - started,
                )
                metrics.write(json.dumps(values) + "\n")
                metrics.flush()
                if iteration % 20 == 0 or iteration + 1 == args.iterations:
                    print(json.dumps(values), flush=True)
                if (
                    (iteration + 1) % 100 == 0
                    or iteration + 1 == args.iterations
                    or iteration == start
                ):
                    _save(
                        output, cfg, learner, iteration + 1, time.monotonic() - started
                    )
        (output / "completed.json").write_text(
            json.dumps(
                {
                    "iterations": args.iterations,
                    "collected": learner.collected,
                    "native_environment": True,
                    "hand": "Wuji",
                    "wall_seconds": time.monotonic() - started,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        if task is not None:
            task.close()


if __name__ == "__main__":
    main()
