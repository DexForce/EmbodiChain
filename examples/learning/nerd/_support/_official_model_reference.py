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

"""Generate official data and model predictions without importing EmbodiChain.

Run this worker with Python 3.8.20 / Torch 2.2.2 and the robot's pinned Warp.
Original source and model objects execute unchanged. The CPU stream adapter
only maps a requested CUDA stream on a CPU device to Torch's no-op stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

__all__: list[str] = []
_COMMIT = "370e89d2b8ce68aa3578aa7812859c681b7502bc"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--robot", choices=["cartpole", "ant"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-external-code", action="store_true", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--window-inputs",
        type=Path,
        help="Evaluate immutable common-frame window tensors without generating new physics data.",
    )
    args = parser.parse_args()
    if args.checkpoint is None and not args.generate_only:
        parser.error("Model reference execution requires --checkpoint")
    if args.window_inputs is not None and args.generate_only:
        parser.error("Choose generation or reference-window evaluation")
    source = args.source.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    changed = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if commit != _COMMIT or changed:
        raise ValueError("Reference source must be the pinned, unchanged checkout")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.horizon < 10 or args.count < 1:
        raise ValueError("horizon >= 10 and count >= 1 required")
    sys.path.insert(0, str(source))
    import h5py
    import numpy as np
    import torch
    import warp as wp
    from envs.neural_environment import NeuralEnvironment
    from generate.trajectory_sampler import TrajectorySampler
    from generate.trajectory_sampler_ant import TrajectorySamplerAnt
    from utils.commons import (
        JOINT_Q_MIN,
        JOINT_Q_MAX,
        JOINT_QD_MIN,
        JOINT_QD_MAX,
        JOINT_ACT_SCALE,
    )
    from utils.python_utils import set_random_seed

    version = "1.8.0" if args.robot == "cartpole" else "1.5.1"
    if (
        platform.python_version() != "3.8.20"
        or torch.__version__.split("+")[0] != "2.2.2"
        or wp.__version__ != version
    ):
        raise ValueError(
            "Reference requires Python 3.8.20, Torch 2.2.2 and Warp " + version
        )
    torch.set_num_threads(1)
    set_random_seed(args.seed)
    model = (
        None
        if args.generate_only
        else torch.load(args.checkpoint, map_location="cpu")[0].eval()
    )
    cfg = {
        "name": "TransformerNeuralIntegrator",
        "num_states_history": 10,
        "states_frame": "body" if args.robot == "ant" else "world",
        "anchor_frame_step": "every",
        "states_embedding_type": "identical",
        "prediction_type": "relative",
        "orientation_prediction_parameterization": "quaternion",
    }
    env = NeuralEnvironment(
        env_name=args.robot.capitalize(),
        num_envs=args.num_envs or min(args.count, 512),
        neural_model=model,
        neural_integrator_cfg=cfg,
        default_env_mode="ground-truth",
        warp_env_cfg={"seed": args.seed, "random_reset": False},
        device="cpu",
        render=False,
    )
    if args.window_inputs is not None:
        # Both trainers must see identical floating-point coordinates. Comparing
        # the reference in Y-up against its rotated Z-up inputs additionally
        # measures the reference's own finite-precision frame non-invariance.
        args.output.mkdir(parents=True)
        expected = {}
        hashes = {}
        with np.load(args.window_inputs, allow_pickle=False) as windows:
            for length in [1, 9, 10]:
                prefix = str(length) + "__raw__"
                batch = {}
                for key in windows.files:
                    if key.startswith(prefix):
                        value = np.ascontiguousarray(windows[key])
                        hashes[key] = hashlib.sha256(value.tobytes()).hexdigest()
                        batch[key[len(prefix) :]] = torch.from_numpy(value.copy())
                batch["contact_masks"] = env.integrator_neural.get_contact_masks(
                    batch["contact_depths"], batch["contact_thicknesses"]
                )
                env.integrator_neural.process_neural_model_inputs(batch)
                target = env.integrator_neural.convert_next_states_to_prediction(
                    batch["states"], batch["next_states"], env.frame_dt
                )
                with torch.no_grad():
                    prediction = model(
                        {key: value.clone() for key, value in batch.items()}
                    )
                for key in model.low_dim_input_names:
                    expected[str(length) + "__features__" + key] = batch[key].numpy()
                expected[str(length) + "__prediction"] = prediction.numpy()
                expected[str(length) + "__target"] = target.numpy()
        np.savez(args.output / "official-windows.npz", **expected)
        identity = {
            "scope": "Original model on byte-identical common-frame comparison inputs",
            "source_commit": commit,
            "checkpoint_sha256": _hash(args.checkpoint),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "warp": wp.__version__,
            "numpy": np.__version__,
            "window_inputs_sha256": _hash(args.window_inputs),
            "window_field_sha256": hashes,
            "predictions_sha256": _hash(args.output / "official-windows.npz"),
            "runtime_imports_embodichain": any(
                k.split(".")[0] == "embodichain" for k in sys.modules
            ),
        }
        (args.output / "reference-identity.json").write_text(
            json.dumps(identity, indent=2)
        )
        print(json.dumps(identity, indent=2))
        return
    name = env.robot_name
    sampler_cls = TrajectorySamplerAnt if args.robot == "ant" else TrajectorySampler
    sampler = sampler_cls(
        env,
        JOINT_Q_MIN[name],
        JOINT_Q_MAX[name],
        JOINT_QD_MIN[name],
        JOINT_QD_MAX[name],
        JOINT_ACT_SCALE[name],
        contact_prob=0.0,
    )
    rows = sampler.sample_trajectories_joint_act_mode(
        args.count * args.horizon,
        args.horizon,
        passive=False,
        initial_states_source="sample",
    )
    data = {}
    for key, value in rows.items():
        if isinstance(value, dict):
            data.update(
                {
                    k: v.detach()
                    .cpu()
                    .numpy()
                    .reshape(
                        args.horizon,
                        -1,
                        v.shape[-1] if v.ndim == 3 else v.shape[-2] * v.shape[-1],
                    )
                    for k, v in value.items()
                }
            )
        else:
            data[key] = value.detach().cpu().numpy()
    args.output.mkdir(parents=True)
    with h5py.File(args.output / "test.raw.h5", "w") as h5:
        group = h5.create_group("data")
        group.attrs["env"] = name
        group.attrs["mode"] = "trajectory"
        group.attrs["total_trajectories"] = data["states"].shape[1]
        group.attrs["total_transitions"] = (
            data["states"].shape[0] * data["states"].shape[1]
        )
        for key, value in data.items():
            group.create_dataset(key, data=value)
    if args.generate_only:
        physical = {}
        for key in (
            "body_mass",
            "body_inertia",
            "body_com",
            "joint_armature",
            "joint_limit_lower",
            "joint_limit_upper",
            "joint_limit_ke",
            "joint_limit_kd",
            "joint_axis_mode",
            "joint_target_ke",
            "joint_target_kd",
        ):
            value = getattr(env.model, key, None)
            if value is not None:
                physical[key] = value.numpy().tolist()
        for key in ("mu", "ke", "kd", "kf"):
            physical["shape_material_" + key] = (
                getattr(env.model.shape_materials, key).numpy().tolist()
            )
        physical["shape_thickness"] = env.model.shape_geo.thickness.numpy().tolist()
        physical["shape_body"] = env.model.shape_body.numpy().tolist()
        physical["ground_plane"] = env.model.ground_plane.numpy().tolist()
        physical["gravity"] = np.asarray(env.model.gravity, dtype=np.float64).tolist()
        asset = (
            source
            / "envs/warp_sim_envs/assets"
            / ("ant.xml" if args.robot == "ant" else "cartpole_single.urdf")
        )
        generator_files = [
            "generate/trajectory_sampler.py",
            "generate/simulation_sampler.py",
            "utils/commons.py",
            "utils/env_utils.py",
            "utils/warp_utils.py",
            "envs/neural_environment.py",
            "envs/abstract_contact.py",
            "envs/abstract_contact_environment.py",
            "envs/warp_sim_envs/environment.py",
            "envs/warp_sim_envs/env_" + args.robot + ".py",
        ]
        if args.robot == "ant":
            generator_files.append("generate/trajectory_sampler_ant.py")
        file_hashes = {name: _hash(source / name) for name in generator_files}
        metadata = {
            "robot": args.robot,
            "dt": env.frame_dt,
            "q_dim": env.dof_q_per_env,
            "qd_dim": env.dof_qd_per_env,
            "codec": {
                "robot": args.robot,
                "base_com": env.model.body_com.numpy()[0].tolist(),
                "continuous_q_indices": [1] if args.robot == "cartpole" else [],
            },
            "joint_names": [
                joint.attrib["name"]
                for joint in ET.parse(asset).getroot().iter("joint")
                if "name" in joint.attrib and joint.attrib.get("type") != "free"
            ],
            "asset_sha256": _hash(asset),
            "source_physical_arrays": physical,
            "physics_contract_version": "official-warp-physical-v2",
            "teacher_origin": {
                "kind": "official_warp",
                "source_commit": commit,
                "warp_version": wp.__version__,
                "integrator": str(env.integrator_type_gt),
                "num_substeps": env.sim_substeps_gt,
                "up_axis": int(env.model.up_axis),
                "gravity": physical["gravity"],
                "gravity_magnitude": float(np.linalg.norm(physical["gravity"])),
                "physics_contract_version": "official-warp-physical-v2",
                "device": "cpu",
                "bodies_per_env": env.env.bodies_per_env,
            },
            "source_generator_files": file_hashes,
            "source_generator_sha256": hashlib.sha256(
                json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "generation_adapter_sha256": _hash(Path(__file__)),
            "sampling": {
                "seed": args.seed,
                "batch_size": env.num_envs,
                "initial_states_source": "sample",
                "passive": False,
            },
        }
        (args.output / "source-physics.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False)
        )
        identity = {
            "scope": "Official analytical teacher generation; no model supplied or evaluated",
            "source_commit": commit,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "warp": wp.__version__,
            "numpy": np.__version__,
            "robot": args.robot,
            "trajectories": data["states"].shape[1],
            "horizon": args.horizon,
            "num_envs": env.num_envs,
            "seed": args.seed,
            "raw_sha256": _hash(args.output / "test.raw.h5"),
            "metadata_sha256": _hash(args.output / "source-physics.json"),
            "runtime_imports_embodichain": any(
                key.split(".")[0] == "embodichain" for key in sys.modules
            ),
        }
        (args.output / "reference-identity.json").write_text(
            json.dumps(identity, indent=2)
        )
        print(json.dumps(identity, indent=2))
        return
    # Preserve the original uninitialized gravity feature in raw evidence.
    # Model comparison uses the measured physical direction for both models.
    direction = np.asarray(env.model.gravity, dtype=np.float32)
    direction /= np.linalg.norm(direction)
    corrected = {key: value.copy() for key, value in data.items()}
    corrected["gravity_dir"][:] = direction
    selected = np.random.default_rng(17)
    windows = {}
    integrator = env.integrator_neural
    for length in [1, 9, 10]:
        ids = np.stack(
            (
                selected.integers(0, data["states"].shape[1], 64),
                selected.integers(0, args.horizon - length + 1, 64),
            ),
            axis=-1,
        )
        batch = {
            key: torch.from_numpy(
                np.stack(
                    [value[start : start + length, episode] for episode, start in ids]
                )
            )
            for key, value in corrected.items()
        }
        batch["contact_masks"] = integrator.get_contact_masks(
            batch["contact_depths"], batch["contact_thicknesses"]
        )
        integrator.process_neural_model_inputs(batch)
        target = integrator.convert_next_states_to_prediction(
            batch["states"], batch["next_states"], env.frame_dt
        )
        with torch.no_grad():
            # Official forward replaces dictionary entries with normalized
            # tensors. Preserve the physical inputs as independent fixtures.
            prediction = model({key: value.clone() for key, value in batch.items()})
        windows[str(length) + "__indices"] = ids
        for key in model.low_dim_input_names:
            windows[str(length) + "__features__" + key] = batch[key].numpy()
        windows[str(length) + "__prediction"] = prediction.numpy()
        windows[str(length) + "__target"] = target.numpy()
    np.savez(args.output / "official-windows.npz", **windows)
    # Exercise the untouched original runtime, including history rollover.
    original_stream = wp.stream_to_torch
    wp.stream_to_torch = lambda device: (
        None if not wp.get_device(device).is_cuda else original_stream(device)
    )
    try:
        env.set_env_mode("neural")
        env.reset(initial_states=torch.from_numpy(data["states"][0].copy()))
        states = [env.states.detach().cpu().numpy().copy()]
        for step in range(args.horizon):
            state = env.step_with_joint_act(
                torch.from_numpy(data["joint_acts"][step].copy()), env_mode="neural"
            )
            states.append(state.detach().cpu().numpy().copy())
        np.savez(
            args.output / "official-rollout.npz",
            states=np.stack(states),
            joint_acts=data["joint_acts"],
        )
    finally:
        wp.stream_to_torch = original_stream
    identity = {
        "source_commit": commit,
        "robot": args.robot,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "warp": wp.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "dtype": "float32",
        "count": args.count,
        "horizon": args.horizon,
        "checkpoint_sha256": _hash(args.checkpoint),
        "seed": 500,
        "runtime_imports_embodichain": any(
            k.split(".")[0] == "embodichain" for k in sys.modules
        ),
        "cpu_stream_adapter": "CPU stream request mapped to None; numerical source unchanged",
        "gravity_transport": {
            "direction": direction.tolist(),
            "raw_preserved": True,
            "reason": "Official sampler leaves non-up-axis gravity components uninitialized",
        },
        "files": {p.name: _hash(p) for p in args.output.iterdir() if p.is_file()},
        "model_inputs": list(model.low_dim_input_names),
        "state_frame": cfg["states_frame"],
        "body_com": env.model.body_com.numpy()[0].tolist(),
    }
    (args.output / "reference-identity.json").write_text(
        json.dumps(identity, indent=2, allow_nan=False)
    )
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
