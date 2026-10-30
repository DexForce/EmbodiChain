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
"""Compare the native CPU policy with the author's checksum-pinned JavaScript.

Run: python -m scripts.benchmark.assembly.policy_reference --cache-dir outputs/policy_reference_cache --output outputs/policy_reference

Node.js and the ordinary native RL Python dependencies are required. The public
actor weights are a numerical reference only; this script performs no training,
simulator stepping, GPU work or performance measurement. It writes JSON and one
Markdown report with the project's three benchmark tables.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any
from urllib.request import urlopen

__all__: list[str] = []

_BASE = "https://younghyopark.me/blind-proprioceptive-assembly/static/"
_ASSETS = {
    "policy": (
        "policies/peg.json",
        "62d3201d7362a904c50f7e2ba42c0439cead30c0a4fa40f0afade7875ff1b164",
    ),
    "weights": (
        "policies/peg.bin",
        "015ab5aa9e53505b2f5d2edc8234c2d79e78cbaa750bb7b70bae392084f3d900",
    ),
    "reference": (
        "js/policy_net.js",
        "730ff1cba9f8e7a1dc024230830650ce0ce9802c03fcbaec3560a3f5ad519f05",
    ),
}
_INPUT_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "examples/sim/assembly/validation/rl_attempt/policy_reference_inputs.json"
)
_FIXTURE_SHA = "8e919e4b42479b1691e56d301747fb73248a981a0474acce7b7ee31afa3b24cb"
_INPUT_SHA = "8b99c010bb3d1cbe5da1070a8e8cdfde0b3200deb12e9173553a04c11d42cc81"
_ATOL = 1e-5
_RTOL = 1e-4

# Execute the pinned, separately downloaded reference rather than duplicating
# its convolution/activation/normalization calculations in the Python harness.
_NODE_RUNNER = r"""
import fs from 'node:fs';
const [sourcePath, metaPath, weightsPath, inputPath, outputPath] = process.argv.slice(2);
const source = fs.readFileSync(sourcePath).toString('base64');
const { PolicyNet } = await import('data:text/javascript;base64,' + source);
const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
const bytes = fs.readFileSync(weightsPath);
const net = new PolicyNet(meta, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const inputs = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const results = inputs.map(obs => {
  const action = Array.from(net.forward(obs, true));
  return { action, latent: Array.from(net.latent), contact_logits: Array.from(net.logits) };
});
fs.writeFileSync(outputPath, JSON.stringify(results));
"""


def _verify(data: bytes, expected: str, name: str) -> None:
    """Reject changed reference files, including previously cached files."""
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {name}: expected {expected}, got {actual}"
        )


def _cached_asset(cache: Path, relative: str, digest: str) -> Path:
    """Download atomically only after verifying the complete source bytes."""
    path = cache / relative.replace("/", "_")
    if path.exists():
        _verify(path.read_bytes(), digest, str(path))
        return path
    with urlopen(_BASE + relative, timeout=30) as response:
        data = response.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise ValueError(
            f"Reference asset exceeds the bounded download size: {relative}"
        )
    _verify(data, digest, _BASE + relative)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _fixed_inputs(output: Path) -> tuple[Path, dict[str, Any]]:
    """Rebuild the original eight observations from small fixed FP32 frames."""
    data = _INPUT_FIXTURE.read_bytes()
    _verify(data, _FIXTURE_SHA, str(_INPUT_FIXTURE))
    fixture = json.loads(data)
    frames = fixture.pop("frames")
    if len(frames) != 8 or any(len(frame) != 33 for frame in frames):
        raise ValueError("Reference inputs must contain eight 33-value frames.")
    inputs = [
        frame + [value for j in range(29) for value in frames[(i + j) % 8]] + frame
        for i, frame in enumerate(frames)
    ]
    encoded = json.dumps(inputs).encode("utf-8")
    _verify(encoded, _INPUT_SHA, "reconstructed observation minibatch")
    path = output / "inputs.json"
    path.write_bytes(encoded)
    return path, fixture


def _native_outputs(
    paths: dict[str, Path], input_path: Path, threads: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load identical author tensors into the native policy on CPU only."""
    import numpy as np
    import torch
    from tensordict import TensorDict

    from embodichain.learning.rl.policies import build_policy

    torch.set_num_threads(threads)
    metadata = json.loads(paths["policy"].read_text())
    architecture = metadata["architecture"]
    frame_dim = architecture["frame_dim"]
    history_length = architecture["history_length"]
    dimension = frame_dim * (1 + history_length)
    policy = build_policy(
        {
            "name": "proprioceptive_actor_critic",
            "obs_groups": {"actor": ["proprioception"], "critic": ["privileged"]},
            "actor_obs_normalization": True,
            "cfg": {
                "architecture_version": "official_v1",
                "frame_dim": frame_dim,
                "history_length": history_length,
                "latent_dim": architecture["latent_dim"],
                "contact_points": architecture["contact_sample_count"],
                "contact_classes": architecture["contact_class_count"],
                "actor_hidden_sizes": architecture["actor_hidden_dims"],
                "decoder_hidden_sizes": architecture["decoder_hidden_dims"],
            },
        },
        dimension,
        architecture["action_dim"],
        torch.device("cpu"),
        critic_obs_space=235,
    ).eval()
    weights = np.fromfile(paths["weights"], dtype="<f4")
    state = policy.state_dict()
    with torch.no_grad():
        normalizer = policy.actor_obs_normalizer
        normalizer.mean.copy_(torch.from_numpy(weights[:dimension]))
        normalizer.variance.copy_(torch.from_numpy(weights[dimension : 2 * dimension]))
        normalizer.standard_deviation.copy_(normalizer.variance.sqrt())
        for name, shape, offset in metadata["tensors"]:
            target = name.replace("history_encoder.", "actor.encoder.").replace(
                "actor_mean.", "actor.head."
            )
            count = int(np.prod(shape))
            tensor = weights[2 * dimension + offset : 2 * dimension + offset + count]
            state[target].copy_(torch.from_numpy(tensor.reshape(shape)))
    observations = torch.tensor(json.loads(input_path.read_text()), dtype=torch.float32)
    batch = TensorDict(
        {"obs": observations, "critic_obs": torch.zeros(len(observations), 235)},
        batch_size=[len(observations)],
    )
    with torch.inference_mode():
        policy.get_action(batch, deterministic=True)
        evaluated = policy.evaluate_actions(batch)
        _, latent = policy.actor.features(policy.actor_obs_normalizer(observations))
    return (
        {
            "action": batch["action"].numpy(),
            "latent": latent.numpy(),
            "contact_logits": evaluated["contact_logits"].flatten(1).numpy(),
        },
        {"torch": torch.__version__, "numpy": np.__version__, "cpu_threads": threads},
    )


def _write_reports(report: dict[str, Any], output: Path) -> Path:
    """Write machine-readable data and exactly three accuracy-report tables."""
    (output / "policy_reference.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    status = report["status"]
    native_success = report.get("success_rate", "unmeasured")
    reference_status = "passed" if "fields" in report else "unavailable"
    lines = [
        "# Assembly policy reference",
        "",
        "CPU fixed-input inference comparison; author weights are never used for native training.",
        "",
        "## Time & Memory",
        "",
        "| implementation | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | --- | --- | --- | --- |",
        "| Public JavaScript | unmeasured | unmeasured | N/A | N/A |",
        "| Native PyTorch CPU | unmeasured | unmeasured | N/A | N/A |",
        "",
        "## Success & Other Metrics",
        "",
        "| implementation | field | status | success_rate | max_abs_error |",
        "| --- | --- | --- | --- | --- |",
        f"| Public JavaScript | reference | {reference_status} | {'1.0' if 'fields' in report else 'unmeasured'} | N/A |",
    ]
    for name, field in report.get("fields", {}).items():
        lines.append(
            f"| Native PyTorch CPU | {name} | {field['status']} | {field['success_rate']} | {field['max_abs_error']:.9g} |"
        )
    if "fields" not in report:
        lines.append(
            f"| Native PyTorch CPU | all | {status} | unmeasured | unmeasured |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | overall_success_rate | scope |",
        "| --- | --- | --- | --- |",
        f"| 1 | Public JavaScript | {'1.0' if 'fields' in report else 'unmeasured'} | finite reference output |",
        f"| {'1 (tie)' if native_success == 1.0 else '2'} | Native PyTorch CPU | {native_success} | all fields within tolerance per input |",
        "",
        "No latency, memory, learning success or dynamics equivalence is measured.",
        "Histories mix fixed reset frames deliberately; they are not physical rollouts.",
        f"Tolerance: atol={_ATOL}, rtol={_RTOL}. Status: {status}.",
    ]
    if "reason" in report:
        lines.extend(["", "Reason: " + report["reason"]])
    path = output / "policy_reference.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_all_benchmarks(args: argparse.Namespace) -> int:
    """Run the independent reference and native CPU network, preserving failures."""
    import numpy as np

    output = Path(args.output).expanduser().resolve()
    cache = Path(args.cache_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture_version": "official_v1",
        "batch_size": 8,
        "observation_dimension": 1023,
        "precision": "float32",
        "device": "cpu",
        "atol": _ATOL,
        "rtol": _RTOL,
        "python": sys.version,
        "platform": platform.platform(),
        "source_assets": {
            key: {"url": _BASE + relative, "sha256": digest}
            for key, (relative, digest) in _ASSETS.items()
        },
        "inputs_sha256": _INPUT_SHA,
        "fixture_sha256": _FIXTURE_SHA,
        "performance": "unmeasured; fixed-input accuracy comparison only",
        "checkpoint_use": "numerical reference only; no training or model initialization",
    }
    try:
        node = shutil.which(args.node)
        if node is None:
            raise FileNotFoundError(f"Node.js executable is unavailable: {args.node}")
        report["node"] = subprocess.check_output([node, "--version"], text=True).strip()
        paths = {
            key: _cached_asset(cache, relative, digest)
            for key, (relative, digest) in _ASSETS.items()
        }
        input_path, fixture = _fixed_inputs(output)
        report["input_construction"] = fixture
        reference_path = output / "reference_outputs.json"
        subprocess.run(
            [
                node,
                "--input-type=module",
                "-",
                str(paths["reference"]),
                str(paths["policy"]),
                str(paths["weights"]),
                str(input_path),
                str(reference_path),
            ],
            input=_NODE_RUNNER,
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
        )
        reference = json.loads(reference_path.read_text())
        actual, runtime = _native_outputs(paths, input_path, args.cpu_threads)
        report["runtime"] = runtime
        passed = np.ones(len(reference), dtype=bool)
        fields = {}
        for name, values in actual.items():
            expected = np.asarray([row[name] for row in reference], dtype=np.float32)
            if values.shape != expected.shape or not np.isfinite(expected).all():
                raise ValueError(f"Invalid reference shape or values for {name}.")
            if not np.isfinite(values).all():
                raise ValueError(f"Native policy produced non-finite {name}.")
            matches = np.isclose(values, expected, atol=_ATOL, rtol=_RTOL).all(axis=1)
            passed &= matches
            error = np.abs(values - expected)
            fields[name] = {
                "status": "passed" if matches.all() else "failed",
                "success_rate": float(matches.mean()),
                "max_abs_error": float(error.max()),
                "mean_abs_error": float(error.mean()),
            }
        report.update(
            status="passed" if passed.all() else "failed",
            success_rate=float(passed.mean()),
            fields=fields,
        )
        (output / "native_outputs.json").write_text(
            json.dumps({key: value.tolist() for key, value in actual.items()}) + "\n"
        )
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        report.update(status="skipped", reason=str(exc))
    except subprocess.CalledProcessError as exc:
        report.update(status="failed", reason=f"Node reference failed: {exc.stderr}")
    except Exception as exc:
        report.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    path = _write_reports(report, output)
    print(f"{report['status']}: Markdown report saved: {path}", flush=True)
    return {"passed": 0, "failed": 1, "skipped": 2}[report["status"]]


def main() -> int:
    """Parse the bounded CPU-reference comparison options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="outputs/policy_reference_cache")
    parser.add_argument("--output", default="outputs/policy_reference")
    parser.add_argument(
        "--node", default="node", help="Node.js executable name or path"
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    return run_all_benchmarks(args)


if __name__ == "__main__":
    raise SystemExit(main())
