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

"""Qualify installed image environments with real codecs, Blender and CUDA."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

__all__ = []


def _versions() -> dict[str, str]:
    import importlib.util
    import embodichain
    import cv2
    import torch

    selected = os.environ.get("EMBODICHAIN_ENV")
    if selected:
        assert selected == f"py{sys.version_info.major}{sys.version_info.minor}", (
            selected,
            sys.executable,
            sys.version,
        )

    expected = (
        ("0.4.4", "2.7.0", "0.22.0", "0.5", "0.13.0")
        if sys.version_info[:2] == (3, 11)
        else ("0.6.1", "2.11.0", "0.26.0", "0.11.1", "0.15.3")
    )
    names = ("lerobot", "torch", "torchvision", "torchcodec", "tensordict")
    actual = {
        name: version(name)
        for name in (
            *names,
            "numpy",
            "dexsim-engine",
            "onnxruntime-gpu",
            "gradio",
            "nvidia-curobo",
            "opencv-python",
            "opencv-python-headless",
        )
    }
    assert sys.version_info[:2] in {(3, 11), (3, 12)}
    assert (
        importlib.util.find_spec("bpy") is None
    ), "Blender must stay in its own runtime"
    assert [actual[name].split("+", 1)[0] for name in names] == list(expected), actual
    assert torch.version.cuda == "12.8", torch.version.cuda
    assert actual["opencv-python"] == actual["opencv-python-headless"], actual
    assert cv2.__version__ == actual["opencv-python"].rsplit(".", 1)[0], actual
    subprocess.run(["uv", "pip", "check", "--python", sys.executable], check=True)
    return {
        "python": sys.version,
        "executable": sys.executable,
        "embodichain_file": embodichain.__file__,
        **actual,
    }


def _media() -> dict[str, object]:
    import av
    import h5py
    import numpy as np
    from torchcodec.decoders import VideoDecoder

    assert av.codec.Codec("libx264", "w").is_encoder
    assert av.codec.Codec("libx265", "w").is_encoder
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with h5py.File(root / "data.h5", "w") as dataset:
            dataset["values"] = np.arange(12)
        with h5py.File(root / "data.h5", "r") as dataset:
            np.testing.assert_array_equal(dataset["values"][:], np.arange(12))
        path = root / "rgb.mp4"
        with av.open(str(path), "w") as output:
            stream = output.add_stream("libx264", rate=10)
            stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
            for color in (20, 80, 140):
                frame = av.VideoFrame.from_ndarray(
                    np.full((24, 32, 3), color, dtype=np.uint8), format="rgb24"
                )
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        decoder = VideoDecoder(str(path), device="cpu")
        assert len(decoder) == 3
        frames = decoder.get_frames_at(indices=[0, 1, 2]).data
        assert tuple(frames.shape) == (3, 3, 24, 32)
        np.testing.assert_allclose(
            frames.float().mean(dim=(1, 2, 3)).numpy(), [20, 80, 140], atol=4
        )
    return {"av": av.__version__, "hdf5": h5py.version.hdf5_version}


def _blender() -> dict[str, object]:
    import numpy as np
    from PIL import Image
    import trimesh
    from embodichain.gen_sim._blender_runtime import _run_blender_job
    from embodichain.gen_sim.scene_engine.pipeline.utils.simready_processor_utils import (
        render_object_front_top_views,
        render_object_pose_switch_candidates,
    )
    from embodichain.gen_sim.scene_engine.pipeline.utils.visual_yaw_optimizer import (
        _render_canonical_oblique_view,
    )
    from embodichain.gen_sim.asset_engine.utils.geometry_utils import process_obj
    from embodichain.gen_sim.asset_engine.utils.ingest_utils import (
        blender_remesh_bake,
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        mesh = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
        source = root / "cube.glb"
        mesh.export(source)
        outputs = [
            render_object_front_top_views(
                glb_path=source, output_path=root / "views.png", resolution=64
            ),
            render_object_pose_switch_candidates(
                glb_path=source, output_path=root / "candidates.png", resolution=64
            ),
            _render_canonical_oblique_view(
                glb_path=source,
                baked_scale_y_up=[1, 1, 1],
                z_up_yaw_degrees=30,
                output_path=root / "yaw.png",
                resolution=64,
            ),
        ]
        for output in outputs:
            with Image.open(output) as image:
                assert image.height == 64 and np.asarray(image).std() > 1
        obj = root / "cube.obj"
        mesh.export(obj)
        process_obj(obj, root / "clean.obj", remove_non_manifold=False)
        assert (root / "clean.obj").is_file()
        baked = blender_remesh_bake(source, root / "baked", texture_size=64)
        assert all(
            (root / "baked" / baked[key]).is_file() for key in ("obj", "png", "mtl")
        )
    return {"python": os.environ.get("EMBODICHAIN_BLENDER_PYTHON"), "operations": 5}


def _cuda_act() -> dict[str, object]:
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    assert torch.cuda.is_available()
    config = ACTConfig(
        device="cuda",
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
            "observation.images.camera": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        pretrained_backbone_weights=None,
        chunk_size=2,
        n_action_steps=1,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=4,
        n_vae_encoder_layers=1,
        dropout=0.0,
    )
    policy = ACTPolicy(config).to("cuda")
    batch = {
        "observation.state": torch.zeros(2, 2, device="cuda"),
        "observation.images.camera": torch.zeros(2, 3, 32, 32, device="cuda"),
        "action": torch.ones(2, 2, 2, device="cuda"),
        "action_is_pad": torch.zeros(2, 2, dtype=torch.bool, device="cuda"),
    }
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
    before = policy.model.action_head.weight.detach().clone()
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(policy.model.action_head.weight.grad).all()
    optimizer.step()
    assert not torch.equal(before, policy.model.action_head.weight)
    with tempfile.TemporaryDirectory() as directory:
        policy.save_pretrained(directory)
        restored = ACTPolicy.from_pretrained(directory).to("cuda")
        torch.testing.assert_close(
            restored.model.action_head.weight, policy.model.action_head.weight
        )
        restored.reset()
    torch.cuda.synchronize()
    return {"gpu": torch.cuda.get_device_name(), "loss": loss.item()}


def _onnx_cuda() -> dict[str, object]:
    import base64
    import numpy as np
    import onnxruntime as ort

    # Fixed opset-13/IR-8 graph: y = MatMul(x, [[1, 2], [3, 4]]).
    # Shipping the tiny graph avoids adding the ONNX authoring SDK to runtime.
    model = base64.b64decode(
        "CAg6aQoRCgF4CgF3EgF5IgZNYXRNdWwSDXF1YWxpZmljYXRpb24qGwgCCAIQASIQAACAPwAAAEAAAEBAAACAQEIBd1oTCgF4Eg4KDAgBEggKAggCCgIIAmITCgF5Eg4KDAgBEggKAggCCgIIAkIECgAQDQ=="
    )
    ort.preload_dlls(directory="")
    with tempfile.TemporaryDirectory() as directory:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(directory) / "onnx")
        session = ort.InferenceSession(
            model, sess_options=options, providers=["CUDAExecutionProvider"]
        )
        actual = session.run(
            None, {"x": np.asarray([[1, 2], [3, 4]], dtype=np.float32)}
        )[0]
        np.testing.assert_array_equal(actual, [[7, 10], [15, 22]])
        events = json.loads(Path(session.end_profiling()).read_text())
        assert any(
            event.get("args", {}).get("provider") == "CUDAExecutionProvider"
            for event in events
        ), "MatMul did not execute on CUDA"
    return {"onnxruntime": ort.__version__, "provider": "CUDAExecutionProvider"}


def _installed_pytest(provenance: str, arguments: list[str]) -> int:
    """Run staged tests against this interpreter's installed distribution."""
    from importlib.metadata import distribution
    from importlib.util import find_spec

    import pytest

    # Only tests and their supporting scripts are staged here, never packages.
    sys.path.insert(0, str(Path.cwd()))
    installed = distribution("embodichain")
    roots = {
        name: Path(installed.locate_file(name)).resolve()
        for name in ("embodichain", "embodichain_tasks", "embodichain_robots")
    }

    def check_imports() -> dict[str, str]:
        for name, root in roots.items():
            spec = find_spec(name)
            locations = (
                list(spec.submodule_search_locations or ()) if spec is not None else []
            )
            if (
                not root.is_relative_to(Path(sys.prefix).resolve())
                or not locations
                or any(Path(location).resolve() != root for location in locations)
            ):
                raise RuntimeError(
                    f"Qualification requires installed wheel: {name} resolved to "
                    f"{locations}, expected {root} inside {sys.prefix}"
                )
        for name, module in list(sys.modules.items()):
            root = roots.get(name.split(".", 1)[0])
            origin = getattr(module, "__file__", None)
            if root is not None and origin is not None:
                if not Path(origin).resolve().is_relative_to(root):
                    raise RuntimeError(f"Source package imported: {name} from {origin}")
        return {name: str(root) for name, root in roots.items()}

    check_imports()
    result = pytest.main(arguments)
    Path(provenance).write_text(json.dumps(check_imports()), encoding="utf-8")
    return result


def _pytest_checks(
    repo: Path, selections: list[str], *, gpu: bool = False
) -> dict[str, object]:
    import xml.etree.ElementTree as xml

    with tempfile.TemporaryDirectory(prefix="embodichain-qualification-") as directory:
        report = Path(directory) / "tests.xml"
        provenance = Path(directory) / "imports.json"
        for name in ("tests", "scripts"):
            shutil.copytree(
                repo / name,
                Path(directory) / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        shutil.copy2(repo / "pyproject.toml", Path(directory) / "pyproject.toml")
        # Isolated Python ignores inherited PYTHONPATH and user site packages.
        # The verifier loads by filename; the checkout never enters sys.path.
        command = [
            sys.executable,
            "-I",
            "-c",
            "import runpy, sys; "
            "raise SystemExit(runpy.run_path(sys.argv[1])['_installed_pytest']("
            "sys.argv[2], sys.argv[3:]))",
            str(Path(__file__).resolve()),
            str(provenance),
            "-q",
            "--import-mode=importlib",
            "-o",
            f"cache_dir={directory}/cache",
        ]
        if gpu:
            command.append("--run-gpu")
        environment = os.environ.copy()
        for name in ("PYTHONPATH", "PYTHONHOME"):
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        result = subprocess.run(
            [*command, *selections, f"--junitxml={report}"],
            cwd=directory,
            env=environment,
        )
        if result.returncode:
            raise RuntimeError(f"Qualification pytest failed: exit {result.returncode}")
        cases = list(xml.parse(report).iter("testcase"))
        skipped = [
            case.attrib.get("name")
            for case in cases
            if case.find("skipped") is not None
        ]
        assert (
            cases and not skipped
        ), f"Required checks must execute, skipped: {skipped}"
        return {
            "cases": len(cases),
            "imports": json.loads(provenance.read_text(encoding="utf-8")),
        }


def _dataset_tests(repo: Path) -> dict[str, object]:
    return _pytest_checks(
        repo,
        [
            "tests/test_lerobot_compatibility.py",
            "tests/data_pipeline/test_lerobot_source.py",
            "tests/gym/envs/managers/test_dataset_functors.py",
            "tests/gym/envs/managers/test_async_dataset_functors.py",
            "tests/lab/scripts/test_preview_lerobot_data.py",
            "tests/gen_sim/test_blender_runtime.py",
            "tests/gen_sim/scene_engine/test_simready_processor_utils.py",
            "tests/gen_sim/scene_engine/test_scene_generation.py",
            "tests/gen_sim/asset_engine",
        ],
    )


def _native_tests(repo: Path) -> dict[str, object]:
    # Existing tests cover the production native boundary, not a second copy
    # of simulation/planner setup logic in this verifier.
    selections = [
        "tests/sim/sensors/test_camera.py::test_camera_backend_smoke",
        "tests/sim/sensors/test_contact_default_e2e.py",
        "tests/sim/sensors/test_contact_newton_e2e.py",
        "tests/sim/motion/planners/test_curobo_integration.py::test_curobo_v2_plans_a_joint_space_move",
        "tests/data_pipeline/depth_video/test_writer.py",
    ]
    return _pytest_checks(repo, selections, gpu=True)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("cpu", "blender", "gpu", "all"), default="all"
    )
    parser.add_argument(
        "--repo", type=Path, help="Checkout used for existing native integration tests"
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    checks = {"versions": _versions}
    if args.suite in {"cpu", "all"}:
        checks["media"] = _media
        if args.repo is not None:
            checks["datasets"] = lambda: _dataset_tests(args.repo.resolve())
    if args.suite in {"blender", "all"}:
        checks["blender"] = _blender
    if args.suite in {"gpu", "all"}:
        checks["cuda_act"] = _cuda_act
        checks["onnx_cuda"] = _onnx_cuda
        if args.repo is None:
            parser.error("--repo is required for GPU/native qualification")
        checks["native"] = lambda: _native_tests(args.repo.resolve())
    report = {"suite": args.suite, "checks": {}}
    for name, check in checks.items():
        started = time.monotonic()
        try:
            value = check()
            report["checks"][name] = {"passed": True, "result": value}
        except Exception as exc:
            import traceback

            traceback.print_exc()
            report["checks"][name] = {"passed": False, "error": str(exc)}
        report["checks"][name]["seconds"] = round(time.monotonic() - started, 3)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"{name}: {'PASS' if report['checks'][name]['passed'] else 'FAIL'}",
            flush=True,
        )
    if not all(value["passed"] for value in report["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
