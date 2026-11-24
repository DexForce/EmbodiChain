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

"""Prepare the pinned RP1, USD converter and official PartyMimic ONNX."""

from __future__ import annotations

import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from urllib.request import urlopen

__all__ = ["main"]

_CACHE = Path.home() / ".cache/embodichain/partymimic"
_PARTYMIMIC = "c3da040573cb9ce06dfa6a2ee007873dcec0c076"
_RP1 = "19081d6f9fd5b613a304a7749f5886b30eb7a022"
_ROBOT_DIR = Path("data/rp1_28dof_mjlab")
_XML = "rp1_28dof.xml"
_MODEL = "rp1_sparse5"
_CONVERTER_REQUIREMENTS = (
    "mujoco-usd-converter==0.10.0",
    "mujoco-core==4.0.1",
    "newton-usd-schemas==0.7.0",
    "usd-exchange==3.0.1",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _upstream(path: Path) -> None:
    path.mkdir(exist_ok=True)
    git = ["git", "-C", str(path)]
    if not (path / ".git").is_dir():
        subprocess.run([*git, "init", "--quiet"], check=True)
    current = subprocess.run(
        [*git, "rev-parse", "--verify", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.strip()
    if current != _PARTYMIMIC:
        url = "https://github.com/Roboparty/PartyMimic.git"
        subprocess.run([*git, "config", "remote.origin.url", url], check=True)
        subprocess.run(
            [
                *git,
                "sparse-checkout",
                "set",
                "src",
                "tools",
                "assets/rp1_28dof_mjlab",
                "assets/motions/kneel_crawl/rp1",
            ],
            check=True,
        )
        subprocess.run(
            [
                *git,
                "-c",
                "http.version=HTTP/1.1",
                "fetch",
                "--depth=1",
                "--filter=blob:none",
                "origin",
                _PARTYMIMIC,
            ],
            check=True,
        )
        subprocess.run(
            [*git, "checkout", "--detach", "--quiet", _PARTYMIMIC], check=True
        )
    for name in (
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
    ):
        if not (path / name).is_file():
            data = subprocess.check_output([*git, "show", f"{_PARTYMIMIC}:{name}"])
            (path / name).write_bytes(data)


def _robot(upstream: Path) -> None:
    path = upstream / _ROBOT_DIR
    if (path / _XML).is_file():
        return
    path.parent.mkdir(exist_ok=True)
    url = f"https://codeload.github.com/Roboparty/rp1_description/tar.gz/{_RP1}"
    with TemporaryDirectory(dir=path.parent) as directory:
        prepared = Path(directory) / path.name
        mesh_dir = prepared / "meshes"
        mesh_dir.mkdir(parents=True)
        with (
            urlopen(url, timeout=60) as response,
            tarfile.open(fileobj=response, mode="r|gz") as archive,
        ):
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if (
                    member.isfile()
                    and len(parts) >= 3
                    and parts[:2] == (f"rp1_description-{_RP1}", "meshes")
                    and ".." not in parts
                ):
                    target = mesh_dir.joinpath(*parts[2:])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        archive.extractfile(member) as data,
                        target.open("wb") as stream,
                    ):
                        shutil.copyfileobj(data, stream)
        if not list(mesh_dir.glob("*.STL")):
            raise RuntimeError("The pinned RP1 archive contained no expected meshes.")
        shutil.copyfile(upstream / "assets/rp1_28dof_mjlab" / _XML, prepared / _XML)
        prepared.replace(path)


def _converter(cache: Path) -> None:
    path = cache / "converter"
    python = path / "bin/python"
    if not python.is_file():
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(path)], check=True
        )
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), *_CONVERTER_REQUIREMENTS],
        check=True,
    )


def _export(checkpoint: Path, upstream: Path, output: Path) -> None:
    names = tuple(f"{_MODEL}{suffix}" for suffix in (".onnx", ".yaml", ".check.npz"))
    receipt = output / f"{_MODEL}.provenance.json"
    provenance = {
        "checkpoint_sha256": _sha256(checkpoint),
        "partymimic_revision": _PARTYMIMIC,
        "rp1_description_revision": _RP1,
        "export_check_clips": "assets/motions/kneel_crawl/rp1",
        "robot_xml_sha256": _sha256(upstream / _ROBOT_DIR / _XML),
    }
    if receipt.is_file() and all((output / name).is_file() for name in names):
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous == {
            **provenance,
            "artifacts": {name: _sha256(output / name) for name in names},
        }:
            return
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
        "PYTHONPATH": str(upstream / "src"),
        "MOTION_DIR": str(upstream / "assets/motions/kneel_crawl/rp1"),
    }
    with TemporaryDirectory(dir=output) as directory:
        temporary = Path(directory)
        subprocess.run(
            [
                "uv",
                "run",
                "--locked",
                "--with",
                "onnx",
                "--python",
                sys.executable,
                "python",
                "tools/export_deploy_onnx.py",
                str(checkpoint),
                str(temporary / names[0]),
            ],
            cwd=upstream,
            env=env,
            check=True,
        )
        provenance["artifacts"] = {name: _sha256(temporary / name) for name in names}
        prepared_receipt = temporary / receipt.name
        prepared_receipt.write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        for name in names:
            (temporary / name).replace(output / name)
        prepared_receipt.replace(receipt)


def main() -> None:
    """Prepare the assets and export a local RP1 sparse5 checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    from embodichain.data.dataset import _dataset_download_lock

    _CACHE.mkdir(parents=True, exist_ok=True)
    upstream = _CACHE / "upstream"
    with _dataset_download_lock(str(_CACHE), "partymimic"):
        _upstream(upstream)
        _robot(upstream)
        _converter(_CACHE)
        _export(checkpoint, upstream, output)
    print(output / f"{_MODEL}.onnx")


if __name__ == "__main__":
    main()
