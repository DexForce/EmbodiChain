#!/usr/bin/env python3
"""Remove build-host search paths and assign a new immutable wheel build tag."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-tag", required=True)
    parser.add_argument("--patchelf", default="patchelf")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9][A-Za-z0-9_]*", args.build_tag):
        parser.error(
            "build tag must start with a digit and contain only letters/digits/underscores"
        )
    if not args.wheel.name.startswith("open3d-") or not args.wheel.is_file():
        parser.error("input must be an existing Open3D wheel")
    patchelf = shutil.which(args.patchelf)
    if not patchelf:
        parser.error(
            "patchelf is required; select the project's worldparticle-build tooling"
        )
    parts = args.wheel.stem.split("-")
    name = "-".join(parts[:2] + [args.build_tag] + parts[-3:]) + ".whl"
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output.resolve() / name
    if target.exists():
        parser.error("refusing to overwrite an existing immutable wheel")
    records = []
    with tempfile.TemporaryDirectory(prefix="dexsim-open3d-wheel-") as directory:
        root = Path(directory)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "unpack",
                str(args.wheel.resolve()),
                "-d",
                str(root),
            ],
            check=True,
        )
        (unpacked,) = root.iterdir()
        package = unpacked / "open3d"
        for path in sorted(package.rglob("*")):
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    continue
            relatives = [
                os.path.relpath(package, path.parent),
                os.path.relpath(unpacked / "torch/lib", path.parent),
                os.path.relpath(unpacked / "nvidia/cuda_runtime/lib", path.parent),
            ]
            runpath = ":".join(
                "$ORIGIN" if relative == "." else "$ORIGIN/" + relative
                for relative in relatives
            )
            subprocess.run([patchelf, "--set-rpath", runpath, str(path)], check=True)
            actual = subprocess.check_output(
                [patchelf, "--print-rpath", str(path)], text=True
            ).strip()
            if actual != runpath:
                raise RuntimeError("ELF relocation verification failed")
            records.append({"path": str(path.relative_to(unpacked)), "runpath": actual})
        if not records:
            raise RuntimeError("No ELF libraries found in the Open3D wheel")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                str(unpacked),
                "-d",
                str(args.output.resolve()),
                "--build-number",
                args.build_tag,
            ],
            check=True,
        )
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    digest = digest.hexdigest()
    report = {
        "filename": target.name,
        "sha256": digest,
        "size": target.stat().st_size,
        "elf_libraries": records,
    }
    target.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["filename", "sha256", "size"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
