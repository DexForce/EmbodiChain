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

"""Prepare and verify a Pages replacement without losing published versions."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

from packaging.version import Version

from merge_published_site import load_versions_manifest, merge_published_site

__all__ = ["prepare_deployment", "verify_deployment"]


def _read_url(site_url: str, path: str) -> bytes:
    url = f"{site_url.rstrip('/')}/{path}?fresh={time.time_ns()}"
    with urlopen(url, timeout=10) as response:
        return response.read()


def _published_snapshot(site_url: str, repository: str, destination: Path) -> Path:
    """Require the exact deployed artifact, including JavaScript-loaded resources."""
    marker = _read_url(site_url, "pages-deployment-id.txt").decode().strip()
    if not re.fullmatch(r"[0-9]+-[0-9]+", marker):
        raise ValueError(f"Invalid published deployment marker: {marker!r}")
    run_id, attempt = marker.split("-")
    result = subprocess.run(
        [
            "gh",
            "run",
            "download",
            run_id,
            "--repo",
            repository,
            "--name",
            f"github-pages-{attempt}",
            "--dir",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if result.returncode:
        raise RuntimeError(
            f"Published Pages artifact {run_id}/{attempt} is unavailable. "
            "Refusing to replace the site without a complete published snapshot."
        )
    with tarfile.open(destination / "artifact.tar") as archive:
        archive.extractall(destination / "site", filter="data")
    snapshot = destination / "site"
    if (snapshot / "pages-deployment-id.txt").read_text().strip() != marker:
        raise RuntimeError("Published artifact does not match the live deployment")
    return snapshot


def prepare_deployment(
    build_dir: Path,
    version: str,
    *,
    published_root: Path,
    max_versions: int = 5,
) -> None:
    """Merge the current site immediately before replacing it with an artifact.

    Args:
        build_dir: Downloaded candidate site.
        version: Only version rebuilt by this candidate (main or a release tag).
        published_root: Exact previously deployed site, or a local test fixture.
        max_versions: Number of newest stable versions to retain.
    """
    if version != "main":
        if not version.startswith("v") or Path(version).name != version:
            raise ValueError(f"Unsupported documentation version: {version!r}")
        Version(version)
    if max_versions < 1:
        raise ValueError("max_versions must be positive")
    if not (build_dir / version / "index.html").is_file():
        raise FileNotFoundError(f"Candidate is missing {version}/index.html")
    manifest = load_versions_manifest(published_root=published_root, required=True)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("versions"), list):
        raise ValueError("Invalid published versions manifest")
    names = {entry["name"] for entry in manifest["versions"]}
    if manifest.get("latest") not in names:
        raise ValueError("Published latest version is missing from its manifest")
    # Only the rebuilt version belongs to this candidate. Cached versions must
    # never replace content that has been deployed since the build started.
    merge_published_site(
        build_dir,
        published_root=published_root,
        skip_versions=frozenset({version}),
        refresh_versions=frozenset(names),
        required=True,
    )
    # Use the existing PEP 440 ordering, including post releases, before pruning.
    generator = Path(__file__).with_name("generate_versions_json.py")
    subprocess.run(
        [sys.executable, str(generator), "--build-dir", str(build_dir)], check=True
    )
    candidate = json.loads((build_dir / "versions.json").read_text())
    tags = [entry["name"] for entry in candidate["versions"] if entry["type"] == "tag"]
    removed = set(tags[max_versions:])
    if version in removed:
        raise ValueError(f"{version} falls outside the newest {max_versions} releases")
    for name in removed:
        shutil.rmtree(build_dir / name)
    for name in (names | {version}) - removed:
        if not (build_dir / name / "index.html").is_file():
            raise FileNotFoundError(f"Prepared site is missing {name}/index.html")
    subprocess.run(
        [sys.executable, str(generator), "--build-dir", str(build_dir)], check=True
    )


def verify_deployment(build_dir: Path, site_url: str, deployment_id: str) -> None:
    """Check the live marker, complete version manifest and every version homepage.

    Args:
        build_dir: Prepared site containing the expected versions.json.
        site_url: Deployed Pages URL.
        deployment_id: Expected workflow run ID and attempt.
    """
    marker = _read_url(site_url, "pages-deployment-id.txt").decode().strip()
    if marker != deployment_id:
        raise RuntimeError(f"Pages serves {marker}, expected {deployment_id}")
    expected = json.loads((build_dir / "versions.json").read_text())
    actual = json.loads(_read_url(site_url, "versions.json"))
    if actual != expected:
        raise RuntimeError("Published versions manifest differs from the prepared site")
    for entry in expected["versions"]:
        _read_url(site_url, f"{entry['name']}/index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify"])
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--version", default="main")
    parser.add_argument("--repository")
    parser.add_argument("--deployment-id")
    args = parser.parse_args()
    if args.command == "prepare":
        if not args.repository:
            parser.error(
                "prepare requires --repository for the complete published artifact"
            )
        with tempfile.TemporaryDirectory() as directory:
            snapshot = _published_snapshot(
                args.site_url, args.repository, Path(directory)
            )
            prepare_deployment(
                args.build_dir,
                args.version,
                published_root=snapshot,
            )
    else:
        if not args.deployment_id:
            parser.error("verify requires --deployment-id")
        for attempt in range(30):
            try:
                verify_deployment(args.build_dir, args.site_url, args.deployment_id)
                print(
                    f"Verified Pages deployment {args.deployment_id} and all version homepages."
                )
                return
            except (OSError, ValueError, RuntimeError) as exc:
                if attempt == 29:
                    raise
                print(f"Waiting for Pages ({attempt + 1}/30): {exc}", flush=True)
                time.sleep(10)


if __name__ == "__main__":
    main()
