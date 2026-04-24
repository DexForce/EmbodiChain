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

"""Helper script for filtering versions to maintain buffer size."""

import re
from pathlib import Path


def parse_version(tag: str) -> tuple[int, int, int]:
    """Parse a version tag like 'v1.2.3' into a tuple (1, 2, 3)."""
    match = re.match(r"^v(\d+)\.(\d+)\.(\d+)$", tag)
    if not match:
        return (0, 0, 0)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def filter_versions(
    all_versions: list[str],
    buffer_size: int,
    main_branch: str = "main",
) -> list[str]:
    """Filter versions to maintain buffer size.

    Keeps the latest (buffer_size - 1) release versions plus the main branch.

    Args:
        all_versions: List of all available version references
        buffer_size: Total number of versions to keep (releases + main)
        main_branch: Name of the main branch

    Returns:
        List of versions to keep
    """
    # Separate releases from branches
    releases = [v for v in all_versions if re.match(r"^v\d+\.\d+\.\d+$", v)]
    branches = [v for v in all_versions if v not in releases]

    # Sort releases by version (newest first)
    releases.sort(key=parse_version, reverse=True)

    # Keep latest (buffer_size - 1) releases
    releases_to_keep = releases[: (buffer_size - 1)]

    # Always include main branch if it exists
    versions_to_keep = releases_to_keep.copy()
    if main_branch in branches:
        versions_to_keep.append(main_branch)

    return versions_to_keep


def main():
    """CLI entry point for version filtering."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter versions for multi-version docs"
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        required=True,
        help="List of all available versions",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=5,
        help="Total number of versions to keep (releases + main)",
    )
    parser.add_argument(
        "--main-branch",
        default="main",
        help="Name of the main branch",
    )
    args = parser.parse_args()

    filtered = filter_versions(args.versions, args.buffer_size, args.main_branch)
    print(" ".join(filtered))


if __name__ == "__main__":
    main()
