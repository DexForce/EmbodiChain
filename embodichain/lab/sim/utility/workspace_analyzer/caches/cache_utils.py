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

"""
Cache management utility for workspace analyzer.

Usage:
    embodichain workspace-cache list              # List all cache sessions
    embodichain workspace-cache info <session>    # Show cache session info
    embodichain workspace-cache clean <session>   # Clean specific session
    embodichain workspace-cache clean --all       # Clean all cache sessions
    embodichain workspace-cache size              # Show total cache size
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from datetime import datetime

from embodichain.utils import logger


def get_cache_root() -> str:
    """Get the root cache directory."""
    return os.path.expanduser("~/.cache/embodichain/workspace_analyzer")


def get_dir_size(path: str) -> int:
    """Calculate total size of a directory in bytes."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(entry.path)
    except (OSError, PermissionError):
        # Directory access error, return partial total
        pass
    return total


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def list_sessions() -> None:
    """List all cache sessions."""
    cache_root = get_cache_root()

    if not os.path.exists(cache_root):
        logger.log_info("No cache sessions found.")
        logger.log_info(f"Cache directory: {cache_root}")
        return

    sessions = []
    for item in os.listdir(cache_root):
        session_path = os.path.join(cache_root, item)
        if os.path.isdir(session_path):
            size = get_dir_size(session_path)
            mtime = os.path.getmtime(session_path)
            sessions.append(
                {
                    "name": item,
                    "path": session_path,
                    "size": size,
                    "modified": datetime.fromtimestamp(mtime),
                }
            )

    if not sessions:
        logger.log_info("No cache sessions found.")
        return

    # Sort by modification time (newest first)
    sessions.sort(key=lambda x: x["modified"], reverse=True)

    logger.log_info(f"\n{'Session Name':<40} {'Size':<12} {'Last Modified'}")
    logger.log_info("-" * 80)

    total_size = 0
    for session in sessions:
        logger.log_info(
            f"{session['name']:<40} {format_size(session['size']):<12} "
            f"{session['modified'].strftime('%Y-%m-%d %H:%M:%S')}"
        )
        total_size += session["size"]

    logger.log_info("-" * 80)
    logger.log_info(
        f"{'Total':<40} {format_size(total_size):<12} {len(sessions)} session(s)"
    )
    logger.log_info(f"\nCache location: {cache_root}")


def show_session_info(session_name: str) -> None:
    """Show detailed information about a cache session."""
    cache_root = get_cache_root()
    session_path = os.path.join(cache_root, session_name)

    if not os.path.exists(session_path):
        logger.log_info(f"Session '{session_name}' not found.")
        logger.log_info(f"Use 'list' command to see available sessions.")
        return

    logger.log_info(f"\nSession: {session_name}")
    logger.log_info(f"Path: {session_path}")
    logger.log_info(f"Size: {format_size(get_dir_size(session_path))}")
    logger.log_info(
        f"Modified: {datetime.fromtimestamp(os.path.getmtime(session_path))}"
    )

    # Check for batches
    batches_dir = os.path.join(session_path, "batches")
    if os.path.exists(batches_dir):
        batch_files = [f for f in os.listdir(batches_dir) if f.endswith(".npy")]
        logger.log_info(f"Batches: {len(batch_files)} file(s)")

        if batch_files:
            import numpy as np

            total_poses = 0
            for batch_file in batch_files:
                batch_path = os.path.join(batches_dir, batch_file)
                try:
                    data = np.load(batch_path)
                    total_poses += len(data)
                except Exception as e:
                    logger.log_warning(
                        f"Warning: Failed to load batch file '{batch_file}': {e}"
                    )
            logger.log_info(f"Total poses: {total_poses:,}")


def clean_session(session_name: str) -> None:
    """Clean a specific cache session."""
    cache_root = get_cache_root()
    session_path = os.path.join(cache_root, session_name)

    if not os.path.exists(session_path):
        logger.log_info(f"Session '{session_name}' not found.")
        return

    size = get_dir_size(session_path)
    response = input(f"Delete session '{session_name}' ({format_size(size)})? [y/N]: ")

    if response.lower() == "y":
        shutil.rmtree(session_path)
        logger.log_info(f"✓ Deleted session '{session_name}'")
    else:
        logger.log_info("Cancelled.")


def clean_all_sessions() -> None:
    """Clean all cache sessions."""
    cache_root = get_cache_root()

    if not os.path.exists(cache_root):
        logger.log_info("No cache sessions found.")
        return

    total_size = get_dir_size(cache_root)
    sessions = [
        d for d in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, d))
    ]

    if not sessions:
        logger.log_info("No cache sessions found.")
        return

    logger.log_info(
        f"Found {len(sessions)} session(s), total size: {format_size(total_size)}"
    )
    response = input(f"Delete all cache sessions? [y/N]: ")

    if response.lower() == "y":
        shutil.rmtree(cache_root)
        logger.log_info(f"✓ Deleted all cache sessions")
    else:
        logger.log_info("Cancelled.")


def show_total_size() -> None:
    """Show total cache size."""
    cache_root = get_cache_root()

    if not os.path.exists(cache_root):
        logger.log_info("No cache found.")
        logger.log_info(f"Cache directory: {cache_root}")
        return

    total_size = get_dir_size(cache_root)
    sessions = [
        d for d in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, d))
    ]

    logger.log_info(f"\nCache location: {cache_root}")
    logger.log_info(f"Total sessions: {len(sessions)}")
    logger.log_info(f"Total size: {format_size(total_size)}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the backward-compatible workspace analyzer cache CLI.

    Args:
        argv: Arguments excluding the command name. Uses ``sys.argv`` when
            omitted.
    """
    from embodichain.workspace_cache_cli import main as cli_main

    cli_main(argv)


if __name__ == "__main__":
    main()


__all__ = [
    "clean_all_sessions",
    "clean_session",
    "format_size",
    "get_cache_root",
    "get_dir_size",
    "list_sessions",
    "main",
    "show_session_info",
    "show_total_size",
]
