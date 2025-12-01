#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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
    python cache_utils.py list              # List all cache sessions
    python cache_utils.py info <session>    # Show cache session info
    python cache_utils.py clean <session>   # Clean specific session
    python cache_utils.py clean --all       # Clean all cache sessions
    python cache_utils.py size              # Show total cache size
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from embodichain.utils import logger

# Add current directory to path for local imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from disk_cache import DiskCache
except ImportError:
    print("Error: Unable to import DiskCache")
    sys.exit(1)


def get_cache_root():
    """Get the root cache directory."""
    return os.path.expanduser("~/.cache/embodichain/workspace_analyzer")


def get_dir_size(path):
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


def format_size(bytes_size):
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def list_sessions():
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


def show_session_info(session_name):
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


def clean_session(session_name):
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


def clean_all_sessions():
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


def show_total_size():
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


def main():
    parser = argparse.ArgumentParser(
        description="Manage workspace analyzer cache sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                    List all cache sessions
  %(prog)s info session_20241127   Show session details
  %(prog)s clean session_20241127  Clean specific session
  %(prog)s clean --all             Clean all sessions
  %(prog)s size                    Show total cache size
        """,
    )

    parser.add_argument(
        "command", choices=["list", "info", "clean", "size"], help="Command to execute"
    )
    parser.add_argument(
        "session", nargs="?", help="Session name (for info/clean commands)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Apply to all sessions (for clean command)"
    )

    args = parser.parse_args()

    if args.command == "list":
        list_sessions()
    elif args.command == "info":
        if not args.session:
            print("Error: Session name required for 'info' command")
            sys.exit(1)
        show_session_info(args.session)
    elif args.command == "clean":
        if args.all:
            clean_all_sessions()
        elif args.session:
            clean_session(args.session)
        else:
            print("Error: Specify a session name or use --all flag")
            sys.exit(1)
    elif args.command == "size":
        show_total_size()


if __name__ == "__main__":
    main()
