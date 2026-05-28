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

from __future__ import annotations

import shutil
from pathlib import Path

from embodichain.toolkits.scaffold import post_process
from embodichain.toolkits.scaffold.presets import gym_config_to_json
from embodichain.toolkits.scaffold.render import (
    render_extension_file,
    render_task_py,
    render_test_py,
)
from embodichain.toolkits.scaffold.spec import TaskSpec

_APACHE_LICENSE = """Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/LICENSE-2.0
"""


def _write(path: Path, content: str, spec: TaskSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and spec.force:
        path.unlink()
    path.write_text(content, encoding="utf-8")


def _collect_paths(spec: TaskSpec) -> list[Path]:
    paths = list(spec.all_output_paths())
    if (
        spec.target == "extension"
        and spec.output_dir is not None
        and spec.is_new_extension_project()
        and not spec.dry_run
    ):
        if (
            spec.output_dir.exists()
            and any(spec.output_dir.iterdir())
            and not spec.force
        ):
            raise FileExistsError(
                f"Output directory is not empty: {spec.output_dir} (use --force)"
            )
    post_process.check_collisions(spec, paths)
    return paths


def generate_task(spec: TaskSpec) -> list[Path]:
    """Generate task scaffold files. Returns list of written paths."""
    _collect_paths(spec)

    if spec.dry_run:
        return spec.all_output_paths()

    written: list[Path] = []

    task_py = spec.task_py_path()
    _write(task_py, render_task_py(spec), spec)
    written.append(task_py)

    gym_json = spec.gym_json_path()
    _write(gym_json, gym_config_to_json(spec), spec)
    written.append(gym_json)

    if test_path := spec.test_py_path():
        _write(test_path, render_test_py(spec), spec)
        written.append(test_path)

    if spec.target == "inrepo":
        post_process.patch_tasks_init(spec)
    elif spec.is_new_extension_project():
        _generate_extension_tree(spec, written)
    else:
        post_process.patch_tasks_init(spec)

    if spec.run_black:
        post_process.run_black(written)

    if spec.target == "extension" and spec.init_git and spec.output_dir:
        post_process.init_git_repo(spec.output_dir)

    return written


def _generate_extension_tree(spec: TaskSpec, written: list[Path]) -> None:
    assert spec.output_dir is not None
    assert spec.package_name is not None
    root = spec.output_dir
    pkg = root / spec.package_name

    license_text = _APACHE_LICENSE
    repo_license = spec.repo_root / "LICENSE"
    if repo_license.is_file():
        license_text = repo_license.read_text(encoding="utf-8")

    files: list[tuple[Path, str]] = [
        (root / "pyproject.toml", render_extension_file("pyproject.toml", spec)),
        (root / "README.md", render_extension_file("README.md", spec)),
        (root / "LICENSE", license_text),
        (root / ".gitignore", render_extension_file("gitignore", spec)),
        (root / "VERSION", "0.1.0\n"),
        (pkg / "VERSION", "0.1.0\n"),
        (pkg / "__init__.py", render_extension_file("package_init.py", spec)),
        (pkg / "tasks" / "__init__.py", render_extension_file("tasks_init.py", spec)),
        (pkg / "data" / "__init__.py", ""),
        (pkg / "data" / "constants.py", render_extension_file("constants.py", spec)),
        (pkg / "utils" / "__init__.py", ""),
        (root / "scripts" / "run_env.py", render_extension_file("run_env.py", spec)),
    ]

    for path, content in files:
        _write(path, content, spec)
        written.append(path)

    post_process.patch_tasks_init(spec)


def print_summary(spec: TaskSpec, paths: list[Path]) -> None:
    """Print generation summary and next-step commands."""
    mode = "DRY RUN — would write" if spec.dry_run else "Wrote"
    print(f"\n{mode} {len(paths)} file(s):\n")
    for p in paths:
        print(f"  {p}")

    print("\nNext steps:\n")
    if spec.target == "inrepo":
        gym_cfg = spec.gym_json_path().relative_to(spec.repo_root)
        print(f"  python embodichain/lab/scripts/run_env.py " f"--gym_config {gym_cfg}")
        print("  pytest " + str(spec.test_py_path().relative_to(spec.repo_root)))
    else:
        assert spec.output_dir is not None
        print(f"  cd {spec.output_dir}")
        print("  pip install -e .")
        print(
            f"  python scripts/run_env.py "
            f"--gym_config configs/{spec.task_snake}/gym.json --headless"
        )
