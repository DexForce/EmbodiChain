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

"""Configure the real non-editable wheel from an external consumer directory."""

from __future__ import annotations
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Thread
import pytest
from PIL import Image

pytestmark = pytest.mark.no_sim


def test_installed_wheel_cli_reads_external_profiles_without_package_edits(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    source = tmp_path / "source"
    source.mkdir()
    for name in (
        "setup.py",
        "pyproject.toml",
        "VERSION",
        "README.md",
        "LICENSE",
        "MANIFEST.in",
    ):
        shutil.copy2(repo / name, source / name)
    for name in (
        "embodichain",
        "examples",
        "extensions/embodichain_robots",
        "extensions/embodichain_tasks",
        "extensions/embodichain_benchmarks",
        "extensions/embodichain_models",
    ):
        shutil.copytree(
            repo / name,
            source / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", ".env"),
        )

    def execute(command, cwd, env=None):
        result = subprocess.run(
            command, cwd=cwd, env=env, capture_output=True, text=True, timeout=180
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    execute(
        [
            sys.executable,
            "-I",
            "setup.py",
            "bdist_wheel",
            "--dist-dir",
            str(tmp_path / "dist"),
        ],
        source,
    )
    wheel = next((tmp_path / "dist").glob("*.whl"))
    venv = tmp_path / "venv"
    execute(
        [sys.executable, "-I", "-m", "venv", "--system-site-packages", str(venv)],
        tmp_path,
    )
    python = str(venv / "bin/python")
    execute(
        [
            python,
            "-I",
            "-m",
            "pip",
            "--isolated",
            "install",
            "--ignore-installed",
            "--no-deps",
            "--no-compile",
            str(wheel),
        ],
        tmp_path,
    )
    shutil.rmtree(source)
    site = (
        venv
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    # A venv created from another venv only inherits the base interpreter's
    # packages. Append the invoking runtime's installed dependencies as plain
    # paths, without executing its editable-install hooks.
    runtime_sites = [
        path
        for path in sys.path
        if path and Path(path).name in {"site-packages", "dist-packages"}
    ]
    (site / "runtime-dependencies.pth").write_text(
        "\n".join(dict.fromkeys(runtime_sites)) + "\n"
    )
    # Remove source imports without excluding dependencies installed inside the checkout.
    source_paths = [repo.resolve()] + [
        path.resolve() for path in (repo / "extensions").iterdir() if path.is_dir()
    ]
    (site / "sitecustomize.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        f"source_paths={list(map(str, source_paths))!r}\n"
        "sys.meta_path=[x for x in sys.meta_path if '__editable__' not in str(x)]\n"
        "sys.path=[x for x in sys.path if x and str(Path(x).resolve()) not in source_paths]\n"
    )
    package = site / "embodichain/gen_sim"
    bundled = package / "asset_engine/configs/gen_config.json"
    before = hashlib.sha256(bundled.read_bytes()).hexdigest()
    assert (package / "services/model_services.example.json").is_file()
    assert (package / "services/DEPLOYMENT.md").is_file()
    assert (package / "services/spec/v1/README.md").is_file()
    assert not (package / ".env").exists()
    from ..task_engine.test_image_planning import _fixture

    step, grounding, plan = _fixture()
    bodies = []
    auth = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            auth.append(self.headers.get("Authorization"))
            if body.get("response_format"):
                content = {"steps": [step]}
            elif isinstance(body["messages"][1]["content"], list):
                content = grounding
            else:
                content = plan
            payload = json.dumps(
                {"choices": [{"message": {"content": json.dumps(content)}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    external = tmp_path / "external"
    external.mkdir()
    config = external / "model_services.json"
    config.write_text(
        json.dumps(
            {
                "model_services": {
                    "profiles": {
                        "cloud": {
                            "adapter": "openai_compatible",
                            "model": {"env": "WHEEL_MODEL"},
                            "api_key_env": "WHEEL_KEY",
                            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                        }
                    },
                    "roles": {
                        r: "cloud"
                        for r in (
                            "task_interpretation",
                            "task_image_grounding",
                            "task_semantic_planning",
                        )
                    },
                }
            }
        )
    )
    dotenv = external / "models.env"
    dotenv.write_text("WHEEL_MODEL=wheel-model\nWHEEL_KEY=file-key\n")
    image = external / "image.png"
    Image.new("RGB", (100, 80)).save(image)
    env = {
        **os.environ,
        "EMBODICHAIN_MODEL_SERVICES_CONFIG": str(config),
        "EMBODICHAIN_ENV_FILE": str(dotenv),
        "WHEEL_KEY": "process-key",
        "XDG_CONFIG_HOME": str(external / "xdg"),
    }
    env.pop("PYTHONPATH", None)
    try:
        execute(
            [
                python,
                "-I",
                "-B",
                "-m",
                "embodichain",
                "task-engine",
                "--image",
                str(image),
                "--instruction",
                "执行测试任务",
                "--output",
                str(external / "result"),
            ],
            external,
            env,
        )
        records = json.loads((external / "result/service_calls.json").read_text())
        assert set(records["selections"]) == {
            "task_interpretation",
            "task_image_grounding",
            "task_semantic_planning",
        }
        assert all(
            s["source"] == "environment" and s["model"] == "wheel-model"
            for s in records["selections"].values()
        )
        assert auth == ["Bearer process-key"] * 3 and all(
            b["model"] == "wheel-model" for b in bodies
        )
        execute(
            [
                python,
                "-I",
                "-B",
                "-m",
                "embodichain",
                "gensim",
                "task",
                "plan",
                "--image",
                str(image),
                "--instruction",
                "执行测试任务",
                "--output",
                str(external / "canonical"),
            ],
            external,
            env,
        )
        assert (external / "canonical/semantic_task_graph.html").is_file()
        assert auth == ["Bearer process-key"] * 6
        for command in (
            ["gensim", "run"],
            ["gensim", "ui"],
            ["gensim", "scene", "expand"],
            ["gensim", "services", "validate"],
            ["gensim", "services", "doctor"],
        ):
            execute(
                [python, "-I", "-B", "-m", "embodichain", *command, "--help"],
                external,
                env,
            )
        # A second installed consumer discovers XDG configuration without the selector.
        user = external / "xdg/embodichain/model_services.json"
        user.parent.mkdir(parents=True)
        shutil.copyfile(config, user)
        env.pop("EMBODICHAIN_MODEL_SERVICES_CONFIG")
        probe = "from pathlib import Path; import embodichain; from embodichain.gen_sim.services import GenSimServices; assert Path(embodichain.__file__).is_relative_to(Path(__import__('sys').prefix)); r=GenSimServices(); assert r.model('task_interpretation')=='wheel-model'; r.get('task_interpretation'); assert r.selections()['task_interpretation']['source']=='user'; r.close()"
        execute([python, "-I", "-B", "-c", probe], external, env)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert hashlib.sha256(bundled.read_bytes()).hexdigest() == before
    assert not (package / ".env").exists()
