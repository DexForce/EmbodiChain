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

from pathlib import Path
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.scene_engine import randomization
from embodichain.gen_sim.scene_engine.randomization import load_randomization_config


def _write_config(tmp_path: Path, content: str = "schema_version: 1\n") -> Path:
    config_path = tmp_path / "randomization.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


@pytest.mark.parametrize("as_string", [False, True])
def test_workspace_derives_fixed_scene_and_baseline_paths(
    tmp_path: Path, as_string: bool
) -> None:
    output_root = tmp_path / "source"
    config_path = _write_config(tmp_path)

    cfg = load_randomization_config(
        config_path, output_root=str(output_root) if as_string else output_root
    )

    assert cfg.scene_path == output_root / "scene_export"
    assert cfg.base_data_path == cfg.scene_path / "scene_randomization_base.yaml"
    assert cfg.schema_version == 1


def test_relative_workspace_resolves_from_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = _write_config(config_dir)
    working_dir = tmp_path / "working-directory"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)

    cfg = load_randomization_config(config_path, output_root="source")

    assert cfg.scene_path == working_dir / "source" / "scene_export"
    assert cfg.base_data_path == cfg.scene_path / "scene_randomization_base.yaml"


def test_workspace_does_not_guess_a_direct_export_directory(tmp_path: Path) -> None:
    export_path = tmp_path / "scene_export"
    export_path.mkdir()
    (export_path / "scene_config.json").write_text("{}", encoding="utf-8")
    config_path = _write_config(tmp_path)

    cfg = load_randomization_config(config_path, output_root=export_path)

    assert cfg.scene_path == export_path / "scene_export"


def test_workspace_expands_user_home(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    cfg = load_randomization_config(config_path, output_root="~/source")

    assert cfg.scene_path == Path.home() / "source" / "scene_export"


def test_output_symlink_is_preserved_for_persistence_validation(tmp_path: Path) -> None:
    destination = tmp_path / "protected.yaml"
    destination.write_text("protected", encoding="utf-8")
    export_path = tmp_path / "scene_export"
    export_path.mkdir()
    symlink = export_path / "scene_randomization_base.yaml"
    symlink.symlink_to(destination)
    config_path = _write_config(tmp_path)

    assert (
        load_randomization_config(config_path, output_root=tmp_path).base_data_path
        == symlink
    )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("schema_version: true\n", "schema_version"),
        ("schema_version: 2\n", "schema_version"),
        ("schema_version: 1.0\n", "schema_version"),
        ("{}\n", "schema_version"),
        ("null\n", "configuration must be a mapping"),
        ("[]\n", "configuration must be a mapping"),
        ("schema_version: 1\nseed: 2\n", "Unknown configuration fields: seed"),
        ("schema_version: 1\nunknown: true\n", "Unknown configuration fields"),
        ("schema_version: 1\nschema_version: 1\n", "Duplicate"),
        ("schema_version: 1\n1: value\n", "keys"),
        ("schema_version: [\n", "Invalid randomization configuration YAML"),
    ],
)
def test_invalid_configuration_is_rejected(
    tmp_path: Path, content: str, message: str
) -> None:
    config_path = _write_config(tmp_path, content)

    with pytest.raises(ValueError, match=message):
        load_randomization_config(config_path, output_root=tmp_path)


@pytest.mark.parametrize(
    "legacy_paths",
    [
        "input: {scene: source}\n",
        "output: {base_data: baseline.yaml}\n",
        "input: {scene: source}\noutput: {base_data: baseline.yaml}\n",
    ],
)
def test_legacy_paths_explain_output_root_migration(
    tmp_path: Path, legacy_paths: str
) -> None:
    config_path = _write_config(tmp_path, "schema_version: 1\n" + legacy_paths)

    with pytest.raises(ValueError, match="input/output paths.*--output_root"):
        load_randomization_config(config_path, output_root=tmp_path)


@pytest.mark.parametrize("output_root", ["", "   "])
def test_empty_workspace_is_rejected(tmp_path: Path, output_root: str) -> None:
    config_path = _write_config(tmp_path)

    with pytest.raises(ValueError, match="--output_root must be a non-empty"):
        load_randomization_config(config_path, output_root=output_root)


def test_loader_requires_explicit_workspace(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    with pytest.raises(TypeError, match="output_root"):
        load_randomization_config(config_path)


def test_fixed_output_cannot_overwrite_configuration(tmp_path: Path) -> None:
    export_path = tmp_path / "scene_export"
    export_path.mkdir()
    config_path = export_path / "scene_randomization_base.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not overwrite the input configuration"):
        load_randomization_config(config_path, output_root=tmp_path)


def test_unified_cli_passes_workspace_config_and_rebuild_and_reports_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from embodichain.cli.main import main

    config_path = _write_config(tmp_path)
    output_root = tmp_path / "source"
    calls = []

    def prepare_scene_baseline(
        cfg: randomization.SceneRandomizationCfg, *, rebuild: bool = False
    ) -> SimpleNamespace:
        calls.append((cfg, rebuild))
        return SimpleNamespace(
            path=cfg.base_data_path, cache_hit=True, data={"groups": [{}, {}]}
        )

    monkeypatch.setattr(randomization, "prepare_scene_baseline", prepare_scene_baseline)

    main(
        [
            "scene-randomization",
            "--output_root",
            str(output_root),
            "--config",
            str(config_path),
            "--rebuild",
        ]
    )

    assert len(calls) == 1
    cfg, rebuild = calls[0]
    assert cfg.scene_path == output_root / "scene_export"
    assert cfg.base_data_path == cfg.scene_path / "scene_randomization_base.yaml"
    assert rebuild is True
    output = capsys.readouterr().out
    assert str(cfg.base_data_path) in output
    assert "Groups: 2" in output
    assert "Cache: reused" in output


@pytest.mark.parametrize("missing", ["--config", "--output_root"])
def test_cli_requires_config_and_workspace(
    tmp_path: Path, missing: str, capsys: pytest.CaptureFixture[str]
) -> None:
    from embodichain.cli.main import main

    arguments = ["scene-randomization"]
    if missing != "--config":
        arguments.extend(["--config", str(tmp_path / "unused.yaml")])
    if missing != "--output_root":
        arguments.extend(["--output_root", str(tmp_path)])

    with pytest.raises(SystemExit) as exc:
        main(arguments)

    assert exc.value.code == 2
    assert f"required: {missing}" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("schema_version: true\n", "schema_version"),
        ("schema_version: 1\ninput: {scene: old}\n", "--output_root"),
    ],
)
def test_cli_reports_invalid_config_before_baseline_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    content: str,
    message: str,
) -> None:
    from embodichain.gen_sim.scene_engine.cli.randomization import main

    config_path = _write_config(tmp_path, content)

    def prepare_scene_baseline(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid configuration must not reach baseline preparation.")

    monkeypatch.setattr(randomization, "prepare_scene_baseline", prepare_scene_baseline)

    with pytest.raises(SystemExit) as exc:
        main(["--output_root", str(tmp_path), "--config", str(config_path)])

    assert exc.value.code == 2
    assert message in capsys.readouterr().err
