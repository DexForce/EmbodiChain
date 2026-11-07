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

from embodichain.gen_sim.scene_engine.randomization import generation, persistence
from embodichain.gen_sim.scene_engine.randomization.cfg import (
    SceneRandomizationCfg,
    load_randomization_config,
)


def _write_config(tmp_path: Path, content: str = "schema_version: 1\n") -> Path:
    config_path = tmp_path / "randomization.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_schema_only_configuration_uses_generation_defaults(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    cfg = load_randomization_config(config_path, output_root=tmp_path)

    assert cfg.seed == 0
    assert cfg.candidate_count == 10
    assert cfg.max_attempts == 1000
    assert cfg.collision_margin_m == pytest.approx(0.02)


def test_configuration_loads_custom_generation_parameters(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        "schema_version: 1\n"
        "seed: 42\n"
        "candidate_count: 3\n"
        "max_attempts: 8\n"
        "collision_margin_m: 0.05\n",
    )

    cfg = load_randomization_config(config_path, output_root=tmp_path)

    assert cfg.seed == 42
    assert cfg.candidate_count == 3
    assert cfg.max_attempts == 8
    assert cfg.collision_margin_m == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 7),
        ("candidate_count", 4),
        ("max_attempts", 25),
        ("collision_margin_m", 0),
    ],
)
def test_partial_generation_configuration_retains_other_defaults(
    tmp_path: Path, field: str, value: int
) -> None:
    config_path = _write_config(tmp_path, f"schema_version: 1\n{field}: {value}\n")
    expected = {
        "seed": 0,
        "candidate_count": 10,
        "max_attempts": 1000,
        "collision_margin_m": 0.02,
    }
    expected[field] = value

    cfg = load_randomization_config(config_path, output_root=tmp_path)

    assert {name: getattr(cfg, name) for name in expected} == expected


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
        (
            "schema_version: 1\ngenerator: {}\n",
            "Unknown configuration fields: generator",
        ),
        ("schema_version: 1\nunknown: true\n", "Unknown configuration fields"),
        ("schema_version: 1\nschema_version: 1\n", "Duplicate"),
        ("schema_version: 1\nseed: 1\nseed: 2\n", "Duplicate"),
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
    ("field", "invalid_value"),
    [
        ("seed", "-1"),
        ("seed", "true"),
        ("seed", "1.0"),
        ("seed", '"1"'),
        ("seed", "null"),
        ("candidate_count", "0"),
        ("candidate_count", "-1"),
        ("candidate_count", "true"),
        ("candidate_count", "2.0"),
        ("candidate_count", '"2"'),
        ("candidate_count", "null"),
        ("max_attempts", "0"),
        ("max_attempts", "-1"),
        ("max_attempts", "true"),
        ("max_attempts", "10.0"),
        ("max_attempts", '"10"'),
        ("max_attempts", "null"),
        ("collision_margin_m", "-0.01"),
        ("collision_margin_m", "true"),
        ("collision_margin_m", '"0.02"'),
        ("collision_margin_m", "null"),
        ("collision_margin_m", ".nan"),
        ("collision_margin_m", ".inf"),
        ("collision_margin_m", "-.inf"),
        ("collision_margin_m", "[]"),
        ("collision_margin_m", "{}"),
    ],
)
def test_invalid_generation_values_are_rejected_during_configuration_loading(
    tmp_path: Path, field: str, invalid_value: str
) -> None:
    config_path = _write_config(
        tmp_path, f"schema_version: 1\n{field}: {invalid_value}\n"
    )

    with pytest.raises(ValueError, match=field):
        load_randomization_config(config_path, output_root=tmp_path)


@pytest.mark.parametrize(
    "parameters",
    [
        "candidate_count: 4\nmax_attempts: 3\n",
        "max_attempts: 9\n",
        "candidate_count: 1001\n",
    ],
)
def test_attempt_budget_must_cover_requested_candidates(
    tmp_path: Path, parameters: str
) -> None:
    config_path = _write_config(tmp_path, "schema_version: 1\n" + parameters)

    with pytest.raises(ValueError, match="max_attempts.*candidate_count"):
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

    config_path = _write_config(
        tmp_path,
        "schema_version: 1\n"
        "seed: 42\n"
        "candidate_count: 3\n"
        "max_attempts: 8\n"
        "collision_margin_m: 0.05\n",
    )
    output_root = tmp_path / "source"
    calls = []
    saved = []
    run_dir = output_root / "scene_layouts/run_001"

    def randomize_scene(
        cfg: SceneRandomizationCfg, *, rebuild: bool = False
    ) -> SimpleNamespace:
        calls.append((cfg, rebuild))
        result = SimpleNamespace(
            baseline=SimpleNamespace(
                path=cfg.base_data_path, cache_hit=True, data={"groups": [{}, {}]}
            ),
            candidates=(object(),) * cfg.candidate_count,
            status="completed",
            attempts=4,
            solver_status_counts={"solved": 3, "no_solution": 1},
        )
        return result

    def save_layouts(result: SimpleNamespace, *, output_root: str) -> Path:
        saved.append((result, output_root))
        return run_dir

    monkeypatch.setattr(generation, "randomize_scene", randomize_scene)
    monkeypatch.setattr(persistence, "save_layouts", save_layouts)

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
    assert cfg.seed == 42
    assert cfg.candidate_count == 3
    assert cfg.max_attempts == 8
    assert cfg.collision_margin_m == pytest.approx(0.05)
    assert rebuild is True
    output = capsys.readouterr().out
    assert str(cfg.base_data_path) in output
    assert "Groups: 2" in output
    assert "Cache: reused" in output
    assert "Sampling: completed" in output
    assert "Attempts: 4" in output
    assert len(saved) == 1
    assert saved[0][0].status == "completed"
    assert saved[0][1] == str(output_root)
    assert "Layouts: 3" in output
    assert f"Run directory: {run_dir}" in output
    assert f"Manifest: {run_dir / 'run_result.json'}" in output
    assert "no_solution=1" in output


@pytest.mark.parametrize(
    ("status", "candidate_count", "exit_code"),
    [
        ("budget_exhausted", 2, None),
        ("budget_exhausted", 0, None),
        ("unsupported", 0, 1),
    ],
)
def test_cli_reports_sampling_outcome_and_returns_normally_on_budget_exhaustion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    candidate_count: int,
    exit_code: int | None,
) -> None:
    from embodichain.gen_sim.scene_engine.cli.randomization import main

    config_path = _write_config(tmp_path)
    result = SimpleNamespace(
        baseline=SimpleNamespace(
            path=tmp_path / "scene_export/scene_randomization_base.yaml",
            cache_hit=True,
            data={"groups": [{}]},
        ),
        candidates=(object(),) * candidate_count,
        status=status,
        attempts=1,
        solver_status_counts={status: 1},
    )
    monkeypatch.setattr(generation, "randomize_scene", lambda *a, **kw: result)
    saved = []

    def save_layouts(actual: SimpleNamespace, *, output_root: str) -> Path:
        saved.append(actual)
        return Path(output_root) / "scene_layouts/run_001"

    monkeypatch.setattr(persistence, "save_layouts", save_layouts)

    arguments = ["--output_root", str(tmp_path), "--config", str(config_path)]
    if exit_code is None:
        assert main(arguments) is None
    else:
        with pytest.raises(SystemExit) as exc:
            main(arguments)
        assert exc.value.code == exit_code
    output = capsys.readouterr().out
    assert f"Sampling: {status}" in output
    assert saved == [result]
    assert f"Layouts: {candidate_count}" in output


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


def test_cli_reports_persistence_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from embodichain.gen_sim.scene_engine.cli.randomization import main

    config_path = _write_config(tmp_path)
    monkeypatch.setattr(generation, "randomize_scene", lambda *a, **kw: object())

    def save_layouts(*args: object, **kwargs: object) -> Path:
        raise OSError("Layout output cannot be written")

    monkeypatch.setattr(persistence, "save_layouts", save_layouts)
    with pytest.raises(SystemExit) as exc:
        main(["--output_root", str(tmp_path), "--config", str(config_path)])
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert "Layout output cannot be written" in output.err
    assert "Manifest:" not in output.out


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("schema_version: true\n", "schema_version"),
        ("schema_version: 1\ninput: {scene: old}\n", "--output_root"),
        ("schema_version: 1\nseed: true\n", "seed"),
        ("schema_version: 1\nmax_attempts: 2\n", "max_attempts"),
        ("schema_version: 1\ncollision_margin_m: .nan\n", "collision_margin_m"),
    ],
)
def test_cli_reports_invalid_config_before_randomization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    content: str,
    message: str,
) -> None:
    from embodichain.gen_sim.scene_engine.cli.randomization import main

    config_path = _write_config(tmp_path, content)

    def randomize_scene(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid configuration must not reach randomization.")

    monkeypatch.setattr(generation, "randomize_scene", randomize_scene)

    with pytest.raises(SystemExit) as exc:
        main(["--output_root", str(tmp_path), "--config", str(config_path)])

    assert exc.value.code == 2
    assert message in capsys.readouterr().err
