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

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from embodichain.gen_sim.task_engine.orchestration.visual_grounding import (
    _default_visual_transport,
    make_visual_grounding_caller,
    visual_grounding_available,
)
from embodichain.gen_sim.task_engine.orchestration.scene_source import scene_revision_id


def _evidence(tmp_path: Path) -> dict:
    config = tmp_path / "scene_config.json"
    config.write_text(
        json.dumps(
            {
                "format": "embodichain.scene-export/v1",
                "background": [],
                "rigid_object": [],
                "articulation": [],
            }
        ),
        encoding="utf-8",
    )
    views = []
    for name in ("oblique", "top"):
        image = tmp_path / f"{name}.png"
        Image.new("RGB", (16, 16), "white").save(image)
        views.append(
            {
                "name": name,
                "annotated_path": str(image),
                "annotated_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        )
    catalog = tmp_path / "catalog.png"
    Image.new("RGB", (16, 16), "white").save(catalog)
    asset = tmp_path / "cup.glb"
    asset.write_bytes(b"test visual asset")
    return {
        "schema_version": "gen_sim.visual-grounding-evidence/v1",
        "source_config_path": str(config),
        "source_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "scene_revision_id": scene_revision_id(config),
        "render_kind": "configured_static_glb_proxies",
        "articulation_state_untrusted_uids": ["cabinet"],
        "objects": [
            {
                "uid": "cup_001",
                "visible_views": ["oblique"],
                "asset_path": str(asset),
                "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            }
        ],
        "views": views,
        "catalog_path": str(catalog),
        "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
    }


def test_visual_caller_keeps_uid_validator_schema_and_saves_provenance(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    seen = []

    def transport(**kwargs):
        seen.append(kwargs)
        return {
            "bindings": [
                {
                    "reference_id": "step_01.object",
                    "status": "not_found",
                    "uids": [],
                    "confidence": 0.0,
                    "evidence_view": "oblique",
                    "evidence_note": "No labeled coaster UID exists.",
                    "scene_missing": True,
                }
            ]
        }

    caller = make_visual_grounding_caller(
        evidence, tmp_path / "audit", transport=transport
    )
    result = caller(prompt="Ground a coaster", schema={}, model="vision-test")

    assert result == {
        "bindings": [
            {
                "reference_id": "step_01.object",
                "status": "not_found",
                "uids": [],
                "confidence": 0.0,
            }
        ]
    }
    assert len(seen[0]["image_paths"]) == 3
    assert "never infer open/closed state" in seen[0]["prompt"]
    audit = json.loads((tmp_path / "audit/call_01.json").read_text())
    assert audit["response"]["bindings"][0]["scene_missing"] is True


def test_visual_caller_rejects_stale_image_before_transport(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence["catalog_sha256"] = "0" * 64
    caller = make_visual_grounding_caller(
        evidence,
        tmp_path / "audit",
        transport=lambda **_kwargs: pytest.fail("Stale images must not be sent."),
    )

    with pytest.raises(ValueError, match="image changed"):
        caller(prompt="Ground", schema={}, model=None)


def test_visual_caller_rejects_unseen_or_scene_missing_selection(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    response = {
        "bindings": [
            {
                "reference_id": "step_01.object",
                "status": "resolved",
                "uids": ["cup_001"],
                "confidence": 0.9,
                "evidence_view": "oblique",
                "evidence_note": "The cup is in the labeled crop.",
                "scene_missing": True,
            }
        ]
    }
    caller = make_visual_grounding_caller(
        evidence, tmp_path / "audit", transport=lambda **_kwargs: response
    )
    with pytest.raises(ValueError, match="without visible evidence"):
        caller(prompt="Ground", schema={}, model=None)
    assert (tmp_path / "audit/call_01.json").is_file()

    response["bindings"][0]["scene_missing"] = False
    response["bindings"][0]["uids"] = ["not_rendered"]
    with pytest.raises(ValueError, match="without visible evidence"):
        caller(prompt="Ground", schema={}, model=None)

    response["bindings"][0]["uids"] = ["cup_001"]
    response["bindings"][0]["evidence_view"] = "top"
    with pytest.raises(ValueError, match="without visible evidence"):
        caller(prompt="Ground", schema={}, model=None)


def test_visual_provider_reuses_local_text_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "embodichain.gen_sim.task_engine.interpretation._load_local_env",
        lambda: {
            "OPENAI_API_KEY": "tp-test-key",
            "OPENAI_BASE_URL": "https://token-plan-cn.xiaomimimo.com/v1",
            "OPENAI_MODEL": "mimo-v2.6-pro",
        },
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert visual_grounding_available()


def test_visual_provider_reports_missing_text_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*, model):
        raise ValueError("no configured model")

    monkeypatch.setattr(
        "embodichain.gen_sim.task_engine.interpretation._load_llm_settings",
        missing,
    )
    assert not visual_grounding_available()


def test_visual_transport_sends_text_and_images_as_multimodal_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence(tmp_path)
    seen = {}

    class Client:
        def with_structured_output(self, _schema, *, method):
            assert method == "json_mode"

            def invoke(messages):
                seen["messages"] = messages
                return {"bindings": []}

            return SimpleNamespace(invoke=invoke)

    def client_factory(**kwargs):
        seen["settings"] = kwargs
        return Client()

    monkeypatch.setattr(
        "embodichain.gen_sim.task_engine.interpretation._load_local_env",
        lambda: {
            "OPENAI_API_KEY": "tp-test-key",
            "OPENAI_BASE_URL": "https://token-plan-cn.xiaomimimo.com/v1",
            "OPENAI_MODEL": "mimo-v2.6-pro",
        },
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", client_factory)
    images = tuple(Path(view["annotated_path"]) for view in evidence["views"])

    response = _default_visual_transport(
        prompt="Ground scene references",
        schema={"type": "object"},
        model="mimo-v2.6-pro",
        image_paths=images,
    )

    assert response == {"bindings": []}
    content = seen["messages"][1].content
    assert content[0]["type"] == "text"
    assert [item["type"] for item in content[1:]] == ["image_url", "image_url"]
    encoded = content[1]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == images[0].read_bytes()
    assert seen["settings"]["base_url"] == "https://token-plan-cn.xiaomimimo.com/v1"
