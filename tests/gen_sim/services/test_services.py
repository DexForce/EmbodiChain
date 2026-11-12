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

"""Observable replacement, configuration, lifecycle and protocol contracts."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace
import pytest
from PIL import Image
from embodichain.gen_sim.services import (
    GenSimServices,
    GenSimServicesCfg,
    ServiceRegistry,
    ChatRequest,
    ChatResponse,
    ChatMessage,
    TextPart,
    ImagePart,
    ServiceConfigurationError,
    CapabilityError,
    SubmissionUnknown,
    GeometryRequest,
    ReconstructedObject,
    GeometryResult,
)
from embodichain.gen_sim.services.config import resolve_profile, load_services_config


class Chat:
    def __init__(self, text="ok"):
        self.text, self.closed = text, 0
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return ChatResponse(self.text)

    def close(self):
        self.closed += 1


def configured(adapter="test", roles=("task_interpretation",)):
    return GenSimServicesCfg(
        profiles={"chosen": {"adapter": adapter}},
        roles={role: "chosen" for role in roles},
    )


def test_profiles_and_concurrent_runs_are_isolated():
    created = []
    registry = ServiceRegistry()

    def factory(profile):
        client = Chat(profile["model"])
        created.append(client)
        return client

    registry.register("test", factory, roles=("task_interpretation",))
    cfg = configured()
    cfg.profiles["chosen"]["model"] = "first"
    with GenSimServices(cfg, registry=registry) as first:
        cfg.profiles["chosen"]["model"] = "second"
        with GenSimServices(cfg, registry=registry) as second:

            def complete(run):
                return run.get("task_interpretation").complete(ChatRequest(())).text

            with ThreadPoolExecutor(max_workers=4) as pool:
                assert list(pool.map(complete, [first, second, first, second])) == [
                    "first",
                    "second",
                    "first",
                    "second",
                ]
    assert len(created) == 2
    assert [client.closed for client in created] == [1, 1]


def test_borrowed_instances_bypass_configuration_and_are_not_closed():
    client = Chat()
    cfg = GenSimServicesCfg(
        profiles={
            "bad": {
                "adapter": "not_registered",
                "api_key_env": "MISSING_SERVICE_TEST_KEY",
            }
        },
        roles={"task_interpretation": "bad"},
    )
    with GenSimServices(cfg, injected={"task_interpretation": client}) as run:
        assert run.get("task_interpretation").complete(ChatRequest(())).text == "ok"
        assert run.selections()["task_interpretation"]["source"] == "injected"
    assert client.closed == 0


def test_partial_initialization_closes_owned_clients_on_failure():
    created = []
    registry = ServiceRegistry()

    def factory(profile):
        client = Chat()
        created.append(client)
        if profile.get("model") == "bad":

            def health():
                raise RuntimeError("unhealthy")

            client.check_health = health
        return client

    registry.register(
        "test", factory, roles=("task_interpretation", "scene_understanding")
    )
    cfg = configured(roles=("task_interpretation", "scene_understanding"))
    cfg.profiles["bad"] = {"adapter": "test", "model": "bad"}
    cfg.roles["scene_understanding"] = "bad"
    with pytest.raises(RuntimeError, match="unhealthy"):
        with GenSimServices(cfg, registry=registry) as run:
            run.get("task_interpretation")
            run.get("scene_understanding")
    assert [client.closed for client in created] == [1, 1]


def test_role_capability_is_checked_before_factory():
    registry = ServiceRegistry()
    registry.register(
        "test",
        lambda _: pytest.fail("must not construct"),
        roles=("task_interpretation",),
    )
    with GenSimServices(
        configured(roles=("geometry_reconstruction",)), registry=registry
    ) as run:
        with pytest.raises(CapabilityError):
            run.get("geometry_reconstruction")


def test_explicit_environment_references_are_atomic_and_lazy(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "TEST_SERVICE_KEY=file-key\nTEST_SERVICE_MODEL=file-model\nOPENAI_BASE_URL=http://unrelated\n"
    )
    monkeypatch.setenv("EMBODICHAIN_ENV_FILE", str(path))
    monkeypatch.setenv("TEST_SERVICE_KEY", "process-key")
    profile = {
        "adapter": "test",
        "api_key_env": "TEST_SERVICE_KEY",
        "model": {"env": "TEST_SERVICE_MODEL"},
    }
    resolved = resolve_profile(profile)
    assert resolved["api_key"] == "process-key"
    assert resolved["model"] == "file-model"
    assert "base_url" not in resolved
    assert profile["api_key_env"] == "TEST_SERVICE_KEY"
    with GenSimServices(
        GenSimServicesCfg(
            profiles={"unused": {"adapter": "test", "api_key_env": "MISSING_KEY"}},
            roles={},
        )
    ):
        pass


def test_legacy_resolution_does_not_mix_key_and_endpoint_sources(tmp_path, monkeypatch):
    from embodichain.gen_sim.services import legacy

    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_API_KEY=file-key\nOPENAI_BASE_URL=http://file/v1\nOPENAI_MODEL=file-model\nTASK_ENGINE_LLM_MODEL=task-model\n"
    )
    config = tmp_path / "config.json"
    config.write_text("{}")
    monkeypatch.setattr(legacy, "_GEN_SIM_ENV_PATH", path)
    monkeypatch.setattr(legacy, "_GEN_CONFIG_PATH", config)
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "LLM_URL",
        "TASK_ENGINE_LLM_MODEL",
        "ACTION_ENGINE_LLM_MODEL",
        "OPENAI_MODEL",
        "LLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "process-key-with-no-endpoint")
    settings = legacy._load_llm_settings(model=None)
    assert (settings["api_key"], settings["base_url"], settings["model"]) == (
        "file-key",
        "http://file/v1",
        "task-model",
    )
    monkeypatch.setenv("OPENAI_MODEL", "process-model")
    assert legacy.instruction_model(None) == "process-model"
    assert legacy.instruction_model("explicit") == "explicit"


def test_json_configuration_is_opt_in(tmp_path, monkeypatch):
    from embodichain.gen_sim.services import config

    monkeypatch.delenv("EMBODICHAIN_MODEL_SERVICES_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-home"))
    path = tmp_path / "gen_config.json"
    path.write_text(json.dumps({"llm": {}}))
    monkeypatch.setattr(config, "DEFAULT_JSON", path)
    assert load_services_config() is None
    path.write_text(
        json.dumps(
            {
                "model_services": {
                    "profiles": {"one": {"adapter": "test"}},
                    "roles": {"task_interpretation": "one"},
                }
            }
        )
    )
    assert load_services_config(path).roles == {"task_interpretation": "one"}


def test_multimodal_order_and_provider_options_stay_in_adapter(tmp_path, monkeypatch):
    from embodichain.gen_sim.services.adapters import chat

    image = tmp_path / "view.png"
    Image.new("RGB", (2, 3)).save(image)
    captured = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

    def urlopen(request, **kwargs):
        captured.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(chat, "urlopen", urlopen)
    adapter = chat.ChatCompletionsAdapter(
        model="mimo-test", api_key="secret", base_url="http://test/v1"
    )
    adapter.complete(
        ChatRequest(
            (
                ChatMessage(
                    "user",
                    (
                        TextPart("front"),
                        ImagePart(image),
                        TextPart("back"),
                        ImagePart(image),
                    ),
                ),
            )
        )
    )
    parts = captured[0]["messages"][0]["content"]
    assert [part["type"] for part in parts] == [
        "text",
        "image_url",
        "text",
        "image_url",
    ]
    assert [parts[0]["text"], parts[2]["text"]] == ["front", "back"]
    assert captured[0]["thinking"] == {"type": "disabled"}


def test_openai_compatible_chat_can_omit_authentication(monkeypatch):
    from embodichain.gen_sim.services.adapters import chat

    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def urlopen(request, **_):
        requests.append(request)
        return Response()

    monkeypatch.setattr(chat, "urlopen", urlopen)
    cfg = GenSimServicesCfg(
        profiles={
            "local": {
                "adapter": "openai_compatible",
                "model": "local-model",
                "base_url": "http://localhost/v1",
            }
        },
        roles={"task_interpretation": "local"},
    )
    with GenSimServices(cfg) as services:
        result = services.get("task_interpretation").complete(ChatRequest(()))
    assert result.text == "ok"
    assert requests[0].get_header("Authorization") is None


def test_selected_adapter_entry_point_is_loaded_lazily(monkeypatch):
    from embodichain.gen_sim.services import runtime

    loaded = []

    class EntryPoint:
        def __init__(self, name):
            self.name = name

        def load(self):
            loaded.append(self.name)
            if self.name == "unused":
                pytest.fail("unselected adapter must not be imported")

            def register(registry):
                registry.register(
                    "plugin_chat",
                    lambda _: Chat("plugin"),
                    roles=("task_interpretation",),
                )

            return register

    monkeypatch.setattr(
        runtime,
        "entry_points",
        lambda **_: [EntryPoint("plugin_chat"), EntryPoint("unused")],
    )
    with GenSimServices(configured("plugin_chat")) as services:
        result = services.get("task_interpretation").complete(ChatRequest(()))
    assert result.text == "plugin"
    assert loaded == ["plugin_chat"]


def test_records_count_transport_attempts_without_leaking_credentials(monkeypatch):
    from urllib.error import HTTPError
    from embodichain.gen_sim.services.adapters import chat

    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def urlopen(request, **kwargs):
        calls.append(request)
        if len(calls) == 1:
            raise HTTPError("http://private", 503, "SECRET", {}, None)
        return Response()

    monkeypatch.setattr(chat, "urlopen", urlopen)
    cfg = GenSimServicesCfg(
        profiles={
            "chat": {
                "adapter": "openai_compatible",
                "model": "model",
                "api_key": "SECRET",
                "base_url": "http://private",
                "max_attempts": 2,
            }
        },
        roles={"task_interpretation": "chat"},
    )
    with GenSimServices(cfg) as run:
        run.get("task_interpretation").complete(ChatRequest(()))
        records = run.records()
    assert len(calls) == 2
    assert records[0]["attempts"] == 2
    assert "SECRET" not in json.dumps(records)
    assert "http://private" not in json.dumps(records)


@pytest.mark.parametrize("kind", ["image", "geometry", "articulation"])
def test_uncertain_generation_submissions_are_never_retried(tmp_path, kind):
    import requests
    from embodichain.gen_sim.services.adapters.normalized import (
        ImageAdapter,
        GeometryAdapter,
        ArticulationAdapter,
    )

    image = tmp_path / "input.png"
    Image.new("RGB", (2, 2)).save(image)

    class Session:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            raise requests.Timeout("credential or provider details")

        def close(self):
            pass

    session = Session()
    kwargs = dict(
        base_url="http://test",
        timeout_s=1,
        max_attempts=3,
        health_path="/health",
        session=session,
    )
    if kind == "image":
        client = ImageAdapter(**kwargs, generate_image_by_prompt_path="/generate")
        invoke = lambda: client.generate_image_by_prompt(
            prompt="test", output_path=tmp_path / "output.png"
        )
    elif kind == "geometry":
        client = GeometryAdapter(**kwargs, generate_objects_path="/generate")
        invoke = lambda: client.generate_objects(
            image_path=image, object_masks=[("one", image)], output_root=tmp_path
        )
    else:
        client = ArticulationAdapter(**kwargs, generate_path="/generate")
        invoke = lambda: client.generate_articulated_object(
            prompt="test", image_path=image
        )
    with pytest.raises(SubmissionUnknown):
        invoke()
    assert session.calls == 1


def test_contract_import_has_no_configuration_or_provider_side_effects():
    script = "from embodichain.gen_sim.services import ChatRequest, ChatService; import sys; assert not any(name in sys.modules for name in ('torch', 'openai', 'requests', 'bpy', 'embodichain.lab.sim'))"
    result = subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


def test_services_never_import_business_engines():
    import ast

    root = Path(__file__).parents[3] / "embodichain/gen_sim/services"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    (
                        "embodichain.gen_sim.task_engine",
                        "embodichain.gen_sim.scene_engine",
                        "embodichain.gen_sim.asset_engine",
                        "embodichain.lab",
                        "bpy",
                    )
                ), path


def test_registered_chat_runs_image_task_without_provider_credentials(tmp_path):
    from embodichain.gen_sim.task_engine.image_planning import (
        generate_semantic_task_graph_from_image,
    )

    fixture = json.loads(
        (
            Path(__file__).parents[1] / "task_engine/data/semantic_graph_contract.json"
        ).read_text()
    )
    step = fixture["candidate"]["draft"]["steps"][0]
    step.update(task_type="E1", relation="inside", orientation_goal="none")
    step["object"]["reference"] = "red cup"
    step["target"] = dict(step["object"], reference="tray")
    intent = {"steps": [step]}
    grounding = {
        "objects": [
            {"id": "red_cup", "description": "red cup", "bbox": [200, 400, 400, 800]},
            {"id": "tray", "description": "tray", "bbox": [500, 200, 900, 600]},
        ],
        "bindings": {"step_01.object": ["red_cup"], "step_01.target": ["tray"]},
    }

    for entity in grounding["objects"]:
        entity["provenance"] = "observed"
    plan = {
        "nodes": [
            {
                "id": "pick",
                "step_id": "step_01",
                "call": {"kind": "pick", "object": "red_cup", "resources": {}},
                "depends_on": [],
                "semantic": {
                    "object": "red_cup",
                    "reference": "tray",
                    "relation": "inside",
                    "intent": {"relation": "inside"},
                },
            },
            {
                "id": "place",
                "step_id": "step_01",
                "call": {
                    "kind": "place",
                    "object": "red_cup",
                    "inside": "cup_in_tray",
                    "resources": {},
                },
                "depends_on": ["pick"],
                "semantic": {
                    "object": "red_cup",
                    "reference": "tray",
                    "relation": "inside",
                    "intent": {"relation": "inside"},
                },
            },
        ],
        "requirements": [
            {
                "id": "placement",
                "quote": "place the red cup in the tray",
                "node_ids": ["pick", "place"],
            }
        ],
    }

    class IndependentChat:
        def complete(self, request):
            value = (
                intent
                if request.schema is not None
                else (
                    grounding
                    if any(
                        isinstance(part, ImagePart)
                        for message in request.messages
                        for part in message.content
                    )
                    else plan
                )
            )
            return ChatResponse(
                json.dumps(value),
                structured=value if request.schema is not None else None,
            )

    registry = ServiceRegistry()
    registry.register(
        "independent",
        lambda _: IndependentChat(),
        roles=("task_interpretation", "task_image_grounding", "task_semantic_planning"),
    )
    image = tmp_path / "image.png"
    Image.new("RGB", (100, 80)).save(image)
    cfg = configured(
        "independent",
        roles=("task_interpretation", "task_image_grounding", "task_semantic_planning"),
    )
    with GenSimServices(cfg, registry=registry) as run:
        result = generate_semantic_task_graph_from_image(
            image, "place the red cup in the tray", services=run
        )
        assert result["image_grounding"]["bindings"] == grounding["bindings"]
        assert result["semantic_task_graph"]["nodes"][1]["call"]["kind"] == "place"
        assert {record["role"] for record in run.records()} == {
            "task_interpretation",
            "task_image_grounding",
            "task_semantic_planning",
        }


def test_registered_services_run_scene_stages_with_canonical_geometry(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.scene_engine.pipeline import api
    from embodichain.gen_sim.scene_engine.core.scene import Scene
    from embodichain.gen_sim.scene_engine.core.scene_object import SceneObject
    from embodichain.gen_sim.scene_engine.core.scene_graph import (
        SceneGraph,
        SceneGraphNode,
    )
    from embodichain.gen_sim.services import SegmentationResult

    image = tmp_path / "image.png"
    Image.new("RGB", (10, 8)).save(image)
    scene = Scene(
        objects=[
            SceneObject(
                id="table",
                kind="table",
                category="table",
                name="table",
                description="table",
            )
        ]
    )
    graph = SceneGraph(nodes=[SceneGraphNode(object_id="table", parent_id=None)])

    class IndependentGeometry:
        def reconstruct(self, request):
            path = request.output_root / "table.glb"
            path.write_bytes(b"glTF-test")
            return GeometryResult(
                (
                    ReconstructedObject(
                        "table",
                        path,
                        (1.0, 2.0, 3.0),
                        (0.0, 0.0, 0.0, 1.0),
                        (1.0, 1.0, 1.0),
                    ),
                )
            )

    class IndependentSegmentation:
        def segment(self, request):
            return SegmentationResult((), (10, 8))

    registry = ServiceRegistry()
    registry.register("chat", lambda _: Chat(), roles=("scene_understanding",))
    registry.register(
        "geometry", lambda _: IndependentGeometry(), roles=("geometry_reconstruction",)
    )
    registry.register(
        "segmentation",
        lambda _: IndependentSegmentation(),
        roles=("image_segmentation",),
    )
    cfg = GenSimServicesCfg(
        profiles={
            name: {"adapter": name} for name in ("chat", "geometry", "segmentation")
        },
        roles={
            "scene_understanding": "chat",
            "geometry_reconstruction": "geometry",
            "image_segmentation": "segmentation",
        },
    )

    def understand(**kwargs):
        assert (
            kwargs["vlm_client"].complete(
                system_prompt="scene", user_prompt="describe", image_path=image
            )
            == "ok"
        )
        assert (
            kwargs["image_segmentation_client"].segment_single_object(
                image_path=image, prompt="table"
            )
            == []
        )
        return scene, graph

    def refine(**kwargs):
        result = kwargs["geometry_generation_client"].reconstruct(
            GeometryRequest(image, (("table", image),), tmp_path)
        )
        assert result.objects[0].quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)
        assert result.objects[0].position == (1.0, 2.0, 3.0)
        return kwargs["scene"]

    monkeypatch.setattr(api, "understand_scene", understand)
    monkeypatch.setattr(api, "generate_scene_and_refine", refine)
    monkeypatch.setattr(
        api, "_export_materialization", lambda **kwargs: kwargs["scene"]
    )
    with GenSimServices(cfg, registry=registry) as run:
        blueprint = api.analyze_image(image, tmp_path, services=run)
        original = blueprint.scene.to_dict()
        api.materialize_blueprint(blueprint, services=run)
        assert blueprint.scene.to_dict() == original
        assert {record["role"] for record in run.records()} == {
            "scene_understanding",
            "geometry_reconstruction",
            "image_segmentation",
        }


def test_legacy_public_client_uses_same_transport_and_response_contract(tmp_path):
    from embodichain.gen_sim.scene_engine.clients.geometry_generation import (
        GeometryGenerationClient,
    )

    class Response:
        content = b"glTF-test"

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "ok": True,
                "result": {
                    "objects": [
                        {
                            "name": "one",
                            "mesh": "/one.glb",
                            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            "translation": [1.0, 2.0, 3.0],
                            "scale": [2.0, 2.0, 2.0],
                        }
                    ]
                },
            }

    class Session:
        def post(self, *args, **kwargs):
            return Response()

        def get(self, *args, **kwargs):
            return Response()

        def close(self):
            pass

    image = tmp_path / "input.png"
    Image.new("RGB", (2, 2)).save(image)
    client = GeometryGenerationClient(
        base_url="http://test",
        timeout_s=1,
        max_attempts=1,
        health_path="/health",
        generate_objects_path="/objects",
        session=Session(),
    )
    request = GeometryRequest(image, (("one", image),), tmp_path)
    result = client.reconstruct(request)
    assert result.objects[0].quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert result.objects[0].position == (1.0, 2.0, 3.0)
    _, original = client.generate_objects(
        image_path=image, object_masks=[("one", image)], output_root=tmp_path
    )
    assert original[0]["rotation_quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0]


def test_closed_run_rejects_use_and_cannot_write_into_sealed_bundle(tmp_path):
    client = Chat()
    run = GenSimServices(configured(), injected={"task_interpretation": client})
    service = run.get("task_interpretation")
    run.close()
    from embodichain.gen_sim.services import ServiceError

    with pytest.raises(ServiceError, match="closed"):
        service.complete(ChatRequest(()))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "integration_fingerprint.json").write_text("{}")
    with pytest.raises(ServiceConfigurationError, match="sealed"):
        run.write_records(bundle / "service_calls.json")


@pytest.mark.parametrize("outcome", ["poll_failure", "malformed_response"])
def test_accepted_geometry_job_is_not_resubmitted(tmp_path, outcome):
    from embodichain.gen_sim.services.adapters.normalized import GeometryAdapter
    from embodichain.gen_sim.services import ServiceResponseError

    image = tmp_path / "input.png"
    Image.new("RGB", (2, 2)).save(image)

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return (
                {"status": "waiting", "request_id": "accepted"}
                if outcome == "poll_failure"
                else {"unexpected": True}
            )

    class Session:
        posts = 0

        def post(self, *args, **kwargs):
            self.posts += 1
            return Response()

        def get(self, *args, **kwargs):
            raise ValueError("malformed task JSON")

        def close(self):
            pass

    session = Session()
    client = GeometryAdapter(
        base_url="http://test",
        timeout_s=1,
        max_attempts=4,
        health_path="/health",
        generate_objects_path="/objects",
        session=session,
    )
    with pytest.raises(ServiceResponseError):
        client.reconstruct(GeometryRequest(image, (("one", image),), tmp_path))
    assert session.posts == 1


def test_legacy_run_and_simready_imports_do_not_initialize_clients():
    script = "from embodichain.gen_sim.services import GenSimServices; import sys; run = GenSimServices(); run.close(); assert 'torch' not in sys.modules; import embodichain.gen_sim.asset_engine.utils.simready_utils as module; assert not hasattr(module, 'client'); assert not hasattr(module, 'DEPLOYMENT')"
    result = subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


def test_shared_owned_instance_closes_once_and_model_lookup_needs_no_credentials(
    monkeypatch,
):
    shared = Chat()
    registry = ServiceRegistry()
    registry.register(
        "shared", lambda _: shared, roles=("task_interpretation", "graph_localization")
    )
    cfg = configured("shared", roles=("task_interpretation", "graph_localization"))
    with GenSimServices(cfg, registry=registry) as run:
        run.get("task_interpretation")
        run.get("graph_localization")
    assert shared.closed == 1
    monkeypatch.setenv("MODEL_ONLY_TEST", "selected-model")
    cfg.profiles["chosen"].update(
        model={"env": "MODEL_ONLY_TEST"}, api_key_env="MISSING_KEY"
    )
    with GenSimServices(cfg, registry=registry) as run:
        assert run.model("task_interpretation") == "selected-model"


@pytest.mark.parametrize("selection", ["env", "xdg", "home", "explicit"])
def test_external_discovery_precedence_and_python_override(
    tmp_path, monkeypatch, selection
):
    from embodichain.gen_sim.services import config

    monkeypatch.delenv("EMBODICHAIN_MODEL_SERVICES_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    def write(path, model):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model_services": {
                        "profiles": {"one": {"adapter": "test", "model": model}},
                        "roles": {"task_interpretation": "one"},
                    }
                }
            )
        )
        return path

    bundled = write(tmp_path / "bundled.json", "bundled")
    monkeypatch.setattr(config, "DEFAULT_JSON", bundled)
    user = write(tmp_path / "xdg/embodichain/model_services.json", "xdg")
    write(tmp_path / "home/.config/embodichain/model_services.json", "home")
    explicit = write(tmp_path / "explicit.json", "explicit")
    if selection in {"env", "explicit"}:
        monkeypatch.setenv("EMBODICHAIN_MODEL_SERVICES_CONFIG", str(explicit))
    elif selection == "home":
        monkeypatch.delenv("XDG_CONFIG_HOME")
    with GenSimServices() as run:
        assert (
            run.model("task_interpretation")
            == {
                "env": "explicit",
                "explicit": "explicit",
                "xdg": "xdg",
                "home": "home",
            }[selection]
        )
    if selection == "explicit":
        assert load_services_config(bundled).profiles["one"]["model"] == "bundled"
    if selection == "explicit":
        from ..task_engine.test_image_planning import _fixture
        from embodichain.gen_sim.task_engine import TaskAgent

        step, _, _ = _fixture()
        monkeypatch.setenv(
            "EMBODICHAIN_MODEL_SERVICES_CONFIG", str(tmp_path / "missing")
        )
        with GenSimServices(
            configured(), injected={"task_interpretation": Chat()}
        ) as run:
            assert run.get("task_interpretation").complete(ChatRequest(())).text == "ok"
            result = TaskAgent(
                services=run, caller=lambda **_: {"steps": [step]}
            ).generate("priority", "执行测试任务", candidate_count=1)
            assert (
                result["candidates"][0]["draft"]["steps"][0]["object"]["reference"]
                == "source"
            )


@pytest.mark.parametrize(
    "content", [None, "not json", "[]", "{}", '{"model_services":null}']
)
def test_selected_external_config_fails_without_provider_fallback(
    tmp_path, monkeypatch, content
):
    path = tmp_path / "selected.json"
    if content is not None:
        path.write_text(content)
    monkeypatch.setenv("EMBODICHAIN_MODEL_SERVICES_CONFIG", str(path))
    with pytest.raises(ServiceConfigurationError):
        GenSimServices()


def test_invalid_default_user_file_is_not_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("EMBODICHAIN_MODEL_SERVICES_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    p = tmp_path / "embodichain/model_services.json"
    p.parent.mkdir()
    p.write_text("{}")
    with pytest.raises(ServiceConfigurationError):
        load_services_config()
