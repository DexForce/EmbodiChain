# GenSim model services

GenSim business code consumes capability contracts; adapters own provider HTTP,
authentication, response decoding, coordinate conventions and task polling.
The service layer imports no business engines, simulator or Blender. Importing
its contracts does not load Torch, a provider SDK or credentials. Configuration
uses the repository's `@configclass` and loads only when constructing a run.

The Sphinx [model-service guide](../../../docs/source/features/generative_sim/model_services.md)
provides the user-facing configuration and workflow overview.

## Existing installations

Existing service settings remain usable. When no external/user configuration
is selected and the bundled Asset Engine `configs/gen_config.json` has no
`model_services` section, each role retains its previous selection rules:

| Role | Legacy settings |
|---|---|
| Task interpretation / graph localization | Process environment, local GenSim `.env`, then `llm.openai_compatible`; a key and endpoint pair stays together. Model overrides and Task/Action aliases retain their order. |
| Scene understanding/editing, Scene orientation/dimensions, image grounding / semantic planning | Required fixed GenSim `.env` Scene VLM settings, including the historical body defaults. |
| SimReady asset orientation/dimensions/semantics/physics | Process `OPENAI_*` overrides the existing JSON `llm.openai_compatible` settings. |
| Image / segmentation / geometry | Existing `SCENE_ENGINE_*` endpoint, timeout, attempt and path keys in the fixed GenSim `.env`. |
| Articulated generation | `ASSET_ENGINE_ARTICULATED_GENERATION_*`, including the existing Scene-key aliases. |
| Gradio remote articulation jobs | Existing Gradio-resolved URL and timeout settings. |

Historical client classes, `from_dotenv()` and method signatures remain importable.
The old geometry method still returns wxyz response dictionaries; the new
reconstruction capability returns typed xyzw results. Business pipeline pose
conversion uses only that canonical result. Generation submission timeouts and
connection failures are intentionally no longer blindly retried: they report
`SubmissionUnknown` because an earlier submission may already be running.

## Ordinary wheel installs: external files

Set `EMBODICHAIN_MODEL_SERVICES_CONFIG` to a JSON file containing
`model_services` with `profiles` and `roles`. Set `EMBODICHAIN_ENV_FILE` separately
for referenced credentials. The existing CLI commands require no additional flags
and no installed files need editing.

Without the JSON selector, discovery checks
`$XDG_CONFIG_HOME/embodichain/model_services.json` (default
`~/.config/embodichain/model_services.json`) before the bundled JSON. Explicit
Python configuration overrides discovery. Each run selects a whole configuration;
files are not merged. A selected external file that is missing, unreadable or
invalid raises `ServiceConfigurationError`; a missing default user file simply
allows the bundled/legacy path. `load_services_config(path)` selects that exact
file and applies the same strict external-file checks. Paths expand `~`.

## Explicit role configuration

Copy the `model_services` section of [the example](model_services.example.json)
into the existing `gen_config.json`. This opts into explicit service selection;
it does not replace the existing ingest or mesh-processing sections. Alternatively,
pass `GenSimServicesCfg` from Python, or use `load_services_config(path)` to load
an existing-format JSON file. No new YAML or automatic cross-model fallback exists.

Profiles contain an adapter ID and optional model, URL, API key, timeout, attempt
budget and adapter options. Fields can use `{"env": "NAME"}`; `api_key_env` is
shorthand for an API-key reference. Only references are resolved: process values
win over the selected dotenv (`EMBODICHAIN_ENV_FILE` or the default GenSim `.env`).
Unrelated `OPENAI_*` values never fill missing fields in an explicit profile.
Unused roles require no credentials, connection or health check. The example's
optional job-control profile is only needed when using Gradio's remote panel.

Python configuration overrides the JSON section. A supplied legacy caller or
client overrides its role. The Python `model=` override applies to interpretation, image grounding and
semantic planning. The current Task Engine CLI reads GenSim configuration and
always stops at the unbound semantic graph, matching the upstream interface.
Structured chat preserves the upstream user-message boundary, sorted schema
prompt and JSON-mode parsing. Invalid model content remains eligible for the
business owner's existing bounded diagnostic repair; transport errors do not.

```python
from embodichain.gen_sim.services import GenSimServices, GenSimServicesCfg
from embodichain.gen_sim.task_engine import TaskAgent

cfg = GenSimServicesCfg(
    profiles={"text": {
        "adapter": "openai_compatible",
        "model": "my-model",
        "base_url": "https://my-service.example/v1",
        "api_key_env": "MY_MODEL_KEY",
    }},
    roles={"task_interpretation": "text"},
)
with GenSimServices(cfg) as services:
    candidates = TaskAgent(services=services).generate("example", "Pick up the cup")
    records = services.records()
```

## Implement an adapter

Implement the relevant Protocol; inheritance from shipped clients is unnecessary.
Chat receives ordered messages whose content interleaves `TextPart` and `ImagePart`,
so multi-view SimReady labels retain their relation to images. Return `ChatResponse`;
when available supply a structured mapping and numeric usage counters. Domain
owners remain responsible for JSON Schema, semantics and physical acceptance.

Geometry returns one `ReconstructedObject` per requested object, in request order,
with local mesh paths, Y-up metre positions, unit xyzw quaternions and separate
positive scales. A mesh-only adapter must compose a layout estimator; it must not
invent a zero pose. The service layer does not implement a new estimator.
Articulation returns a validated USDC and an optional request-evidence sidecar.
The custom `gen_sim:closedPosition` field is optional at admission. When present,
it must be a finite, non-boolean joint-limit endpoint; absence preserves unknown
open/closed semantics and does not prevent scene export. A task's open/closed
target binding must supply the missing semantics before resolving its targets.
The existing UI job-control API is a separate optional capability; it preserves
submit/status/cancel/download behavior without blocking the Gradio event stream.

```python
from embodichain.gen_sim.services import ServiceRegistry, ChatResponse

class MyChat:
    def complete(self, request):
        return ChatResponse(text='{"steps": []}', structured={"steps": []})
    def close(self):
        pass

registry = ServiceRegistry.defaults()
registry.register("my_chat", lambda profile: MyChat(), roles=("task_interpretation",))
# Bind that ID in a profile and pass registry=registry to GenSimServices.
```

Factories receive resolved profile mappings, including timeout/attempt defaults.
A factory must clean up resources if construction fails before returning. Adapters
must support concurrent calls; shipped Chat calls own their HTTP transport and
HTTP artifact adapters serialize access to their sessions. Health checks run only
on first use. Adapter-owned polling does not create an agent loop.

## Ownership, failures and diagnostics

Pass `services=` to TaskAgent, image planning, Scene top-level/staged APIs,
ParserManager or asset ingestion. A supplied run is borrowed. Otherwise an entry
owns a fresh run and closes it after all joined work, including failures. Injected
instances are borrowed and never closed by the run. Explicit legacy client
injection retains its prior ownership contract. Do not close a run during an
active call; join concurrent work first.

Use classified service errors for configuration, capability, transport, timeout,
malformed output and unknown submission. Adapter retries own transport budgets;
semantic repair stays with business code. There is no hidden SDK retry layer.
Once accepted, a generated job is polled without resubmitting. A timed-out or
connection-lost POST with unknown acceptance must not be automatically repeated.

`selections()` reports used role/adapter/model/source without URLs or keys.
`records()` adds operation, elapsed time, status, transport attempt count and
artifact paths; numeric usage is included only when supplied. No prompts or
provider error bodies are recorded. CLI runs write `service_calls.json` alongside
run outputs; Python users can explicitly call `write_records(path)`. Records must
remain outside sealed bundle directories. Remote Gradio job clients retain the
run until completion or successful cancellation, then release owned resources.

Run provider-free checks with `python -m pytest tests/gen_sim/services` and the
existing Task/Scene/Asset/Gradio regression groups. Live provider and simulator
qualification remain separate; offline tests do not establish generation quality.
