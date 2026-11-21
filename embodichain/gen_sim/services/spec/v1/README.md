# GenSim model-service protocol v1

This protocol lets operators deploy any model or serving framework behind the
stable capability boundary consumed by GenSim. It specifies wire behavior, not
Docker images, ports, GPU allocation, model names or scheduling infrastructure.

The built-in adapter ID selects the protocol. A breaking wire change requires a
new adapter ID or an installed adapter plugin; it must not silently change v1.

| Capability | Adapter ID | Required operation |
|---|---|---|
| OpenAI-compatible chat/VLM | `openai_compatible` | Chat Completions |
| Prompt-to-image | `image_http` | JSON prompt to PNG |
| Prompted segmentation | `segmentation_http` | Image and prompt to RLE masks |
| Geometry reconstruction | `geometry_http` | Image and masks to GLB meshes and poses |
| Articulated generation | `articulation_http` | Prompt/image to asynchronous USDC artifact |
| Articulation UI jobs | `articulation_jobs_http` | Submit, status, cancel and download |

Endpoint paths are profile options. Paths shown below are the conventional v1
values and match the existing SAM3, SAM3D and Z-Image deployment.

## Common rules

- Base URLs must use HTTP or HTTPS. Production deployments should use TLS or a
  trusted private network.
- A configured `api_key` is sent as `Authorization: Bearer <value>`. Omit it for
  a service that does not authenticate.
- Health endpoints return HTTP 200. Image, geometry and articulation health
  payloads contain `{"ok": true}`.
- Relative artifact URLs resolve against the configured service base URL.
- Non-2xx responses are transport failures. JSON operations may additionally
  return `{"ok": false, "error": "safe summary"}`.
- A POST timeout or lost connection can mean that a job was accepted. Clients do
  not blindly resubmit generation requests.
- Service logs and errors must not expose credentials or complete private input.

## OpenAI-compatible chat and VLM

The adapter sends `POST <base_url>/chat/completions`, unless `base_url` already
ends in `/chat/completions`. The endpoint follows the OpenAI Chat Completions
shape:

- ordered `messages` with text or `image_url` data-URL parts;
- optional `response_format: {"type": "json_object"}`;
- `stream: false` for structured requests;
- response text at `choices[0].message.content`.

Image-grounding and scene/asset vision roles require data-URL image support.
Text-only models should be bound only to roles that never receive `ImagePart`.

## Image generation

Conventional paths:

```text
GET  /health
POST /generate_image_by_prompt
```

Request:

```http
Content-Type: application/json

{"prompt": "a red cup on a white background"}
```

The response is HTTP 200 with `Content-Type: image/png` and raw PNG bytes.

## Image segmentation

Conventional paths:

```text
GET  /health
POST /segment_by_prompt
```

The inference request is `multipart/form-data`:

| Field | Type | Meaning |
|---|---|---|
| `image` | one PNG or JPEG file | Source image |
| `prompt` | text | Concept to segment |

Preferred response:

```json
{
  "ok": true,
  "result": {
    "masks": [
      {"size": [480, 640], "counts": [10, 4, 20]}
    ]
  }
}
```

Each mask is a COCO-style RLE mapping. `size` is `[height, width]`. For legacy
compatibility, the client also accepts `data.masks` and `instances[*].mask_rle`,
`mask` or `segmentation`.

## Geometry reconstruction

Conventional paths:

```text
GET  /health
POST /generate_multiple_objects
GET  /tasks/<request_id>
GET  /assets/<request_id>/<filename>
```

The POST is `multipart/form-data` with one `image` and one or more repeated
`masks` files. Each mask filename stem is the requested object ID and must be
unique and safe as a filename.

The service may respond immediately:

```json
{
  "ok": true,
  "request_id": "job-id",
  "result": {
    "objects": [
      {
        "name": "cup",
        "mesh": "/assets/job-id/cup.glb",
        "pose_coordinate_system": "glb_y_up",
        "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "translation": [0.1, 0.2, 0.3],
        "scale": [1.0, 1.0, 1.0]
      }
    ]
  }
}
```

Alternatively it returns `status` as `queued`, `running` or a string containing
`waiting`, plus a non-empty `request_id`. The client polls
`GET /tasks/<request_id>` until `succeeded`, `failed` or `cancelled`.

Objects remain in request order and their names equal the mask IDs. Mesh URLs
return binary GLB content beginning with the `glTF` magic bytes. Translation is
in metres in the editable GLB Y-up frame. The wire quaternion is explicitly
wxyz; the adapter converts it once to GenSim's canonical xyzw contract. Scale is
a separate positive XYZ vector.

## Articulated generation

Conventional paths:

```text
GET  /health
POST /generate_articulation
GET  /tasks/<request_id>
GET  <result.artifacts.usdc>
```

The POST accepts either `{"prompt": "..."}` JSON or multipart `prompt` plus an
optional `image`. It returns a non-empty `request_id`. Status values are
`queued`, `running`, `waiting`, `succeeded`, `failed` or `cancelled`.

A successful response provides a relative USDC URL:

```json
{
  "status": "succeeded",
  "result": {"artifacts": {"usdc": "/artifacts/job-id/asset.usdc"}}
}
```

The client downloads and validates the stage. It must contain an articulation
root and at least one supported enabled movable joint with valid bodies, axis and
limits. A prismatic joint also declares `gen_sim:closedPosition` at one limit.

## Optional articulation job control

The Gradio job-control adapter additionally uses:

```text
POST /generate_articulation
GET  /tasks/<request_id>
POST /tasks/<request_id>/cancel
GET  <result.artifacts[artifact]>
```

This UI capability is separate from the synchronous pipeline-facing
`articulated_generation` role.

## Installed adapter plugins

A service with a different wire API can provide an installed adapter instead of
changing its server. Register one entry point whose name equals the profile's
`adapter` ID:

```toml
[project.entry-points."embodichain.gensim_service_adapters"]
my_geometry = "my_package.gensim:register"
```

The loaded callable receives a run-local `ServiceRegistry` and registers the
same ID for one or more roles:

```python
def register(registry):
    registry.register(
        "my_geometry",
        lambda profile: MyGeometryAdapter(profile),
        roles=("geometry_reconstruction",),
    )
```

Only entry points selected by a bound profile are imported. Arbitrary module
paths in JSON are not supported. Plugin adapters implement the Protocol for the
selected role, are concurrent-call safe, and expose `close()` and optionally
`check_health()` when they own resources or health behavior.

## Validation

Use the installed CLI before running a business workflow:

```bash
embodichain gensim services validate --config model_services.json
embodichain gensim services doctor --config model_services.json
```

`validate` resolves selected environment references, checks capabilities and
constructs/closes adapters without provider requests. Plugin factories therefore
keep network checks in `check_health()`. `doctor` initializes each selected role
and invokes `check_health()` when its adapter provides one. A health pass proves
connectivity and wire readiness, not model quality or simulation success.
