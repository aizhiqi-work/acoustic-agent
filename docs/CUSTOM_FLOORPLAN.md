# Custom Floorplans

The Custom workflow turns a compact room program or an editable metric JSON
specification into the same multi-room geometry used by indexed Floorplan
scenes. It is intentionally usable without an OpenAI, GPT, or VLM API key.

## What Works Without An API

| Workflow | API key | Behavior |
| --- | --- | --- |
| Text to floor plan (Python/HTTP) | No | Deterministic local partitioning, doors, windows, validation, and RIR solving |
| JSON editing | No | Edit, validate, compile, export, and solve a complete floor plan |
| ChatGPT image handoff | No | Copy the image prompt, attach the image to ChatGPT, and paste the returned JSON |
| ChatGPT text handoff | No | Describe the desired home, copy the generated prompt to ChatGPT, and paste the returned JSON |
| Automatic server-side VLM | Optional | Not enabled in this release; a provider must be added explicitly |

The workbench never claims that an image was understood when no VLM provider is
configured. Uploading an image does not send it to the Acoustic Agent server.

## Start And Test

```bash
acoustic-agent web
```

Open <http://127.0.0.1:8765/custom>. The workbench opens with a lightweight
template but does not run the expensive RIR solver. Replace it with returned
JSON, select rooms, and press **Run static simulation** when the scene is ready.

For either input with no API key:

1. Choose **Floor-plan image** and select a clear image, or choose **Text description** and describe the desired home.
2. Click **Copy ChatGPT prompt**.
3. For image input, attach the same image with the copied prompt in ChatGPT. The text prompt already contains the description.
4. Paste the returned JSON into **Floorplan JSON**.
5. Click **Apply floor plan** and inspect validation.
6. Calibrate Width or Depth; the other dimension follows with uniform scaling.
7. Set Height, select rooms, then run the simulation.

The JSON remains the review boundary: a generated or VLM-derived plan must pass
the same geometry checks before it can reach the solver.

After applying JSON, choose a furnishing compactness and press **Auto place**.
The layout uses room semantics and verified door/window geometry, remains
deterministic for the selected seed, and can still be edited manually.

## Python API

```python
from acoustic_agent import AcousticAgent, FloorplanBuilder

spec = FloorplanBuilder.from_text(
    "10m x 8m，两室一厅一厨一卫",
    seed=42,
)
report = FloorplanBuilder.validate(spec)
assert report["valid"], report["errors"]

agent = AcousticAgent.from_floorplan_spec(
    spec,
    source_room="living_0",
    receiver_room="bedroom_1",
    quality="preview",
    fs=16000,
    duration_s=1.0,
)
rir = agent.run().rir
```

Run the packaged example with:

```bash
python examples/custom_floorplan.py
```

## JSON Contract

Coordinates use meters in an XY floor plane. A minimal specification contains:

```json
{
  "schema_version": 1,
  "title": "Two-room example",
  "units": "m",
  "coordinate_system": "image_top_left",
  "height_m": 2.8,
  "wall_depth_m": 0.12,
  "outer_boundary": [[0, 0], [6, 0], [6, 4], [0, 4]],
  "rooms": [
    {"id": "living_0", "type": "living", "corners": [[0, 0], [3, 0], [3, 4], [0, 4]]},
    {"id": "bedroom_0", "type": "bedroom", "corners": [[3, 0], [6, 0], [6, 4], [3, 4]]}
  ],
  "openings": [
    {"id": "door_0", "type": "door", "room_ids": ["living_0", "bedroom_0"], "segment": [[3, 1.5], [3, 2.4]], "height_m": 2.1, "sill_height_m": 0, "connection": "interior_room", "open": true, "confidence": 1.0}
  ],
  "provenance": {"source": "manual"}
}
```

Supported room types are `living`, `kitchen`, `bedroom`, `bathroom`, `storage`,
and `balcony`. Supported opening types are `door`, `window`, and `opening`.
Interior open connections reference two rooms; exterior entries and facade
windows reference one room.

Validation rejects overlapping or uncovered rooms, out-of-bound polygons,
misaligned opening segments, unknown room references, and disconnected indoor
door graphs. This protects both the WebGL model and acoustic portal solver.
Image-derived JSON uses a top-left origin with X right and Y down, matching the
uploaded bitmap, minimap, and top view. Explicit legacy
`cartesian_bottom_left` JSON is vertically normalized during validation.

## HTTP API

Generate locally:

```bash
curl -X POST http://127.0.0.1:8765/api/v1/custom/generate \
  -H 'content-type: application/json' \
  --data '{"description":"10m x 8m，两室一厅一厨一卫","seed":42}'
```

Validate or compile edited JSON with `POST /api/v1/custom/validate` and
`POST /api/v1/custom/compile`, using `{"spec": {...}}`. Read provider
capabilities at `GET /api/v1/custom/capabilities`. Read the image extraction
contract at `GET /api/v1/custom/prompt?mode=image`, or create a text generation
prompt with `GET /api/v1/custom/prompt?mode=text&description=...`.

The local HTTP server has no authentication or TLS. Do not expose it publicly
without adding authentication, request limits, and a secure reverse proxy.
