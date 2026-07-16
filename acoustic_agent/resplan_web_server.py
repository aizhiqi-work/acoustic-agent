from __future__ import annotations

import argparse
from functools import lru_cache
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .resplan import DEFAULT_RESPLAN_PATH, ResPlanDataset
from .web_server import AcousticWorkbenchHandler, WEB_ROOT, _warm_simulation_kernels


_SCENE_START = '      <section class="panelBlock editorBlock">'
_SCENE_END = '      <section class="panelBlock positionBlock">'
_RESPLAN_SCENE_SETUP = '''      <section class="panelBlock editorBlock resplanEditorBlock">
        <div class="inspectorHeader">
          <h2>ResPlan scene</h2>
          <span id="status">Loading</span>
        </div>

        <div class="resplanIndexControl">
          <button id="resplanPrev" class="iconButton" type="button" aria-label="Previous plan" title="Previous plan">&larr;</button>
          <label for="resplanIdx">Plan idx
            <input id="resplanIdx" type="number" min="0" step="1" value="0">
          </label>
          <button id="resplanNext" class="iconButton" type="button" aria-label="Next plan" title="Next plan">&rarr;</button>
          <button id="resplanRandom" type="button">Random</button>
        </div>

        <label class="presetSelect" for="resplanRoom">Room
          <select id="resplanRoom"></select>
        </label>

        <canvas id="resplanPlanCanvas" width="320" height="210" aria-label="Selected ResPlan floor plan"></canvas>
        <div id="resplanMeta" class="resplanMeta"></div>

        <div class="propertyGrid materialGrid">
          <label>Wall
            <select id="wallMaterial">
              <option value="wall">Brick</option>
              <option value="wood">Wood</option>
              <option value="curtain">Curtain</option>
            </select>
          </label>
          <label>Floor
            <select id="floorMaterial">
              <option value="floor">Marble</option>
              <option value="floor_carpet">Carpet</option>
              <option value="wood">Wood</option>
            </select>
          </label>
          <label>Ceiling
            <select id="ceilingMaterial">
              <option value="ceiling">Wood</option>
              <option value="ceiling_tile">Tile</option>
              <option value="fabric">Fabric</option>
            </select>
          </label>
        </div>

        <div class="propertyGrid resplanHeightControl">
          <label>Height (m)<input id="height" type="number" min="2" max="6" step="0.1" value="2.8"></label>
        </div>

        <div class="editorCommandRow">
          <button id="reset" type="button">Reset</button>
        </div>
      </section>

'''


class ResPlanWorkbenchHandler(AcousticWorkbenchHandler):
    dataset: ResPlanDataset

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/v1/resplan/scene":
            try:
                query = parse_qs(parsed.query)
                index = int(query.get("idx", ["0"])[0])
                room_id = query.get("room", [None])[0]
                height = float(query.get("height", ["2.8"])[0])
                self._send_json(self.dataset.scene(index, room_id, height_m=height))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/v1/resplan/index":
            try:
                query = parse_qs(parsed.query)
                index = int(query.get("idx", ["0"])[0])
                direction = query.get("direction", ["nearest"])[0]
                self._send_json({"index": self.dataset.resolve_index(index, direction)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/v1/resplan/stats":
            self._send_json(self.dataset.stats())
            return
        if parsed.path in {"/", "/viewer.html"}:
            self._send_html(_resplan_viewer_html())
            return
        super().do_GET()

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(data)


@lru_cache(maxsize=1)
def _resplan_viewer_html() -> str:
    html = (WEB_ROOT / "viewer.html").read_text(encoding="utf-8")
    start = html.index(_SCENE_START)
    end = html.index(_SCENE_END, start)
    html = html[:start] + _RESPLAN_SCENE_SETUP + html[end:]
    html = html.replace('<main id="app">', '<main id="app" data-scene-source="resplan">', 1)
    html = html.replace("AcousticAgent WebGL Workbench", "AcousticAgent ResPlan Workbench", 1)
    html = html.replace("Scene setup</p>", "ResPlan room</p>", 1)
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Acoustic Agent ResPlan workbench.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8766, type=int)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RESPLAN_PATH)
    args = parser.parse_args()
    ResPlanWorkbenchHandler.dataset = ResPlanDataset(args.dataset)
    _warm_simulation_kernels()
    server = ThreadingHTTPServer((args.host, args.port), ResPlanWorkbenchHandler)
    print(f"Acoustic Agent ResPlan workbench: http://{args.host}:{args.port}")
    stats = ResPlanWorkbenchHandler.dataset.stats()
    print(
        f"ResPlan scenes: {stats['eligible_records']} eligible / {stats['records']} total "
        f"from {ResPlanWorkbenchHandler.dataset.path}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
