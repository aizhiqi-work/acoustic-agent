from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .resplan import DEFAULT_RESPLAN_PATH, ResPlanDataset
from .resplan_resource import DEFAULT_RESPLAN_RESOURCE, ResPlanResource
from .web_server import AcousticWorkbenchHandler, WEB_ROOT, _warm_simulation_kernels


_SCENE_START = '      <section id="sceneSection" class="panelBlock editorBlock setupSection">'
_SCENE_END = '      <section id="materialsSection" class="panelBlock materialBlock setupSection">'
_RESPLAN_SCENE_SETUP = '''      <section id="sceneSection" class="panelBlock editorBlock resplanEditorBlock setupSection">
        <div class="inspectorHeader">
          <h2>ResPlan scene</h2>
        </div>

        <div class="resplanIndexControl">
          <button id="resplanPrev" class="iconButton" type="button" aria-label="Previous plan" title="Previous plan">&larr;</button>
          <label for="resplanIdx">Plan idx
            <input id="resplanIdx" type="number" min="0" step="1" value="0">
          </label>
          <button id="resplanNext" class="iconButton" type="button" aria-label="Next plan" title="Next plan">&rarr;</button>
          <button id="resplanRandom" type="button">Random</button>
        </div>

        <label class="presetSelect" for="resplanRoom">Source room
          <select id="resplanRoom"></select>
        </label>

        <label class="presetSelect" for="resplanReceiverRoom">Microphone room
          <select id="resplanReceiverRoom"></select>
        </label>

        <canvas id="resplanPlanCanvas" width="320" height="210" aria-label="Selected ResPlan floor plan"></canvas>
        <div id="resplanMeta" class="resplanMeta"></div>

        <div class="propertyGrid resplanHeightControl">
          <label>Height (m)<input id="height" type="number" min="2" max="6" step="0.1" value="2.8"></label>
        </div>

        <div class="editorCommandRow">
          <button id="reset" type="button">Reset</button>
        </div>
      </section>

'''


class ResPlanWorkbenchHandler(AcousticWorkbenchHandler):
    dataset: Any

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/v1/resplan/scene":
            try:
                query = parse_qs(parsed.query)
                index = int(query.get("idx", ["0"])[0])
                room_id = query.get("room", [None])[0]
                receiver_room_id = query.get("receiver_room", [None])[0]
                height = float(query.get("height", ["2.8"])[0])
                self._send_json(
                    self.dataset.scene(
                        index,
                        room_id,
                        receiver_room_id=receiver_room_id,
                        height_m=height,
                    )
                )
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


def _resplan_viewer_html() -> str:
    html = (WEB_ROOT / "viewer.html").read_text(encoding="utf-8")
    start = html.index(_SCENE_START)
    end = html.index(_SCENE_END, start)
    html = html[:start] + _RESPLAN_SCENE_SETUP + html[end:]
    html = html.replace('data-scene-source="geometry"', 'data-scene-source="resplan"', 1)
    html = html.replace("AcousticAgent WebGL Workbench", "AcousticAgent ResPlan Workbench", 1)
    html = html.replace("Scene setup</p>", "ResPlan room</p>", 1)
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Acoustic Agent ResPlan workbench.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8766, type=int)
    parser.add_argument(
        "--resource",
        type=Path,
        default=DEFAULT_RESPLAN_RESOURCE,
        help="Compiled ResPlan SQLite resource. Used by default.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=f"Legacy raw ResPlan pickle path. Defaults to {DEFAULT_RESPLAN_PATH}.",
    )
    args = parser.parse_args()
    if args.dataset is not None:
        ResPlanWorkbenchHandler.dataset = ResPlanDataset(args.dataset)
    else:
        ResPlanWorkbenchHandler.dataset = ResPlanResource(args.resource)
    _warm_simulation_kernels()
    server = ThreadingHTTPServer((args.host, args.port), ResPlanWorkbenchHandler)
    print(f"Acoustic Agent ResPlan workbench: http://{args.host}:{args.port}")
    stats = ResPlanWorkbenchHandler.dataset.stats()
    source_path = ResPlanWorkbenchHandler.dataset.path
    print(
        f"ResPlan scenes: {stats['eligible_records']} eligible / {stats['records']} total "
        f"from {source_path}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
