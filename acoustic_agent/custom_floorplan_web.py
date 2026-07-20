from __future__ import annotations

from .floorplan_web_server import _SCENE_END, _SCENE_START
from .web_server import WEB_ROOT


_CUSTOM_SCENE_SETUP = '''      <section id="sceneSection" class="panelBlock editorBlock customEditorBlock setupSection">
        <div class="inspectorHeader">
          <h2>Custom floor plan</h2>
          <span id="customGeneratorBadge">Codex</span>
        </div>

        <label class="customUploadField" for="customImageFile">Floor-plan image
          <input id="customImageFile" type="file" accept="image/png,image/jpeg,image/webp">
        </label>
        <div class="customImageCommands">
          <label>Overlay
            <span class="rangeField"><input id="customImageOpacity" type="range" min="0" max="1" step="0.05" value="1"><output id="customImageOpacityValue">100%</output></span>
          </label>
          <button id="customVlmPrompt" type="button">Copy Codex prompt</button>
        </div>
        <p id="customVlmStatus" class="customProviderStatus">Attach the image and copied prompt to Codex, then paste its JSON below.</p>

        <label class="customJsonPaste" for="customSpecJson">Floorplan JSON
          <textarea id="customSpecJson" rows="10" spellcheck="false" placeholder="Paste JSON returned by Codex"></textarea>
        </label>
        <div class="editorCommandRow customApplyRow">
          <button id="customApplyJson" class="primaryAction" type="button">Apply floor plan</button>
          <button id="reset" type="button">Reset</button>
        </div>

        <div class="propertyGrid customDimensions">
          <label>Width (m)<input id="customWidth" type="number" min="3" max="40" step="0.1" value="10"></label>
          <label>Depth (m)<input id="customDepth" type="number" min="3" max="40" step="0.1" value="8"></label>
          <label>Height (m)<input id="height" type="number" min="2" max="6" step="0.1" value="2.8"></label>
        </div>

        <label class="presetSelect" for="floorplanRoom">Source room
          <select id="floorplanRoom"></select>
        </label>

        <label class="presetSelect" for="floorplanReceiverRoom">Microphone room
          <select id="floorplanReceiverRoom"></select>
        </label>

        <canvas id="floorplanPlanCanvas" width="320" height="230" aria-label="Custom floor plan editor"></canvas>
        <div id="customValidation" class="customValidation" aria-live="polite"></div>
        <div id="floorplanMeta" class="floorplanMeta"></div>
      </section>

'''


def custom_viewer_html() -> str:
    html = (WEB_ROOT / "viewer.html").read_text(encoding="utf-8")
    start = html.index(_SCENE_START)
    end = html.index(_SCENE_END, start)
    html = html[:start] + _CUSTOM_SCENE_SETUP + html[end:]
    html = html.replace('data-scene-source="geometry"', 'data-scene-source="custom"', 1)
    html = html.replace("AcousticAgent WebGL Workbench", "AcousticAgent Custom Workbench", 1)
    html = html.replace("Scene setup</p>", "Custom floor plan</p>", 1)
    return html
