from __future__ import annotations

from .floorplan_web_server import _SCENE_END, _SCENE_START
from .web_server import WEB_ROOT


_CUSTOM_SCENE_SETUP = '''      <section id="sceneSection" class="panelBlock editorBlock customEditorBlock setupSection">
        <div class="inspectorHeader">
          <h2>Custom floor plan</h2>
          <span id="customGeneratorBadge">Local</span>
        </div>

        <div class="customSourceTabs" role="tablist" aria-label="Custom scene source">
          <button id="customTextTab" class="active" type="button" role="tab" aria-selected="true">Text</button>
          <button id="customImageTab" type="button" role="tab" aria-selected="false">Image</button>
        </div>

        <div id="customTextPane" class="customSourcePane active">
          <label for="customDescription">Description
            <textarea id="customDescription" rows="4">10m x 8m，两室一厅一厨一卫</textarea>
          </label>
        </div>

        <div id="customImagePane" class="customSourcePane" hidden>
          <label class="customUploadField" for="customImageFile">Floor-plan image
            <input id="customImageFile" type="file" accept="image/png,image/jpeg,image/webp">
          </label>
          <div class="customImageCommands">
            <label>Overlay
              <span class="rangeField"><input id="customImageOpacity" type="range" min="0" max="1" step="0.05" value="0.5"><output id="customImageOpacityValue">50%</output></span>
            </label>
            <button id="customVlmPrompt" type="button">Copy Codex prompt</button>
          </div>
          <p id="customVlmStatus" class="customProviderStatus">Attach the image and copied prompt to Codex, then paste its JSON below.</p>
        </div>

        <div class="propertyGrid customDimensions">
          <label>Width (m)<input id="customWidth" type="number" min="3" max="40" step="0.1" value="10"></label>
          <label>Depth (m)<input id="customDepth" type="number" min="3" max="40" step="0.1" value="8"></label>
          <label>Height (m)<input id="height" type="number" min="2" max="6" step="0.1" value="2.8"></label>
        </div>

        <div class="customGenerateRow">
          <label>Seed<input id="customSeed" type="number" min="0" max="2147483647" step="1" value="42"></label>
          <button id="customGenerate" class="primaryAction" type="button">Generate</button>
          <button id="customVariant" class="iconButton" type="button" title="Generate another layout" aria-label="Generate another layout">&#8635;</button>
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

        <details class="customSpecEditor">
          <summary>Floorplan JSON</summary>
          <textarea id="customSpecJson" rows="10" spellcheck="false"></textarea>
          <div class="editorCommandRow">
            <button id="customApplyJson" type="button">Apply JSON</button>
            <button id="customDownloadJson" type="button">Export JSON</button>
          </div>
        </details>

        <div class="editorCommandRow">
          <button id="reset" type="button">Reset</button>
        </div>
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
