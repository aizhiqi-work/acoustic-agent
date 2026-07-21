from __future__ import annotations

from .floorplan_web_server import _SCENE_END, _SCENE_START
from .web_server import WEB_ROOT


_CUSTOM_SCENE_SETUP = '''      <section id="sceneSection" class="panelBlock editorBlock customEditorBlock setupSection">
        <div class="inspectorHeader">
          <h2>Custom floor plan</h2>
          <span id="customGeneratorBadge">ChatGPT</span>
        </div>

        <fieldset class="customInputMode">
          <legend>Input</legend>
          <div class="customModeSwitch" role="radiogroup" aria-label="Custom floor-plan input">
            <label><input type="radio" name="customInputMode" value="image" checked><span>Floor-plan image</span></label>
            <label><input type="radio" name="customInputMode" value="text"><span>Text description</span></label>
          </div>
        </fieldset>

        <div id="customImageSource" class="customSourcePanel">
          <label class="customUploadField" for="customImageFile">Floor-plan image
            <input id="customImageFile" type="file" accept="image/png,image/jpeg,image/webp">
          </label>
        </div>
        <div id="customTextSource" class="customSourcePanel" hidden>
          <label for="customDescription">Home description
            <textarea id="customDescription" rows="4" maxlength="1200" placeholder="For example: a 10 m x 8 m home with two bedrooms, one living room, one kitchen and one bathroom"></textarea>
          </label>
        </div>
        <button id="customVlmPrompt" class="customPromptButton" type="button">Copy ChatGPT prompt</button>
        <p id="customVlmStatus" class="customProviderStatus">Paste ChatGPT's JSON output below.</p>

        <label class="customJsonPaste" for="customSpecJson">Floorplan JSON
          <textarea id="customSpecJson" rows="10" spellcheck="false" placeholder="Paste JSON returned by ChatGPT"></textarea>
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
