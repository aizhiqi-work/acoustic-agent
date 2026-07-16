import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const canvas = document.getElementById("view");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const codeEl = document.getElementById("code");
const copyCodeButton = document.getElementById("copyCode");
const dryAudioEl = document.getElementById("dryAudio");
const wetAudioEl = document.getElementById("wetAudio");
const calibrationAudioMetaEl = document.getElementById("calibrationAudioMeta");

const presets = [
  { id: "rectangle", title: "Rectangle" },
  { id: "triangle", title: "Triangle" },
  { id: "polygon", title: "Polygon" },
  { id: "circle", title: "Ellipse" },
  { id: "l_shape", title: "L Shape" },
  { id: "t_shape", title: "T Shape" },
  { id: "trapezoid", title: "Trapezoid" },
  { id: "u_shape", title: "U Shape" },
  { id: "fan_shape", title: "Fan Shape" }
];

const micOptions = [
  { id: "mono", title: "Mono" },
  { id: "hrtf", title: "HRTF" },
  { id: "linear", title: "Linear array" },
  { id: "circular", title: "Circular array" }
];
const sourceDirectivityOptions = [
  { id: "omni", title: "Omni", dipole_weight: 0.0, dipole_power: 1.0 },
  { id: "cardioid", title: "Cardioid", dipole_weight: 0.5, dipole_power: 1.0 },
  { id: "dipole", title: "Dipole", dipole_weight: 1.0, dipole_power: 1.0 },
  { id: "focused", title: "Focused", dipole_weight: 0.5, dipole_power: 4.0 }
];
const objectTypeOptions = [
  { id: "cuboid", title: "Cuboid" },
  { id: "panel", title: "Thin panel" },
  { id: "low_block", title: "Low block" }
];
const furnitureCatalog = {
  cuboid: { title: "Cuboid", size: [1.2, 0.55, 1.05], color: 0x8d7463, kind: "block", material: "wood", description: "solid obstacle" },
  panel: { title: "Thin panel", size: [1.45, 0.08, 1.35], color: 0x4f6672, kind: "panel", z: 0.675, material: "plaster", description: "reflective slab" },
  low_block: { title: "Low block", size: [1.35, 0.72, 0.45], color: 0x7f8b6f, kind: "block", material: "fabric", description: "low reflector" }
};
const objectMaterialColors = {
  fabric: 0x7f8b6f,
  wood: 0x9a7656,
  glass: 0x7fb4c7,
  screen: 0x242a2f,
  plaster: 0xb7afa5,
};
const MIN_WALL_DISTANCE_M = 0.15;

const defaultState = {
  shape: "rectangle",
  size: [6.0, 4.0, 2.8],
  geometry: {
    triangleApex: 0.5,
    circleSegments: 36,
    polygonSides: 6,
    polygonIrregularity: 0.18,
    polygonSkew: 0.0,
    lCutoutWidth: 0.45,
    lCutoutDepth: 0.45,
    tHeadDepth: 0.38,
    tStemWidth: 0.34,
    tStemOffset: 0.5,
    trapezoidTopWidth: 0.62,
    trapezoidOffset: 0.5,
    uOpeningWidth: 0.42,
    uOpeningDepth: 0.48,
    uOpeningOffset: 0.5,
    fanAngle: 90,
    fanInnerRadius: 0.28,
    fanSegments: 24
  },
  materials: { wall: "wood", floor: "floor_carpet", ceiling: "ceiling" },
  objects: [],
  source: [1.2, 1.1, 1.5],
  receiver: [4.7, 2.8, 1.4],
  config: { fs: 16000, duration_s: 2.0, quality: "simulation", rt_num_rays: 32768, rt_num_bounces: 64, rt_duration_s: 2.0, diffraction_order: 3, max_diffraction_paths: 8 },
  mic: { type: "mono", count: 4, spacing_m: 0.08, radius_m: 0.12, orientation_deg: 0 },
  sourceDirectivity: { type: "omni", orientation_deg: 0, elevation_deg: 0, dipole_weight: 0.0, dipole_power: 1.0 }
};

let state = structuredClone(defaultState);
let simData = makeClientScene(state);
let simulateTimer = null;
let simulationRequestSeq = 0;
let lastSimulationPayload = null;
let calibrationAudioSeq = 0;
let calibrationAudioUrls = [];
let selectedObjectId = null;
let pendingObjectId = null;
let dirtyObjectId = null;
let objectMode = "move";
let objectDrag = null;
let suppressObjectSelectionUntil = 0;
const layerState = { direct: true, diffraction: true, rt: true };

let renderer;
let camera;
let controls;
const raycaster = new THREE.Raycaster();
const pointerNdc = new THREE.Vector2();
const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const threeScene = new THREE.Scene();
const floorGroup = new THREE.Group();
const shellGroup = new THREE.Group();
const pathGroup = new THREE.Group();
const furnitureGroup = new THREE.Group();
const markerGroup = new THREE.Group();
threeScene.add(floorGroup, shellGroup, furnitureGroup, pathGroup, markerGroup);

bootstrap();

function bootstrap() {
  setupThree();
  setupControls();
  renderThumbnails();
  renderMicThumbnails();
  renderSourceDirectivityThumbnails();
  renderObjectThumbnails();
  bindEvents();
  updateControls();
  requestSimulation();
  animate();
}

function setupThree() {
  threeScene.background = new THREE.Color(0xeef2f4);
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;

  camera = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.05, 200);
  camera.position.set(6.6, 7.2, 7.4);
  camera.lookAt(3, 0, 2);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.minPolarAngle = Math.PI * 0.16;
  controls.maxPolarAngle = Math.PI * 0.5;

  threeScene.add(new THREE.HemisphereLight(0xffffff, 0xb9c3c7, 2.45));
  const key = new THREE.DirectionalLight(0xffffff, 2.7);
  key.position.set(-4, 9, 6);
  key.castShadow = true;
  threeScene.add(key);

  window.addEventListener("resize", resize);
  canvas.addEventListener("pointerdown", handleCanvasPointerDown, { capture: true });
  canvas.addEventListener("pointermove", handleCanvasPointerMove, { capture: true });
  canvas.addEventListener("pointerup", handleCanvasPointerUp, { capture: true });
  canvas.addEventListener("pointercancel", handleCanvasPointerUp, { capture: true });
  resize();
}

function setupControls() {
  fillSelect("shape", presets.map((preset) => [preset.id, preset.title]));
}

function geometrySpecs(shape) {
  const specs = {
    triangle: [
      { key: "triangleApex", label: "Apex", min: 0.05, max: 0.95, step: 0.01 }
    ],
    circle: [
      { key: "circleSegments", label: "Segments", min: 12, max: 96, step: 1 }
    ],
    polygon: [
      { key: "polygonSides", label: "Sides", min: 5, max: 12, step: 1 },
      { key: "polygonIrregularity", label: "Irregular", min: 0, max: 0.35, step: 0.01 },
      { key: "polygonSkew", label: "Skew", min: -0.3, max: 0.3, step: 0.01 }
    ],
    l_shape: [
      { key: "lCutoutWidth", label: "Cutout W", min: 0.15, max: 0.8, step: 0.01 },
      { key: "lCutoutDepth", label: "Cutout D", min: 0.15, max: 0.8, step: 0.01 }
    ],
    t_shape: [
      { key: "tHeadDepth", label: "Head D", min: 0.15, max: 0.65, step: 0.01 },
      { key: "tStemWidth", label: "Stem W", min: 0.18, max: 0.85, step: 0.01 },
      { key: "tStemOffset", label: "Stem X", min: 0, max: 1, step: 0.01 }
    ],
    trapezoid: [
      { key: "trapezoidTopWidth", label: "Top W", min: 0.2, max: 1, step: 0.01 },
      { key: "trapezoidOffset", label: "Top X", min: 0, max: 1, step: 0.01 }
    ],
    u_shape: [
      { key: "uOpeningWidth", label: "Gap W", min: 0.2, max: 0.72, step: 0.01 },
      { key: "uOpeningDepth", label: "Gap D", min: 0.18, max: 0.82, step: 0.01 },
      { key: "uOpeningOffset", label: "Gap X", min: 0, max: 1, step: 0.01 }
    ],
    fan_shape: [
      { key: "fanAngle", label: "Angle", min: 45, max: 150, step: 1 },
      { key: "fanInnerRadius", label: "Inner R", min: 0.05, max: 0.55, step: 0.01 },
      { key: "fanSegments", label: "Segments", min: 8, max: 48, step: 1 }
    ]
  };
  return specs[shape] || [];
}

function renderGeometryParams() {
  const container = document.getElementById("geometryParams");
  const specs = geometrySpecs(state.shape);
  state.geometry = { ...defaultState.geometry, ...(state.geometry || {}) };
  container.innerHTML = specs.map((spec) => {
    const value = Number(state.geometry[spec.key] ?? defaultState.geometry[spec.key]);
    return `<label>${spec.label}<input data-geom="${spec.key}" type="number" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${value}"></label>`;
  }).join("");
  container.classList.toggle("empty", specs.length === 0);
}

function readGeometryParams() {
  state.geometry = { ...defaultState.geometry, ...(state.geometry || {}) };
  document.querySelectorAll("#geometryParams [data-geom]").forEach((input) => {
    const key = input.dataset.geom;
    const spec = geometrySpecs(state.shape).find((item) => item.key === key);
    const raw = Number(input.value);
    if (!Number.isFinite(raw)) return;
    const value = spec ? clamp(raw, spec.min, spec.max) : raw;
    state.geometry[key] = key.includes("Sides") || key.includes("Segments") || key.includes("Angle") ? Math.round(value) : value;
    input.value = state.geometry[key];
  });
}

function bindEvents() {
  const ids = ["shape", "sizeX", "sizeY", "height", "wallMaterial", "floorMaterial", "ceilingMaterial", "qualitySelect", "rirDuration", "sourceX", "sourceY", "sourceZ", "receiverX", "receiverY", "receiverZ", "micOrientation", "micCount", "micSpacing", "sourceOrientation", "sourceElevation", "sourcePower", "fs"];
  ids.forEach((id) => document.getElementById(id).addEventListener("input", () => {
    const oldShape = state.shape;
    readControls();
    if (id === "shape" && oldShape !== state.shape) {
      renderGeometryParams();
      applyPresetPoints();
    } else if (["sizeX", "sizeY", "height", "sourceX", "sourceY", "sourceZ", "receiverX", "receiverY", "receiverZ"].includes(id)) {
      clampScenePointsToRoom();
      syncPositionControls();
    }
    if (id === "qualitySelect") {
      applyQualityPreset(true);
    }
    simData = makeClientScene(state);
    rebuildThreeScene();
    scheduleSimulation();
  }));
  document.getElementById("geometryParams").addEventListener("input", () => {
    readControls();
    clampScenePointsToRoom();
    simData = makeClientScene(state);
    rebuildThreeScene();
    scheduleSimulation();
  });

  copyCodeButton?.addEventListener("click", copyCodeSnippet);
  document.getElementById("randomPositions").addEventListener("click", randomizePositions);
  document.getElementById("reset").addEventListener("click", () => {
    state = structuredClone(defaultState);
    selectedObjectId = null;
    pendingObjectId = null;
    dirtyObjectId = null;
    if (camera) camera.userData.fitted = false;
    simData = makeClientScene(state);
    updateControls();
    requestSimulation();
  });
  document.getElementById("addAsset").addEventListener("click", handlePaletteSelection);
  const confirmButton = document.getElementById("confirmFurniture");
  confirmButton.addEventListener("pointerdown", (event) => {
    event.stopPropagation();
  });
  confirmButton.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    confirmSelectedObject();
  });
  document.getElementById("deleteFurniture").addEventListener("click", deleteSelectedObject);
  ["objectX", "objectY", "objectZ", "objectWidth", "objectDepth", "objectHeight", "objectRotation"].forEach((id) => {
    document.getElementById(id).addEventListener("input", handleObjectSettingsInput);
  });
  document.getElementById("pathLimit").addEventListener("input", () => {
    updatePathLimitLabel();
    rebuildThreeScene();
    safeDrawRirPanel();
  });
  [["layerDirect", "direct"], ["layerDiffraction", "diffraction"], ["layerRt", "rt"]].forEach(([id, key]) => {
    document.getElementById(id).addEventListener("change", (event) => {
      layerState[key] = event.target.checked;
      rebuildThreeScene();
      safeDrawRirPanel();
    });
  });
}

function handleCanvasPointerDown(event) {
  if (Date.now() < suppressObjectSelectionUntil) return;
  const hit = pickFurnitureObject(event);
  if (!hit) {
    if (!hasUnconfirmedObjectChange()) clearObjectSelection();
    return;
  }
  const object = sceneObjectById(hit.objectId);
  if (!object) return;
  stopObjectDragEvent(event);
  canvas.setPointerCapture?.(event.pointerId);
  selectSceneObject(object.id);
  rebuildThreeScene();
  const ground = groundPointFromPointer(event);
  const group = furnitureGroup.children.find((item) => item.userData.objectId === object.id);
  objectDrag = {
    pointerId: event.pointerId,
    objectId: object.id,
    startX: event.clientX,
    startRotation: Number(object.rotation || 0),
    changed: false,
    offset: ground && group ? group.position.clone().sub(ground) : new THREE.Vector3(),
  };
  controls.enabled = false;
  canvas.style.cursor = "grabbing";
}

function handleCanvasPointerMove(event) {
  if (!objectDrag || event.pointerId !== objectDrag.pointerId) return;
  const object = sceneObjectById(objectDrag.objectId);
  if (!object) return;
  stopObjectDragEvent(event);
  if (objectMode === "rotate") {
    const nextRotation = Number((objectDrag.startRotation + (event.clientX - objectDrag.startX) * 0.45).toFixed(1));
    if (nextRotation !== object.rotation) {
      object.rotation = nextRotation;
      objectDrag.changed = true;
    }
    const group = furnitureGroup.children.find((item) => item.userData.objectId === object.id);
    if (group) group.rotation.y = roomRotationToThreeY(object.rotation);
    updateSelectionToolbarPosition();
    return;
  }
  const ground = groundPointFromPointer(event);
  if (!ground) return;
  const target = ground.clone().add(objectDrag.offset || new THREE.Vector3());
  const spec = furnitureCatalog[object.type] || furnitureCatalog.cuboid;
  const z = Number(object.z ?? spec.z ?? spec.size[2] * 0.5);
  const safe = nearestSafeRoomPoint([target.x, target.z, z], cornersFor(state.shape, state.size, state.geometry), z);
  if (!safe) return;
  const nextPosition = [Number(safe[0].toFixed(3)), Number(safe[1].toFixed(3))];
  if (nextPosition[0] !== object.position?.[0] || nextPosition[1] !== object.position?.[1]) {
    object.position = nextPosition;
    objectDrag.changed = true;
  }
  const group = furnitureGroup.children.find((item) => item.userData.objectId === object.id);
  if (group) group.position.copy(toVector3([object.position[0], object.position[1], 0]));
  syncSelectedObjectControls(object);
  updateSelectionToolbarPosition();
}

function handleCanvasPointerUp(event) {
  if (!objectDrag || event.pointerId !== objectDrag.pointerId) return;
  stopObjectDragEvent(event);
  const draggedObjectId = objectDrag.objectId;
  const changed = objectDrag.changed;
  canvas.releasePointerCapture?.(event.pointerId);
  objectDrag = null;
  controls.enabled = true;
  canvas.style.cursor = "";
  rebuildThreeScene();
  updatePanels();
  if (!changed) {
    setStatus("Object selected. Move, rotate, or resize to edit.");
    return;
  }
  markObjectEditForConfirmation(draggedObjectId);
}

function stopObjectDragEvent(event) {
  event.preventDefault();
  event.stopPropagation();
  event.stopImmediatePropagation?.();
}

function pickFurnitureObject(event) {
  updatePointerNdc(event);
  raycaster.setFromCamera(pointerNdc, camera);
  const hits = raycaster.intersectObjects(furnitureGroup.children, true);
  const hit = hits.find((item) => item.object?.userData?.objectId);
  return hit ? { objectId: hit.object.userData.objectId } : null;
}

function groundPointFromPointer(event) {
  updatePointerNdc(event);
  raycaster.setFromCamera(pointerNdc, camera);
  const point = new THREE.Vector3();
  return raycaster.ray.intersectPlane(floorPlane, point) ? point : null;
}

function updatePointerNdc(event) {
  const rect = canvas.getBoundingClientRect();
  pointerNdc.x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
  pointerNdc.y = -(((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2 - 1);
}

function sceneObjectById(id) {
  return (state.objects || []).find((object) => object.id === id);
}

function hasPendingObject() {
  return Boolean(pendingObjectId && sceneObjectById(pendingObjectId));
}

function hasDirtyObject() {
  return Boolean(dirtyObjectId && sceneObjectById(dirtyObjectId));
}

function unconfirmedObjectId() {
  if (hasPendingObject()) return pendingObjectId;
  if (hasDirtyObject()) return dirtyObjectId;
  return null;
}

function hasUnconfirmedObjectChange() {
  return Boolean(unconfirmedObjectId());
}

function selectedObjectIsPending() {
  return Boolean(selectedObjectId && pendingObjectId === selectedObjectId && sceneObjectById(selectedObjectId));
}

function selectedObjectNeedsConfirmation() {
  return Boolean(selectedObjectId && selectedObjectId === unconfirmedObjectId());
}

function handlePaletteSelection() {
  const selected = activeObjectType();
  if (furnitureCatalog[selected]) {
    addSceneObject(selected);
  } else {
    selectedObjectId = null;
    setSelection("Scene");
    setStatus("Choose a geometry object to add to the scene.");
  }
}

function readControls() {
  state.shape = value("shape");
  state.size = [number("sizeX"), number("sizeY"), number("height")];
  state.materials = {
    wall: value("wallMaterial"),
    floor: value("floorMaterial"),
    ceiling: value("ceilingMaterial")
  };
  readGeometryParams();
  state.config.quality = value("qualitySelect");
  state.config.duration_s = clamp(number("rirDuration"), 0.3, 6.0);
  state.config.rt_duration_s = state.config.duration_s;
  applyQualityPreset(false);
  state.source = [number("sourceX"), number("sourceY"), number("sourceZ")];
  state.receiver = [number("receiverX"), number("receiverY"), number("receiverZ")];
  if (state.sourceDirectivity.type !== "omni") {
    state.sourceDirectivity.orientation_deg = clamp(number("sourceOrientation"), -180, 180);
    state.sourceDirectivity.elevation_deg = clamp(number("sourceElevation"), -90, 90);
    state.sourceDirectivity.dipole_power = clamp(number("sourcePower"), 0.25, 8.0);
  }
  const previousMicType = state.mic.type;
  if (isDirectionalMic(state.mic.type)) {
    state.mic.orientation_deg = clamp(number("micOrientation"), -180, 180);
  }
  if (isArrayMic(state.mic.type)) {
    state.mic.count = Math.max(1, Math.round(number("micCount")));
    if (state.mic.type === "circular") {
      if (previousMicType === state.mic.type) state.mic.radius_m = number("micSpacing");
    } else if (previousMicType === state.mic.type) {
      state.mic.spacing_m = number("micSpacing");
    }
  }
  syncMicControls();
  state.config.fs = Math.round(number("fs"));
}

function updateControls() {
  renderGeometryParams();
  setValue("shape", state.shape);
  setValue("sizeX", state.size[0]);
  setValue("sizeY", state.size[1]);
  setValue("height", state.size[2]);
  setValue("wallMaterial", state.materials.wall);
  setValue("floorMaterial", state.materials.floor);
  setValue("ceilingMaterial", state.materials.ceiling);
  setValue("qualitySelect", state.config.quality);
  setValue("rirDuration", Number(state.config.duration_s || 2.0).toFixed(1));
  setValue("sourceX", state.source[0]);
  setValue("sourceY", state.source[1]);
  setValue("sourceZ", state.source[2]);
  setValue("receiverX", state.receiver[0]);
  setValue("receiverY", state.receiver[1]);
  setValue("receiverZ", state.receiver[2]);
  syncMicControls();
  syncSourceDirectivityControls();
  syncSelectedObjectControls(sceneObjectById(selectedObjectId));
  setValue("fs", state.config.fs);
  updatePanels();
  rebuildThreeScene();
}

function syncPositionControls() {
  setValue("sourceX", state.source[0]);
  setValue("sourceY", state.source[1]);
  setValue("sourceZ", state.source[2]);
  setValue("receiverX", state.receiver[0]);
  setValue("receiverY", state.receiver[1]);
  setValue("receiverZ", state.receiver[2]);
}

function updateMicControls() {
  const showArrayControls = isArrayMic(state.mic.type);
  const showDirectionControls = isDirectionalMic(state.mic.type);
  document.getElementById("micControlTray").classList.toggle("empty", !showArrayControls && !showDirectionControls);
  setMicControlVisibility(document.getElementById("micArrayControls"), showArrayControls);
  setMicControlVisibility(document.getElementById("micDirectionControls"), showDirectionControls);
  document.getElementById("micSpacingLabel").textContent = state.mic.type === "circular" ? "Radius" : "Spacing";
}

function setMicControlVisibility(group, visible) {
  group.classList.toggle("micControlHidden", !visible);
  group.querySelectorAll("input, select, button").forEach((item) => {
    item.disabled = !visible;
  });
}

function syncMicControls() {
  updateMicControls();
  setValue("micOrientation", state.mic.orientation_deg);
  setValue("micCount", state.mic.count);
  setValue("micSpacing", state.mic.type === "circular" ? state.mic.radius_m : state.mic.spacing_m);
}

function syncSourceDirectivityControls() {
  const directional = state.sourceDirectivity.type !== "omni";
  document.getElementById("sourceDirectionControls").classList.toggle("hidden", !directional);
  setValue("sourceOrientation", state.sourceDirectivity.orientation_deg);
  setValue("sourceElevation", state.sourceDirectivity.elevation_deg);
  setValue("sourcePower", state.sourceDirectivity.dipole_power);
  document.getElementById("sourcePowerValue").textContent = Number(state.sourceDirectivity.dipole_power).toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  document.getElementById("sourceDirectivityType").textContent = state.sourceDirectivity.type;
  refreshSourceDirectivityThumbnails();
}

function isArrayMic(type) {
  return type === "linear" || type === "circular";
}

function isDirectionalMic(type) {
  return type === "hrtf" || isArrayMic(type);
}

function scheduleSimulation() {
  clearTimeout(simulateTimer);
  if (hasUnconfirmedObjectChange()) {
    setStatus("Confirm the object edit to update simulation.");
    return;
  }
  simulateTimer = setTimeout(requestSimulation, 220);
}

async function requestSimulation() {
  if (hasUnconfirmedObjectChange()) {
    clearTimeout(simulateTimer);
    setStatus("Confirm the object edit to update simulation.");
    return;
  }
  const requestSeq = ++simulationRequestSeq;
  const payload = structuredClone(apiPayload());
  let simulationSucceeded = false;
  setStatus("Computing indoor RIR paths...");
  clearCalibrationAudio("reading.wav · waiting for RIR");
  try {
    const response = await fetch("/api/v1/workbench", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) throw new Error(await response.text());
    const nextSimData = await response.json();
    if (requestSeq !== simulationRequestSeq) return;
    simData = nextSimData;
    lastSimulationPayload = payload;
    simulationSucceeded = true;
    setStatus("Simulation updated.");
  } catch (error) {
    if (requestSeq !== simulationRequestSeq) return;
    simData = makeClientScene(state);
    lastSimulationPayload = payload;
    simData.metadata = { ...(simData.metadata || {}), warning: String(error.message || error) };
    setStatus("Local API unavailable; showing editable geometry only.", true);
  }
  if (requestSeq !== simulationRequestSeq) return;
  rebuildThreeScene();
  updatePanels();
  if (simulationSucceeded) void updateCalibrationAudio(simData, requestSeq);
}

function apiPayload() {
  const mic = state.mic.type === "circular"
    ? { type: "circular", count: state.mic.count, radius_m: state.mic.radius_m, orientation_deg: state.mic.orientation_deg }
    : state.mic.type === "linear"
      ? { type: "linear", count: state.mic.count, spacing_m: state.mic.spacing_m, orientation_deg: state.mic.orientation_deg }
      : { type: state.mic.type, orientation_deg: state.mic.orientation_deg };
  return {
    shape: state.shape,
    size: state.size,
    corners: cornersFor(state.shape, state.size, state.geometry),
    geometry: state.geometry,
    materials: state.materials,
    objects: state.objects,
    source: state.source,
    receiver: state.receiver,
    config: state.config,
    receiver_model: mic,
    source_model: {
      type: state.sourceDirectivity.type,
      orientation_deg: state.sourceDirectivity.orientation_deg,
      elevation_deg: state.sourceDirectivity.elevation_deg,
      dipole_weight: state.sourceDirectivity.dipole_weight,
      dipole_power: state.sourceDirectivity.dipole_power
    }
  };
}

function applyQualityPreset(overwriteRays = true) {
  const presetsByQuality = {
    preview: { rt_num_rays: 8192, rt_num_bounces: 32 },
    simulation: { rt_num_rays: 32768, rt_num_bounces: 64 },
    fine: { rt_num_rays: 65536, rt_num_bounces: 96 },
    reference: { rt_num_rays: 131072, rt_num_bounces: 96 }
  };
  const preset = presetsByQuality[state.config.quality] || presetsByQuality.simulation;
  if (overwriteRays) state.config.rt_num_rays = preset.rt_num_rays;
  state.config.rt_num_bounces = preset.rt_num_bounces;
  state.config.rt_duration_s = state.config.duration_s;
}

function rebuildThreeScene(options = {}) {
  clearGroup(floorGroup);
  clearGroup(shellGroup);
  clearGroup(furnitureGroup);
  clearGroup(pathGroup);
  clearGroup(markerGroup);
  addPlan3D();
  addRoomShell3D();
  addFurniture3D();
  addPaths3D();
  addMarkers3D();
  const signature = cameraSceneSignature();
  const shouldFitCamera = Boolean(options.forceFitCamera) || !camera.userData.fitted || camera.userData.sceneSignature !== signature;
  if (shouldFitCamera) fitCamera(signature);
  drawMiniMap();
}

function addPlan3D() {
  const corners = simData.room.corners;
  const shape = shapeFromCorners(corners);
  const floor = new THREE.Mesh(
    new THREE.ShapeGeometry(shape),
    new THREE.MeshStandardMaterial({ color: 0xc9cecc, roughness: 0.96, metalness: 0.0 })
  );
  floor.rotation.x = Math.PI / 2;
  floor.position.y = -0.01;
  floor.receiveShadow = true;
  floorGroup.add(floor);

  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(floor.geometry, 30),
    new THREE.LineBasicMaterial({ color: 0x59636a, transparent: true, opacity: 0.55 })
  );
  edges.rotation.copy(floor.rotation);
  edges.position.copy(floor.position);
  floorGroup.add(edges);

  const bounds = getBounds(corners);
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(bounds.w + 8, bounds.h + 8),
    new THREE.ShadowMaterial({ color: 0x6f777c, opacity: 0.12 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set((bounds.x0 + bounds.x1) * 0.5, -0.06, (bounds.y0 + bounds.y1) * 0.5);
  ground.receiveShadow = true;
  floorGroup.add(ground);

  const grid = new THREE.GridHelper(Math.max(bounds.w, bounds.h) + 6, Math.ceil(Math.max(bounds.w, bounds.h) + 6), 0xb8c2c6, 0xd8dee1);
  grid.position.set((bounds.x0 + bounds.x1) * 0.5, -0.055, (bounds.y0 + bounds.y1) * 0.5);
  floorGroup.add(grid);
}

function addRoomShell3D() {
  const corners = simData.room.corners;
  const height = Number(simData.room.height_m || state.size[2] || 2.8);
  const wallMat = new THREE.MeshStandardMaterial({ color: 0xc4cbd0, transparent: true, opacity: 0.34, roughness: 0.88, side: THREE.DoubleSide, depthWrite: false });
  for (let i = 0; i < corners.length; i += 1) {
    const a = corners[i];
    const b = corners[(i + 1) % corners.length];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const length = Math.hypot(dx, dy);
    if (length < 0.02) continue;
    const wall = new THREE.Mesh(new THREE.BoxGeometry(length, height, 0.12), wallMat.clone());
    wall.position.set((a[0] + b[0]) * 0.5, height * 0.5, (a[1] + b[1]) * 0.5);
    wall.rotation.y = -Math.atan2(dy, dx);
    wall.receiveShadow = true;
    shellGroup.add(wall);
    const edge = new THREE.LineSegments(new THREE.EdgesGeometry(wall.geometry, 30), new THREE.LineBasicMaterial({ color: 0x46535a, transparent: true, opacity: 0.5 }));
    edge.position.copy(wall.position);
    edge.rotation.copy(wall.rotation);
    shellGroup.add(edge);
  }

  const ceiling = new THREE.Mesh(
    new THREE.ShapeGeometry(shapeFromCorners(corners)),
    new THREE.MeshStandardMaterial({ color: 0x9fb8c0, transparent: true, opacity: 0.08, roughness: 0.9, side: THREE.DoubleSide, depthWrite: false })
  );
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.y = height;
  shellGroup.add(ceiling);
  const ceilingEdges = new THREE.LineSegments(new THREE.EdgesGeometry(ceiling.geometry, 30), new THREE.LineBasicMaterial({ color: 0x55717a, transparent: true, opacity: 0.46 }));
  ceilingEdges.rotation.copy(ceiling.rotation);
  ceilingEdges.position.copy(ceiling.position);
  shellGroup.add(ceilingEdges);
}

function addPaths3D() {
  const limit = Number(document.getElementById("pathLimit")?.value || 512);
  const visible = displayPaths(simData.paths || []).slice(0, limit);
  for (const path of visible) {
    const style = pathStyle(path);
    const points = (path.points || []).map(toVector3);
    if (points.length < 2) continue;
    pathGroup.add(pathObject3D(points, style));
    if (path.kind === "diffraction") {
      for (const point of points.slice(1, -1)) pathGroup.add(pathHitMarker(point, style.color, 0.045));
    }
  }
}

function pathLayerVisible(path) {
  if (path.kind === "direct" || path.kind === "direct_transmitted") return layerState.direct;
  if (path.kind === "diffraction") return layerState.diffraction;
  if (path.kind === "rt_reflection") return layerState.rt;
  return true;
}

function displayPaths(paths) {
  const shownObjectDiffractions = new Map();
  return (paths || []).filter((path) => {
    if (!pathLayerVisible(path)) return false;
    if (!isObjectDiffraction(path)) return true;
    const key = `${path.metadata?.object_index ?? "object"}:${path.metadata?.object_id ?? ""}`;
    const count = shownObjectDiffractions.get(key) || 0;
    if (count >= 2) return false;
    shownObjectDiffractions.set(key, count + 1);
    return true;
  });
}

function isObjectDiffraction(path) {
  return path?.kind === "diffraction" && path?.metadata?.model === "steam_audio_utd_object_edge_approx";
}

function pathStyle(path) {
  if (path.kind === "direct" || path.kind === "direct_transmitted") return { color: 0xef476f, radius: 0.015, opacity: 0.94, dashed: path.kind === "direct_transmitted" };
  if (path.kind === "diffraction") return { color: 0x7d8cff, radius: isObjectDiffraction(path) ? 0.008 : 0.011, opacity: isObjectDiffraction(path) ? 0.66 : 0.82, dashed: false };
  if (path.kind === "rt_reflection") return { color: 0x126f5d, radius: 0.004, opacity: 0.43, dashed: false, line: true };
  return { color: 0xf2a541, radius: 0.009, opacity: 0.72, dashed: false };
}

function pathObject3D(points, style) {
  const group = new THREE.Group();
  if (style.line) {
    group.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color: style.color, transparent: true, opacity: style.opacity })
    ));
    return group;
  }
  for (let index = 0; index < points.length - 1; index += 1) {
    if (style.dashed) addDashedTube(group, points[index], points[index + 1], style);
    else group.add(tubeSegment(points[index], points[index + 1], style));
  }
  return group;
}

function addDashedTube(group, start, end, style) {
  const delta = end.clone().sub(start);
  const length = delta.length();
  if (length <= 1e-6) return;
  const direction = delta.clone().divideScalar(length);
  for (let cursor = 0; cursor < length; cursor += 0.22) {
    const a = start.clone().addScaledVector(direction, cursor);
    const b = start.clone().addScaledVector(direction, Math.min(cursor + 0.13, length));
    group.add(tubeSegment(a, b, style));
  }
}

function tubeSegment(start, end, style) {
  const delta = end.clone().sub(start);
  const length = delta.length();
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(style.radius, style.radius, Math.max(length, 1e-6), 7, 1, true),
    new THREE.MeshBasicMaterial({ color: style.color, transparent: true, opacity: style.opacity, depthWrite: false })
  );
  mesh.position.copy(start).add(end).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize());
  return mesh;
}

function pathHitMarker(point, color, radius) {
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 14, 10),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.94, depthWrite: false })
  );
  marker.position.copy(point);
  return marker;
}

function addMarkers3D() {
  addSphere(state.source, 0xef476f, 0.13);
  addSourceDirection3D();
  addSphere(state.receiver, 0x0f7f9f, 0.13);
  for (const point of microphonePoints()) addSphere(point, 0x7d8cff, 0.055);
}

function addSourceDirection3D() {
  if (state.sourceDirectivity.type === "omni") return;
  const yaw = THREE.MathUtils.degToRad(Number(state.sourceDirectivity.orientation_deg || 0));
  const pitch = THREE.MathUtils.degToRad(Number(state.sourceDirectivity.elevation_deg || 0));
  const cosPitch = Math.cos(pitch);
  const direction = new THREE.Vector3(
    cosPitch * Math.cos(yaw),
    Math.sin(pitch),
    cosPitch * Math.sin(yaw)
  ).normalize();
  const origin = toVector3(state.source).addScaledVector(direction, 0.12);
  markerGroup.add(new THREE.ArrowHelper(direction, origin, 0.72, 0xef476f, 0.18, 0.1));
}

function addFurniture3D() {
  (state.objects || []).forEach((object) => {
    const spec = furnitureCatalog[object.type] || furnitureCatalog.cuboid;
    const group = new THREE.Group();
    group.name = object.id;
    group.position.copy(toVector3([object.position[0], object.position[1], 0]));
    group.rotation.y = roomRotationToThreeY(object.rotation);
    group.add(furnitureMesh(object, spec));
    if (object.id === selectedObjectId) group.add(selectionRing(object, spec));
    group.userData.objectId = object.id;
    group.traverse((child) => {
      child.userData.objectId = object.id;
    });
    furnitureGroup.add(group);
  });
}

function furnitureMesh(object, spec) {
  const [width, depth, height] = object.size || spec.size;
  const color = objectMaterialColors[object.material] || spec.color;
  if (spec.kind === "table") return tableMesh(width, depth, height, color);
  if (spec.kind === "shelves") return shelfMesh(width, depth, height, color);
  const geometry = new THREE.BoxGeometry(width, height, depth);
  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: spec.kind === "panel" ? 0.42 : 0.72,
    metalness: spec.kind === "panel" && object.type === "tv" ? 0.12 : 0.0,
    transparent: object.type === "window",
    opacity: object.type === "window" ? 0.62 : 1.0,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.y = Number(object.z ?? spec.z ?? height * 0.5);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function tableMesh(width, depth, height, color) {
  const group = new THREE.Group();
  const top = new THREE.Mesh(
    new THREE.BoxGeometry(width, 0.08, depth),
    new THREE.MeshStandardMaterial({ color, roughness: 0.72 })
  );
  top.position.y = height;
  group.add(top);
  const legMaterial = new THREE.MeshStandardMaterial({ color: 0x6b5643, roughness: 0.76 });
  [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach(([sx, sz]) => {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.08, height, 0.08), legMaterial);
    leg.position.set(sx * width * 0.42, height * 0.5, sz * depth * 0.38);
    group.add(leg);
  });
  return group;
}

function shelfMesh(width, depth, height, color) {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.78 });
  const outer = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), material);
  outer.position.y = height * 0.5;
  group.add(outer);
  const shelfMaterial = new THREE.MeshStandardMaterial({ color: 0xb99b7d, roughness: 0.7 });
  [0.32, 0.55, 0.78].forEach((ratio) => {
    const shelf = new THREE.Mesh(new THREE.BoxGeometry(width * 0.92, 0.025, depth * 1.04), shelfMaterial);
    shelf.position.y = height * ratio;
    group.add(shelf);
  });
  return group;
}

function selectionRing(object, spec) {
  const [width, depth] = object.size || spec.size;
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(Math.max(width, depth) * 0.58, Math.max(width, depth) * 0.62, 48),
    new THREE.MeshBasicMaterial({ color: 0x0f7f9f, transparent: true, opacity: 0.72, side: THREE.DoubleSide })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.015;
  return ring;
}

function addSphere(point, color, radius) {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 24, 16),
    new THREE.MeshStandardMaterial({ color, roughness: 0.55 })
  );
  mesh.position.copy(toVector3(point));
  mesh.castShadow = true;
  markerGroup.add(mesh);
}

function microphonePoints() {
  const receiver = state.receiver;
  const yaw = Math.PI * Number(state.mic.orientation_deg || 0) / 180;
  const axis = [Math.cos(yaw), Math.sin(yaw)];
  const lateral = [-Math.sin(yaw), Math.cos(yaw)];
  if (state.mic.type === "linear") {
    const count = Math.max(2, state.mic.count);
    const center = (count - 1) * 0.5;
    return Array.from({ length: count }, (_, i) => {
      const offset = (i - center) * state.mic.spacing_m;
      return [receiver[0] + axis[0] * offset, receiver[1] + axis[1] * offset, receiver[2]];
    });
  }
  if (state.mic.type === "circular") {
    const count = Math.max(3, state.mic.count);
    return Array.from({ length: count }, (_, i) => {
      const angle = yaw + Math.PI * 2 * i / count;
      return [receiver[0] + Math.cos(angle) * state.mic.radius_m, receiver[1] + Math.sin(angle) * state.mic.radius_m, receiver[2]];
    });
  }
  if (state.mic.type === "hrtf") {
    return [
      [receiver[0] - lateral[0] * 0.09, receiver[1] - lateral[1] * 0.09, receiver[2]],
      [receiver[0] + lateral[0] * 0.09, receiver[1] + lateral[1] * 0.09, receiver[2]]
    ];
  }
  return [];
}

function animate() {
  requestAnimationFrame(animate);
  controls?.update();
  updateSelectionToolbarPosition();
  updateViewMeta();
  renderer.render(threeScene, camera);
}

function resize() {
  const width = canvas.clientWidth || window.innerWidth;
  const height = canvas.clientHeight || window.innerHeight;
  renderer.setSize(width, height, false);
  const aspect = width / Math.max(height, 1);
  const size = camera.userData.viewSize || 7;
  camera.left = -size * aspect;
  camera.right = size * aspect;
  camera.top = size;
  camera.bottom = -size;
  camera.updateProjectionMatrix();
}

function fitCamera(signature = cameraSceneSignature()) {
  const bounds = getBounds(simData.room.corners);
  const center = new THREE.Vector3((bounds.x0 + bounds.x1) * 0.5, 0, (bounds.y0 + bounds.y1) * 0.5);
  const span = Math.max(bounds.w, bounds.h, Number(simData.room.height_m || 2.8), 1);
  camera.userData.viewSize = span * 0.86 + 2.3;
  controls.target.set(center.x, Number(simData.room.height_m || 2.8) * 0.35, center.z);
  camera.position.set(center.x + span * 0.85, span * 0.95, center.z + span * 1.05);
  camera.lookAt(controls.target);
  camera.userData.fitted = true;
  camera.userData.sceneSignature = signature;
  controls.update();
  resize();
}

function cameraSceneSignature() {
  const corners = simData.room?.corners || [];
  const cornerKey = corners.map((point) => `${Number(point[0]).toFixed(3)},${Number(point[1]).toFixed(3)}`).join(";");
  return `${cornerKey}|h=${Number(simData.room?.height_m || 0).toFixed(3)}`;
}

function updateViewMeta() {
  const element = document.getElementById("viewMeta");
  if (!element || !camera || !controls) return;
  const offset = camera.position.clone().sub(controls.target);
  const spherical = new THREE.Spherical().setFromVector3(offset);
  const pitch = Math.round(THREE.MathUtils.radToDeg(spherical.phi));
  let yaw = Math.round(THREE.MathUtils.radToDeg(spherical.theta));
  if (yaw > 180) yaw -= 360;
  if (yaw <= -180) yaw += 360;
  const zoom = Math.max(0.1, Number(camera.zoom || 1));
  element.textContent = `View pitch ${pitch}° / yaw ${yaw}° / zoom ${zoom.toFixed(1)}x`;
}

function shapeFromCorners(corners) {
  const shape = new THREE.Shape();
  corners.forEach((point, index) => {
    if (index === 0) shape.moveTo(point[0], point[1]);
    else shape.lineTo(point[0], point[1]);
  });
  shape.closePath();
  return shape;
}

function toVector3(point) {
  return new THREE.Vector3(Number(point[0]), Number(point[2] || 0), Number(point[1]));
}

function roomRotationToThreeY(rotationDeg) {
  return -Number(rotationDeg || 0) * Math.PI / 180;
}

function clearGroup(group) {
  while (group.children.length) {
    const child = group.children.pop();
    child.geometry?.dispose?.();
    child.material?.dispose?.();
    group.remove(child);
  }
}

function makeClientScene(current) {
  const corners = cornersFor(current.shape, current.size, current.geometry);
  return {
    room: {
      id: "room",
      name: presetTitle(current.shape),
      corners,
      height_m: current.size[2],
      materials: {
        wall: { id: current.materials.wall, name: current.materials.wall },
        floor: { id: current.materials.floor, name: current.materials.floor },
        ceiling: { id: current.materials.ceiling, name: current.materials.ceiling }
      },
      metadata: {
        shape: current.shape,
        geometry_model: current.shape === "rectangle" ? "rectangular room" : "extruded polygon",
        geometry_params: { ...(current.geometry || {}) }
      }
    },
    sources: [current.source],
    receivers: [current.receiver],
    objects: current.objects || [],
    paths: [],
    rt60: {},
    metadata: { source_model: { ...(current.sourceDirectivity || {}) } }
  };
}

function cornersFor(shape, size, params = state.geometry) {
  const [x, y] = size;
  const p = { ...defaultState.geometry, ...(params || {}) };
  if (shape === "triangle") return [[0, 0], [x, 0], [x * clamp(p.triangleApex, 0.05, 0.95), y]];
  if (shape === "circle") {
    const segments = Math.max(12, Math.round(p.circleSegments));
    return Array.from({ length: segments }, (_, i) => {
      const angle = Math.PI * 2 * i / segments;
      return [x * 0.5 + Math.cos(angle) * x * 0.5, y * 0.5 + Math.sin(angle) * y * 0.5];
    });
  }
  if (shape === "polygon") return polygonCorners(x, y, p);
  if (shape === "l_shape") {
    const cutoutW = clamp(p.lCutoutWidth, 0.15, 0.8);
    const cutoutD = clamp(p.lCutoutDepth, 0.15, 0.8);
    const innerX = x * (1 - cutoutW);
    const innerY = y * (1 - cutoutD);
    return [[0, 0], [x, 0], [x, innerY], [innerX, innerY], [innerX, y], [0, y]];
  }
  if (shape === "t_shape") {
    const stemW = x * clamp(p.tStemWidth, 0.18, 0.85);
    const stemX = (x - stemW) * clamp(p.tStemOffset, 0, 1);
    const headH = y * clamp(p.tHeadDepth, 0.15, 0.65);
    return [[0, 0], [x, 0], [x, headH], [stemX + stemW, headH], [stemX + stemW, y], [stemX, y], [stemX, headH], [0, headH]];
  }
  if (shape === "trapezoid") {
    const topW = x * clamp(p.trapezoidTopWidth, 0.2, 1);
    const topX = (x - topW) * clamp(p.trapezoidOffset, 0, 1);
    return [[0, 0], [x, 0], [topX + topW, y], [topX, y]];
  }
  if (shape === "u_shape") {
    const gapW = x * clamp(p.uOpeningWidth, 0.2, 0.72);
    const gapD = y * clamp(p.uOpeningDepth, 0.18, 0.82);
    const gapX = (x - gapW) * clamp(p.uOpeningOffset, 0, 1);
    const leftX = gapX;
    const rightX = gapX + gapW;
    const innerY = y - gapD;
    return [[0, 0], [x, 0], [x, y], [rightX, y], [rightX, innerY], [leftX, innerY], [leftX, y], [0, y]];
  }
  if (shape === "fan_shape") return fanCorners(x, y, p);
  return [[0, 0], [x, 0], [x, y], [0, y]];
}

function polygonCorners(width, depth, params) {
  const sides = Math.max(5, Math.min(12, Math.round(params.polygonSides)));
  const irregularity = clamp(params.polygonIrregularity, 0, 0.35);
  const skew = clamp(params.polygonSkew, -0.3, 0.3);
  const cx = width * 0.5;
  const cy = depth * 0.5;
  const raw = Array.from({ length: sides }, (_, i) => {
    const angle = -Math.PI * 0.5 + Math.PI * 2 * i / sides;
    const ripple = Math.sin((i + 1) * 1.7) * 0.5 + Math.cos((i + 2) * 2.3) * 0.5;
    const scale = 1 - irregularity * 0.5 + ripple * irregularity;
    const px = cx + Math.cos(angle) * width * 0.47 * scale;
    const py = cy + Math.sin(angle) * depth * 0.47 * scale;
    return [px + (py - cy) * skew, py];
  });
  return normalizeCorners(raw, width, depth, 0.02);
}

function normalizeCorners(corners, width, depth, padRatio = 0) {
  const bounds = getBounds(corners);
  const padX = width * padRatio;
  const padY = depth * padRatio;
  const spanX = Math.max(bounds.w, 1e-6);
  const spanY = Math.max(bounds.h, 1e-6);
  return corners.map(([px, py]) => [
    padX + ((px - bounds.x0) / spanX) * Math.max(width - padX * 2, 0.1),
    padY + ((py - bounds.y0) / spanY) * Math.max(depth - padY * 2, 0.1)
  ]);
}

function fanCorners(width, depth, params) {
  const angleDeg = clamp(params.fanAngle, 45, 150);
  const innerRatio = clamp(params.fanInnerRadius, 0.05, 0.55);
  const segments = Math.max(8, Math.min(48, Math.round(params.fanSegments)));
  const half = angleDeg * Math.PI / 360;
  const outer = Array.from({ length: segments + 1 }, (_, i) => {
    const a = -half + (2 * half * i / segments);
    return [Math.sin(a), Math.cos(a)];
  });
  const inner = Array.from({ length: segments + 1 }, (_, i) => {
    const a = half - (2 * half * i / segments);
    return [Math.sin(a) * innerRatio, Math.cos(a) * innerRatio];
  });
  return normalizeCorners([...outer, ...inner], width, depth, 0.02);
}

function renderThumbnails() {
  const container = document.getElementById("thumbs");
  container.innerHTML = "";
  presets.forEach((preset) => {
    const button = document.createElement("button");
    button.className = "thumb";
    button.dataset.shape = preset.id;
    button.innerHTML = `<canvas width="160" height="108"></canvas><span>${preset.title}</span>`;
    button.addEventListener("click", () => {
      state.shape = preset.id;
      applyPresetPoints();
      simData = makeClientScene(state);
      updateControls();
      requestSimulation();
    });
    container.appendChild(button);
    drawThumb(button.querySelector("canvas"), staticThumbnailCorners(preset.id));
  });
  refreshThumbnails();
}

function renderMicThumbnails() {
  const container = document.getElementById("micThumbs");
  container.innerHTML = "";
  micOptions.forEach((option) => {
    const button = document.createElement("button");
    button.className = "thumb micThumb";
    button.dataset.mic = option.id;
    button.innerHTML = `<canvas width="160" height="72"></canvas><span>${option.title}</span>`;
    button.addEventListener("click", () => {
      state.mic.type = option.id;
      syncMicControls();
      refreshMicThumbnails();
      simData = makeClientScene(state);
      rebuildThreeScene();
      requestSimulation();
    });
    container.appendChild(button);
    drawMicThumb(button.querySelector("canvas"), option.id);
  });
  refreshMicThumbnails();
}

function renderSourceDirectivityThumbnails() {
  const container = document.getElementById("sourceDirectivityThumbs");
  container.innerHTML = "";
  sourceDirectivityOptions.forEach((option) => {
    const button = document.createElement("button");
    button.className = "thumb sourceDirectivityThumb";
    button.dataset.sourceDirectivity = option.id;
    button.innerHTML = `<canvas width="160" height="72"></canvas><span>${option.title}</span>`;
    button.addEventListener("click", () => {
      state.sourceDirectivity = {
        ...state.sourceDirectivity,
        type: option.id,
        dipole_weight: option.dipole_weight,
        dipole_power: option.dipole_power
      };
      syncSourceDirectivityControls();
      simData = makeClientScene(state);
      rebuildThreeScene();
      requestSimulation();
    });
    container.appendChild(button);
    drawSourceDirectivityThumb(button.querySelector("canvas"), option);
  });
  refreshSourceDirectivityThumbnails();
}

function refreshSourceDirectivityThumbnails() {
  document.querySelectorAll(".sourceDirectivityThumb").forEach((button) => {
    button.classList.toggle("active", button.dataset.sourceDirectivity === state.sourceDirectivity.type);
  });
}

function drawSourceDirectivityThumb(thumbCanvas, option) {
  const ctx = thumbCanvas.getContext("2d");
  const w = thumbCanvas.width;
  const h = thumbCanvas.height;
  const cx = w * 0.44;
  const cy = h * 0.5;
  const radius = Math.min(w * 0.29, h * 0.38);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#d4dde1";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(14, cy);
  ctx.lineTo(w - 14, cy);
  ctx.stroke();
  ctx.beginPath();
  for (let index = 0; index <= 144; index += 1) {
    const angle = Math.PI * 2 * index / 144;
    const gain = Math.abs((1 - option.dipole_weight) + option.dipole_weight * Math.cos(angle)) ** option.dipole_power;
    const x = cx + Math.cos(angle) * radius * gain;
    const y = cy + Math.sin(angle) * radius * gain;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = "rgba(239,71,111,.14)";
  ctx.strokeStyle = "#ef476f";
  ctx.lineWidth = 2;
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#ef476f";
  ctx.beginPath();
  ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
  ctx.fill();
}

function renderObjectThumbnails() {
  const container = document.getElementById("objectThumbs");
  if (!container) return;
  container.innerHTML = "";
  objectTypeOptions.forEach((option) => {
    const spec = furnitureCatalog[option.id];
    const button = document.createElement("button");
    button.className = "thumb objectThumb";
    button.dataset.objectType = option.id;
    button.innerHTML = `<canvas width="160" height="108"></canvas><span>${option.title}</span>`;
    button.addEventListener("click", () => {
      if (hasUnconfirmedObjectChange() && selectedObjectId !== unconfirmedObjectId()) {
        selectSceneObject(unconfirmedObjectId());
        return;
      }
      const selected = sceneObjectById(selectedObjectId);
      if (selected) handleObjectTypeChange(option.id);
      else {
        setActiveObjectType(option.id);
        setObjectControlDraft(option.id);
        setStatus(`${spec.title} selected. Set dimensions, then Add geometry.`);
      }
    });
    container.appendChild(button);
    drawObjectThumb(button.querySelector("canvas"), option.id);
  });
  refreshObjectThumbnails();
}

function refreshObjectThumbnails(type = activeObjectType()) {
  document.querySelectorAll(".objectThumb").forEach((button) => {
    button.classList.toggle("active", button.dataset.objectType === type);
  });
}

function setActiveObjectType(type) {
  refreshObjectThumbnails(furnitureCatalog[type] ? type : objectTypeOptions[0].id);
}

function activeObjectType() {
  return document.querySelector(".objectThumb.active")?.dataset.objectType || objectTypeOptions[0].id;
}

function drawObjectThumb(thumbCanvas, type) {
  const ctx = thumbCanvas.getContext("2d");
  const w = thumbCanvas.width;
  const h = thumbCanvas.height;
  const spec = furnitureCatalog[type] || furnitureCatalog.cuboid;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#d4dde1";
  ctx.lineWidth = 1;
  ctx.strokeRect(10.5, 10.5, w - 21, h - 21);
  const color = `#${(spec.color || 0x8d7463).toString(16).padStart(6, "0")}`;
  ctx.fillStyle = color;
  ctx.strokeStyle = "#20282e";
  ctx.lineWidth = 1.5;
  if (type === "panel") {
    ctx.save();
    ctx.translate(w * 0.5, h * 0.55);
    ctx.rotate(-0.28);
    ctx.fillRect(-42, -7, 84, 14);
    ctx.strokeRect(-42, -7, 84, 14);
    ctx.restore();
  } else if (type === "low_block") {
    ctx.fillRect(w * 0.23, h * 0.58, w * 0.54, h * 0.16);
    ctx.strokeRect(w * 0.23, h * 0.58, w * 0.54, h * 0.16);
  } else {
    ctx.fillRect(w * 0.28, h * 0.35, w * 0.44, h * 0.34);
    ctx.strokeRect(w * 0.28, h * 0.35, w * 0.44, h * 0.34);
  }
}

function refreshMicThumbnails() {
  document.querySelectorAll(".micThumb").forEach((button) => {
    button.classList.toggle("active", button.dataset.mic === state.mic.type);
  });
}

function drawMicThumb(thumbCanvas, type) {
  const ctx = thumbCanvas.getContext("2d");
  const w = thumbCanvas.width;
  const h = thumbCanvas.height;
  const cx = w * 0.5;
  const cy = h * 0.5;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#d4dde1";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(18, cy);
  ctx.lineTo(w - 18, cy);
  ctx.stroke();
  const drawDot = (x, y, r = 5, color = "#0f7f9f") => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  };
  if (type === "mono") {
    drawDot(cx, cy, 7);
  } else if (type === "hrtf") {
    drawDot(cx - 15, cy, 5, "#7d8cff");
    drawDot(cx + 15, cy, 5, "#7d8cff");
    ctx.strokeStyle = "#7d8cff";
    ctx.beginPath();
    ctx.arc(cx, cy, 20, Math.PI * 1.15, Math.PI * 1.85);
    ctx.stroke();
  } else if (type === "linear") {
    for (let i = 0; i < 4; i += 1) drawDot(cx - 30 + i * 20, cy, 4.8, "#7d8cff");
  } else if (type === "circular") {
    for (let i = 0; i < 8; i += 1) {
      const a = Math.PI * 2 * i / 8;
      drawDot(cx + Math.cos(a) * 25, cy + Math.sin(a) * 25, 4.2, "#7d8cff");
    }
    ctx.strokeStyle = "rgba(125,140,255,.45)";
    ctx.beginPath();
    ctx.arc(cx, cy, 25, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function refreshThumbnails() {
  document.querySelectorAll(".thumb[data-shape]").forEach((button) => {
    button.classList.toggle("active", button.dataset.shape === state.shape);
  });
}

function staticThumbnailCorners(shape) {
  return cornersFor(shape, [6.0, 4.0, 2.8], defaultState.geometry);
}

function drawThumb(thumbCanvas, corners) {
  const ctx = thumbCanvas.getContext("2d");
  drawPlanCanvas(ctx, thumbCanvas.width, thumbCanvas.height, corners, [], [0, 0], [0, 0], false);
}

function updatePanels() {
  const paths = simData.paths || [];
  updatePathLimitLabel();
  document.getElementById("hudMeta").textContent = `${presetTitle(state.shape)} | ${paths.length} paths | ${state.config.fs} Hz`;
  document.getElementById("receiverType").textContent = state.mic.type;
  document.getElementById("sourceDirectivityType").textContent = state.sourceDirectivity.type;
  statsEl.innerHTML = statsHtml(paths);
  codeEl.textContent = acousticAgentCode();
  drawMiniMap();
  safeDrawRirPanel();
  refreshMicThumbnails();
  refreshSourceDirectivityThumbnails();
  refreshThumbnails();
  const countLabel = document.getElementById("sceneObjectCount");
  if (countLabel) {
    const count = (state.objects || []).length;
    countLabel.textContent = `${count} object${count === 1 ? "" : "s"}`;
  }
  refreshObjectThumbnails(sceneObjectById(selectedObjectId)?.type || activeObjectType());
}

function updatePathLimitLabel() {
  const value = Number(document.getElementById("pathLimit")?.value || 0);
  document.getElementById("pathCount").textContent = `${value} max`;
}

function safeDrawRirPanel() {
  try {
    drawRirPanel();
  } catch (error) {
    drawRirFallback(String(error?.message || error));
    console.warn("RIR panel render failed", error);
  }
}

function drawRirFallback(message) {
  const canvasEl = document.getElementById("rirCanvas");
  if (!canvasEl) return;
  const ctx = canvasEl.getContext("2d");
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  ctx.fillStyle = "#f8fafb";
  ctx.fillRect(0, 0, canvasEl.width, canvasEl.height);
  ctx.fillStyle = "#a73e30";
  ctx.font = "11px system-ui";
  ctx.fillText("RIR preview unavailable", 12, 22);
  ctx.fillStyle = "#69767d";
  ctx.fillText(message.slice(0, 44), 12, 40);
}

function statsHtml(paths) {
  const rt60 = simData.rt60 || {};
  const rir = simData.rir || {};
  const metrics = rir.metrics || {};
  const rirRt60 = rt60.rir_rt60_s ?? rt60.rt60_s;
  const materialRt60 = rt60.material_rt60_s;
  const rows = [
    ["RIR RT60", formatSeconds(rirRt60)],
    ["Material RT60", formatSeconds(materialRt60)],
    ["DRR", formatDb(metrics.drr_db)],
    ["C50 / C80", `${formatDb(metrics.c50_db)} / ${formatDb(metrics.c80_db)}`],
    ["RIR peak", `${formatDb(metrics.peak_dbfs)} @ ${formatMs(metrics.peak_time_ms)}`],
    ["RMS level", formatDb(metrics.rms_dbfs)],
    ["Length", `${formatSeconds(rir.duration_s)} | ${Number(rir.channel_count || 1)} ch`]
  ];
  return rows.map(([k, v]) => `<div class="stat"><span>${k}</span><strong>${v}</strong></div>`).join("");
}

function drawMiniMap() {
  const mini = document.getElementById("miniCanvas");
  if (!mini) return;
  const ctx = mini.getContext("2d");
  drawAcousticMiniMap(ctx, mini.width, mini.height);
}

function drawAcousticMiniMap(ctx, width, height) {
  const corners = simData.room.corners;
  const bounds = getBounds(corners);
  const pad = 17;
  const scale = Math.min((width - pad * 2) / Math.max(bounds.w, 1e-6), (height - pad * 2) / Math.max(bounds.h, 1e-6));
  const toCanvas = ([x, y]) => [pad + (x - bounds.x0) * scale, pad + (y - bounds.y0) * scale];
  const selection = miniMapPathSelection(simData.paths || []);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f7fafb";
  ctx.fillRect(0, 0, width, height);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  ctx.beginPath();
  corners.forEach((point, index) => {
    const [x, y] = toCanvas(point);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = "#e8eef0";
  ctx.strokeStyle = "#344149";
  ctx.lineWidth = 1.8;
  ctx.fill();
  ctx.stroke();

  drawMiniFurniture(ctx, toCanvas, scale, 1);

  if (selection.direct) {
    drawMiniAcousticPath(ctx, selection.direct, toCanvas, "rgba(239,71,111,.96)", 2.25, false, selection.nlos);
  }
  selection.diffractions.forEach((path, index) => {
    drawMiniAcousticPath(ctx, path, toCanvas, index === 0 ? "rgba(125,140,255,.96)" : "rgba(125,140,255,.62)", index === 0 ? 2.15 : 1.35, true, false);
  });

  drawMiniSourceDirectivity(ctx, toCanvas(state.source));
  drawMiniMarker(ctx, toCanvas(state.source), "#ef476f", "SRC", 1);
  drawMiniMarker(ctx, toCanvas(state.receiver), "#0f7f9f", "MIC", -1);

  const stateLabel = selection.nlos ? "NLOS / UTD" : "LOS";
  ctx.font = "700 9px system-ui";
  const badgeWidth = ctx.measureText(stateLabel).width + 12;
  ctx.fillStyle = selection.nlos ? "rgba(125,140,255,.14)" : "rgba(239,71,111,.12)";
  ctx.fillRect(width - badgeWidth - 6, 6, badgeWidth, 17);
  ctx.fillStyle = selection.nlos ? "#5968d8" : "#c93658";
  ctx.fillText(stateLabel, width - badgeWidth, 18);

  ctx.fillStyle = "#69767d";
  ctx.font = "600 9px system-ui";
  ctx.fillText(`RT ${selection.rtCount}`, 7, height - 7);
}

function drawMiniSourceDirectivity(ctx, point) {
  const model = state.sourceDirectivity;
  const yaw = Number(model.orientation_deg || 0) * Math.PI / 180;
  const radius = model.type === "omni" ? 9 : 15;
  ctx.save();
  ctx.translate(point[0], point[1]);
  ctx.rotate(yaw);
  ctx.beginPath();
  for (let index = 0; index <= 96; index += 1) {
    const angle = Math.PI * 2 * index / 96;
    const gain = Math.abs((1 - model.dipole_weight) + model.dipole_weight * Math.cos(angle)) ** model.dipole_power;
    const x = Math.cos(angle) * radius * gain;
    const y = Math.sin(angle) * radius * gain;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = "rgba(239,71,111,.11)";
  ctx.strokeStyle = "rgba(239,71,111,.72)";
  ctx.lineWidth = 1.2;
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawMiniFurniture(ctx, project, scale, rotationSign = -1) {
  (state.objects || []).forEach((object) => {
    const spec = furnitureCatalog[object.type] || furnitureCatalog.cuboid;
    const [width, depth] = object.size || spec.size;
    const [cx, cy] = project(object.position || [0, 0]);
    const rotation = rotationSign * Number(object.rotation || 0) * Math.PI / 180;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(rotation);
    ctx.fillStyle = object.id === selectedObjectId ? "rgba(18, 111, 93, 0.18)" : "rgba(74, 88, 95, 0.13)";
    ctx.strokeStyle = object.id === selectedObjectId ? "#126f5d" : "rgba(74, 88, 95, 0.72)";
    ctx.lineWidth = object.id === selectedObjectId ? 1.8 : 1.1;
    ctx.fillRect(-width * scale * 0.5, -depth * scale * 0.5, width * scale, depth * scale);
    ctx.strokeRect(-width * scale * 0.5, -depth * scale * 0.5, width * scale, depth * scale);
    ctx.restore();
  });
}

function miniMapPathSelection(paths) {
  const direct = paths.find((path) => path.kind === "direct" || path.kind === "direct_transmitted");
  const visibleByGeometry = segmentInsidePolygon2D(state.source, state.receiver, simData.room?.corners || cornersFor(state.shape, state.size, state.geometry));
  const nlos = direct?.kind === "direct_transmitted" || Number(simData.metadata?.steam_audio?.direct?.occlusion ?? (visibleByGeometry ? 1 : 0)) < 1;
  const diffraction = displayPaths(paths)
    .filter((path) => path.kind === "diffraction")
    .sort((a, b) => Number(a.delay_s || 0) - Number(b.delay_s || 0));
  return {
    nlos,
    direct: direct || { kind: nlos ? "direct_transmitted" : "direct", points: [state.source, state.receiver] },
    diffractions: diffraction.slice(0, 3),
    rtCount: paths.filter((path) => path.kind === "rt_reflection").length,
  };
}

function drawMiniAcousticPath(ctx, path, project, color, width, markHits, dashed = false) {
  const points = path.points || [];
  if (points.length < 2) return;
  ctx.beginPath();
  points.forEach((point, index) => {
    const [x, y] = project(point);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dashed ? [5, 4] : []);
  ctx.stroke();
  ctx.setLineDash([]);
  if (!markHits) return;
  points.slice(1, -1).forEach((point) => {
    const [x, y] = project(point);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 2.1, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawMiniMarker(ctx, point, color, label, direction) {
  ctx.fillStyle = "rgba(255,255,255,.96)";
  ctx.beginPath();
  ctx.arc(point[0], point[1], 5.8, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(point[0], point[1], 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.font = "750 8px system-ui";
  ctx.textAlign = direction > 0 ? "left" : "right";
  ctx.fillText(label, point[0] + direction * 7, point[1] - 6);
  ctx.textAlign = "left";
}

function drawRirPanel() {
  const canvasEl = document.getElementById("rirCanvas");
  if (!canvasEl) return;
  const ctx = canvasEl.getContext("2d");
  const w = canvasEl.width;
  const h = canvasEl.height;
  const rir = simData.rir || {};
  const channels = rirWaveChannels(rir);
  const stride = Math.max(1, Number(rir.sample_stride || 1));
  const fs = Math.max(1, Number(rir.fs || state.config.fs));
  const paths = (simData.paths || []).filter(pathLayerVisible);
  const rt60 = simData.rt60 || {};
  const duration = Number(rir.duration_s || state.config.duration_s);
  const maxDelay = Math.max(0.05, duration);
  const maxChannelLength = Math.max(0, ...channels.map((channel) => channel.samples.length));
  const visibleSamples = Math.min(maxChannelLength, Math.ceil(maxDelay * fs / stride));
  const padL = 38;
  const padR = 12;
  const padT = 26;
  const padB = 24;
  const gap = 12;
  const plotW = w - padL - padR;
  const waveH = 74;
  const decayH = h - padT - padB - gap - waveH;
  const waveTop = padT;
  const waveMid = waveTop + waveH * 0.52;
  const decayTop = waveTop + waveH + gap;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#f8fafb";
  ctx.fillRect(0, 0, w, h);
  drawRirLegend(ctx, padL, 9, channels);
  drawRirPlotFrame(ctx, padL, waveTop, plotW, waveH, rirWaveAxisLabel());
  drawRirPlotFrame(ctx, padL, decayTop, plotW, decayH, "Energy decay", ["0", "-20", "-40", "-60", "-80"]);
  drawLateTailRegion(ctx, padL, waveTop, plotW, waveH + gap + decayH, maxDelay);

  if (visibleSamples > 0) {
    drawRirWaveforms(ctx, channels, visibleSamples, padL, waveMid, plotW, waveH, maxDelay, stride, fs);
    drawRirEnergyDecayCurve(ctx, channels, visibleSamples, maxChannelLength, padL, decayTop, plotW, decayH, maxDelay, stride, fs);
  }
  drawRirRt60Badge(ctx, padL + plotW, 9, rt60);
  drawRirEventTicks(ctx, paths, padL, waveTop, plotW, waveH, maxDelay);
  drawRirPeakMarker(ctx, padL, waveTop, plotW, waveH, maxDelay);
  ctx.fillStyle = "#69767d";
  ctx.font = "10px system-ui";
  ctx.fillText("0 ms", padL, h - 7);
  ctx.textAlign = "right";
  ctx.fillText(`${Math.round(maxDelay * 1000)} ms`, padL + plotW, h - 7);
  ctx.textAlign = "left";
}

function rirWaveChannels(rir) {
  const channelSamples = Array.isArray(rir.channel_samples) ? rir.channel_samples : null;
  const labels = Array.isArray(rir.channel_labels) ? rir.channel_labels : [];
  if (channelSamples && channelSamples.length > 0) {
    return channelSamples.slice(0, 2).map((samples, index) => ({
      label: labels[index] || `Ch ${index + 1}`,
      color: index === 0 ? "rgba(211,126,42,0.96)" : "rgba(33,135,170,0.9)",
      samples: Array.isArray(samples) ? samples.map((item) => Number(item) || 0) : [],
    }));
  }
  const samples = Array.isArray(rir.samples) ? rir.samples.map((item) => Number(item) || 0) : [];
  return [{ label: "Ch 1", color: "rgba(211,126,42,0.96)", samples }];
}

function rirWaveAxisLabel() {
  if (state.mic.type === "hrtf") return "Binaural";
  if (isArrayMic(state.mic.type)) return "Channels";
  return "Pressure";
}

function drawRirLegend(ctx, x, y, channels) {
  const channelItems = channels.length > 1
    ? channels.map((channel) => [channel.label, channel.color])
    : [];
  const items = [...channelItems, ["Direct", "#ef476f"], ["Late field", "#7d8cff"]];
  let cursor = x;
  ctx.font = "10px system-ui";
  items.forEach(([label, color]) => {
    ctx.fillStyle = color;
    ctx.fillRect(cursor, y + 4, 13, 3);
    ctx.fillStyle = "#69767d";
    ctx.fillText(label, cursor + 17, y + 9);
    cursor += ctx.measureText(label).width + 38;
  });
}

function drawRirRt60Badge(ctx, right, y, rt60) {
  const value = rt60?.rir_rt60_s ?? rt60?.rt60_s;
  if (!Number.isFinite(Number(value))) return;
  const text = `RIR RT60 ${formatSeconds(value)}`;
  ctx.save();
  ctx.font = "700 10px system-ui";
  const width = ctx.measureText(text).width + 14;
  const x = right - width;
  ctx.fillStyle = "rgba(18,111,93,0.10)";
  ctx.strokeStyle = "rgba(18,111,93,0.22)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y - 2, width, 17, 7);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#126f5d";
  ctx.fillText(text, x + 7, y + 10);
  ctx.restore();
}

function drawRirWaveforms(ctx, channels, visibleSamples, padL, waveMid, plotW, waveH, maxDelay, stride, fs) {
  let maxAbs = 1e-9;
  channels.forEach((channel) => {
    for (let index = 0; index < visibleSamples; index += 1) {
      maxAbs = Math.max(maxAbs, Math.abs(channel.samples[index] || 0));
    }
  });
  const step = Math.max(1, Math.ceil(visibleSamples / Math.max(96, plotW)));
  channels.forEach((channel) => {
    ctx.strokeStyle = channel.color;
    ctx.lineWidth = channels.length > 1 ? 1.0 : 1.15;
    ctx.beginPath();
    for (let start = 0; start < visibleSamples; start += step) {
      let lo = 0;
      let hi = 0;
      for (let index = start; index < Math.min(visibleSamples, start + step); index += 1) {
        const sample = channel.samples[index] || 0;
        lo = Math.min(lo, sample);
        hi = Math.max(hi, sample);
      }
      const time = start * stride / fs;
      const x = padL + time / maxDelay * plotW;
      ctx.moveTo(x, waveMid - hi / maxAbs * waveH * 0.42);
      ctx.lineTo(x, waveMid - lo / maxAbs * waveH * 0.42);
    }
    ctx.stroke();
  });
}

function drawRirPlotFrame(ctx, x, y, width, height, axisLabel, dbTicks = null) {
  ctx.strokeStyle = "rgba(76, 90, 99, 0.18)";
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, width, height);
  drawYAxisLabel(ctx, axisLabel, x - 28, y + height * 0.5);
  ctx.strokeStyle = "rgba(76, 90, 99, 0.12)";
  ctx.beginPath();
  for (let i = 1; i < 4; i += 1) {
    const gy = y + height * i / 4;
    ctx.moveTo(x, gy);
    ctx.lineTo(x + width, gy);
  }
  ctx.stroke();
  if (!dbTicks) return;
  ctx.fillStyle = "#8a969d";
  ctx.font = "9px system-ui";
  dbTicks.forEach((tick, index) => {
    const ty = y + height * index / Math.max(1, dbTicks.length - 1);
    ctx.textAlign = "right";
    ctx.fillText(tick, x - 5, ty + 3);
  });
  ctx.textAlign = "left";
}

function drawYAxisLabel(ctx, label, x, y) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#69767d";
  ctx.font = "9px system-ui";
  ctx.textAlign = "center";
  ctx.fillText(label, 0, 0);
  ctx.restore();
  ctx.textAlign = "left";
}

function drawRirEnergyDecayCurve(ctx, channels, visibleSamples, fullSamples, padL, top, plotW, plotH, maxDelay, stride, fs) {
  const minDb = -80;
  const decayDb = rirEnergyDecayDbSamples(channels, visibleSamples, fullSamples);
  drawMaterialDecayReference(ctx, padL, top, plotW, plotH, maxDelay, minDb);
  ctx.strokeStyle = "rgba(18,111,93,0.92)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const plotSamples = Math.min(visibleSamples, decayDb.length);
  const step = Math.max(1, Math.ceil(plotSamples / Math.max(96, plotW)));
  const indices = [];
  for (let index = 0; index < plotSamples; index += step) indices.push(index);
  if (plotSamples > 0 && indices[indices.length - 1] !== plotSamples - 1) indices.push(plotSamples - 1);
  let finalDb = 0;
  indices.forEach((index, pointIndex) => {
    const db = Math.max(minDb, Number(decayDb[index]) || minDb);
    finalDb = db;
    const time = index * stride / fs;
    const x = padL + time / maxDelay * plotW;
    const y = rirDecayDbToY(db, top, plotH, minDb);
    if (pointIndex === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  if (indices.length > 0) {
    ctx.lineTo(padL + plotW, rirDecayDbToY(finalDb, top, plotH, minDb));
  }
  ctx.stroke();
  drawDecayReachStatus(ctx, finalDb, padL, top, plotW, plotH, minDb);

  ctx.save();
  ctx.fillStyle = "#789097";
  ctx.font = "8.5px system-ui";
  ctx.textAlign = "right";
  ctx.fillText("Full-RIR Schroeder decay", padL + plotW - 4, top + 10);
  ctx.restore();
  ctx.textAlign = "left";
}

function rirEnergyDecayDbSamples(channels, visibleSamples, fullSamples) {
  const backendDecay = Array.isArray(simData.rir?.decay_db)
    ? simData.rir.decay_db.map((value) => Number(value)).filter((value) => Number.isFinite(value))
    : [];
  if (backendDecay.length > 0) return backendDecay;
  const totalSamples = Math.max(visibleSamples, fullSamples);
  const energy = Array.from({ length: totalSamples }, (_, index) => {
    let sum = 0;
    channels.forEach((channel) => {
      const sample = channel.samples[index] || 0;
      sum += sample * sample;
    });
    return sum;
  });
  const schroeder = new Array(energy.length);
  let cumulative = 0;
  for (let index = energy.length - 1; index >= 0; index -= 1) {
    cumulative += energy[index];
    schroeder[index] = cumulative;
  }
  const totalEnergy = Math.max(1e-18, schroeder[0] || 0);
  return schroeder.map((value) => 10 * Math.log10(Math.max(value, 1e-18) / totalEnergy));
}

function drawDecayReachStatus(ctx, finalDb, padL, top, plotW, plotH, minDb) {
  const targetDb = -60;
  const y60 = rirDecayDbToY(targetDb, top, plotH, minDb);
  ctx.save();
  ctx.strokeStyle = finalDb > targetDb ? "rgba(239,71,111,0.28)" : "rgba(18,111,93,0.18)";
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(padL, y60);
  ctx.lineTo(padL + plotW, y60);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = finalDb > targetDb ? "#c93658" : "#789097";
  ctx.font = "8.5px system-ui";
  ctx.textAlign = "left";
  ctx.fillText(finalDb > targetDb ? `tail still ${finalDb.toFixed(1)} dB` : "reaches -60 dB", padL + 4, y60 - 3);
  ctx.restore();
  ctx.textAlign = "left";
}

function drawMaterialDecayReference(ctx, padL, top, plotW, plotH, maxDelay, minDb) {
  const materialRt60 = Number(simData.rt60?.material_rt60_s);
  if (!Number.isFinite(materialRt60) || materialRt60 <= 0) return;
  ctx.save();
  ctx.strokeStyle = "rgba(105,118,125,0.58)";
  ctx.lineWidth = 1.1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  const points = 48;
  for (let i = 0; i <= points; i += 1) {
    const t = maxDelay * i / points;
    const db = Math.max(minDb, -60 * t / materialRt60);
    const x = padL + t / maxDelay * plotW;
    const y = rirDecayDbToY(db, top, plotH, minDb);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#789097";
  ctx.font = "8.5px system-ui";
  ctx.textAlign = "left";
  ctx.fillText("Material slope", padL + 4, top + 10);
  ctx.restore();
  ctx.textAlign = "left";
}

function rirDecayDbToY(value, top, plotH, minDb) {
  const db = Number(value);
  if (!Number.isFinite(db)) return NaN;
  return top + (0 - Math.max(minDb, Math.min(0, db))) / (0 - minDb) * plotH;
}

function drawRirEventTicks(ctx, paths, padL, top, plotW, plotH, maxDelay) {
  const limit = Number(document.getElementById("pathLimit")?.value || 512);
  const visible = paths.slice(0, limit);
  const maxGain = Math.max(1e-9, ...visible.map((path) => Math.abs(Number(path.gain || 0))));
  visible.forEach((path) => {
    if (Number(path.delay_s || 0) > maxDelay) return;
    const x = padL + Number(path.delay_s || 0) / maxDelay * plotW;
    const height = 6 + Math.abs(Number(path.gain || 0)) / maxGain * 18;
    const style = pathStyle(path);
    ctx.strokeStyle = `#${style.color.toString(16).padStart(6, "0")}`;
    ctx.globalAlpha = Math.min(0.9, style.opacity + 0.05);
    ctx.lineWidth = path.kind === "direct" || path.kind === "direct_transmitted" ? 2 : 1;
    ctx.beginPath();
    ctx.moveTo(x, top + plotH - 3);
    ctx.lineTo(x, top + plotH - 3 - height);
    ctx.stroke();
  });
  ctx.globalAlpha = 1;
}

function drawRirPeakMarker(ctx, padL, top, plotW, plotH, maxDelay) {
  const metrics = simData.rir?.metrics || {};
  const delayMs = Number(metrics.peak_time_ms ?? metrics.direct_delay_ms);
  if (!Number.isFinite(delayMs)) return;
  const delay = delayMs / 1000;
  if (delay < 0 || delay > maxDelay) return;
  const x = padL + delay / maxDelay * plotW;
  ctx.strokeStyle = "rgba(239,71,111,0.58)";
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(x, top);
  ctx.lineTo(x, top + plotH);
  ctx.stroke();
  ctx.setLineDash([]);
  const peakLevel = formatDb(metrics.peak_dbfs).replace(" dB", " dBFS");
  const label = `peak ${peakLevel} @ ${formatMs(delayMs)}`;
  ctx.save();
  ctx.font = "700 8.5px system-ui";
  const labelWidth = ctx.measureText(label).width + 8;
  const lx = Math.min(Math.max(x + 4, padL + 2), padL + plotW - labelWidth - 2);
  ctx.fillStyle = "rgba(255,255,255,0.86)";
  ctx.fillRect(lx, top + 3, labelWidth, 14);
  ctx.fillStyle = "#c93658";
  ctx.fillText(label, lx + 4, top + 13);
  ctx.restore();
}

function drawLateTailRegion(ctx, padL, top, plotW, plotH, maxDelay) {
  const reflections = simData.metadata?.steam_audio?.reflections || {};
  if (!reflections.late_tail_enabled) return;
  const cutoff = Number(reflections.late_tail?.transition_start_s ?? reflections.late_tail_cutoff_s ?? 0.75);
  const x0 = padL + Math.min(cutoff, maxDelay) / maxDelay * plotW;
  ctx.fillStyle = "rgba(125,140,255,0.08)";
  ctx.fillRect(x0, top, padL + plotW - x0, plotH);
}

function drawPlanCanvas(ctx, width, height, corners, paths, source, receiver, includePaths) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  const bounds = getBounds(corners);
  const pad = 14;
  const scale = Math.min((width - pad * 2) / Math.max(bounds.w, 1e-6), (height - pad * 2) / Math.max(bounds.h, 1e-6));
  const toCanvas = ([x, y]) => [pad + (x - bounds.x0) * scale, height - pad - (y - bounds.y0) * scale];
  ctx.beginPath();
  corners.forEach((point, index) => {
    const [x, y] = toCanvas(point);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = "#e8eef0";
  ctx.strokeStyle = "#20282e";
  ctx.lineWidth = 2;
  ctx.fill();
  ctx.stroke();
  if (includePaths) {
    displayPaths(paths).slice(0, 36).forEach((path) => {
      const points = path.points || [];
      ctx.beginPath();
      points.forEach((point, index) => {
        const [x, y] = toCanvas(point);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = path.kind === "direct" || path.kind === "direct_transmitted" ? "rgba(239,71,111,.9)" : path.kind === "diffraction" ? "rgba(125,140,255,.82)" : "rgba(18,111,93,.62)";
      ctx.lineWidth = path.kind === "direct" || path.kind === "direct_transmitted" ? 2 : 1;
      ctx.setLineDash(path.kind === "direct_transmitted" ? [5, 4] : []);
      ctx.stroke();
      ctx.setLineDash([]);
      if (path.kind === "diffraction") {
        points.slice(1, -1).forEach((point) => {
          const [x, y] = toCanvas(point);
          ctx.fillStyle = "rgba(125,140,255,.9)";
          ctx.beginPath();
          ctx.arc(x, y, 2.1, 0, Math.PI * 2);
          ctx.fill();
        });
      }
    });
    drawCanvasPoint(ctx, toCanvas(source), "#ef476f");
    drawCanvasPoint(ctx, toCanvas(receiver), "#0f7f9f");
  }
}

function drawCanvasPoint(ctx, point, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(point[0], point[1], 4, 0, Math.PI * 2);
  ctx.fill();
}

async function updateCalibrationAudio(scene, requestSeq) {
  const token = ++calibrationAudioSeq;
  const rirInfo = scene?.rir || {};
  const fs = Number(rirInfo.fs || state.config.fs);
  try {
    if (!rirInfo.wav_url || !Array.isArray(rirInfo.shape)) throw new Error("exact RIR is unavailable");
    setCalibrationAudioMeta(`reading.wav · 44.1 → ${formatSampleRate(fs)}`);
    const [dryResponse, rirResponse] = await Promise.all([
      fetch(`/api/calibration-audio?fs=${encodeURIComponent(fs)}`, { cache: "no-store" }),
      fetch(rirInfo.wav_url, { cache: "no-store" }),
    ]);
    if (!dryResponse.ok) throw new Error(await dryResponse.text());
    if (!rirResponse.ok) throw new Error(await rirResponse.text());
    const dry = decodePcm16Wav(await dryResponse.arrayBuffer());
    if (dry.fs !== fs) throw new Error(`dry sample rate is ${dry.fs} Hz`);
    if (token !== calibrationAudioSeq || requestSeq !== simulationRequestSeq) return;
    const rir = decodeFloat32WavFirstChannel(await rirResponse.arrayBuffer());
    if (rir.fs !== fs) throw new Error(`RIR sample rate is ${rir.fs} Hz`);
    const wet = await convolveMono(dry.samples, rir.samples, fs);
    if (token !== calibrationAudioSeq || requestSeq !== simulationRequestSeq) return;

    const sharedGain = 0.98 / Math.max(1.0, maxAbs(dry.samples), maxAbs(wet));
    const nextUrls = [
      URL.createObjectURL(encodePcm16Wav(scaledSamples(dry.samples, sharedGain), fs)),
      URL.createObjectURL(encodePcm16Wav(scaledSamples(wet, sharedGain), fs)),
    ];
    replaceCalibrationAudioUrls(nextUrls);
    dryAudioEl.src = nextUrls[0];
    wetAudioEl.src = nextUrls[1];
    dryAudioEl.load();
    wetAudioEl.load();
    setCalibrationAudioMeta(`reading.wav · 44.1 → ${formatSampleRate(fs)} · ready`);
  } catch (error) {
    if (token !== calibrationAudioSeq || requestSeq !== simulationRequestSeq) return;
    clearCalibrationAudio(`Audio unavailable · ${String(error?.message || error).slice(0, 42)}`);
  }
}

function clearCalibrationAudio(label = "reading.wav · waiting") {
  calibrationAudioSeq += 1;
  [dryAudioEl, wetAudioEl].forEach((audio) => {
    if (!audio) return;
    audio.pause();
    audio.removeAttribute("src");
    audio.load();
  });
  replaceCalibrationAudioUrls([]);
  setCalibrationAudioMeta(label);
}

function replaceCalibrationAudioUrls(nextUrls) {
  calibrationAudioUrls.forEach((url) => URL.revokeObjectURL(url));
  calibrationAudioUrls = nextUrls;
}

function setCalibrationAudioMeta(label) {
  if (calibrationAudioMetaEl) calibrationAudioMetaEl.textContent = label;
}

function formatSampleRate(fs) {
  const khz = Number(fs) / 1000;
  return `${Number.isInteger(khz) ? khz.toFixed(0) : khz.toFixed(1)} kHz`;
}

function decodePcm16Wav(buffer) {
  const view = new DataView(buffer);
  if (readAscii(view, 0, 4) !== "RIFF" || readAscii(view, 8, 4) !== "WAVE") throw new Error("invalid dry WAV");
  let offset = 12;
  let format = null;
  let dataOffset = -1;
  let dataLength = 0;
  while (offset + 8 <= view.byteLength) {
    const id = readAscii(view, offset, 4);
    const length = view.getUint32(offset + 4, true);
    const start = offset + 8;
    if (id === "fmt " && length >= 16) {
      format = {
        tag: view.getUint16(start, true),
        channels: view.getUint16(start + 2, true),
        fs: view.getUint32(start + 4, true),
        bits: view.getUint16(start + 14, true),
      };
    } else if (id === "data") {
      dataOffset = start;
      dataLength = Math.min(length, view.byteLength - start);
    }
    offset = start + length + (length & 1);
  }
  if (!format || format.tag !== 1 || format.bits !== 16 || dataOffset < 0) throw new Error("dry WAV must be PCM16");
  const frames = Math.floor(dataLength / (2 * format.channels));
  const samples = new Float32Array(frames);
  for (let frame = 0; frame < frames; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < format.channels; channel += 1) {
      sum += view.getInt16(dataOffset + (frame * format.channels + channel) * 2, true) / 32768;
    }
    samples[frame] = sum / format.channels;
  }
  return { samples, fs: format.fs };
}

function decodeFloat32WavFirstChannel(buffer) {
  const view = new DataView(buffer);
  if (readAscii(view, 0, 4) !== "RIFF" || readAscii(view, 8, 4) !== "WAVE") throw new Error("invalid RIR WAV");
  let offset = 12;
  let format = null;
  let dataOffset = -1;
  let dataLength = 0;
  while (offset + 8 <= view.byteLength) {
    const id = readAscii(view, offset, 4);
    const length = view.getUint32(offset + 4, true);
    const start = offset + 8;
    if (id === "fmt " && length >= 16) {
      format = {
        tag: view.getUint16(start, true),
        channels: view.getUint16(start + 2, true),
        fs: view.getUint32(start + 4, true),
        bits: view.getUint16(start + 14, true),
      };
    } else if (id === "data") {
      dataOffset = start;
      dataLength = Math.min(length, view.byteLength - start);
    }
    offset = start + length + (length & 1);
  }
  if (!format || format.tag !== 3 || format.bits !== 32 || format.channels < 1 || dataOffset < 0) {
    throw new Error("RIR WAV must be IEEE float32");
  }
  const frames = Math.floor(dataLength / (4 * format.channels));
  const samples = new Float32Array(frames);
  for (let frame = 0; frame < frames; frame += 1) {
    samples[frame] = view.getFloat32(dataOffset + frame * format.channels * 4, true);
  }
  return { samples, fs: format.fs };
}

async function convolveMono(dry, rir, fs) {
  const OfflineContext = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!OfflineContext) throw new Error("offline audio rendering is unsupported");
  const outputLength = dry.length + rir.length - 1;
  const context = new OfflineContext(1, outputLength, fs);
  const sourceBuffer = context.createBuffer(1, dry.length, fs);
  sourceBuffer.copyToChannel(dry, 0);
  const impulseBuffer = context.createBuffer(1, rir.length, fs);
  impulseBuffer.copyToChannel(rir, 0);
  const source = context.createBufferSource();
  const convolver = context.createConvolver();
  convolver.normalize = false;
  source.buffer = sourceBuffer;
  convolver.buffer = impulseBuffer;
  source.connect(convolver).connect(context.destination);
  source.start(0);
  const rendered = await context.startRendering();
  return new Float32Array(rendered.getChannelData(0));
}

function maxAbs(samples) {
  let peak = 0;
  for (let index = 0; index < samples.length; index += 1) peak = Math.max(peak, Math.abs(samples[index]));
  return peak;
}

function scaledSamples(samples, gain) {
  const output = new Float32Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) output[index] = samples[index] * gain;
  return output;
}

function encodePcm16Wav(samples, fs) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, fs, true);
  view.setUint32(28, fs * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index += 1) {
    const value = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(44 + index * 2, Math.round(value < 0 ? value * 32768 : value * 32767), true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function readAscii(view, offset, length) {
  let value = "";
  for (let index = 0; index < length; index += 1) value += String.fromCharCode(view.getUint8(offset + index));
  return value;
}

function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
}

function acousticAgentCode(payload = lastSimulationPayload || apiPayload()) {
  const shape = String(payload.shape || "rectangle");
  const geometry = payload.geometry || {};
  const room = {
    shape,
    size: (payload.size || [6, 4, 2.8]).map(Number),
    materials: payload.materials || { wall: "wall", floor: "floor", ceiling: "ceiling" },
  };
  const geometryParams = {
    triangle: { apex: Number(geometry.triangleApex ?? 0.5) },
    circle: { segments: Number(geometry.circleSegments ?? 36) },
    polygon: {
      sides: Number(geometry.polygonSides ?? 6),
      irregularity: Number(geometry.polygonIrregularity ?? 0.18),
      skew: Number(geometry.polygonSkew ?? 0),
    },
    l_shape: {
      cutout_width: Number(geometry.lCutoutWidth ?? 0.45),
      cutout_depth: Number(geometry.lCutoutDepth ?? 0.45),
    },
    t_shape: {
      head_depth: Number(geometry.tHeadDepth ?? 0.38),
      stem_width: Number(geometry.tStemWidth ?? 0.34),
      stem_offset: Number(geometry.tStemOffset ?? 0.5),
    },
    trapezoid: {
      top_width: Number(geometry.trapezoidTopWidth ?? 0.62),
      top_offset: Number(geometry.trapezoidOffset ?? 0.5),
    },
    u_shape: {
      opening_width: Number(geometry.uOpeningWidth ?? 0.42),
      opening_depth: Number(geometry.uOpeningDepth ?? 0.48),
      opening_offset: Number(geometry.uOpeningOffset ?? 0.5),
    },
    fan_shape: {
      angle_deg: Number(geometry.fanAngle ?? 90),
      inner_radius: Number(geometry.fanInnerRadius ?? 0.28),
      segments: Number(geometry.fanSegments ?? 24),
    },
  };
  Object.assign(room, geometryParams[shape] || {});

  const acousticGeometry = (payload.objects || []).map((object) => ({
    type: String(object.type || "cuboid"),
    material: String(object.material || "wood"),
    position: (object.position || [0, 0]).slice(0, 2).map(Number),
    z: Number(object.z ?? Number(object.size?.[2] || 1) * 0.5),
    size: (object.size || [1, 1, 1]).slice(0, 3).map(Number),
    rotation_deg: Number(object.rotation ?? object.rotation_deg ?? 0),
  }));

  const source = JSON.stringify((payload.source || [1.2, 1.1, 1.5]).map(Number));
  const mic = JSON.stringify((payload.receiver || [4.7, 2.8, 1.4]).map(Number));
  const micModel = payload.receiver_model || { type: "mono" };
  const micType = String(micModel.type || "mono");
  const micParams = micType === "linear" || micType === "linear_array"
    ? {
        count: Number(micModel.count ?? 4),
        spacing_m: Number(micModel.spacing_m ?? 0.08),
        orientation_deg: Number(micModel.orientation_deg ?? 0),
      }
    : micType === "circular" || micType === "circular_array"
      ? {
          count: Number(micModel.count ?? 8),
          radius_m: Number(micModel.radius_m ?? 0.12),
          orientation_deg: Number(micModel.orientation_deg ?? 0),
        }
      : micType === "hrtf"
        ? { orientation_deg: Number(micModel.orientation_deg ?? 0) }
        : {};
  const emitterModel = payload.source_model || { type: "omni" };
  const sourceType = String(emitterModel.pattern || emitterModel.type || "omni");
  const sourceModel = sourceType === "omni"
    ? { type: "omni" }
    : {
        type: sourceType,
        orientation_deg: Number(emitterModel.orientation_deg ?? 0),
        elevation_deg: Number(emitterModel.elevation_deg ?? 0),
        dipole_weight: Number(emitterModel.dipole_weight ?? 0.5),
        dipole_power: Number(emitterModel.dipole_power ?? 1),
      };
  const quality = String(payload.config?.quality || payload.quality || "simulation");
  const rirLength = Number(payload.config?.duration_s || 2.0);
  const sampleRate = Number(payload.config?.fs || 16000);
  return [
    "from acoustic_agent import AcousticAgent",
    "",
    `room = ${JSON.stringify(room, null, 4)}`,
    `acoustic_geometry = ${JSON.stringify(acousticGeometry, null, 4)}`,
    "",
    `source = ${source}         # [x, y, z] m`,
    `source_directivity = ${JSON.stringify(sourceModel, null, 4)}`,
    `mic = ${mic}               # [x, y, z] m`,
    `mic_type = "${micType}"   # mono / hrtf / linear / circular`,
    `mic_params = ${JSON.stringify(micParams, null, 4)}`,
    `quality = "${quality}"    # preview / simulation / fine / reference`,
    `rir_length = ${rirLength}  # seconds`,
    `sample_rate = ${sampleRate} # Hz`,
    "",
    "agent = AcousticAgent(",
    "    room=room, acoustic_geometry=acoustic_geometry,",
    "    source_model=source_directivity,",
    '    receiver_model={"type": mic_type, **mic_params},',
    "    quality=quality, duration_s=rir_length, fs=sample_rate,",
    ")",
    "rir = agent.run(source=source, receiver=mic).rir",
  ].join("\n");
}

function applySceneToState(loaded) {
  const room = loaded.room || {};
  const size = estimateSize(room.corners || cornersFor(state.shape, state.size, state.geometry));
  state.shape = room.metadata?.shape || state.shape;
  state.geometry = { ...defaultState.geometry, ...(room.metadata?.geometry_params || state.geometry || {}) };
  state.size = [size[0], size[1], Number(room.height_m || state.size[2])];
  state.source = loaded.sources?.[0] || state.source;
  state.receiver = loaded.receivers?.[0] || state.receiver;
  const loadedSourceModel = loaded.metadata?.source_model;
  if (loadedSourceModel) {
    state.sourceDirectivity = {
      ...defaultState.sourceDirectivity,
      ...loadedSourceModel,
      type: loadedSourceModel.pattern || loadedSourceModel.type || "omni"
    };
  }
  clampScenePointsToRoom();
}

function applyPresetPoints() {
  const [x, y, z] = state.size;
  const h = Math.min(1.4, z - 0.2);
  const presetsByShape = {
    rectangle: [[x * 0.22, y * 0.28, h], [x * 0.78, y * 0.70, h]],
    triangle: [[x * 0.34, y * 0.24, h], [x * 0.58, y * 0.36, h]],
    polygon: [[x * 0.24, y * 0.26, h], [x * 0.58, y * 0.58, h]],
    circle: [[x * 0.34, y * 0.38, h], [x * 0.62, y * 0.58, h]],
    l_shape: [[x * 0.22, y * 0.24, h], [x * 0.42, y * 0.72, h]],
    t_shape: [[x * 0.28, y * 0.18, h], [x * 0.50, y * 0.74, h]],
    trapezoid: [[x * 0.25, y * 0.24, h], [x * 0.66, y * 0.68, h]],
    u_shape: [[x * 0.24, y * 0.24, h], [x * 0.74, y * 0.24, h]],
    fan_shape: [[x * 0.50, y * 0.22, h], [x * 0.50, y * 0.70, h]]
  };
  const pair = presetsByShape[state.shape] || presetsByShape.rectangle;
  const corners = cornersFor(state.shape, state.size, state.geometry);
  state.source = safeRoomPoint(pair[0], corners);
  state.receiver = safeRoomPoint(pair[1], corners);
}

function clampScenePointsToRoom() {
  const corners = cornersFor(state.shape, state.size, state.geometry);
  state.source = safeRoomPoint(state.source, corners);
  state.receiver = safeRoomPoint(state.receiver, corners);
  state.source[2] = clamp(state.source[2], 0.05, Math.max(0.05, state.size[2] - 0.05));
  state.receiver[2] = clamp(state.receiver[2], 0.05, Math.max(0.05, state.size[2] - 0.05));
  normalizeAllObjectPlacements();
}

function randomizePositions() {
  const corners = cornersFor(state.shape, state.size, state.geometry);
  const bounds = getBounds(corners);
  const minDistance = Math.max(0.65, Math.min(bounds.w, bounds.h) * 0.22);
  const source = randomRoomPoint(corners, state.size[2]);
  let receiver = randomRoomPoint(corners, state.size[2]);
  for (let attempt = 0; attempt < 80 && distance2D(source, receiver) < minDistance; attempt += 1) {
    receiver = randomRoomPoint(corners, state.size[2]);
  }
  state.source = source;
  state.receiver = receiver;
  simData = makeClientScene(state);
  updateControls();
  requestSimulation();
}

function randomRoomPoint(corners, height) {
  const bounds = getBounds(corners);
  for (let attempt = 0; attempt < 500; attempt += 1) {
    const x = bounds.x0 + Math.random() * bounds.w;
    const y = bounds.y0 + Math.random() * bounds.h;
    if (!pointIsSafelyInsideRoom([x, y], corners)) continue;
    return [Number(x.toFixed(3)), Number(y.toFixed(3)), randomHeight(height)];
  }
  const centroid = polygonCentroid(corners);
  return [Number(centroid[0].toFixed(3)), Number(centroid[1].toFixed(3)), randomHeight(height)];
}

function randomHeight(height) {
  const maxZ = Math.max(0.1, Number(height) - 0.2);
  const minZ = Math.min(1.0, maxZ);
  const preferredMax = Math.min(maxZ, 1.8);
  return Number((minZ + Math.random() * Math.max(0.01, preferredMax - minZ)).toFixed(3));
}

function distance2D(a, b) {
  return Math.hypot(Number(a[0]) - Number(b[0]), Number(a[1]) - Number(b[1]));
}

function safeRoomPoint(point, corners) {
  const z = point[2] ?? Math.min(1.4, state.size[2] - 0.2);
  if (pointIsSafelyInsideRoom(point, corners)) return [point[0], point[1], z];
  const bounds = getBounds(corners);
  const ratioCandidates = [
    [0.25, 0.25], [0.75, 0.72], [0.35, 0.62], [0.62, 0.35], [0.5, 0.5],
    [0.2, 0.5], [0.5, 0.2], [0.8, 0.35], [0.35, 0.8]
  ];
  const candidates = [
    [Number(point[0]), Number(point[1]), z],
    ...ratioCandidates.map(([rx, ry]) => [bounds.x0 + bounds.w * rx, bounds.y0 + bounds.h * ry, z])
  ];
  for (const candidate of candidates) {
    const safe = nearestSafeRoomPoint(candidate, corners, z);
    if (safe) return safe;
  }
  const centroid = polygonCentroid(corners);
  return nearestSafeRoomPoint([centroid[0], centroid[1], z], corners, z) || [centroid[0], centroid[1], z];
}

function addSceneObject(type) {
  const spec = furnitureCatalog[type];
  if (!spec) return;
  const unconfirmedId = unconfirmedObjectId();
  if (unconfirmedId) {
    selectSceneObject(unconfirmedId);
    rebuildThreeScene();
    updatePanels();
    setStatus("Confirm or delete the current object edit before adding another.");
    return;
  }
  clearTimeout(simulateTimer);
  simulationRequestSeq += 1;
  state.objects = Array.isArray(state.objects) ? state.objects : [];
  const draft = objectDraftFromControls(type);
  const object = {
    id: `obj_${Date.now().toString(36)}_${state.objects.length}`,
    type,
    title: spec.title,
    position: draft.position,
    rotation: draft.rotation,
    size: draft.size,
    z: draft.z,
    material: spec.material,
    pending: true,
  };
  normalizeObjectVerticalPlacement(object);
  state.objects.push(object);
  pendingObjectId = object.id;
  selectSceneObject(object.id);
  rebuildThreeScene();
  updatePanels();
  setStatus(`${spec.title} added. Adjust it, then update simulation.`);
}

function objectDraftFromControls(type = activeObjectType()) {
  const spec = furnitureCatalog[type] || furnitureCatalog.cuboid;
  const defaults = defaultObjectDraft(type);
  const size = [
    clamp(controlNumber("objectWidth", defaults.size[0]), 0.05, Math.max(0.05, state.size[0])),
    clamp(controlNumber("objectDepth", defaults.size[1]), 0.03, Math.max(0.03, state.size[1])),
    clamp(controlNumber("objectHeight", defaults.size[2]), 0.05, Math.max(0.05, state.size[2])),
  ];
  const halfHeight = size[2] * 0.5;
  const z = clamp(controlNumber("objectZ", defaults.z), halfHeight, Math.max(halfHeight, state.size[2] - halfHeight));
  const corners = cornersFor(state.shape, state.size, state.geometry);
  const safe = nearestSafeRoomPoint([controlNumber("objectX", defaults.position[0]), controlNumber("objectY", defaults.position[1]), z], corners, z);
  return {
    position: safe ? [Number(safe[0].toFixed(3)), Number(safe[1].toFixed(3))] : defaults.position,
    z,
    size,
    rotation: clamp(controlNumber("objectRotation", defaultObjectRotation(type)), -180, 180),
    material: spec.material,
  };
}

function defaultObjectDraft(type = activeObjectType()) {
  const spec = furnitureCatalog[type] || furnitureCatalog.cuboid;
  const position = nextObjectPosition(spec);
  const size = [...spec.size];
  const halfHeight = size[2] * 0.5;
  return {
    position,
    size,
    z: clamp(Number(spec.z ?? halfHeight), halfHeight, Math.max(halfHeight, state.size[2] - halfHeight)),
    rotation: defaultObjectRotation(type),
  };
}

function setObjectControlDraft(type = activeObjectType()) {
  const draft = defaultObjectDraft(type);
  setValue("objectX", roundControl(draft.position[0]));
  setValue("objectY", roundControl(draft.position[1]));
  setValue("objectZ", roundControl(draft.z));
  setValue("objectWidth", roundControl(draft.size[0]));
  setValue("objectDepth", roundControl(draft.size[1]));
  setValue("objectHeight", roundControl(draft.size[2]));
  setValue("objectRotation", Number(draft.rotation || 0).toFixed(1));
}

function nextObjectPosition(spec) {
  const corners = cornersFor(state.shape, state.size, state.geometry);
  const bounds = getBounds(corners);
  const index = (state.objects || []).length;
  const candidates = [
    [0.28, 0.28], [0.72, 0.28], [0.72, 0.72], [0.28, 0.72], [0.5, 0.5],
    [0.18, 0.5], [0.82, 0.5], [0.5, 0.18], [0.5, 0.82],
  ];
  for (let offset = 0; offset < candidates.length; offset += 1) {
    const [rx, ry] = candidates[(index + offset) % candidates.length];
    const point = [bounds.x0 + bounds.w * rx, bounds.y0 + bounds.h * ry, Math.min(state.size[2] - 0.05, spec.z ?? spec.size[2] * 0.5)];
    const safe = nearestSafeRoomPoint(point, corners, point[2]);
    if (safe) return [Number(safe[0].toFixed(3)), Number(safe[1].toFixed(3))];
  }
  const centroid = polygonCentroid(corners);
  return [Number(centroid[0].toFixed(3)), Number(centroid[1].toFixed(3))];
}

function defaultObjectRotation(type) {
  return type === "panel" ? 0 : 12;
}

function normalizeObjectVerticalPlacement(object) {
  if (!object) return;
  const spec = furnitureCatalog[object.type] || furnitureCatalog.cuboid;
  const size = object.size || spec.size;
  const height = clamp(Number(size[2] ?? spec.size[2] ?? 0.5), 0.05, Math.max(0.05, state.size[2]));
  object.size = [
    clamp(Number(size[0] ?? spec.size[0] ?? 0.5), 0.05, Math.max(0.05, state.size[0])),
    clamp(Number(size[1] ?? spec.size[1] ?? 0.5), 0.03, Math.max(0.03, state.size[1])),
    height,
  ];
  const halfHeight = height * 0.5;
  const maxCenter = Math.max(halfHeight, state.size[2] - halfHeight);
  object.z = clamp(Number(object.z ?? spec.z ?? halfHeight), halfHeight, maxCenter);
}

function normalizeAllObjectPlacements() {
  (state.objects || []).forEach((object) => normalizeObjectVerticalPlacement(object));
}

function selectSceneObject(id) {
  const unconfirmedId = unconfirmedObjectId();
  if (unconfirmedId && id !== unconfirmedId) {
    selectedObjectId = unconfirmedId;
    const unconfirmed = sceneObjectById(unconfirmedId);
    setSelection(unconfirmed ? unconfirmed.title : "Scene");
    syncSelectedObjectControls(unconfirmed);
    setStatus("Confirm the current object edit before selecting another object.");
    return;
  }
  selectedObjectId = id;
  const object = (state.objects || []).find((item) => item.id === id);
  setSelection(object ? object.title : "Scene");
  syncSelectedObjectControls(object);
}

function syncSelectedObjectControls(object = sceneObjectById(selectedObjectId)) {
  const settings = document.getElementById("objectSettings");
  const confirmButton = document.getElementById("confirmFurniture");
  const editHint = document.getElementById("objectEditHint");
  const commandRow = document.querySelector(".objectCommandRow");
  const addButton = document.getElementById("addAsset");
  const countLabel = document.getElementById("sceneObjectCount");
  if (countLabel) {
    const count = (state.objects || []).length;
    countLabel.textContent = `${count} object${count === 1 ? "" : "s"}`;
  }
  if (!settings) return;
  if (!object) {
    if (confirmButton) confirmButton.hidden = true;
    if (commandRow) commandRow.classList.remove("hasConfirm");
    if (addButton) addButton.textContent = "Add geometry";
    if (addButton) addButton.disabled = false;
    if (editHint) editHint.textContent = "Choose a geometry card, add it, then place it in the scene.";
    refreshObjectThumbnails();
    if (!document.activeElement || !settings.contains(document.activeElement)) setObjectControlDraft(activeObjectType());
    return;
  }
  const spec = furnitureCatalog[object.type] || furnitureCatalog.cuboid;
  const [width, depth, height] = object.size || spec.size;
  setActiveObjectType(object.type);
  setValue("objectX", roundControl(object.position?.[0] ?? 0));
  setValue("objectY", roundControl(object.position?.[1] ?? 0));
  setValue("objectZ", roundControl(object.z ?? spec.z ?? height * 0.5));
  setValue("objectWidth", roundControl(width));
  setValue("objectDepth", roundControl(depth));
  setValue("objectHeight", roundControl(height));
  setValue("objectRotation", Number(object.rotation || 0).toFixed(1));
  const needsConfirm = object.id === pendingObjectId || object.id === dirtyObjectId;
  if (confirmButton) confirmButton.hidden = !needsConfirm;
  if (commandRow) commandRow.classList.toggle("hasConfirm", needsConfirm);
  if (addButton) addButton.textContent = needsConfirm ? "Add later" : "Add geometry";
  if (addButton) addButton.disabled = needsConfirm;
  if (editHint) {
    editHint.textContent = needsConfirm
      ? "Preview is live. Press Update simulation to commit acoustics."
      : "Selected geometry. Drag in WebGL to move; edit dimensions here.";
  }
}

function handleObjectTypeChange(nextType = activeObjectType()) {
  const object = sceneObjectById(selectedObjectId);
  const spec = furnitureCatalog[nextType];
  if (!spec) return;
  if (!object) {
    setActiveObjectType(nextType);
    setObjectControlDraft(nextType);
    return;
  }
  object.type = nextType;
  object.title = spec.title;
  object.size = [...spec.size];
  object.z = spec.z;
  object.material = spec.material;
  normalizeObjectVerticalPlacement(object);
  setSelection(object.title);
  syncSelectedObjectControls(object);
  applyObjectEdit();
}

function handleObjectSettingsInput() {
  const object = sceneObjectById(selectedObjectId);
  if (!object) return;
  const draft = objectDraftFromControls(object.type);
  object.position = draft.position;
  object.z = draft.z;
  object.size = draft.size;
  object.rotation = draft.rotation;
  normalizeObjectVerticalPlacement(object);
  applyObjectEdit();
}

function applyObjectEdit() {
  rebuildThreeScene();
  updatePanels();
  markObjectEditForConfirmation(selectedObjectId);
}

function markObjectEditForConfirmation(objectId) {
  const object = sceneObjectById(objectId);
  if (!object) return;
  clearTimeout(simulateTimer);
  simulationRequestSeq += 1;
  const isPending = object.id === pendingObjectId;
  if (!isPending) {
    dirtyObjectId = object.id;
    object.dirty = true;
  }
  syncSelectedObjectControls(object);
  rebuildThreeScene();
  updatePanels();
  setStatus(isPending
    ? `${object.title} pending. Set size/position, then Confirm to simulate.`
    : `${object.title} edited. Confirm to update simulation.`);
}

function clearObjectSelection() {
  selectedObjectId = null;
  setSelection("Scene");
  syncSelectedObjectControls(null);
  rebuildThreeScene();
  updatePanels();
}

function confirmSelectedObject() {
  suppressObjectSelectionUntil = Date.now() + 350;
  const object = sceneObjectById(selectedObjectId);
  if (!object || object.id !== unconfirmedObjectId()) {
    setStatus("Edit an object before confirming.");
    return;
  }
  const title = object.title;
  normalizeObjectVerticalPlacement(object);
  delete object.pending;
  delete object.dirty;
  pendingObjectId = null;
  dirtyObjectId = null;
  simData = makeClientScene(state);
  clearObjectSelection();
  setTimeout(clearObjectSelection, 0);
  setTimeout(clearObjectSelection, 120);
  setStatus(`${title} confirmed. Updating simulation...`);
  requestSimulation();
}

function duplicateSelectedObject() {
  const object = (state.objects || []).find((item) => item.id === selectedObjectId);
  if (!object) {
    setStatus("Select an object before duplicating.");
    return;
  }
  if (hasUnconfirmedObjectChange()) {
    setStatus("Confirm or delete the current object edit before duplicating.");
    return;
  }
  const copy = structuredClone(object);
  copy.id = `obj_${Date.now().toString(36)}_${state.objects.length}`;
  copy.title = `${object.title} copy`;
  delete copy.pending;
  delete copy.dirty;
  copy.pending = true;
  const corners = cornersFor(state.shape, state.size, state.geometry);
  const shifted = nearestSafeRoomPoint([object.position[0] + 0.35, object.position[1] + 0.28, copy.z ?? 0.5], corners, copy.z ?? 0.5);
  copy.position = shifted ? [Number(shifted[0].toFixed(3)), Number(shifted[1].toFixed(3))] : [...object.position];
  state.objects.push(copy);
  pendingObjectId = copy.id;
  selectSceneObject(copy.id);
  rebuildThreeScene();
  updatePanels();
  setStatus(`${copy.title} created. Confirm to update simulation.`);
}

function deleteSelectedObject() {
  if (!selectedObjectId) {
    setStatus("Select an object before deleting.");
    return;
  }
  const before = (state.objects || []).length;
  state.objects = (state.objects || []).filter((object) => object.id !== selectedObjectId);
  if (state.objects.length === before) return;
  if (selectedObjectId === pendingObjectId) {
    pendingObjectId = null;
  }
  if (selectedObjectId === dirtyObjectId) {
    dirtyObjectId = null;
  }
  selectedObjectId = null;
  setSelection("Scene");
  syncSelectedObjectControls(null);
  simData = makeClientScene(state);
  rebuildThreeScene();
  updatePanels();
  scheduleSimulation();
}

function setObjectMode(mode) {
  objectMode = mode;
  document.getElementById("modeMove")?.classList.toggle("active", objectMode === "move");
  document.getElementById("modeRotate")?.classList.toggle("active", objectMode === "rotate");
}

function setSelection(name) {
  document.getElementById("selectionName").textContent = name;
  if (!selectedObjectId) syncSelectedObjectControls(null);
  updateSelectionToolbarPosition();
}

function updateSelectionToolbarPosition() {
  const toolbar = document.getElementById("selectionToolbar");
  if (!toolbar) return;
  const group = selectedObjectId
    ? furnitureGroup.children.find((item) => item.userData.objectId === selectedObjectId)
    : null;
  if (!group || !camera || !canvas) {
    toolbar.hidden = true;
    toolbar.classList.remove("objectFollow");
    toolbar.style.left = "50%";
    toolbar.style.top = "";
    toolbar.style.bottom = "18px";
    toolbar.style.transform = "translateX(-50%)";
    return;
  }
  toolbar.hidden = false;
  const box = new THREE.Box3().setFromObject(group);
  const anchor = box.getCenter(new THREE.Vector3());
  anchor.y = box.max.y + 0.34;
  anchor.project(camera);
  const canvasRect = canvas.getBoundingClientRect();
  const parentRect = toolbar.offsetParent?.getBoundingClientRect() || canvasRect;
  const toolbarWidth = toolbar.offsetWidth || 300;
  const toolbarHeight = toolbar.offsetHeight || 46;
  const x = canvasRect.left + ((anchor.x + 1) * 0.5 * canvasRect.width) - parentRect.left;
  const y = canvasRect.top + ((1 - anchor.y) * 0.5 * canvasRect.height) - parentRect.top;
  const margin = 14;
  const minX = margin + toolbarWidth * 0.5;
  const maxX = Math.max(minX, parentRect.width - margin - toolbarWidth * 0.5);
  const minY = margin;
  const maxY = Math.max(minY, parentRect.height - margin - toolbarHeight);
  toolbar.classList.add("objectFollow");
  toolbar.style.left = `${clamp(x, minX, maxX)}px`;
  toolbar.style.top = `${clamp(y - toolbarHeight, minY, maxY)}px`;
  toolbar.style.bottom = "auto";
  toolbar.style.transform = "translateX(-50%)";
}

async function copyCodeSnippet() {
  const text = codeEl.textContent || "";
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    flashCopyButton("Copied");
  } catch {
    if (copyTextFallback(text)) flashCopyButton("Copied");
    else {
      selectCodeSnippet();
      flashCopyButton("Selected");
    }
  }
}

function copyTextFallback(text) {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  textarea.style.top = "0";
  document.body.appendChild(textarea);
  textarea.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } finally {
    document.body.removeChild(textarea);
  }
  return ok;
}

function selectCodeSnippet() {
  const selection = window.getSelection();
  if (!selection) return;
  const range = document.createRange();
  range.selectNodeContents(codeEl);
  selection.removeAllRanges();
  selection.addRange(range);
}

function flashCopyButton(label) {
  if (!copyCodeButton) return;
  const original = copyCodeButton.textContent;
  copyCodeButton.textContent = label;
  window.setTimeout(() => {
    copyCodeButton.textContent = original || "Copy code";
  }, 1200);
}

function fillSelect(id, options) {
  document.getElementById(id).innerHTML = options.map(([value, label]) => `<option value="${value}">${label}</option>`).join("");
}

function value(id) { return document.getElementById(id).value; }
function setValue(id, v) { document.getElementById(id).value = v; }
function number(id) { return Number(document.getElementById(id).value); }
function controlNumber(id, fallback = 0) {
  const element = document.getElementById(id);
  if (!element || element.value === "") return fallback;
  const numeric = Number(element.value);
  return Number.isFinite(numeric) ? numeric : fallback;
}
function roundControl(value) { return Number(value || 0).toFixed(2); }
function presetTitle(id) { return presets.find((preset) => preset.id === id)?.title || id; }
function formatSeconds(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? `${numeric.toFixed(3)} s` : "-";
}
function formatMs(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${numeric.toFixed(1)} ms` : "-";
}
function formatDb(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${numeric.toFixed(1)} dB` : "-";
}
function solverModeLabel() {
  return {
    preview: "Preview",
    simulation: "Simulation",
    fine: "Fine",
    reference: "Reference"
  }[state.config.quality] || "Simulation";
}
function fmt3(point) { return point.map((v) => Number(v).toFixed(2)).join(", "); }

function getBounds(corners) {
  const xs = corners.map((p) => p[0]);
  const ys = corners.map((p) => p[1]);
  return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys), w: Math.max(...xs) - Math.min(...xs), h: Math.max(...ys) - Math.min(...ys) };
}

function estimateSize(corners) {
  const b = getBounds(corners);
  return [Number(b.w.toFixed(2)), Number(b.h.toFixed(2))];
}

function pointInPolygon2D(point, corners) {
  const x = Number(point[0]);
  const y = Number(point[1]);
  let inside = false;
  for (let i = 0, j = corners.length - 1; i < corners.length; j = i++) {
    const xi = corners[i][0], yi = corners[i][1];
    const xj = corners[j][0], yj = corners[j][1];
    const crosses = (yi > y) !== (yj > y);
    const denom = yj - yi;
    if (crosses && Math.abs(denom) > 1e-12 && x < ((xj - xi) * (y - yi)) / denom + xi) inside = !inside;
  }
  return inside;
}

function pointIsSafelyInsideRoom(point, corners, margin = MIN_WALL_DISTANCE_M) {
  return pointInPolygon2D(point, corners) && distanceToRoomBoundary(point, corners) >= margin;
}

function distanceToRoomBoundary(point, corners) {
  let best = Infinity;
  for (let index = 0; index < corners.length; index += 1) {
    best = Math.min(best, pointSegmentDistance2D(point, corners[index], corners[(index + 1) % corners.length]));
  }
  return best;
}

function pointSegmentDistance2D(point, a, b) {
  const px = Number(point[0]);
  const py = Number(point[1]);
  const ax = Number(a[0]);
  const ay = Number(a[1]);
  const bx = Number(b[0]);
  const by = Number(b[1]);
  const dx = bx - ax;
  const dy = by - ay;
  const denom = dx * dx + dy * dy;
  const t = denom <= 1e-12 ? 0 : clamp(((px - ax) * dx + (py - ay) * dy) / denom, 0, 1);
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

function nearestSafeRoomPoint(point, corners, z) {
  if (pointIsSafelyInsideRoom(point, corners)) return [Number(point[0]), Number(point[1]), z];
  const centroid = polygonCentroid(corners);
  const start = [Number(point[0]), Number(point[1])];
  for (let step = 1; step <= 40; step += 1) {
    const t = step / 40;
    const candidate = [
      start[0] + (centroid[0] - start[0]) * t,
      start[1] + (centroid[1] - start[1]) * t,
      z
    ];
    if (pointIsSafelyInsideRoom(candidate, corners)) return candidate.map((value) => Number(value.toFixed(3)));
  }
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const candidate = randomRoomPoint(corners, state.size[2]);
    if (pointIsSafelyInsideRoom(candidate, corners)) return [candidate[0], candidate[1], z];
  }
  return null;
}

function segmentInsidePolygon2D(start, end, corners) {
  for (let index = 1; index < 32; index += 1) {
    const t = index / 32;
    const point = [
      Number(start[0]) + (Number(end[0]) - Number(start[0])) * t,
      Number(start[1]) + (Number(end[1]) - Number(start[1])) * t
    ];
    if (!pointInPolygon2D(point, corners)) return false;
  }
  return true;
}

function polygonCentroid(corners) {
  const sum = corners.reduce((acc, point) => [acc[0] + point[0], acc[1] + point[1]], [0, 0]);
  return [sum[0] / Math.max(corners.length, 1), sum[1] / Math.max(corners.length, 1)];
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, Number(value)));
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}
