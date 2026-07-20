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
const runSimulationButton = document.getElementById("runSimulation");
const motionPlayButton = document.getElementById("motionPlay");
const motionTimelineEl = document.getElementById("motionTimeline");
const appRoot = document.getElementById("app");
const floorplanMode = appRoot?.dataset.sceneSource === "floorplan";
const customMode = appRoot?.dataset.sceneSource === "custom";
const multiRoomMode = floorplanMode || customMode;

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
const absorptionOptions = [
  ["auto", "Auto"],
  ["reflective", "Reflective"],
  ["semi_reflective", "Semi-reflective"],
  ["absorptive", "Absorptive"],
  ["highly_absorptive", "Highly absorptive"],
];
const boundaryMaterialControls = [
  ["wall", "wall", "wallAbsorption"],
  ["floor", "floor", "floorAbsorption"],
  ["ceiling", "ceiling", "ceilingAbsorption"],
  ["door", "door", "doorAbsorption"],
  ["window", "window_glass", "windowAbsorption"],
];
const activeBoundaryMaterialControls = multiRoomMode
  ? boundaryMaterialControls
  : boundaryMaterialControls.slice(0, 3);
const objectTypeOptions = [
  { id: "sofa", title: "Sofa" },
  { id: "bed", title: "Bed" },
  { id: "table", title: "Table" },
  { id: "cabinet", title: "Cabinet" },
  { id: "chair", title: "Chair" },
  { id: "rug", title: "Rug" },
  { id: "curtain", title: "Curtain" },
  { id: "tv_mirror", title: "TV / Mirror" },
  { id: "fridge", title: "Fridge" },
  { id: "washing_machine", title: "Washing Machine" },
  { id: "acoustic_panel", title: "Panel" },
  { id: "tile_surface", title: "Tile Surface" },
  { id: "sanitary_fixture", title: "Bath / Sink" },
  { id: "structural_element", title: "Column / Beam" },
  { id: "person", title: "Person" }
];
const furnitureCatalog = {
  sofa: { title: "Sofa", semantic: "sofa_couch", size: [2.05, 0.9, 0.72], color: 0x7c8f78, kind: "sofa", material: "chairs_heavy_upholstered", description: "large soft absorber" },
  bed: { title: "Bed", semantic: "bed_mattress", size: [2.1, 1.55, 0.55], color: 0x8f9bb5, kind: "bed", material: "chairs_medium_upholstered", description: "large bedroom absorber" },
  table: { title: "Table", semantic: "table_desk_counter", size: [1.35, 0.78, 0.74], color: 0x9a7656, kind: "table", material: "hard_object_wood_16mm", description: "hard reflecting surface" },
  cabinet: { title: "Cabinet", semantic: "cabinet_shelf_wardrobe", size: [1.25, 0.42, 1.75], color: 0x8d7463, kind: "shelves", material: "hard_object_wood_16mm", description: "large vertical reflector" },
  chair: { title: "Chair", semantic: "chair_seating", size: [0.55, 0.55, 0.86], color: 0x7d8b70, kind: "chair", material: "chairs_upholstered_moderate", description: "small seating absorber" },
  rug: { title: "Rug", semantic: "carpet_rug", size: [1.85, 1.2, 0.04], color: 0xa67b6b, kind: "rug", z: 0.02, material: "carpet_soft_10mm", description: "local floor absorber" },
  curtain: { title: "Curtain", semantic: "curtain_blind", size: [1.75, 0.06, 2.1], color: 0x7f8b9f, kind: "panel", z: 1.05, material: "curtains_fabric", description: "soft wall covering" },
  tv_mirror: { title: "TV / Mirror", semantic: "screen_mirror", size: [1.1, 0.05, 0.65], color: 0x242a2f, kind: "panel", z: 1.15, material: "window_door_glass_3mm", description: "hard reflective screen" },
  fridge: { title: "Fridge", semantic: "appliance", size: [0.75, 0.68, 1.75], color: 0xc6d0d6, kind: "appliance", material: "ptb_0583_iron_door", description: "large hard appliance" },
  washing_machine: { title: "Washing Machine", semantic: "appliance", size: [0.65, 0.62, 0.86], color: 0xbfc9cf, kind: "appliance", material: "ptb_0583_iron_door", description: "hard appliance block" },
  acoustic_panel: { title: "Panel", semantic: "acoustic_treatment", size: [1.2, 0.08, 1.2], color: 0x6f8f93, kind: "panel", z: 1.25, material: "mineral_wool_50mm_40kgm3", description: "absorptive acoustic panel" },
  tile_surface: { title: "Tile Surface", semantic: "ceramic_tile_surface", size: [1.6, 1.2, 0.05], color: 0xb8c7c9, kind: "tile_surface", z: 0.025, description: "hard local tiled surface" },
  sanitary_fixture: { title: "Bath / Sink", semantic: "sanitary_fixture", size: [1.55, 0.76, 0.62], color: 0xe1e6e4, kind: "sanitary_fixture", description: "hard ceramic sanitary fixture" },
  structural_element: { title: "Column / Beam", semantic: "structural_element", size: [0.46, 0.46, 2.45], color: 0x92999a, kind: "structural_element", description: "structural concrete reflector" },
  person: { title: "Person", semantic: "human_person", size: [0.42, 0.32, 1.7], color: 0xb79078, kind: "person", z: 0.85, material: "audience_1_m2", description: "human absorber" },
  cuboid: { title: "Cuboid", semantic: "structural_element", size: [1.2, 0.55, 1.05], color: 0x8d7463, kind: "block", material: "hard_object_wood_16mm", description: "legacy solid obstacle" },
  panel: { title: "Thin panel", semantic: "structural_element", size: [1.45, 0.08, 1.35], color: 0x4f6672, kind: "panel", z: 0.675, material: "wall_plasterboard", description: "legacy reflective slab" },
  low_block: { title: "Low block", semantic: "structural_element", size: [1.35, 0.72, 0.45], color: 0x7f8b6f, kind: "block", material: "chairs_upholstered_moderate", description: "legacy low reflector" }
};
const objectMaterialColors = {
  chairs_heavy_upholstered: 0x7c8f78,
  chairs_medium_upholstered: 0x8f9bb5,
  chairs_upholstered_moderate: 0x7d8b70,
  hard_object_wood_16mm: 0x9a7656,
  carpet_soft_10mm: 0xa67b6b,
  curtains_fabric: 0x7f8b9f,
  window_door_glass_3mm: 0x242a2f,
  ptb_0583_iron_door: 0xc6d0d6,
  mineral_wool_50mm_40kgm3: 0x6f8f93,
  audience_1_m2: 0xb79078,
  wall_plasterboard: 0xb7afa5,
  fabric: 0x7f8b6f,
  wood: 0x9a7656,
  glass: 0x7fb4c7,
  screen: 0x242a2f,
  plaster: 0xb7afa5,
};
const MIN_WALL_DISTANCE_M = 0.15;
const RIR_DECAY_MIN_DB = -60;
const RIR_DECAY_DB_TICKS = ["0", "-20", "-40", "-60"];

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
  materialProfile: { wall: "auto", floor: "auto", ceiling: "auto", door: "auto", window: "auto" },
  materialSeed: 42,
  objects: [],
  source: [1.2, 1.1, 1.5],
  receiver: [4.7, 2.8, 1.4],
  motion: { mode: "static", moving: "source", distance_m: 0.8, keyframe_spacing_m: 0.25, random_seed: 42 },
  config: { fs: 16000, duration_s: 2.0, quality: "simulation", rt_num_rays: 32768, rt_num_bounces: 64, rt_duration_s: 2.0, diffraction_order: 3, max_diffraction_paths: 8 },
  mic: { type: "mono", count: 4, spacing_m: 0.08, radius_m: 0.12, orientation_deg: 0 },
  sourceDirectivity: { type: "omni", orientation_deg: 0, elevation_deg: 0, dipole_weight: 0.0, dipole_power: 1.0 },
  floorplan: { index: 0, count: 0, roomId: null, roomType: null, receiverRoomId: null, receiverRoomType: null, corners: null, roomOptions: [], plan: null, dataset: null, selectedRoom: null, receiverRoom: null, roomMetadata: null },
  custom: { spec: null, validation: null, imageOpacity: 1.0 }
};

let state = structuredClone(defaultState);
let simData = makeClientScene(state);
let simulateTimer = null;
let simulationRequestSeq = 0;
let lastSimulationPayload = null;
let calibrationAudioSeq = 0;
let calibrationAudioUrls = [];
let simulationRunning = false;
let displayedMotionFrameIndex = -1;
let motionDisplayPhase = 0;
let motionPlayback = { active: false, startedAt: 0, startPhase: 0, duration_s: 4.0 };
let selectedObjectId = null;
let pendingObjectId = null;
let dirtyObjectId = null;
let objectMode = "move";
let objectDrag = null;
let suppressObjectSelectionUntil = 0;
let materialSemanticCatalog = {};
let randomMotionRouteCache = { signature: "", value: null };
let customImageElement = null;
let customImageUrl = null;
let customEditTimer = null;
const layerState = { direct: true, portal: true, diffraction: true, rt: true };

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
const motionGroup = new THREE.Group();
threeScene.add(floorGroup, shellGroup, furnitureGroup, pathGroup, motionGroup, markerGroup);

void bootstrap();

async function bootstrap() {
  setupThree();
  setupViewControls();
  setupControls();
  setupMotionControls();
  setupMaterialControls();
  renderThumbnails();
  renderMicThumbnails();
  renderSourceDirectivityThumbnails();
  renderObjectThumbnails();
  bindEvents();
  setupSectionNavigation();
  setupResultNavigation();
  await loadMaterialCatalog();
  if (floorplanMode) {
    try {
      await loadFloorplanScene(0, null, "auto", { simulate: false });
    } catch (error) {
      setStatus(String(error?.message || error), true);
    }
  } else if (customMode) {
    try {
      await loadCustomCapabilities();
      await generateCustomScene();
    } catch (error) {
      setStatus(String(error?.message || error), true);
    }
  }
  updateControls();
  markSimulationPending("Scene ready · run simulation when needed.");
  animate();
}

function setupSectionNavigation() {
  const panel = document.getElementById("setupScroll");
  const links = [...document.querySelectorAll("[data-section-link]")];
  if (!panel || links.length === 0) return;
  const sections = links
    .map((link) => document.getElementById(link.dataset.sectionLink))
    .filter(Boolean);

  const activate = (sectionId) => {
    links.forEach((link) => link.classList.toggle("active", link.dataset.sectionLink === sectionId));
  };
  const scrollToSection = (sectionId, behavior = "smooth") => {
    const section = document.getElementById(sectionId);
    if (!section) return;
    const panelTop = panel.getBoundingClientRect().top;
    const targetTop = panel.scrollTop + section.getBoundingClientRect().top - panelTop - 2;
    panel.scrollTo({ top: Math.max(0, targetTop), behavior });
    activate(section.id);
  };
  links.forEach((link) => link.addEventListener("click", (event) => {
    const section = document.getElementById(link.dataset.sectionLink);
    if (!section) return;
    event.preventDefault();
    history.replaceState(null, "", `#${section.id}`);
    scrollToSection(section.id);
  }));

  let navigationFrame = 0;
  const updateActiveSection = () => {
    navigationFrame = 0;
    const boundary = panel.getBoundingClientRect().top + 24;
    let activeSection = sections[0];
    const atBottom = panel.scrollTop + panel.clientHeight >= panel.scrollHeight - 4;
    if (atBottom) {
      activeSection = sections[sections.length - 1];
    } else {
      sections.forEach((section) => {
        if (section.getBoundingClientRect().top <= boundary) activeSection = section;
      });
    }
    if (activeSection) activate(activeSection.id);
  };
  panel.addEventListener("scroll", () => {
    if (navigationFrame) return;
    navigationFrame = requestAnimationFrame(updateActiveSection);
  });
  updateActiveSection();
  const initialSectionId = location.hash.slice(1);
  if (initialSectionId) {
    setTimeout(() => scrollToSection(initialSectionId, "auto"), 0);
    setTimeout(() => scrollToSection(initialSectionId, "auto"), 900);
  }
}

function setupResultNavigation() {
  const panel = document.getElementById("resultsPanel");
  const chrome = panel?.querySelector(".resultsChrome");
  const links = [...document.querySelectorAll("[data-result-link]")];
  if (!panel || !chrome || links.length === 0) return;
  const sections = links
    .map((link) => document.getElementById(link.dataset.resultLink))
    .filter(Boolean);
  const activate = (sectionId) => {
    links.forEach((link) => link.classList.toggle("active", link.dataset.resultLink === sectionId));
  };
  links.forEach((link) => link.addEventListener("click", (event) => {
    const section = document.getElementById(link.dataset.resultLink);
    if (!section) return;
    event.preventDefault();
    const panelTop = panel.getBoundingClientRect().top;
    const offset = chrome.getBoundingClientRect().height + 10;
    const targetTop = panel.scrollTop + section.getBoundingClientRect().top - panelTop - offset;
    panel.scrollTo({ top: Math.max(0, targetTop), behavior: "smooth" });
    activate(section.id);
  }));
  let navigationFrame = 0;
  const updateActiveSection = () => {
    navigationFrame = 0;
    const boundary = chrome.getBoundingClientRect().bottom + 16;
    let activeSection = sections[0];
    sections.forEach((section) => {
      if (section.getBoundingClientRect().top <= boundary) activeSection = section;
    });
    if (activeSection) activate(activeSection.id);
  };
  panel.addEventListener("scroll", () => {
    if (navigationFrame) return;
    navigationFrame = requestAnimationFrame(updateActiveSection);
  });
  updateActiveSection();
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
  controls.minZoom = 0.55;
  controls.maxZoom = 4.0;

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

function setupViewControls() {
  document.getElementById("viewFit")?.addEventListener("click", () => fitCurrentView());
  document.getElementById("viewIso")?.addEventListener("click", () => fitCamera());
  document.getElementById("viewTop")?.addEventListener("click", () => setTopView());
  controls?.addEventListener("start", () => {
    camera.userData.viewMode = "custom";
    setActiveViewControl(null);
  });
}

function setActiveViewControl(id) {
  ["viewFit", "viewIso", "viewTop"].forEach((buttonId) => {
    document.getElementById(buttonId)?.classList.toggle("active", buttonId === id);
  });
}

function fitCurrentView() {
  const bounds = sceneDisplayBounds();
  const height = Number(simData.room?.height_m || 2.8);
  const center = new THREE.Vector3((bounds.x0 + bounds.x1) * 0.5, height * 0.35, (bounds.y0 + bounds.y1) * 0.5);
  const span = Math.max(bounds.w, bounds.h, height, 1);
  const direction = camera.position.clone().sub(controls.target);
  if (direction.lengthSq() < 1e-8) direction.set(1, 1, 1);
  direction.normalize();
  camera.userData.viewSize = span * 0.86 + 2.3;
  camera.zoom = 1;
  controls.target.copy(center);
  camera.position.copy(center).addScaledVector(direction, span * 2.25 + height);
  camera.lookAt(center);
  controls.update();
  resize();
  setActiveViewControl("viewFit");
}

function setTopView() {
  const bounds = sceneDisplayBounds();
  const height = Number(simData.room?.height_m || 2.8);
  const center = new THREE.Vector3((bounds.x0 + bounds.x1) * 0.5, 0, (bounds.y0 + bounds.y1) * 0.5);
  const span = Math.max(bounds.w, bounds.h, height, 1);
  camera.userData.viewSize = span * 0.72 + 1.8;
  camera.userData.viewMode = "top";
  camera.zoom = 1;
  camera.up.set(0, 0, -1);
  controls.minPolarAngle = 0;
  controls.target.copy(center);
  camera.position.set(center.x, span * 2.5 + height, center.z + 1e-4);
  camera.lookAt(center);
  controls.update();
  resize();
  setActiveViewControl("viewTop");
}

function setupControls() {
  if (multiRoomMode) return;
  fillSelect("shape", presets.map((preset) => [preset.id, preset.title]));
}

function setupMotionControls() {
  const select = document.getElementById("motionMode");
  if (!select) return;
  [...select.options].forEach((option) => {
    if (!["static", "approach", "random"].includes(option.value)) option.remove();
  });
  const travelOption = select.querySelector('option[value="approach"]');
  if (travelOption) travelOption.textContent = "Approach";
  let randomOption = select.querySelector('option[value="random"]');
  if (!randomOption) {
    randomOption = document.createElement("option");
    randomOption.value = "random";
    select.append(randomOption);
  }
  randomOption.textContent = "Random";
  if (!["static", "approach", "random"].includes(state.motion.mode)) state.motion.mode = "static";
}

function setupMaterialControls() {
  activeBoundaryMaterialControls.forEach(([, , id]) => {
    if (document.getElementById(id)) fillSelect(id, absorptionOptions);
  });
}

async function loadMaterialCatalog() {
  try {
    const response = await fetch("/api/v1/materials/semantics", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Unable to load acoustic materials");
    materialSemanticCatalog = Object.fromEntries((payload.semantics || []).map((item) => [item.semantic, item]));
    applyMaterialAvailability();
  } catch (error) {
    console.warn("Material catalog unavailable", error);
  }
}

function applyMaterialAvailability() {
  activeBoundaryMaterialControls.forEach(([, semantic, id]) => {
    const select = document.getElementById(id);
    const available = new Set(materialSemanticCatalog[semantic]?.available_absorption_classes || []);
    if (!select || available.size === 0) return;
    [...select.options].forEach((option) => {
      option.disabled = option.value !== "auto" && !available.has(option.value);
    });
    if (select.selectedOptions[0]?.disabled) select.value = "auto";
  });
}

async function loadFloorplanScene(index, roomId = null, receiverRoomId = "auto", options = {}) {
  if (!floorplanMode) return;
  clearTimeout(simulateTimer);
  simulationRequestSeq += 1;
  setStatus("Loading Floorplan room...");
  const boundedIndex = Math.max(0, Math.round(Number(index) || 0));
  const query = new URLSearchParams({ idx: String(boundedIndex), height: String(state.size[2] || 2.8) });
  if (roomId) query.set("room", roomId);
  if (receiverRoomId) query.set("receiver_room", receiverRoomId);
  const response = await fetch(`/api/v1/floorplan/scene?${query.toString()}`, { cache: "no-store" });
  const scene = await response.json();
  if (!response.ok) throw new Error(scene.error || `Unable to load Floorplan index ${boundedIndex}`);
  applyMultiRoomScene(scene);
  const roomLabel = state.floorplan.roomId === state.floorplan.receiverRoomId
    ? state.floorplan.roomType
    : `${state.floorplan.roomType} → ${state.floorplan.receiverRoomType}`;
  setStatus(`${roomLabel} · idx ${state.floorplan.index}`);
  if (options.simulate === true) requestSimulation();
}

function applyMultiRoomScene(scene, customPayload = null) {
  state.shape = "floorplan";
  state.size = scene.room.size.map(Number);
  state.source = scene.source.map(Number);
  state.receiver = scene.receiver.map(Number);
  state.objects = [];
  state.floorplan = {
    index: Number(scene.dataset.index),
    count: Number(scene.dataset.count),
    roomId: String(scene.selected_room.id),
    roomType: String(scene.selected_room.type),
    receiverRoomId: String(scene.receiver_room?.id || scene.selected_room.id),
    receiverRoomType: String(scene.receiver_room?.type || scene.selected_room.type),
    corners: scene.room.corners.map((point) => point.map(Number)),
    roomOptions: Array.isArray(scene.rooms) ? scene.rooms : [],
    plan: scene.plan || null,
    dataset: scene.dataset || null,
    selectedRoom: scene.selected_room || null,
    receiverRoom: scene.receiver_room || scene.selected_room || null,
    roomMetadata: scene.room.metadata || null,
  };
  if (customPayload) {
    state.custom.spec = customPayload.spec || state.custom.spec;
    state.custom.validation = customPayload.validation || state.custom.validation;
    const editor = document.getElementById("customSpecJson");
    if (editor && state.custom.spec && customPayload.populateEditor !== false) {
      editor.value = JSON.stringify(state.custom.spec, null, 2);
    }
    updateCustomValidation();
  }
  selectedObjectId = null;
  pendingObjectId = null;
  dirtyObjectId = null;
  lastSimulationPayload = null;
  if (camera) camera.userData.fitted = false;
  simData = makeClientScene(state);
  updateControls();
}

async function loadCustomCapabilities() {
  const response = await fetch("/api/v1/custom/capabilities", { cache: "no-store" });
  const capabilities = await response.json();
  if (!response.ok) throw new Error(capabilities.error || "Unable to read Custom capabilities");
  const status = document.getElementById("customVlmStatus");
  const available = Boolean(capabilities.vlm?.available);
  if (status) {
    status.textContent = available
      ? `${capabilities.vlm.provider || "VLM"} ready`
      : "No API needed · use Codex with the copied prompt, then paste its JSON below";
    status.classList.toggle("ready", available);
  }
}

async function copyCustomVlmPrompt() {
  const response = await fetch("/api/v1/custom/prompt", { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok || !payload.prompt) throw new Error(payload.error || "Unable to load the Codex prompt");
  const button = document.getElementById("customVlmPrompt");
  try {
    await navigator.clipboard.writeText(payload.prompt);
  } catch {
    if (!copyTextFallback(payload.prompt)) throw new Error("Unable to copy the Codex prompt");
  }
  if (button) {
    const label = button.textContent;
    button.textContent = "Copied";
    window.setTimeout(() => { button.textContent = label || "Copy Codex prompt"; }, 1200);
  }
  setStatus("Codex image prompt copied");
}

async function generateCustomScene(options = {}) {
  if (!customMode) return;
  clearTimeout(simulateTimer);
  simulationRequestSeq += 1;
  const seedInput = document.getElementById("customSeed");
  const seed = Math.max(0, Math.round(Number(seedInput?.value || 42) + Number(options.seedOffset || 0)));
  const description = "";
  if (seedInput) seedInput.value = String(seed);
  setStatus("Generating custom floor plan...");
  const response = await fetch("/api/v1/custom/generate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      description,
      width_m: controlNumber("customWidth", 10),
      depth_m: controlNumber("customDepth", 8),
      height_m: controlNumber("height", 2.8),
      seed,
    }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Unable to generate custom floor plan");
  applyMultiRoomScene(result.scene, { ...result, populateEditor: options.populateEditor === true });
  setStatus(`Custom floor plan generated · seed ${seed}`);
}

async function compileCustomScene(spec = state.custom.spec, roomSelection = {}) {
  if (!customMode || !spec) return;
  clearTimeout(simulateTimer);
  simulationRequestSeq += 1;
  setStatus("Validating custom floor plan...");
  const requestedSpec = structuredClone(spec);
  requestedSpec.height_m = controlNumber("height", Number(spec.height_m || 2.8));
  const response = await fetch("/api/v1/custom/compile", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      spec: requestedSpec,
      source_room: roomSelection.sourceRoom || state.floorplan.roomId,
      receiver_room: roomSelection.receiverRoom || state.floorplan.receiverRoomId,
      height_m: requestedSpec.height_m,
      seed: 42,
    }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Unable to compile custom floor plan");
  applyMultiRoomScene(result.scene, result);
  setStatus("Custom floor plan validated");
}

function updateCustomValidation() {
  if (!customMode) return;
  const element = document.getElementById("customValidation");
  if (!element) return;
  const validation = state.custom.validation || {};
  const summary = validation.summary || {};
  const errors = validation.errors || [];
  const warnings = validation.warnings || [];
  element.classList.toggle("error", errors.length > 0);
  element.textContent = errors.length
    ? errors[0]
    : `${summary.rooms || 0} rooms · ${summary.doors || 0} doors · ${summary.windows || 0} windows${warnings.length ? ` · ${warnings[0]}` : ""}`;
}

function handleCustomImageUpload(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setStatus("Unsupported image type", true);
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    setStatus("Floor-plan image must be smaller than 12 MB", true);
    return;
  }
  if (customImageUrl) URL.revokeObjectURL(customImageUrl);
  customImageUrl = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    customImageElement = image;
    drawFloorplanOverview();
    setStatus(`${file.name} loaded locally`);
  };
  image.onerror = () => setStatus("Unable to read floor-plan image", true);
  image.src = customImageUrl;
}

function customSpecBounds(spec) {
  const points = Array.isArray(spec?.outer_boundary) ? spec.outer_boundary : [];
  const xs = points.map((point) => Number(point?.[0])).filter(Number.isFinite);
  const ys = points.map((point) => Number(point?.[1])).filter(Number.isFinite);
  if (xs.length < 3 || ys.length < 3) return null;
  const x0 = Math.min(...xs);
  const y0 = Math.min(...ys);
  const x1 = Math.max(...xs);
  const y1 = Math.max(...ys);
  if (x1 - x0 <= 1e-6 || y1 - y0 <= 1e-6) return null;
  return { x0, y0, width: x1 - x0, depth: y1 - y0 };
}

async function rescaleCustomScene(axis) {
  const bounds = customSpecBounds(state.custom.spec);
  if (!bounds) throw new Error("Apply a valid floor plan before calibrating its size");
  const requested = axis === "depth"
    ? controlNumber("customDepth", bounds.depth)
    : controlNumber("customWidth", bounds.width);
  const base = axis === "depth" ? bounds.depth : bounds.width;
  const minScale = Math.max(3 / bounds.width, 3 / bounds.depth);
  const maxScale = Math.min(40 / bounds.width, 40 / bounds.depth);
  const scale = clamp(requested / base, minScale, maxScale);
  const next = structuredClone(state.custom.spec);
  const scalePoint = (point) => [
    Number((bounds.x0 + (Number(point[0]) - bounds.x0) * scale).toFixed(6)),
    Number((bounds.y0 + (Number(point[1]) - bounds.y0) * scale).toFixed(6)),
  ];
  next.outer_boundary = next.outer_boundary.map(scalePoint);
  next.rooms = (next.rooms || []).map((room) => ({ ...room, corners: room.corners.map(scalePoint) }));
  next.openings = (next.openings || []).map((opening) => ({ ...opening, segment: opening.segment.map(scalePoint) }));
  next.provenance = {
    ...(next.provenance || {}),
    scale_calibration: {
      uniform_scale: Number(scale.toFixed(8)),
      width_m: Number((bounds.width * scale).toFixed(4)),
      depth_m: Number((bounds.depth * scale).toFixed(4)),
    },
  };
  await compileCustomScene(next);
  setStatus("Floor-plan scale calibrated");
}

function scheduleCustomEdit(action, input) {
  clearTimeout(customEditTimer);
  if (input?.value === "") return;
  customEditTimer = setTimeout(() => {
    Promise.resolve(action()).catch((error) => setStatus(String(error.message || error), true));
  }, 300);
}

async function navigateFloorplan(index, direction = "nearest") {
  if (!floorplanMode) return;
  setStatus("Finding eligible Floorplan scene...");
  const query = new URLSearchParams({ idx: String(Math.round(Number(index) || 0)), direction });
  const response = await fetch(`/api/v1/floorplan/index?${query.toString()}`, { cache: "no-store" });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Unable to resolve an eligible Floorplan index");
  await loadFloorplanScene(Number(result.index), null, "auto");
}

function syncFloorplanRoomOptions() {
  const select = document.getElementById("floorplanRoom");
  const receiverSelect = document.getElementById("floorplanReceiverRoom");
  if (!select || !receiverSelect) return;
  const options = state.floorplan.roomOptions || [];
  const signature = options.map((room) => `${room.id}:${room.type}:${Number(room.area_m2).toFixed(3)}`).join("|");
  if (select.dataset.signature !== signature) {
    const makeOption = (room) => {
      const option = document.createElement("option");
      option.value = room.id;
      option.textContent = `${room.id} · ${Number(room.area_m2).toFixed(1)} m²`;
      return option;
    };
    select.replaceChildren(...options.map(makeOption));
    receiverSelect.replaceChildren(...options.map(makeOption));
    select.dataset.signature = signature;
    receiverSelect.dataset.signature = signature;
  }
  select.value = state.floorplan.roomId || "";
  receiverSelect.value = state.floorplan.receiverRoomId || "";
  drawFloorplanOverview();
  updateFloorplanMeta();
}

function updateFloorplanMeta() {
  const container = document.getElementById("floorplanMeta");
  if (!container) return;
  const dataset = state.floorplan.dataset || {};
  const room = state.floorplan.selectedRoom || {};
  const receiverRoom = state.floorplan.receiverRoom || {};
  const features = state.floorplan.roomMetadata?.boundary_features || [];
  const connections = state.floorplan.roomMetadata?.connections || [];
  const exteriorExposures = state.floorplan.roomMetadata?.exterior_exposures || [];
  const rows = [
    [customMode ? "Source" : "Dataset", customMode ? String(dataset.generator || "local") : `${Number(state.floorplan.index) + 1} / ${Number(state.floorplan.count) || 0}`],
    [customMode ? "Rooms" : "Eligible", customMode ? String(state.floorplan.roomOptions?.length || 0) : `${Number(dataset.eligible_count) || 0}`],
    ["Source room", String(room.type || "-")],
    ["Microphone room", String(receiverRoom.type || "-")],
    ["Area", Number.isFinite(Number(room.area_m2)) ? `${Number(room.area_m2).toFixed(1)} m²` : "-"],
    ["Scale", customMode ? "metric" : Number.isFinite(Number(dataset.meters_per_unit)) ? `${Number(dataset.meters_per_unit).toFixed(4)} m/u` : "-"],
    [customMode ? "Gross area" : "Scale source", customMode ? `${Number(dataset.gross_area_m2 || 0).toFixed(1)} m²` : String(dataset.scale_source || "-").replaceAll("_", " ")],
    ["Doors", String(features.filter((item) => item.type === "door").length)],
    ["Openings", String(features.filter((item) => item.type === "opening").length)],
    ["Windows", String(features.filter((item) => item.type === "window").length)],
    ["Route", String(state.floorplan.roomMetadata?.multi_room?.route_portal_ids?.length || 0) + " portals"],
    ["Connections", exteriorExposures.length ? `${connections.length} · ${exteriorExposures.length} exterior` : String(connections.length)],
  ];
  container.replaceChildren(...rows.map(([label, value]) => {
    const item = document.createElement("div");
    const name = document.createElement("span");
    const result = document.createElement("strong");
    name.textContent = label;
    result.textContent = value;
    item.append(name, result);
    return item;
  }));
}

function drawFloorplanOverview() {
  const canvasEl = document.getElementById("floorplanPlanCanvas");
  const plan = state.floorplan.plan;
  if (!canvasEl || !plan) return;
  const ctx = canvasEl.getContext("2d");
  const width = canvasEl.width;
  const height = canvasEl.height;
  const pad = 14;
  const planWidth = Math.max(Number(plan.size?.[0]), 1e-6);
  const planDepth = Math.max(Number(plan.size?.[1]), 1e-6);
  const scale = Math.min((width - pad * 2) / planWidth, (height - pad * 2) / planDepth);
  const toCanvas = ([x, y]) => [pad + Number(x) * scale, pad + Number(y) * scale];
  const simulationOrigin = plan.simulation_origin || [0, 0];
  const toSimulationCanvas = ([x, y]) => toCanvas([
    Number(x) + Number(simulationOrigin[0] || 0),
    Number(y) + Number(simulationOrigin[1] || 0),
  ]);
  const colors = { living: "#c7e4df", bedroom: "#d7def4", kitchen: "#efd8ad", bathroom: "#cce5ee", storage: "#d9dddf", balcony: "#cee0c4" };
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8fafb";
  ctx.fillRect(0, 0, width, height);
  if (customMode && customImageElement) {
    ctx.save();
    ctx.globalAlpha = Number(state.custom.imageOpacity ?? 0.5);
    ctx.drawImage(customImageElement, pad, pad, planWidth * scale, planDepth * scale);
    ctx.restore();
  }
  (plan.rooms || []).forEach((room) => {
    drawCanvasPolygon(ctx, room.polygon || [], toCanvas);
    ctx.fillStyle = colors[room.type] || "#e2e6e8";
    ctx.strokeStyle = room.selected ? "#ef476f" : room.receiver ? "#0f7f9f" : "#65727a";
    ctx.lineWidth = room.selected || room.receiver ? 3 : 1;
    ctx.fill();
    ctx.stroke();
  });
  const semanticFeatures = state.floorplan.roomMetadata?.boundary_features || [];
  if (semanticFeatures.length) {
    drawFloorplanOverviewFeatures(ctx, semanticFeatures, toSimulationCanvas);
  } else {
    (plan.apertures || []).forEach((feature) => {
      drawCanvasPolygon(ctx, feature.polygon || [], toCanvas);
      ctx.fillStyle = feature.type === "window" ? "#54a8c1" : "#ce6545";
      ctx.globalAlpha = 0.9;
      ctx.fill();
      ctx.globalAlpha = 1;
    });
  }
  [[state.source, "#ef476f", "S"], [state.receiver, "#0f7f9f", "M"]].forEach(([point, color, label]) => {
    const [x, y] = toSimulationCanvas(point);
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 8px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, x, y + 0.5);
  });
}

function drawFloorplanOverviewFeatures(ctx, features, toCanvas) {
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  features.forEach((feature) => {
    (feature.segments || []).forEach((segment) => {
      if (!Array.isArray(segment) || segment.length < 2) return;
      drawPlanBoundarySymbol(ctx, feature, segment, toCanvas, 1);
    });
  });
}

function drawPlanBoundarySymbol(ctx, feature, segment, toCanvas, scale = 1) {
  const [ax, ay] = toCanvas(segment[0]);
  const [bx, by] = toCanvas(segment[1]);
  const dx = bx - ax;
  const dy = by - ay;
  const length = Math.hypot(dx, dy);
  if (length < 0.2) return;
  const ux = dx / length;
  const uy = dy / length;
  const stroke = (color, width, from = 0, to = 1) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width * scale;
    ctx.beginPath();
    ctx.moveTo(ax + dx * from, ay + dy * from);
    ctx.lineTo(ax + dx * to, ay + dy * to);
    ctx.stroke();
  };

  ctx.save();
  ctx.lineCap = "round";
  ctx.setLineDash([]);
  if (feature.type === "window") {
    const nx = -uy * 1.25 * scale;
    const ny = ux * 1.25 * scale;
    ctx.strokeStyle = "rgba(70,108,119,.86)";
    ctx.lineWidth = 1.4 * scale;
    for (const offset of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(ax + nx * offset, ay + ny * offset);
      ctx.lineTo(bx + nx * offset, by + ny * offset);
      ctx.stroke();
    }
    stroke("rgba(133,207,224,.96)", 1.2);
  } else if (feature.type === "opening") {
    stroke("rgba(33,105,91,.78)", 2.5, 0, 0.24);
    stroke("rgba(33,105,91,.78)", 2.5, 0.76, 1);
  } else if (feature.open) {
    stroke("rgba(184,126,88,.92)", 2.6, 0, 0.22);
    stroke("rgba(184,126,88,.92)", 2.6, 0.78, 1);
  } else {
    stroke("rgba(104,81,69,.9)", 3.1);
    stroke("rgba(206,168,142,.96)", 1.25, 0.06, 0.94);
  }
  ctx.restore();
}

function drawCanvasPolygon(ctx, polygon, transform) {
  ctx.beginPath();
  polygon.forEach((point, index) => {
    const [x, y] = transform(point);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
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
  if (!container) return;
  const specs = geometrySpecs(state.shape);
  state.geometry = { ...defaultState.geometry, ...(state.geometry || {}) };
  container.innerHTML = specs.map((spec) => {
    const value = Number(state.geometry[spec.key] ?? defaultState.geometry[spec.key]);
    return `<label>${spec.label}<input data-geom="${spec.key}" type="number" min="${spec.min}" max="${spec.max}" step="${spec.step}" value="${value}"></label>`;
  }).join("");
  container.classList.toggle("empty", specs.length === 0);
}

function readGeometryParams() {
  if (!document.getElementById("geometryParams")) return;
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
  const roomIds = floorplanMode
    ? ["height", "materialSeed", ...activeBoundaryMaterialControls.map(([, , id]) => id)]
    : customMode
      ? ["materialSeed", ...activeBoundaryMaterialControls.map(([, , id]) => id)]
    : ["shape", "sizeX", "sizeY", "height", "materialSeed", ...activeBoundaryMaterialControls.map(([, , id]) => id)];
  const ids = [...roomIds, "qualitySelect", "rirDuration", "sourceX", "sourceY", "sourceZ", "receiverX", "receiverY", "receiverZ", "motionMode", "motionMoving", "motionDistance", "motionFrameSpacing", "micOrientation", "micCount", "micSpacing", "sourceOrientation", "sourceElevation", "sourcePower", "fs"];
  ids.forEach((id) => document.getElementById(id)?.addEventListener("input", () => {
    if (floorplanMode && id === "height") {
      state.size[2] = clamp(number("height"), 2.0, 6.0);
      void loadFloorplanScene(state.floorplan.index, state.floorplan.roomId, state.floorplan.receiverRoomId);
      return;
    }
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
    if (["motionDistance", "motionFrameSpacing"].includes(id)) {
      clearTimeout(simulateTimer);
      simulateTimer = setTimeout(() => markSimulationPending(), 100);
      return;
    }
    markSimulationPending();
  }));
  document.getElementById("geometryParams")?.addEventListener("input", () => {
    readControls();
    clampScenePointsToRoom();
    markSimulationPending();
  });

  copyCodeButton?.addEventListener("click", copyCodeSnippet);
  runSimulationButton?.addEventListener("click", () => requestSimulation());
  motionPlayButton?.addEventListener("click", toggleMotionPlayback);
  motionTimelineEl?.addEventListener("input", () => {
    pauseMotionPlayback();
    setMotionDisplayPhase(Number(motionTimelineEl.value || 0) / 1000, true);
  });
  wetAudioEl?.addEventListener("play", () => {
    motionPlayback.active = false;
    updateMotionPlaybackControls();
  });
  wetAudioEl?.addEventListener("pause", updateMotionPlaybackControls);
  wetAudioEl?.addEventListener("ended", () => setMotionDisplayPhase(1, true));
  document.getElementById("randomPositions").addEventListener("click", randomizePositions);
  document.getElementById("resampleMotionPath")?.addEventListener("click", resampleRandomMotionPath);
  document.getElementById("reset").addEventListener("click", async () => {
    const floorplanSelection = multiRoomMode ? {
      index: state.floorplan.index,
      roomId: state.floorplan.roomId,
      receiverRoomId: state.floorplan.receiverRoomId,
    } : null;
    state = structuredClone(defaultState);
    selectedObjectId = null;
    pendingObjectId = null;
    dirtyObjectId = null;
    if (camera) camera.userData.fitted = false;
    if (floorplanSelection) {
      if (floorplanMode) {
        await loadFloorplanScene(floorplanSelection.index, floorplanSelection.roomId, floorplanSelection.receiverRoomId, { simulate: false });
      } else {
        const editor = document.getElementById("customSpecJson");
        if (editor) editor.value = "";
        await generateCustomScene({ populateEditor: false });
      }
    }
    updateControls();
    markSimulationPending();
  });
  if (floorplanMode) {
    document.getElementById("floorplanIdx")?.addEventListener("change", () => navigateFloorplan(controlNumber("floorplanIdx", 0), "nearest"));
    document.getElementById("floorplanPrev")?.addEventListener("click", () => navigateFloorplan(state.floorplan.index, "previous"));
    document.getElementById("floorplanNext")?.addEventListener("click", () => navigateFloorplan(state.floorplan.index, "next"));
    document.getElementById("floorplanRandom")?.addEventListener("click", () => navigateFloorplan(state.floorplan.index, "random"));
    document.getElementById("floorplanRoom")?.addEventListener("change", (event) => loadFloorplanScene(
      state.floorplan.index,
      event.target.value,
      state.floorplan.receiverRoomId === event.target.value ? "auto" : state.floorplan.receiverRoomId,
    ));
    document.getElementById("floorplanReceiverRoom")?.addEventListener("change", (event) => loadFloorplanScene(
      state.floorplan.index,
      state.floorplan.roomId,
      event.target.value,
    ));
  } else if (customMode) {
    document.getElementById("floorplanRoom")?.addEventListener("change", (event) => compileCustomScene(state.custom.spec, {
      sourceRoom: event.target.value,
      receiverRoom: state.floorplan.receiverRoomId === event.target.value ? event.target.value : state.floorplan.receiverRoomId,
    }));
    document.getElementById("floorplanReceiverRoom")?.addEventListener("change", (event) => compileCustomScene(state.custom.spec, {
      sourceRoom: state.floorplan.roomId,
      receiverRoom: event.target.value,
    }));
    document.getElementById("customImageOpacity")?.addEventListener("input", (event) => {
      state.custom.imageOpacity = Number(event.target.value || 1);
      const output = document.getElementById("customImageOpacityValue");
      if (output) output.textContent = `${Math.round(state.custom.imageOpacity * 100)}%`;
      drawFloorplanOverview();
    });
    document.getElementById("customImageFile")?.addEventListener("change", handleCustomImageUpload);
    document.getElementById("customVlmPrompt")?.addEventListener("click", () => copyCustomVlmPrompt().catch((error) => setStatus(String(error.message || error), true)));
    document.getElementById("customWidth")?.addEventListener("input", (event) => scheduleCustomEdit(() => rescaleCustomScene("width"), event.target));
    document.getElementById("customDepth")?.addEventListener("input", (event) => scheduleCustomEdit(() => rescaleCustomScene("depth"), event.target));
    document.getElementById("height")?.addEventListener("input", (event) => scheduleCustomEdit(() => {
      if (!state.custom.spec) return;
      const next = structuredClone(state.custom.spec);
      next.height_m = clamp(controlNumber("height", 2.8), 2.0, 6.0);
      return compileCustomScene(next);
    }, event.target));
    document.getElementById("customApplyJson")?.addEventListener("click", () => {
      try {
        const spec = JSON.parse(document.getElementById("customSpecJson")?.value || "{}");
        void compileCustomScene(spec).catch((error) => setStatus(String(error.message || error), true));
      } catch (error) {
        setStatus(`Invalid JSON · ${String(error.message || error)}`, true);
      }
    });
  }
  document.getElementById("randomMaterials")?.addEventListener("click", () => {
    state.materialSeed = Math.floor(Math.random() * 2147483647);
    setValue("materialSeed", state.materialSeed);
    renderMaterialSelections();
    markSimulationPending();
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
  ["objectX", "objectY", "objectZ", "objectWidth", "objectDepth", "objectHeight", "objectRotation", "objectAbsorption"].forEach((id) => {
    document.getElementById(id).addEventListener("input", handleObjectSettingsInput);
  });
  document.getElementById("pathLimit").addEventListener("input", () => {
    updatePathLimitLabel();
    rebuildThreeScene();
    safeDrawRirPanel();
  });
  [["layerDirect", "direct"], ["layerPortal", "portal"], ["layerDiffraction", "diffraction"], ["layerRt", "rt"]].forEach(([id, key]) => {
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
  if (multiRoomMode) {
    state.size[2] = number("height");
  } else {
    state.shape = value("shape");
    state.size = [number("sizeX"), number("sizeY"), number("height")];
  }
  state.materialSeed = Math.max(0, Math.round(controlNumber("materialSeed", state.materialSeed)));
  state.materialProfile = Object.fromEntries(
    activeBoundaryMaterialControls.map(([surface, , id]) => [surface, value(id) || "auto"]),
  );
  readGeometryParams();
  state.config.quality = value("qualitySelect");
  state.config.duration_s = clamp(number("rirDuration"), 0.3, 6.0);
  state.config.rt_duration_s = state.config.duration_s;
  state.motion = {
    mode: value("motionMode") || "static",
    moving: value("motionMoving") || "source",
    distance_m: clamp(controlNumber("motionDistance", 0.8), 0.2, 6.0),
    keyframe_spacing_m: clamp(controlNumber("motionFrameSpacing", 0.25), 0.1, 1.0),
    random_seed: Math.max(0, Math.round(Number(state.motion?.random_seed ?? 42))),
  };
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
  if (floorplanMode) {
    setValue("floorplanIdx", state.floorplan.index);
    const indexInput = document.getElementById("floorplanIdx");
    if (indexInput) indexInput.max = Math.max(0, state.floorplan.count - 1);
    syncFloorplanRoomOptions();
  } else if (customMode) {
    setValue("customWidth", state.size[0]);
    setValue("customDepth", state.size[1]);
    syncFloorplanRoomOptions();
    updateCustomValidation();
  } else {
    setValue("shape", state.shape);
    setValue("sizeX", state.size[0]);
    setValue("sizeY", state.size[1]);
  }
  setValue("height", state.size[2]);
  setValue("materialSeed", state.materialSeed);
  activeBoundaryMaterialControls.forEach(([surface, , id]) => setValue(id, state.materialProfile?.[surface] || "auto"));
  applyMaterialAvailability();
  setValue("qualitySelect", state.config.quality);
  setValue("rirDuration", Number(state.config.duration_s || 2.0).toFixed(1));
  setValue("sourceX", state.source[0]);
  setValue("sourceY", state.source[1]);
  setValue("sourceZ", state.source[2]);
  setValue("receiverX", state.receiver[0]);
  setValue("receiverY", state.receiver[1]);
  setValue("receiverZ", state.receiver[2]);
  syncMotionControls();
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

function syncMotionControls() {
  const motion = state.motion || defaultState.motion;
  setValue("motionMode", motion.mode);
  setValue("motionMoving", motion.moving);
  setValue("motionDistance", motion.distance_m);
  setValue("motionFrameSpacing", motion.keyframe_spacing_m ?? 0.25);
  const dynamic = motion.mode !== "static";
  ["motionMoving", "motionFrameSpacing"].forEach((id) => {
    const control = document.getElementById(id);
    if (control) control.disabled = !dynamic;
  });
  const distanceControl = document.getElementById("motionDistance");
  if (distanceControl) distanceControl.disabled = !dynamic;
  const resamplePathButton = document.getElementById("resampleMotionPath");
  const randomTravel = dynamic && motion.mode === "random";
  if (resamplePathButton) {
    resamplePathButton.hidden = !randomTravel;
    resamplePathButton.disabled = !randomTravel;
  }
  document.querySelector(".motionTravelField")?.classList.toggle("hasRandomAction", randomTravel);
  document.querySelector(".motionGrid")?.classList.toggle("disabled", !dynamic);
  const sampled = sampleMotionState();
  const distance = Number(sampled.distance_m || 0);
  const distanceLabel = document.getElementById("motionDistanceValue");
  if (distanceLabel) distanceLabel.textContent = `${Number(motion.distance_m).toFixed(1)} m`;
  const summary = document.getElementById("motionSummary");
  if (summary) {
    const portalLabel = sampled.path_model === "portal_route_smoothstep" ? " · portal" : "";
    const randomLabel = sampled.path_model === "random_room_route" ? " · random" : "";
    summary.textContent = dynamic
      ? `${motion.moving === "source" ? "Source" : "Microphone"} · ${sampled.keyframes} frames · ${distance.toFixed(2)} m${portalLabel}${randomLabel}`
      : "Static";
  }
  updateRunControls();
}

function updateRunControls() {
  const dynamic = state.motion?.mode !== "static";
  const readyFrames = Array.isArray(simData.dynamic?.frames) ? simData.dynamic.frames.length : 0;
  const expectedFrames = Number(simData.dynamic?.keyframes || 0);
  const motionReady = dynamic && expectedFrames > 1 && readyFrames === expectedFrames;
  if (runSimulationButton) {
    const plannedFrames = dynamic ? sampleMotionState().keyframes : 1;
    runSimulationButton.disabled = simulationRunning || hasUnconfirmedObjectChange();
    runSimulationButton.textContent = simulationRunning
      ? dynamic ? `Computing ${readyFrames}/${plannedFrames}` : "Computing..."
      : dynamic ? "Compute motion RIRs" : "Run static simulation";
  }
  if (motionPlayButton) motionPlayButton.disabled = !motionReady || simulationRunning;
  if (motionTimelineEl) motionTimelineEl.disabled = !motionReady || simulationRunning;
  updateMotionPlaybackControls();
  updateRirFrameMeta();
}

function updateMotionPlaybackControls() {
  if (!motionPlayButton) return;
  const playingAudio = Boolean(wetAudioEl && !wetAudioEl.paused && !wetAudioEl.ended);
  const playing = motionPlayback.active || playingAudio;
  motionPlayButton.textContent = playing ? "❚❚" : "▶";
  motionPlayButton.title = playing ? "Pause motion" : "Play motion";
  motionPlayButton.setAttribute("aria-label", motionPlayButton.title);
}

function updateRirFrameMeta() {
  const label = document.getElementById("rirFrameMeta");
  if (!label) return;
  const frames = simData.dynamic?.frames || [];
  if (frames.length > 0 && displayedMotionFrameIndex >= 0) {
    label.textContent = `Frame ${displayedMotionFrameIndex + 1} / ${Number(simData.dynamic.keyframes || frames.length)}`;
  } else {
    label.textContent = simData.rir?.samples?.length ? "Static result" : "Not computed";
  }
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
  simulateTimer = setTimeout(() => markSimulationPending(), 80);
}

async function requestSimulation() {
  if (hasUnconfirmedObjectChange()) {
    clearTimeout(simulateTimer);
    setStatus("Confirm the object edit to update simulation.");
    return;
  }
  const requestSeq = ++simulationRequestSeq;
  const payload = structuredClone(apiPayload());
  const dynamic = payload.motion?.mode && payload.motion.mode !== "static";
  let simulationSucceeded = false;
  simulationRunning = true;
  pauseMotionPlayback();
  updateRunControls();
  updateResultStatus();
  setStatus(dynamic ? `Computing motion RIR 0/${payload.motion.frames.length}...` : "Computing indoor RIR paths...");
  clearCalibrationAudio("reading.wav · waiting for RIR");
  try {
    const nextSimData = dynamic
      ? await requestMotionSimulation(payload, requestSeq)
      : await requestWorkbenchFrame(payload);
    if (requestSeq !== simulationRequestSeq) return;
    simData = nextSimData;
    lastSimulationPayload = payload;
    simulationSucceeded = true;
    setStatus(dynamic ? `Dynamic simulation updated · ${nextSimData.dynamic?.keyframes || payload.motion.frames.length} frames.` : "Simulation updated.");
  } catch (error) {
    if (requestSeq !== simulationRequestSeq) return;
    simData = makeClientScene(state);
    lastSimulationPayload = payload;
    simData.metadata = { ...(simData.metadata || {}), warning: String(error.message || error) };
    setStatus(`Simulation failed · ${String(error.message || error).slice(0, 120)}`, true);
  }
  if (requestSeq !== simulationRequestSeq) return;
  simulationRunning = false;
  rebuildThreeScene();
  updatePanels();
  if (simulationSucceeded) {
    if (dynamic) startMotionPlayback();
    void updateCalibrationAudio(simData, requestSeq);
  }
}

async function requestWorkbenchFrame(payload) {
  const response = await fetch("/api/v1/workbench", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function requestMotionSimulation(payload, requestSeq) {
  const motion = payload.motion;
  const plannedFrames = motion.frames;
  const basePayload = structuredClone(payload);
  delete basePayload.motion;
  const computedFrames = [];
  let referenceScene = null;
  for (let index = 0; index < plannedFrames.length; index += 1) {
    if (requestSeq !== simulationRequestSeq) throw new Error("simulation superseded");
    const planned = plannedFrames[index];
    setStatus(`Computing motion RIR ${index + 1}/${plannedFrames.length}...`);
    const frameMetadata = motionRoomMetadata(basePayload.room_metadata, planned.source, planned.receiver);
    const frameScene = await requestWorkbenchFrame({
      ...basePayload,
      source: planned.source,
      receiver: planned.receiver,
      room_metadata: frameMetadata,
    });
    if (requestSeq !== simulationRequestSeq) throw new Error("simulation superseded");
    referenceScene ||= frameScene;
    computedFrames.push({
      index,
      phase: Number(planned.phase),
      source: planned.source,
      receiver: planned.receiver,
      result_id: frameScene.result_id,
      rir: frameScene.rir,
      rt60: frameScene.rt60,
      paths: frameScene.paths,
    });
    referenceScene.dynamic = {
      mode: motion.mode,
      moving: motion.moving,
      distance_m: motion.distance_m,
      requested_distance_m: motion.requested_distance_m,
      keyframes: plannedFrames.length,
      path_model: motion.path_model,
      frames: computedFrames,
      planned_frames: plannedFrames,
      renderer: "time_varying_rir_snapshot_interpolation",
    };
    simData = referenceScene;
    displayedMotionFrameIndex = index;
    motionDisplayPhase = Number(planned.phase);
    rebuildThreeScene();
    updatePanels();
  }
  displayedMotionFrameIndex = 0;
  motionDisplayPhase = 0;
  return referenceScene;
}

function motionRoomMetadata(metadata, source, receiver) {
  if (!metadata?.multi_room?.enabled) return metadata;
  const updated = structuredClone(metadata);
  const multiRoom = updated.multi_room;
  const sourceRoom = roomIdForMotionPoint(source, multiRoom.rooms, multiRoom.source_room_id);
  const receiverRoom = roomIdForMotionPoint(receiver, multiRoom.rooms, multiRoom.receiver_room_id);
  const route = openPortalRoomRoute(sourceRoom, receiverRoom, multiRoom.portals || []);
  if (!route.room_ids.length) return updated;
  updated.source_room_id = sourceRoom;
  updated.receiver_room_id = receiverRoom;
  multiRoom.source_room_id = sourceRoom;
  multiRoom.receiver_room_id = receiverRoom;
  multiRoom.route_room_ids = route.room_ids;
  multiRoom.route_portal_ids = route.portal_ids;
  return updated;
}

function roomIdForMotionPoint(point, rooms, fallback) {
  const candidates = (rooms || [])
    .filter((room) => Array.isArray(room.corners) && pointInPolygon2D(point, room.corners))
    .map((room) => ({ id: room.id, clearance: distanceToRoomBoundary(point, room.corners) }))
    .sort((first, second) => second.clearance - first.clearance);
  return candidates[0]?.id || fallback;
}

function openPortalRoomRoute(sourceRoom, receiverRoom, portals) {
  if (!sourceRoom || !receiverRoom) return { room_ids: [], portal_ids: [] };
  if (sourceRoom === receiverRoom) return { room_ids: [sourceRoom], portal_ids: [] };
  const adjacency = new Map();
  (portals || []).filter((portal) => portal.open).forEach((portal) => {
    const roomIds = portal.room_ids || [];
    if (roomIds.length !== 2) return;
    if (!adjacency.has(roomIds[0])) adjacency.set(roomIds[0], []);
    if (!adjacency.has(roomIds[1])) adjacency.set(roomIds[1], []);
    adjacency.get(roomIds[0]).push([roomIds[1], portal.id]);
    adjacency.get(roomIds[1]).push([roomIds[0], portal.id]);
  });
  const queue = [sourceRoom];
  const visited = new Set([sourceRoom]);
  const previous = new Map();
  while (queue.length) {
    const current = queue.shift();
    if (current === receiverRoom) break;
    (adjacency.get(current) || []).forEach(([neighbor, portalId]) => {
      if (visited.has(neighbor)) return;
      visited.add(neighbor);
      previous.set(neighbor, [current, portalId]);
      queue.push(neighbor);
    });
  }
  if (!visited.has(receiverRoom)) return { room_ids: [], portal_ids: [] };
  const roomIds = [receiverRoom];
  const portalIds = [];
  while (roomIds[roomIds.length - 1] !== sourceRoom) {
    const [prior, portalId] = previous.get(roomIds[roomIds.length - 1]);
    roomIds.push(prior);
    portalIds.push(portalId);
  }
  roomIds.reverse();
  portalIds.reverse();
  return { room_ids: roomIds, portal_ids: portalIds };
}

function markSimulationPending(message = "Changes ready · run simulation to update RIR.") {
  clearTimeout(simulateTimer);
  simulationRequestSeq += 1;
  simulationRunning = false;
  pauseMotionPlayback();
  displayedMotionFrameIndex = -1;
  motionDisplayPhase = 0;
  lastSimulationPayload = null;
  simData = makeClientScene(state);
  clearCalibrationAudio("reading.wav · waiting for simulation");
  rebuildThreeScene();
  updatePanels();
  setStatus(message);
}

function apiPayload() {
  const mic = state.mic.type === "circular"
    ? { type: "circular", count: state.mic.count, radius_m: state.mic.radius_m, orientation_deg: state.mic.orientation_deg }
    : state.mic.type === "linear"
      ? { type: "linear", count: state.mic.count, spacing_m: state.mic.spacing_m, orientation_deg: state.mic.orientation_deg }
      : { type: state.mic.type, orientation_deg: state.mic.orientation_deg };
  const materialProfile = Object.fromEntries(
    activeBoundaryMaterialControls.map(([surface]) => [surface, state.materialProfile?.[surface] || "auto"]),
  );
  const motion = sampleMotionState();
  return {
    shape: state.shape,
    size: state.size,
    corners: cornersFor(state.shape, state.size, state.geometry),
    geometry: state.geometry,
    room_metadata: multiRoomMode ? state.floorplan.roomMetadata : undefined,
    materials: undefined,
    material_profile: materialProfile,
    material_seed: state.materialSeed,
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
    },
    motion,
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
  clearGroup(motionGroup);
  clearGroup(markerGroup);
  addPlan3D();
  addRoomShell3D();
  addFurniture3D();
  addPaths3D();
  addMotionTrajectory3D();
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

  const bounds = sceneDisplayBounds();
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
  if (hasVerticalSurfaceSegments()) {
    addSegmentedRoomShell3D(corners, height);
    return;
  }
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
  addBoundaryFeatures3D(height);

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

function hasVerticalSurfaceSegments() {
  const segments = simData.room?.metadata?.surface_segments || [];
  return segments.some((segment) => Number.isFinite(Number(segment.z_min)) && Number.isFinite(Number(segment.z_max)));
}

function addSegmentedRoomShell3D(corners, roomHeight) {
  const segments = simData.room?.metadata?.surface_segments || [];
  segments.forEach((segment) => {
    const a = segment.a;
    const b = segment.b;
    if (!Array.isArray(a) || !Array.isArray(b)) return;
    const dx = Number(b[0]) - Number(a[0]);
    const dy = Number(b[1]) - Number(a[1]);
    const length = Math.hypot(dx, dy);
    const zMin = clamp(Number(segment.z_min || 0), 0, roomHeight);
    const zMax = clamp(Number(segment.z_max ?? roomHeight), zMin, roomHeight);
    const segmentHeight = zMax - zMin;
    if (length < 0.015 || segmentHeight < 0.015) return;
    const isWindow = segment.type === "window";
    const isDoor = segment.type === "door";
    if (isWindow || isDoor) {
      addArchitecturalPanel3D(a, b, zMin, zMax, isWindow ? "window" : "door", false);
      return;
    }
    const material = new THREE.MeshStandardMaterial({
      color: 0xbfc7cb,
      transparent: true,
      opacity: 0.35,
      roughness: 0.88,
      metalness: 0,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const wall = new THREE.Mesh(new THREE.BoxGeometry(length, segmentHeight, 0.1), material);
    wall.position.set((Number(a[0]) + Number(b[0])) * 0.5, zMin + segmentHeight * 0.5, (Number(a[1]) + Number(b[1])) * 0.5);
    wall.rotation.y = -Math.atan2(dy, dx);
    wall.receiveShadow = true;
    shellGroup.add(wall);
  });
  (simData.room?.metadata?.boundary_features || []).forEach((feature) => {
    if (feature.type !== "door" || !feature.open) return;
    const zMin = clamp(Number(feature.sill_height_m || 0), 0, roomHeight);
    const zMax = clamp(zMin + Number(feature.height_m || 2.1), zMin, roomHeight);
    (feature.segments || []).forEach((segment) => {
      if (!Array.isArray(segment) || segment.length < 2) return;
      addArchitecturalPanel3D(segment[0], segment[1], zMin, zMax, "door", true);
    });
  });
  const ceiling = new THREE.Mesh(
    new THREE.ShapeGeometry(shapeFromCorners(corners)),
    new THREE.MeshStandardMaterial({ color: 0x9fb8c0, transparent: true, opacity: 0.055, roughness: 0.9, side: THREE.DoubleSide, depthWrite: false })
  );
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.y = roomHeight;
  shellGroup.add(ceiling);
}

function addArchitecturalPanel3D(a, b, zMin, zMax, kind, open) {
  const dx = Number(b[0]) - Number(a[0]);
  const dy = Number(b[1]) - Number(a[1]);
  const length = Math.hypot(dx, dy);
  const height = zMax - zMin;
  if (length < 0.04 || height < 0.04) return;

  const group = new THREE.Group();
  group.position.set(
    (Number(a[0]) + Number(b[0])) * 0.5,
    zMin + height * 0.5,
    (Number(a[1]) + Number(b[1])) * 0.5
  );
  group.rotation.y = -Math.atan2(dy, dx);
  const frameDepth = kind === "window" ? 0.12 : 0.085;
  const frameSize = Math.min(0.065, Math.max(0.025, Math.min(length, height) * 0.1));
  const frameMaterial = new THREE.MeshStandardMaterial({
    color: kind === "window" ? 0x506a72 : 0x5e5550,
    transparent: true,
    opacity: 0.86,
    roughness: 0.72,
    metalness: kind === "window" ? 0.12 : 0.02,
    depthWrite: false,
  });
  const addBar = (width, barHeight, x, y, depth = frameDepth) => {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(width, barHeight, depth), frameMaterial);
    bar.position.set(x, y, 0);
    group.add(bar);
  };

  if (kind === "window") {
    const glass = new THREE.Mesh(
      new THREE.BoxGeometry(Math.max(0.015, length - frameSize * 1.35), Math.max(0.015, height - frameSize * 1.35), 0.045),
      new THREE.MeshStandardMaterial({
        color: 0x8ccbd8,
        transparent: true,
        opacity: 0.3,
        roughness: 0.08,
        metalness: 0.05,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    group.add(glass);
    addBar(frameSize, height, -length * 0.5 + frameSize * 0.5, 0);
    addBar(frameSize, height, length * 0.5 - frameSize * 0.5, 0);
    addBar(length, frameSize, 0, -height * 0.5 + frameSize * 0.5);
    addBar(length, frameSize, 0, height * 0.5 - frameSize * 0.5);
    if (length > 0.72) addBar(frameSize * 0.72, Math.max(0.02, height - frameSize * 1.4), 0, 0, frameDepth * 0.78);
    if (height > 1.05) addBar(Math.max(0.02, length - frameSize * 1.4), frameSize * 0.66, 0, 0, frameDepth * 0.78);
  } else {
    addBar(frameSize, height, -length * 0.5 + frameSize * 0.5, 0);
    addBar(frameSize, height, length * 0.5 - frameSize * 0.5, 0);
    addBar(length, frameSize, 0, height * 0.5 - frameSize * 0.5);
    if (open) {
      addBar(Math.max(0.02, length - frameSize * 1.5), 0.025, 0, -height * 0.5 + 0.014, frameDepth * 0.75);
    } else {
      const leaf = new THREE.Mesh(
        new THREE.BoxGeometry(Math.max(0.02, length - frameSize * 1.55), Math.max(0.02, height - frameSize * 1.35), 0.045),
        new THREE.MeshStandardMaterial({ color: 0xa8795f, transparent: true, opacity: 0.76, roughness: 0.82, depthWrite: false })
      );
      leaf.position.z = -0.006;
      group.add(leaf);
      const inset = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(Math.max(0.03, length * 0.68), Math.max(0.04, height * 0.34), 0.052), 30),
        new THREE.LineBasicMaterial({ color: 0x6f5141, transparent: true, opacity: 0.5 })
      );
      inset.position.set(0, -height * 0.13, 0.012);
      group.add(inset);
      const handle = new THREE.Mesh(
        new THREE.SphereGeometry(Math.min(0.035, Math.max(0.018, length * 0.04)), 10, 8),
        new THREE.MeshStandardMaterial({ color: 0xd8c6a5, roughness: 0.26, metalness: 0.7 })
      );
      handle.position.set(length * 0.31, 0.02, frameDepth * 0.55);
      group.add(handle);
    }
  }
  shellGroup.add(group);
}

function addBoundaryFeatures3D(roomHeight) {
  const features = simData.room?.metadata?.boundary_features || [];
  features.forEach((feature, featureIndex) => {
    if (feature.open) return;
    const kind = feature.type === "window" ? "window" : "door";
    const sill = clamp(Number(feature.sill_height_m || 0), 0, roomHeight);
    const featureHeight = Math.min(Number(feature.height_m || (kind === "door" ? 2.1 : 1.2)), Math.max(0.05, roomHeight - sill));
    const color = kind === "window" ? 0x54a8c1 : 0xce6545;
    (feature.segments || []).forEach((segment, segmentIndex) => {
      if (!Array.isArray(segment) || segment.length < 2) return;
      const a = segment[0];
      const b = segment[1];
      const dx = Number(b[0]) - Number(a[0]);
      const dy = Number(b[1]) - Number(a[1]);
      const length = Math.hypot(dx, dy);
      if (length < 0.015) return;
      const material = new THREE.MeshStandardMaterial({
        color,
        transparent: true,
        opacity: kind === "window" ? 0.58 : 0.34,
        roughness: kind === "window" ? 0.28 : 0.72,
        metalness: kind === "window" ? 0.08 : 0,
        side: THREE.DoubleSide,
        depthWrite: kind !== "door",
      });
      const panel = new THREE.Mesh(new THREE.BoxGeometry(length, featureHeight, kind === "door" ? 0.045 : 0.145), material);
      panel.position.set((Number(a[0]) + Number(b[0])) * 0.5, sill + featureHeight * 0.5, (Number(a[1]) + Number(b[1])) * 0.5);
      panel.rotation.y = -Math.atan2(dy, dx);
      panel.userData.boundaryFeature = `${kind}_${featureIndex}_${segmentIndex}`;
      shellGroup.add(panel);
      if (kind === "door") {
        const edge = new THREE.LineSegments(
          new THREE.EdgesGeometry(panel.geometry, 30),
          new THREE.LineBasicMaterial({ color: 0x9a4f2f, transparent: true, opacity: 0.62 })
        );
        edge.position.copy(panel.position);
        edge.rotation.copy(panel.rotation);
        shellGroup.add(edge);
      }
    });
  });
}

function addPaths3D() {
  const limit = Number(document.getElementById("pathLimit")?.value || 512);
  const visible = displayPaths(pathsForDisplay()).slice(0, limit);
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
  if (path.kind === "portal_path") return layerState.portal;
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
  if (path.kind === "portal_path") return { color: 0xf2a541, radius: 0.013, opacity: 0.94, dashed: false };
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
  markerGroup.add(sourcePoint3D());
  addSourceDirection3D();
  markerGroup.add(receiverDevice3D());
  updateMotionMarkerAtPhase(0);
}

function addSourceDirection3D() {
  if (state.sourceDirectivity.type === "omni") return;
  const direction = sourceForwardVector3D();
  const origin = toVector3(state.source).addScaledVector(direction, 0.18);
  const arrow = new THREE.ArrowHelper(direction, origin, 0.72, 0xef476f, 0.18, 0.1);
  arrow.userData.motionRole = "source";
  arrow.userData.basePosition = origin.toArray();
  markerGroup.add(arrow);
}

function motionFramesForDisplay() {
  const plannedFrames = simData.dynamic?.planned_frames;
  if (Array.isArray(plannedFrames) && plannedFrames.length > 1) return plannedFrames;
  const remoteFrames = simData.dynamic?.frames;
  return Array.isArray(remoteFrames) && remoteFrames.length === Number(simData.dynamic?.keyframes)
    ? remoteFrames
    : sampleMotionState().frames;
}

function addMotionTrajectory3D() {
  if (state.motion?.mode === "static") return;
  const frames = motionFramesForDisplay();
  if (frames.length < 2) return;
  const role = state.motion.moving === "receiver" ? "receiver" : "source";
  const color = role === "source" ? 0xef476f : 0x0f7f9f;
  const points = frames.map((frame) => toVector3(frame[role]));
  motionGroup.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.88 })
  ));
  points.forEach((point, index) => {
    const endpoint = index === 0 || index === points.length - 1;
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(endpoint ? 0.055 : 0.032, 14, 10),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: endpoint ? 0.92 : 0.54, depthWrite: false })
    );
    marker.position.copy(point);
    motionGroup.add(marker);
  });
  const end = points.at(-1);
  const previous = points.at(-2);
  const direction = end.clone().sub(previous);
  if (direction.lengthSq() > 1e-8) {
    motionGroup.add(new THREE.ArrowHelper(direction.normalize(), previous, Math.min(0.28, previous.distanceTo(end)), color, 0.11, 0.07));
  }
}

function motionPositionAtPhase(role, phase) {
  const frames = motionFramesForDisplay();
  if (frames.length < 2) return role === "source" ? state.source : state.receiver;
  const bounded = clamp(phase, 0, 1);
  let upper = frames.findIndex((frame) => Number(frame.phase) >= bounded);
  if (upper <= 0) return frames[0][role];
  if (upper < 0) return frames.at(-1)[role];
  const lower = upper - 1;
  const startPhase = Number(frames[lower].phase);
  const endPhase = Number(frames[upper].phase);
  const mix = clamp((bounded - startPhase) / Math.max(endPhase - startPhase, 1e-9), 0, 1);
  return frames[lower][role].map((value, axis) => Number(value) + (Number(frames[upper][role][axis]) - Number(value)) * mix);
}

function updateMotionMarkerAtPhase(phase) {
  if (state.motion?.mode === "static") return;
  const role = state.motion.moving === "receiver" ? "receiver" : "source";
  const position = toVector3(motionPositionAtPhase(role, phase));
  const device = markerGroup.children.find((child) => child.userData.role === role);
  if (device) device.position.copy(position);
  markerGroup.children.filter((child) => child.userData.motionRole === role).forEach((child) => {
    const base = toVector3(role === "source" ? state.source : state.receiver);
    const initial = new THREE.Vector3().fromArray(child.userData.basePosition || [0, 0, 0]);
    child.position.copy(initial.add(position.clone().sub(base)));
  });
}

function dynamicResultReady() {
  const frames = simData.dynamic?.frames;
  return Array.isArray(frames) && frames.length > 1 && frames.length === Number(simData.dynamic?.keyframes);
}

function currentDynamicFrame() {
  const frames = simData.dynamic?.frames;
  if (!Array.isArray(frames) || frames.length === 0) return null;
  const index = clamp(displayedMotionFrameIndex, 0, frames.length - 1);
  return frames[index] || null;
}

function pathsForDisplay() {
  return currentDynamicFrame()?.paths || simData.paths || [];
}

function rirForDisplay() {
  return currentDynamicFrame()?.rir || simData.rir || {};
}

function rt60ForDisplay() {
  return currentDynamicFrame()?.rt60 || simData.rt60 || {};
}

function setMotionDisplayPhase(phase, force = false) {
  motionDisplayPhase = clamp(Number(phase) || 0, 0, 1);
  if (motionTimelineEl) motionTimelineEl.value = String(Math.round(motionDisplayPhase * 1000));
  updateMotionMarkerAtPhase(motionDisplayPhase);
  updateStageReadout();
  const frames = simData.dynamic?.frames || [];
  if (frames.length === 0) return;
  let nearest = 0;
  let nearestDelta = Number.POSITIVE_INFINITY;
  frames.forEach((frame, index) => {
    const delta = Math.abs(Number(frame.phase) - motionDisplayPhase);
    if (delta < nearestDelta) {
      nearest = index;
      nearestDelta = delta;
    }
  });
  if (!force && nearest === displayedMotionFrameIndex) {
    updateMotionFrameValue();
    return;
  }
  displayedMotionFrameIndex = nearest;
  clearGroup(pathGroup);
  addPaths3D();
  statsEl.innerHTML = statsHtml(pathsForDisplay());
  safeDrawRirPanel();
  drawMiniMap();
  updateMotionFrameValue();
  updateRirFrameMeta();
}

function updateMotionFrameValue() {
  const output = document.getElementById("motionFrameValue");
  if (!output) return;
  const total = Number(simData.dynamic?.keyframes || 0);
  output.textContent = total > 1 && displayedMotionFrameIndex >= 0
    ? `${displayedMotionFrameIndex + 1} / ${total}`
    : "-- / --";
}

function startMotionPlayback() {
  if (!dynamicResultReady()) return;
  wetAudioEl?.pause();
  motionPlayback = {
    active: true,
    startedAt: performance.now(),
    startPhase: 0,
    duration_s: clamp(Number(simData.dynamic?.distance_m || 1) / 0.8, 3.0, 10.0),
  };
  setMotionDisplayPhase(0, true);
  updateMotionPlaybackControls();
}

function pauseMotionPlayback() {
  motionPlayback.active = false;
  if (wetAudioEl && !wetAudioEl.paused) wetAudioEl.pause();
  updateMotionPlaybackControls();
}

function toggleMotionPlayback() {
  if (!dynamicResultReady()) return;
  const audioPlaying = wetAudioEl && !wetAudioEl.paused && !wetAudioEl.ended;
  if (motionPlayback.active || audioPlaying) {
    pauseMotionPlayback();
    return;
  }
  if (wetAudioEl?.src) {
    const dryDuration = Number(dryAudioEl?.duration || 0);
    if (wetAudioEl.ended || (dryDuration > 0 && wetAudioEl.currentTime >= dryDuration)) wetAudioEl.currentTime = 0;
    wetAudioEl.play().catch(() => startMotionPlayback());
    updateMotionPlaybackControls();
    return;
  }
  motionPlayback = {
    active: true,
    startedAt: performance.now(),
    startPhase: motionDisplayPhase >= 1 ? 0 : motionDisplayPhase,
    duration_s: clamp(Number(simData.dynamic?.distance_m || 1) / 0.8, 3.0, 10.0),
  };
  if (motionDisplayPhase >= 1) setMotionDisplayPhase(0, true);
  updateMotionPlaybackControls();
}

function sourceForwardVector3D() {
  const yaw = THREE.MathUtils.degToRad(Number(state.sourceDirectivity.orientation_deg || 0));
  const pitch = THREE.MathUtils.degToRad(Number(state.sourceDirectivity.elevation_deg || 0));
  const cosPitch = Math.cos(pitch);
  return new THREE.Vector3(
    cosPitch * Math.cos(yaw),
    Math.sin(pitch),
    cosPitch * Math.sin(yaw)
  ).normalize();
}

function receiverForwardVector3D() {
  const yaw = THREE.MathUtils.degToRad(Number(state.mic.orientation_deg || 0));
  return new THREE.Vector3(Math.cos(yaw), 0, Math.sin(yaw)).normalize();
}

function sourcePoint3D() {
  const group = new THREE.Group();
  group.name = "source-point";
  group.position.copy(toVector3(state.source));
  const core = new THREE.Mesh(
    new THREE.SphereGeometry(0.13, 24, 18),
    new THREE.MeshStandardMaterial({ color: 0xef476f, roughness: 0.52, metalness: 0.04 })
  );
  group.add(core);
  for (const radius of [0.17, 0.215]) {
    const wave = new THREE.Mesh(
      new THREE.TorusGeometry(radius, 0.008, 8, 40),
      new THREE.MeshBasicMaterial({ color: 0xef476f, transparent: true, opacity: 0.5, depthWrite: false })
    );
    wave.rotation.x = Math.PI / 2;
    group.add(wave);
  }
  group.add(markerHalo3D(0xef476f, 0.18, -0.155));
  markDeviceGroup(group, "source");
  return group;
}

function receiverDevice3D() {
  if (state.mic.type === "hrtf") return binauralHead3D();
  if (state.mic.type === "linear") return linearMicrophoneArray3D();
  if (state.mic.type === "circular") return circularMicrophoneArray3D();
  return monoReceiver3D();
}

function monoReceiver3D() {
  const group = new THREE.Group();
  group.name = "receiver-mono-point";
  group.position.copy(toVector3(state.receiver));
  const core = new THREE.Mesh(
    new THREE.SphereGeometry(0.105, 22, 16),
    new THREE.MeshStandardMaterial({ color: 0x0f7f9f, roughness: 0.44, metalness: 0.12 })
  );
  group.add(core);
  for (const rotation of [
    [Math.PI / 2, 0, 0],
    [0, Math.PI / 2, 0],
  ]) {
    const pickupRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.14, 0.008, 8, 36),
      new THREE.MeshBasicMaterial({ color: 0x65b8cc, transparent: true, opacity: 0.64, depthWrite: false })
    );
    pickupRing.rotation.set(...rotation);
    group.add(pickupRing);
  }
  group.add(markerHalo3D(0x0f7f9f, 0.16, -0.125));
  markDeviceGroup(group, "receiver");
  return group;
}

function binauralHead3D() {
  const group = new THREE.Group();
  group.name = "receiver-binaural-head";
  group.position.copy(toVector3(state.receiver));
  group.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), receiverForwardVector3D());
  const skinMaterial = new THREE.MeshStandardMaterial({ color: 0xcaa894, roughness: 0.78, metalness: 0.0 });
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.14, 28, 20), skinMaterial);
  head.scale.set(0.88, 1.22, 0.8);
  group.add(head);
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.035, 0.09, 16), skinMaterial);
  nose.rotation.z = -Math.PI / 2;
  nose.position.set(0.145, 0.005, 0);
  group.add(nose);
  for (const side of [-1, 1]) {
    const ear = new THREE.Mesh(
      new THREE.TorusGeometry(0.038, 0.012, 9, 24),
      new THREE.MeshStandardMaterial({ color: 0xb98f7c, roughness: 0.8 })
    );
    ear.scale.y = 1.25;
    ear.position.set(0, 0, side * 0.125);
    group.add(ear);
    const capsule = new THREE.Mesh(
      new THREE.SphereGeometry(0.018, 14, 10),
      new THREE.MeshStandardMaterial({ color: 0x0f7f9f, roughness: 0.36, metalness: 0.5 })
    );
    capsule.position.set(0, 0, side * 0.132);
    group.add(capsule);
  }
  group.add(markerHalo3D(0x0f7f9f, 0.19, -0.19));
  markDeviceGroup(group, "receiver");
  return group;
}

function linearMicrophoneArray3D() {
  const group = new THREE.Group();
  group.name = "receiver-linear-array";
  const points = microphonePoints().map(toVector3);
  const center = toVector3(state.receiver);
  group.position.copy(center);
  if (points.length >= 2) group.add(deviceTubeBetween3D(points[0].clone().sub(center), points.at(-1).clone().sub(center), 0.018, 0x40545d));
  points.forEach((point) => {
    const capsule = microphoneCapsule3D();
    capsule.position.copy(point.sub(center));
    group.add(capsule);
  });
  group.add(markerHalo3D(0x0f7f9f, Math.max(0.15, Number(state.mic.spacing_m || 0.08) * Number(state.mic.count || 4) * 0.58), -0.09));
  markDeviceGroup(group, "receiver");
  return group;
}

function circularMicrophoneArray3D() {
  const group = new THREE.Group();
  group.name = "receiver-circular-array";
  const center = toVector3(state.receiver);
  group.position.copy(center);
  const radius = Math.max(0.04, Number(state.mic.radius_m || 0.12));
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(radius, 0.014, 9, 48),
    new THREE.MeshStandardMaterial({ color: 0x40545d, roughness: 0.48, metalness: 0.48 })
  );
  ring.rotation.x = Math.PI / 2;
  group.add(ring);
  microphonePoints().map(toVector3).forEach((point) => {
    const capsule = microphoneCapsule3D();
    capsule.position.copy(point.sub(center));
    group.add(capsule);
  });
  group.add(markerHalo3D(0x0f7f9f, radius + 0.07, -0.09));
  markDeviceGroup(group, "receiver");
  return group;
}

function microphoneCapsule3D() {
  const group = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(0.023, 0.023, 0.075, 14),
    new THREE.MeshStandardMaterial({ color: 0x176f87, roughness: 0.38, metalness: 0.52 })
  );
  group.add(body);
  const grille = new THREE.Mesh(
    new THREE.SphereGeometry(0.027, 14, 10),
    new THREE.MeshStandardMaterial({ color: 0xa8bac1, roughness: 0.32, metalness: 0.7 })
  );
  grille.position.y = 0.038;
  group.add(grille);
  return group;
}

function deviceTubeBetween3D(start, end, radius, color) {
  const delta = end.clone().sub(start);
  const length = Math.max(delta.length(), 1e-6);
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, length, 14),
    new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.45 })
  );
  mesh.position.copy(start).add(end).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize());
  return mesh;
}

function markerHalo3D(color, radius, yOffset) {
  const halo = new THREE.Mesh(
    new THREE.TorusGeometry(radius, 0.011, 8, 40),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.76, depthWrite: false })
  );
  halo.rotation.x = Math.PI / 2;
  halo.position.y = yOffset;
  return halo;
}

function markDeviceGroup(group, role) {
  group.userData.role = role;
  group.traverse((child) => {
    child.userData.role = role;
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });
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
  let mesh = null;
  if (spec.kind === "sofa") mesh = sofaMesh(width, depth, height, color);
  else if (spec.kind === "bed") mesh = bedMesh(width, depth, height, color);
  else if (spec.kind === "chair") mesh = chairMesh(width, depth, height, color);
  else if (spec.kind === "rug") mesh = rugMesh(width, depth, height, color);
  else if (spec.kind === "appliance") mesh = applianceMesh(width, depth, height, color, object.type);
  else if (spec.kind === "person") mesh = personMesh(width, depth, height, color);
  else if (spec.kind === "table") mesh = tableMesh(width, depth, height, color);
  else if (spec.kind === "shelves") mesh = shelfMesh(width, depth, height, color);
  else if (spec.kind === "tile_surface") mesh = tileSurfaceMesh(width, depth, height, color);
  else if (spec.kind === "sanitary_fixture") mesh = sanitaryFixtureMesh(width, depth, height, color);
  else if (spec.kind === "structural_element") mesh = structuralElementMesh(width, depth, height, color);
  if (mesh) {
    mesh.position.y += Number(object.z ?? spec.z ?? height * 0.5) - height * 0.5;
    return mesh;
  }
  const geometry = new THREE.BoxGeometry(width, height, depth);
  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: spec.kind === "panel" ? 0.42 : 0.72,
    metalness: spec.kind === "panel" && object.type === "tv_mirror" ? 0.18 : 0.0,
    transparent: object.type === "window",
    opacity: object.type === "window" ? 0.62 : 1.0,
  });
  const bodyMesh = new THREE.Mesh(geometry, material);
  bodyMesh.position.y = Number(object.z ?? spec.z ?? height * 0.5);
  bodyMesh.castShadow = true;
  bodyMesh.receiveShadow = true;
  return bodyMesh;
}

function sofaMesh(width, depth, height, color) {
  const group = new THREE.Group();
  const fabric = new THREE.MeshStandardMaterial({ color, roughness: 0.86 });
  const base = new THREE.Mesh(new THREE.BoxGeometry(width, height * 0.42, depth), fabric);
  base.position.y = height * 0.24;
  group.add(base);
  const back = new THREE.Mesh(new THREE.BoxGeometry(width, height * 0.7, depth * 0.18), fabric);
  back.position.set(0, height * 0.48, -depth * 0.41);
  group.add(back);
  [-1, 1].forEach((side) => {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(width * 0.11, height * 0.55, depth), fabric);
    arm.position.set(side * width * 0.445, height * 0.36, 0);
    group.add(arm);
  });
  return group;
}

function bedMesh(width, depth, height, color) {
  const group = new THREE.Group();
  const mattress = new THREE.Mesh(
    new THREE.BoxGeometry(width, height * 0.62, depth),
    new THREE.MeshStandardMaterial({ color, roughness: 0.84 })
  );
  mattress.position.y = height * 0.34;
  group.add(mattress);
  const headboard = new THREE.Mesh(
    new THREE.BoxGeometry(width, height * 0.82, depth * 0.08),
    new THREE.MeshStandardMaterial({ color: 0x7a6a62, roughness: 0.76 })
  );
  headboard.position.set(0, height * 0.48, -depth * 0.48);
  group.add(headboard);
  return group;
}

function chairMesh(width, depth, height, color) {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({ color, roughness: 0.78 });
  const seat = new THREE.Mesh(new THREE.BoxGeometry(width, height * 0.18, depth), material);
  seat.position.y = height * 0.45;
  group.add(seat);
  const back = new THREE.Mesh(new THREE.BoxGeometry(width, height * 0.52, depth * 0.12), material);
  back.position.set(0, height * 0.72, -depth * 0.44);
  group.add(back);
  const legMaterial = new THREE.MeshStandardMaterial({ color: 0x5d5046, roughness: 0.8 });
  [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach(([sx, sz]) => {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.045, height * 0.42, 0.045), legMaterial);
    leg.position.set(sx * width * 0.36, height * 0.21, sz * depth * 0.34);
    group.add(leg);
  });
  return group;
}

function rugMesh(width, depth, height, color) {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(width, Math.max(0.018, height), depth),
    new THREE.MeshStandardMaterial({ color, roughness: 0.92 })
  );
  mesh.position.y = Math.max(0.012, height * 0.5);
  mesh.receiveShadow = true;
  return mesh;
}

function applianceMesh(width, depth, height, color, type) {
  const group = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(width, height, depth),
    new THREE.MeshStandardMaterial({ color, roughness: 0.48, metalness: 0.18 })
  );
  body.position.y = height * 0.5;
  group.add(body);
  const face = new THREE.Mesh(
    new THREE.BoxGeometry(width * 0.82, height * (type === "fridge" ? 0.78 : 0.56), 0.018),
    new THREE.MeshStandardMaterial({ color: 0xe8eef1, roughness: 0.42, metalness: 0.08 })
  );
  face.position.set(0, height * (type === "fridge" ? 0.52 : 0.5), depth * 0.51);
  group.add(face);
  if (type === "washing_machine") {
    const drum = new THREE.Mesh(
      new THREE.CylinderGeometry(Math.min(width, height) * 0.22, Math.min(width, height) * 0.22, 0.022, 32),
      new THREE.MeshStandardMaterial({ color: 0x6f8794, roughness: 0.35, metalness: 0.22 })
    );
    drum.rotation.x = Math.PI / 2;
    drum.position.set(0, height * 0.5, depth * 0.535);
    group.add(drum);
  } else {
    const split = new THREE.Mesh(
      new THREE.BoxGeometry(width * 0.84, 0.014, 0.024),
      new THREE.MeshBasicMaterial({ color: 0x849198 })
    );
    split.position.set(0, height * 0.56, depth * 0.535);
    group.add(split);
  }
  return group;
}

function personMesh(width, depth, height, color) {
  const group = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CapsuleGeometry(Math.max(width, depth) * 0.33, height * 0.52, 8, 16),
    new THREE.MeshStandardMaterial({ color, roughness: 0.82 })
  );
  body.position.y = height * 0.43;
  group.add(body);
  const head = new THREE.Mesh(
    new THREE.SphereGeometry(Math.max(width, depth) * 0.28, 20, 14),
    new THREE.MeshStandardMaterial({ color: 0xcaa894, roughness: 0.8 })
  );
  head.position.y = height * 0.9;
  group.add(head);
  return group;
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

function tileSurfaceMesh(width, depth, height, color) {
  const group = new THREE.Group();
  const slabHeight = Math.max(0.018, height);
  const slab = new THREE.Mesh(
    new THREE.BoxGeometry(width, slabHeight, depth),
    new THREE.MeshStandardMaterial({ color, roughness: 0.32, metalness: 0.02 })
  );
  slab.position.y = slabHeight * 0.5;
  group.add(slab);
  const grout = new THREE.MeshBasicMaterial({ color: 0x6f7d80 });
  [-0.25, 0.25].forEach((ratio) => {
    const seamX = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.004, depth * 0.98), grout);
    seamX.position.set(width * ratio, slabHeight + 0.002, 0);
    group.add(seamX);
    const seamZ = new THREE.Mesh(new THREE.BoxGeometry(width * 0.98, 0.004, 0.008), grout);
    seamZ.position.set(0, slabHeight + 0.002, depth * ratio);
    group.add(seamZ);
  });
  return group;
}

function sanitaryFixtureMesh(width, depth, height, color) {
  const group = new THREE.Group();
  const ceramic = new THREE.MeshStandardMaterial({ color, roughness: 0.3, metalness: 0.02 });
  const basin = new THREE.MeshStandardMaterial({ color: 0xb9d3d6, roughness: 0.24 });
  const wallHeight = height * 0.72;
  const wallY = wallHeight * 0.5;
  const addPart = (partWidth, partHeight, partDepth, x, y, z, material = ceramic) => {
    const part = new THREE.Mesh(new THREE.BoxGeometry(partWidth, partHeight, partDepth), material);
    part.position.set(x, y, z);
    group.add(part);
  };
  addPart(width * 0.82, height * 0.16, depth * 0.72, 0, height * 0.08, 0);
  addPart(width, wallHeight, depth * 0.12, 0, wallY, -depth * 0.44);
  addPart(width, wallHeight, depth * 0.12, 0, wallY, depth * 0.44);
  addPart(width * 0.11, wallHeight, depth * 0.76, -width * 0.445, wallY, 0);
  addPart(width * 0.11, wallHeight, depth * 0.76, width * 0.445, wallY, 0);
  addPart(width * 0.72, 0.012, depth * 0.58, 0, height * 0.18, 0, basin);
  const chrome = new THREE.MeshStandardMaterial({ color: 0x8c999e, roughness: 0.2, metalness: 0.72 });
  const faucet = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, height * 0.32, 12), chrome);
  faucet.position.set(width * 0.28, height * 0.86, -depth * 0.38);
  group.add(faucet);
  return group;
}

function structuralElementMesh(width, depth, height, color) {
  const group = new THREE.Group();
  const concrete = new THREE.MeshStandardMaterial({ color, roughness: 0.88 });
  const base = new THREE.Mesh(new THREE.BoxGeometry(width, height * 0.06, depth), concrete);
  base.position.y = height * 0.03;
  group.add(base);
  const shaft = new THREE.Mesh(new THREE.BoxGeometry(width * 0.78, height * 0.88, depth * 0.78), concrete);
  shaft.position.y = height * 0.5;
  group.add(shaft);
  const capital = new THREE.Mesh(new THREE.BoxGeometry(width, height * 0.06, depth), concrete);
  capital.position.y = height * 0.97;
  group.add(capital);
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
  if (state.motion?.mode !== "static" && dynamicResultReady()) {
    const dryDuration = Number(dryAudioEl?.duration || 0);
    const audioPlaying = wetAudioEl && !wetAudioEl.paused && !wetAudioEl.ended && dryDuration > 0;
    if (audioPlaying) {
      setMotionDisplayPhase(Number(wetAudioEl.currentTime || 0) / dryDuration);
    } else if (motionPlayback.active) {
      const elapsed = (performance.now() - motionPlayback.startedAt) / 1000;
      const phase = motionPlayback.startPhase + elapsed / Math.max(motionPlayback.duration_s, 0.1);
      setMotionDisplayPhase(phase);
      if (phase >= 1) {
        motionPlayback.active = false;
        updateMotionPlaybackControls();
      }
    }
  }
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
  const bounds = sceneDisplayBounds();
  const center = new THREE.Vector3((bounds.x0 + bounds.x1) * 0.5, 0, (bounds.y0 + bounds.y1) * 0.5);
  const span = Math.max(bounds.w, bounds.h, Number(simData.room.height_m || 2.8), 1);
  camera.userData.viewSize = span * 0.86 + 2.3;
  camera.userData.viewMode = "iso";
  camera.zoom = 1;
  camera.up.set(0, 1, 0);
  controls.minPolarAngle = Math.PI * 0.16;
  controls.target.set(center.x, Number(simData.room.height_m || 2.8) * 0.35, center.z);
  camera.position.set(center.x + span * 0.85, span * 0.95, center.z + span * 1.05);
  camera.lookAt(controls.target);
  camera.userData.fitted = true;
  camera.userData.sceneSignature = signature;
  controls.update();
  resize();
  setActiveViewControl("viewIso");
}

function cameraSceneSignature() {
  const bounds = sceneDisplayBounds();
  const roomIds = (simData.room?.metadata?.multi_room?.rooms || []).map((room) => room.id).join(",");
  return [
    floorplanMode ? `idx=${state.floorplan.index}` : customMode ? `custom=${state.custom.spec?.title || "scene"}` : `shape=${state.shape}`,
    `rooms=${roomIds}`,
    `bounds=${bounds.x0.toFixed(3)},${bounds.y0.toFixed(3)},${bounds.x1.toFixed(3)},${bounds.y1.toFixed(3)}`,
    `h=${Number(simData.room?.height_m || 0).toFixed(3)}`,
  ].join("|");
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
    child.traverse?.((node) => {
      node.geometry?.dispose?.();
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      materials.filter(Boolean).forEach((material) => material.dispose?.());
    });
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
      materials: {},
      metadata: {
        shape: current.shape,
        geometry_model: current.shape === "rectangle" ? "rectangular room" : current.shape === "floorplan" ? "floorplan room extrusion" : "extruded polygon",
        geometry_params: { ...(current.geometry || {}) },
        ...(current.shape === "floorplan" ? (current.floorplan.roomMetadata || {}) : {})
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
  if (shape === "floorplan" && Array.isArray(state.floorplan?.corners) && state.floorplan.corners.length >= 3) {
    return state.floorplan.corners.map((point) => point.map(Number));
  }
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
  if (!container) return;
  container.innerHTML = "";
  presets.forEach((preset) => {
    const button = document.createElement("button");
    button.className = "thumb";
    button.dataset.shape = preset.id;
    button.innerHTML = `<canvas width="160" height="108"></canvas><span>${preset.title}</span>`;
    button.addEventListener("click", () => {
      state.shape = preset.id;
      applyPresetPoints();
      updateControls();
      markSimulationPending();
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
      markSimulationPending();
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
      markSimulationPending();
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
        setStatus(`${spec.title} selected. Set dimensions, then Add object.`);
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
  if (type === "rug") {
    ctx.fillRect(w * 0.18, h * 0.62, w * 0.64, h * 0.12);
    ctx.strokeRect(w * 0.18, h * 0.62, w * 0.64, h * 0.12);
  } else if (type === "curtain" || type === "tv_mirror" || type === "acoustic_panel" || type === "panel") {
    ctx.save();
    ctx.translate(w * 0.5, h * 0.55);
    ctx.rotate(-0.28);
    ctx.fillRect(-42, -7, 84, 14);
    ctx.strokeRect(-42, -7, 84, 14);
    ctx.restore();
  } else if (type === "fridge") {
    ctx.fillRect(w * 0.38, h * 0.22, w * 0.24, h * 0.52);
    ctx.strokeRect(w * 0.38, h * 0.22, w * 0.24, h * 0.52);
    ctx.beginPath();
    ctx.moveTo(w * 0.38, h * 0.49);
    ctx.lineTo(w * 0.62, h * 0.49);
    ctx.stroke();
  } else if (type === "washing_machine") {
    ctx.fillRect(w * 0.34, h * 0.34, w * 0.32, h * 0.34);
    ctx.strokeRect(w * 0.34, h * 0.34, w * 0.32, h * 0.34);
    ctx.beginPath();
    ctx.arc(w * 0.5, h * 0.52, h * 0.095, 0, Math.PI * 2);
    ctx.stroke();
  } else if (type === "tile_surface") {
    ctx.fillRect(w * 0.2, h * 0.48, w * 0.6, h * 0.24);
    ctx.strokeRect(w * 0.2, h * 0.48, w * 0.6, h * 0.24);
    ctx.beginPath();
    [0.4, 0.6].forEach((ratio) => {
      ctx.moveTo(w * ratio, h * 0.48);
      ctx.lineTo(w * ratio, h * 0.72);
    });
    ctx.moveTo(w * 0.2, h * 0.6);
    ctx.lineTo(w * 0.8, h * 0.6);
    ctx.stroke();
  } else if (type === "sanitary_fixture") {
    ctx.fillRect(w * 0.22, h * 0.4, w * 0.56, h * 0.3);
    ctx.strokeRect(w * 0.22, h * 0.4, w * 0.56, h * 0.3);
    ctx.fillStyle = "#b9d3d6";
    ctx.fillRect(w * 0.3, h * 0.46, w * 0.4, h * 0.14);
    ctx.strokeRect(w * 0.3, h * 0.46, w * 0.4, h * 0.14);
  } else if (type === "structural_element") {
    ctx.fillRect(w * 0.38, h * 0.22, w * 0.24, h * 0.54);
    ctx.strokeRect(w * 0.38, h * 0.22, w * 0.24, h * 0.54);
    ctx.fillRect(w * 0.33, h * 0.19, w * 0.34, h * 0.08);
    ctx.strokeRect(w * 0.33, h * 0.19, w * 0.34, h * 0.08);
    ctx.fillRect(w * 0.33, h * 0.71, w * 0.34, h * 0.08);
    ctx.strokeRect(w * 0.33, h * 0.71, w * 0.34, h * 0.08);
  } else if (type === "person") {
    ctx.beginPath();
    ctx.arc(w * 0.5, h * 0.3, h * 0.08, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillRect(w * 0.43, h * 0.4, w * 0.14, h * 0.28);
    ctx.strokeRect(w * 0.43, h * 0.4, w * 0.14, h * 0.28);
  } else if (type === "bed" || type === "low_block") {
    ctx.fillRect(w * 0.23, h * 0.58, w * 0.54, h * 0.16);
    ctx.strokeRect(w * 0.23, h * 0.58, w * 0.54, h * 0.16);
  } else if (type === "chair") {
    ctx.fillRect(w * 0.38, h * 0.46, w * 0.24, h * 0.22);
    ctx.strokeRect(w * 0.38, h * 0.46, w * 0.24, h * 0.22);
    ctx.fillRect(w * 0.42, h * 0.27, w * 0.16, h * 0.22);
    ctx.strokeRect(w * 0.42, h * 0.27, w * 0.16, h * 0.22);
  } else if (type === "table") {
    ctx.fillRect(w * 0.24, h * 0.42, w * 0.52, h * 0.12);
    ctx.strokeRect(w * 0.24, h * 0.42, w * 0.52, h * 0.12);
    [[0.29, 0.55], [0.69, 0.55]].forEach(([rx, ry]) => ctx.fillRect(w * rx, h * ry, 5, 18));
  } else if (type === "cabinet") {
    ctx.fillRect(w * 0.33, h * 0.24, w * 0.34, h * 0.48);
    ctx.strokeRect(w * 0.33, h * 0.24, w * 0.34, h * 0.48);
    ctx.beginPath();
    ctx.moveTo(w * 0.33, h * 0.4);
    ctx.lineTo(w * 0.67, h * 0.4);
    ctx.moveTo(w * 0.33, h * 0.56);
    ctx.lineTo(w * 0.67, h * 0.56);
    ctx.stroke();
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
  const paths = pathsForDisplay();
  syncMotionControls();
  updatePathLimitLabel();
  const sampledMotion = sampleMotionState();
  const motionLabel = state.motion?.mode === "static"
    ? "Static"
    : `${multiRoomMode ? state.motion.mode.replaceAll("_", "-") : state.motion.mode === "random" ? "random travel" : "approach travel"} · ${sampledMotion.keyframes} frames`;
  document.getElementById("hudMeta").textContent = `${presetTitle(state.shape)} | ${motionLabel} | ${paths.length} paths | ${state.config.fs} Hz`;
  document.getElementById("receiverType").textContent = state.mic.type;
  document.getElementById("sourceDirectivityType").textContent = state.sourceDirectivity.type;
  updateStageReadout();
  updateResultStatus();
  statsEl.innerHTML = statsHtml(paths);
  codeEl.textContent = acousticAgentCode();
  drawMiniMap();
  safeDrawRirPanel();
  refreshMicThumbnails();
  refreshSourceDirectivityThumbnails();
  refreshThumbnails();
  if (multiRoomMode) syncFloorplanRoomOptions();
  renderMaterialSelections();
  renderObjectMaterialSelection();
  const countLabel = document.getElementById("sceneObjectCount");
  if (countLabel) {
    const count = (state.objects || []).length;
    countLabel.textContent = `${count} object${count === 1 ? "" : "s"}`;
  }
  refreshObjectThumbnails(sceneObjectById(selectedObjectId)?.type || activeObjectType());
  updateMotionFrameValue();
  updateRunControls();
}

function updateStageReadout() {
  const element = document.getElementById("stageDistance");
  if (!element) return;
  const source = (state.motion?.mode === "static"
    ? state.source
    : motionPositionAtPhase("source", motionDisplayPhase)) || state.source;
  const receiver = (state.motion?.mode === "static"
    ? state.receiver
    : motionPositionAtPhase("receiver", motionDisplayPhase)) || state.receiver;
  const distance = Math.hypot(
    Number(source?.[0] || 0) - Number(receiver?.[0] || 0),
    Number(source?.[1] || 0) - Number(receiver?.[1] || 0),
    Number(source?.[2] || 0) - Number(receiver?.[2] || 0),
  );
  element.textContent = `${distance.toFixed(2)} m`;
}

function updateResultStatus() {
  const element = document.getElementById("resultStatus");
  if (!element) return;
  const rir = rirForDisplay();
  const warning = simData.metadata?.warning;
  const ready = Number(rir.duration_s || 0) > 0;
  const stateName = warning ? "error" : simulationRunning ? "running" : ready ? "ready" : "pending";
  element.dataset.state = stateName;
  element.textContent = warning ? "Error" : simulationRunning ? "Running" : ready ? "Ready" : "Pending";
}

function renderMaterialSelections() {
  const materials = simData.room?.materials || {};
  activeBoundaryMaterialControls.forEach(([surface, semantic]) => {
    const label = document.getElementById(`sampled-${surface}`);
    if (!label) return;
    const material = materials[surface];
    if (!material) {
      label.textContent = `VLM semantic · ${semantic.replaceAll("_", " ")}`;
      label.title = "";
      return;
    }
    const level = String(material.resolved_absorption_class || material.absorption_class || "auto").replaceAll("_", " ");
    const family = String(material.material_type || semantic).replaceAll("_", " ");
    label.textContent = `${level} · ${family}`;
    const bands = Object.entries(material.absorption || {}).map(([band, alpha]) => `${band} Hz ${Number(alpha).toFixed(2)}`).join(" · ");
    label.title = `${material.name || material.id}${bands ? `\n${bands}` : ""}`;
  });
}

function renderObjectMaterialSelection() {
  const label = document.getElementById("objectMaterialResult");
  if (!label) return;
  const selected = sceneObjectById(selectedObjectId);
  if (!selected) {
    label.textContent = "Semantic material sampled after simulation.";
    label.title = "";
    return;
  }
  const simulated = (simData.objects || []).find((item) => item.id === selected.id);
  const material = simulated?.material_selection;
  if (!material) {
    label.textContent = `${String(selected.semantic || selected.type).replaceAll("_", " ")} · pending simulation`;
    label.title = "";
    return;
  }
  const level = String(material.resolved_absorption_class || material.absorption_class || "auto").replaceAll("_", " ");
  label.textContent = `${level} · ${String(material.material_type || material.semantic).replaceAll("_", " ")}`;
  label.title = material.material_name || material.material_id || "";
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
  const rt60 = rt60ForDisplay();
  const rir = rirForDisplay();
  const metrics = rir.metrics || {};
  const rirRt60 = rt60.rir_rt60_s ?? rt60.rt60_s;
  const rows = [
    ["Broadband EDC RT60", formatSeconds(rirRt60)],
    ["DRR", formatDb(metrics.drr_db)],
    ["C50 / C80", `${formatDb(metrics.c50_db)} / ${formatDb(metrics.c80_db)}`],
    ["Dominant path", `${formatDb(metrics.dominant_path_gain_db)} @ ${formatMs(metrics.dominant_path_time_ms)}`],
    ["RIR max sample", `${formatDb(metrics.peak_dbfs)} @ ${formatMs(metrics.peak_time_ms)}`],
    ["RMS level", formatDb(metrics.rms_dbfs)],
    ["Length", `${formatSeconds(rir.duration_s)} | ${Number(rir.channel_count || 1)} ch`]
  ];
  return [
    `<div class="stat"><span>${rows[0][0]}</span><strong>${rows[0][1]}</strong></div>`,
    ...rows.slice(1).map(([k, v]) => `<div class="stat"><span>${k}</span><strong>${v}</strong></div>`)
  ].join("");
}

function drawMiniMap() {
  const mini = document.getElementById("miniCanvas");
  if (!mini) return;
  const ctx = mini.getContext("2d");
  drawAcousticMiniMap(ctx, mini.width, mini.height);
}

function drawAcousticMiniMap(ctx, width, height) {
  const corners = simData.room.corners;
  const bounds = sceneDisplayBounds();
  const pad = 17;
  const scale = Math.min((width - pad * 2) / Math.max(bounds.w, 1e-6), (height - pad * 2) / Math.max(bounds.h, 1e-6));
  const offsetX = (width - bounds.w * scale) * 0.5;
  const offsetY = (height - bounds.h * scale) * 0.5;
  const toCanvas = ([x, y]) => [offsetX + (x - bounds.x0) * scale, offsetY + (y - bounds.y0) * scale];
  const selection = miniMapPathSelection(pathsForDisplay());

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
  ctx.strokeStyle = hasVerticalSurfaceSegments() ? "rgba(52,65,73,.24)" : "#344149";
  ctx.lineWidth = hasVerticalSurfaceSegments() ? 0.8 : 1.8;
  ctx.fill();
  ctx.stroke();

  drawMiniRoomPlan(ctx, toCanvas);
  drawMiniBoundaryFeatures(ctx, toCanvas);
  drawMiniFurniture(ctx, toCanvas, scale, 1);

  if (selection.portal) {
    drawMiniAcousticPath(ctx, selection.portal, toCanvas, "rgba(242,165,65,.96)", 2.35, true, false);
  } else if (selection.direct) {
    drawMiniAcousticPath(ctx, selection.direct, toCanvas, "rgba(239,71,111,.96)", 2.25, false, selection.nlos);
  }
  selection.diffractions.forEach((path, index) => {
    drawMiniAcousticPath(ctx, path, toCanvas, index === 0 ? "rgba(125,140,255,.96)" : "rgba(125,140,255,.62)", index === 0 ? 2.15 : 1.35, true, false);
  });

  const motionFrames = motionFramesForDisplay();
  if (state.motion?.mode !== "static" && motionFrames.length > 1) {
    const role = state.motion.moving === "receiver" ? "receiver" : "source";
    ctx.beginPath();
    motionFrames.forEach((frame, index) => {
      const [x, y] = toCanvas(frame[role]);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = role === "source" ? "rgba(239,71,111,.86)" : "rgba(15,127,159,.86)";
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const displayedFrame = {
    source: state.motion?.mode === "static" ? state.source : motionPositionAtPhase("source", motionDisplayPhase),
    receiver: state.motion?.mode === "static" ? state.receiver : motionPositionAtPhase("receiver", motionDisplayPhase),
  };
  drawMiniSourceDirectivity(ctx, toCanvas(displayedFrame.source));
  drawMiniMarker(ctx, toCanvas(displayedFrame.source), "#ef476f", "SRC", 1);
  drawMiniMarker(ctx, toCanvas(displayedFrame.receiver), "#0f7f9f", "MIC", -1);

  const routeIsPortal = selection.routeType === "portal";
  const stateLabel = routeIsPortal ? "PORTAL" : selection.nlos ? "NLOS / UTD" : "LOS";
  ctx.font = "700 9px system-ui";
  const badgeWidth = ctx.measureText(stateLabel).width + 12;
  ctx.fillStyle = routeIsPortal ? "rgba(242,165,65,.16)" : selection.nlos ? "rgba(125,140,255,.14)" : "rgba(239,71,111,.12)";
  ctx.fillRect(width - badgeWidth - 6, 6, badgeWidth, 17);
  ctx.fillStyle = routeIsPortal ? "#a75c10" : selection.nlos ? "#5968d8" : "#c93658";
  ctx.fillText(stateLabel, width - badgeWidth, 18);

}

function drawMiniRoomPlan(ctx, toCanvas) {
  const metadata = simData.room?.metadata || {};
  const multiRoom = metadata.multi_room;
  const roomColors = {
    living: "#eef6f4",
    bedroom: "#f0f2f9",
    kitchen: "#faf4e8",
    bathroom: "#edf6f8",
    storage: "#f1f3f4",
    balcony: "#f1f6ed",
  };
  ctx.save();
  if (multiRoom?.enabled) {
    (multiRoom.rooms || []).forEach((room) => {
      if (!Array.isArray(room.corners) || room.corners.length < 3) return;
      drawCanvasPolygon(ctx, room.corners, toCanvas);
      ctx.fillStyle = roomColors[room.type] || "#f1f4f5";
      ctx.globalAlpha = 0.92;
      ctx.fill();
    });
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "rgba(52,65,73,.9)";
  ctx.lineWidth = 2.1;
  ctx.lineCap = "square";
  (metadata.surface_segments || []).forEach((segment) => {
    if (segment.type !== "wall") return;
    const zMin = Number(segment.z_min || 0);
    const zMax = Number(segment.z_max || 0);
    if (zMin > 0.18 || zMax < 0.55) return;
    if (!Array.isArray(segment.a) || !Array.isArray(segment.b)) return;
    const [ax, ay] = toCanvas(segment.a);
    const [bx, by] = toCanvas(segment.b);
    if (Math.hypot(bx - ax, by - ay) < 0.25) return;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  });
  ctx.restore();
}

function drawMiniBoundaryFeatures(ctx, toCanvas) {
  const features = simData.room?.metadata?.boundary_features || [];
  features.forEach((feature) => {
    (feature.segments || []).forEach((segment) => {
      if (!Array.isArray(segment) || segment.length < 2) return;
      drawPlanBoundarySymbol(ctx, feature, segment, toCanvas, 0.92);
    });
  });
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
  const solvedPortal = paths
    .filter((path) => path.kind === "portal_path")
    .sort((a, b) => Number(a.delay_s || 0) - Number(b.delay_s || 0))[0] || null;
  const multiRoom = simData.room?.metadata?.multi_room;
  const sourceRoomId = multiRoom?.source_room_id || state.floorplan.roomId;
  const receiverRoomId = multiRoom?.receiver_room_id || state.floorplan.receiverRoomId;
  const crossRoom = Boolean(sourceRoomId && receiverRoomId && sourceRoomId !== receiverRoomId);
  const portalById = new Map((multiRoom?.portals || []).map((portal) => [portal.id, portal]));
  const routeCenters = (multiRoom?.route_portal_ids || [])
    .map((portalId) => portalById.get(portalId)?.center)
    .filter((point) => Array.isArray(point) && point.length >= 2);
  const provisionalPortal = crossRoom && routeCenters.length
    ? { kind: "portal_path", points: [state.source, ...routeCenters, state.receiver], provisional: true }
    : null;
  const portalRoute = solvedPortal || provisionalPortal;
  const portal = layerState.portal ? portalRoute : null;
  const direct = paths.find((path) => path.kind === "direct" || path.kind === "direct_transmitted");
  const sourceRoomCorners = (multiRoom?.rooms || []).find((room) => room.id === sourceRoomId)?.corners;
  const visibilityPolygon = crossRoom
    ? null
    : sourceRoomCorners || simData.room?.corners || cornersFor(state.shape, state.size, state.geometry);
  const visibleByGeometry = Boolean(visibilityPolygon) && segmentInsidePolygon2D(state.source, state.receiver, visibilityPolygon);
  const nlos = !portalRoute && (
    crossRoom
    || direct?.kind === "direct_transmitted"
    || Number(simData.metadata?.steam_audio?.direct?.occlusion ?? (visibleByGeometry ? 1 : 0)) < 1
  );
  const diffraction = displayPaths(paths)
    .filter((path) => path.kind === "diffraction")
    .sort((a, b) => Number(a.delay_s || 0) - Number(b.delay_s || 0));
  return {
    nlos,
    routeType: portalRoute ? "portal" : nlos ? "nlos" : "los",
    portal,
    direct: portalRoute ? null : direct || { kind: nlos ? "direct_transmitted" : "direct", points: [state.source, state.receiver] },
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
  const rir = rirForDisplay();
  const channels = rirWaveChannels(rir);
  const stride = Math.max(1, Number(rir.sample_stride || 1));
  const fs = Math.max(1, Number(rir.fs || state.config.fs));
  const paths = pathsForDisplay().filter(pathLayerVisible);
  const rt60 = rt60ForDisplay();
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
  drawRirPlotFrame(ctx, padL, decayTop, plotW, decayH, "Energy decay", RIR_DECAY_DB_TICKS);
  drawLateTailRegion(ctx, padL, waveTop, plotW, waveH + gap + decayH, maxDelay);

  if (visibleSamples > 0) {
    drawRirWaveforms(ctx, channels, visibleSamples, padL, waveMid, plotW, waveH, maxDelay, stride, fs);
    drawRirEnergyDecayCurve(ctx, channels, visibleSamples, maxChannelLength, padL, decayTop, plotW, decayH, maxDelay, stride, fs, rir, rt60);
  }
  drawRirRt60Badge(ctx, padL + plotW, 9, rt60);
  drawRirEventTicks(ctx, paths, padL, waveTop, plotW, waveH, maxDelay);
  drawRirPeakMarker(ctx, padL, waveTop, plotW, waveH, maxDelay, rir);
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
  const gridDivisions = dbTicks ? Math.max(1, dbTicks.length - 1) : 4;
  ctx.strokeStyle = "rgba(76, 90, 99, 0.12)";
  ctx.beginPath();
  for (let i = 1; i < gridDivisions; i += 1) {
    const gy = y + height * i / gridDivisions;
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

function drawRirEnergyDecayCurve(ctx, channels, visibleSamples, fullSamples, padL, top, plotW, plotH, maxDelay, stride, fs, rir, rt60) {
  const minDb = RIR_DECAY_MIN_DB;
  const decayDb = rirEnergyDecayDbSamples(channels, visibleSamples, fullSamples, rir);
  drawMaterialDecayReference(ctx, padL, top, plotW, plotH, maxDelay, minDb, rt60);
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

function rirEnergyDecayDbSamples(channels, visibleSamples, fullSamples, rir) {
  const backendDecay = Array.isArray(rir?.decay_db)
    ? rir.decay_db.map((value) => Number(value)).filter((value) => Number.isFinite(value))
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

function drawMaterialDecayReference(ctx, padL, top, plotW, plotH, maxDelay, minDb, rt60) {
  const materialRt60 = Number(rt60?.material_rt60_s);
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

function drawRirPeakMarker(ctx, padL, top, plotW, plotH, maxDelay, rir) {
  const metrics = rir?.metrics || {};
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
  const dynamicFrames = Array.isArray(scene?.dynamic?.frames) ? scene.dynamic.frames : [];
  try {
    if (!rirInfo.wav_url || !Array.isArray(rirInfo.shape)) throw new Error("exact RIR is unavailable");
    const rirUrls = dynamicFrames.length > 1
      ? dynamicFrames.map((frame) => frame.rir?.wav_url).filter(Boolean)
      : [rirInfo.wav_url];
    if (dynamicFrames.length > 1 && rirUrls.length !== dynamicFrames.length) throw new Error("dynamic RIR frames are incomplete");
    setCalibrationAudioMeta(dynamicFrames.length > 1
      ? `reading.wav · rendering ${dynamicFrames.length} motion frames`
      : `reading.wav · 44.1 → ${formatSampleRate(fs)}`);
    const [dryResponse, ...rirResponses] = await Promise.all([
      fetch(`/api/calibration-audio?fs=${encodeURIComponent(fs)}`, { cache: "no-store" }),
      ...rirUrls.map((url) => fetch(url, { cache: "no-store" })),
    ]);
    if (!dryResponse.ok) throw new Error(await dryResponse.text());
    const failedRir = rirResponses.find((response) => !response.ok);
    if (failedRir) throw new Error(await failedRir.text());
    const dry = decodePcm16Wav(await dryResponse.arrayBuffer());
    if (dry.fs !== fs) throw new Error(`dry sample rate is ${dry.fs} Hz`);
    if (token !== calibrationAudioSeq || requestSeq !== simulationRequestSeq) return;
    const rirs = [];
    for (const response of rirResponses) {
      const rir = decodeFloat32WavChannels(await response.arrayBuffer());
      if (rir.fs !== fs) throw new Error(`RIR sample rate is ${rir.fs} Hz`);
      rirs.push(monitorRirChannels(rir.channels));
    }
    const wetChannels = rirs.length > 1
      ? await convolveDynamicChannels(dry.samples, rirs, dynamicFrames.map((frame) => Number(frame.phase)), fs)
      : await convolveChannels(dry.samples, rirs[0], fs);
    if (token !== calibrationAudioSeq || requestSeq !== simulationRequestSeq) return;

    const sharedGain = 0.98 / Math.max(1.0, maxAbs(dry.samples), maxAbsChannels(wetChannels));
    const nextUrls = [
      URL.createObjectURL(encodePcm16WavChannels([scaledSamples(dry.samples, sharedGain)], fs)),
      URL.createObjectURL(encodePcm16WavChannels(scaledChannels(wetChannels, sharedGain), fs)),
    ];
    replaceCalibrationAudioUrls(nextUrls);
    dryAudioEl.src = nextUrls[0];
    wetAudioEl.src = nextUrls[1];
    dryAudioEl.load();
    wetAudioEl.load();
    setCalibrationAudioMeta(dynamicFrames.length > 1
      ? `reading.wav · ${dynamicFrames.length} moving RIR frames · ready`
      : `reading.wav · 44.1 → ${formatSampleRate(fs)} · ready`);
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

function decodeFloat32WavChannels(buffer) {
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
  const channels = Array.from({ length: format.channels }, () => new Float32Array(frames));
  for (let frame = 0; frame < frames; frame += 1) {
    for (let channel = 0; channel < format.channels; channel += 1) {
      channels[channel][frame] = view.getFloat32(dataOffset + (frame * format.channels + channel) * 4, true);
    }
  }
  return { channels, fs: format.fs };
}

function monitorRirChannels(channels) {
  if (channels.length <= 2) return channels;
  const length = Math.max(...channels.map((samples) => samples.length));
  const downmix = new Float32Array(length);
  for (const samples of channels) {
    for (let index = 0; index < samples.length; index += 1) downmix[index] += samples[index] / channels.length;
  }
  return [downmix];
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

async function convolveChannels(dry, rirChannels, fs) {
  return Promise.all(rirChannels.map((rir) => convolveMono(dry, rir, fs)));
}

async function convolveDynamicChannels(dry, rirs, phases, fs) {
  const renderedFrames = [];
  for (const rir of rirs) renderedFrames.push(await convolveChannels(dry, rir, fs));
  const channelCount = renderedFrames[0]?.length || 1;
  const outputLength = Math.max(...renderedFrames.flat().map((samples) => samples.length));
  const output = Array.from({ length: channelCount }, () => new Float32Array(outputLength));
  let upper = 1;
  for (let sample = 0; sample < outputLength; sample += 1) {
    const phase = clamp(sample / Math.max(dry.length - 1, 1), 0, 1);
    while (upper < phases.length - 1 && phase > phases[upper]) upper += 1;
    const lower = Math.max(0, upper - 1);
    const start = Number(phases[lower] ?? 0);
    const end = Number(phases[upper] ?? 1);
    const mix = lower === upper ? 0 : clamp((phase - start) / Math.max(end - start, 1e-9), 0, 1);
    for (let channel = 0; channel < channelCount; channel += 1) {
      output[channel][sample] = (renderedFrames[lower]?.[channel]?.[sample] || 0) * (1 - mix)
        + (renderedFrames[upper]?.[channel]?.[sample] || 0) * mix;
    }
  }
  return output;
}

function maxAbs(samples) {
  let peak = 0;
  for (let index = 0; index < samples.length; index += 1) peak = Math.max(peak, Math.abs(samples[index]));
  return peak;
}

function maxAbsChannels(channels) {
  return Math.max(0, ...channels.map((samples) => maxAbs(samples)));
}

function scaledSamples(samples, gain) {
  const output = new Float32Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) output[index] = samples[index] * gain;
  return output;
}

function scaledChannels(channels, gain) {
  return channels.map((samples) => scaledSamples(samples, gain));
}

function encodePcm16WavChannels(channels, fs) {
  const channelCount = Math.max(1, channels.length);
  const frameCount = Math.max(...channels.map((samples) => samples.length));
  const dataLength = frameCount * channelCount * 2;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channelCount, true);
  view.setUint32(24, fs, true);
  view.setUint32(28, fs * channelCount * 2, true);
  view.setUint16(32, channelCount * 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataLength, true);
  for (let frame = 0; frame < frameCount; frame += 1) {
    for (let channel = 0; channel < channelCount; channel += 1) {
      const value = Math.max(-1, Math.min(1, channels[channel]?.[frame] || 0));
      const offset = 44 + (frame * channelCount + channel) * 2;
      view.setInt16(offset, Math.round(value < 0 ? value * 32768 : value * 32767), true);
    }
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
  const motion = payload.motion || { mode: "static" };
  const dynamicMotion = motion.mode && motion.mode !== "static";
  const dynamicRunLines = dynamicMotion ? [
    "",
    "motion = agent.sample_motion(",
    `    mode="${motion.mode}", moving="${motion.moving || "source"}",`,
    `    distance_m=${Number(motion.requested_distance_m || motion.distance_m || 0.8)},`,
    `    keyframe_spacing_m=${Number(state.motion?.keyframe_spacing_m || 0.25)},`,
    ...(motion.mode === "random" ? [`    seed=${Number(motion.random_seed ?? state.motion?.random_seed ?? 42)},`] : []),
    ")",
    "result = agent.run_dynamic(motion)",
    "rir_frames = result.rirs",
  ] : ["rir = agent.run().rir"];
  const acousticGeometry = (payload.objects || []).map((object) => ({
    type: String(object.type || "sofa"),
    semantic: String(object.semantic || furnitureCatalog[object.type]?.semantic || object.type || "furniture"),
    absorption_class: String(object.absorption_class || "auto"),
    ...(object.material ? { material: String(object.material) } : {}),
    position: (object.position || [0, 0]).slice(0, 2).map(Number),
    z: Number(object.z ?? Number(object.size?.[2] || 1) * 0.5),
    size: (object.size || [1, 1, 1]).slice(0, 3).map(Number),
    rotation_deg: Number(object.rotation ?? object.rotation_deg ?? 0),
  }));

  if (customMode && shape === "floorplan") {
    const materialProfile = payload.material_profile || state.materialProfile || {};
    const materialSeed = Number(payload.material_seed ?? state.materialSeed ?? 42);
    const sourcePosition = JSON.stringify((payload.source || state.source).map(Number));
    const receiverPosition = JSON.stringify((payload.receiver || state.receiver).map(Number));
    return [
      "from acoustic_agent import AcousticAgent",
      "",
      `floorplan_spec = ${JSON.stringify(state.custom.spec || {}, null, 4)}`,
      `source = ${sourcePosition}  # [x, y, z] m`,
      `mic = ${receiverPosition}     # [x, y, z] m`,
      `material_seed = ${materialSeed}`,
      `material_profile = ${JSON.stringify(materialProfile, null, 4)}`,
      `mic_type = "${micType}"   # mono / hrtf / linear / circular`,
      `mic_params = ${JSON.stringify(micParams, null, 4)}`,
      `source_directivity = ${JSON.stringify(sourceModel, null, 4)}`,
      `acoustic_geometry = ${JSON.stringify(acousticGeometry, null, 4)}`,
      `quality = "${quality}"`,
      `rir_length = ${rirLength}`,
      `sample_rate = ${sampleRate}`,
      "",
      "agent = AcousticAgent.from_floorplan_spec(",
      "    floorplan_spec, source=source, receiver=mic,",
      "    material_seed=material_seed, material_profile=material_profile,",
      '    receiver_model={"type": mic_type, **mic_params},',
      "    source_model=source_directivity, acoustic_geometry=acoustic_geometry,",
      "    quality=quality, duration_s=rir_length, fs=sample_rate,",
      ")",
      ...dynamicRunLines,
    ].join("\n");
  }

  if (shape === "floorplan") {
    const floorplan = payload.room_metadata?.floorplan || state.floorplan?.dataset || {};
    const idx = Number(floorplan.index ?? state.floorplan?.index ?? 0);
    const materialProfile = payload.material_profile || state.materialProfile || {};
    const materialSeed = Number(payload.material_seed ?? state.materialSeed ?? 42);
    const sourcePosition = JSON.stringify((payload.source || state.source || [1.2, 1.1, 1.5]).map(Number));
    const receiverPosition = JSON.stringify((payload.receiver || state.receiver || [4.7, 2.8, 1.4]).map(Number));
    return [
      "from acoustic_agent import AcousticAgent",
      "",
      `idx = ${idx}`,
      `source = ${sourcePosition}  # [x, y, z] m`,
      `mic = ${receiverPosition}     # [x, y, z] m`,
      `material_seed = ${materialSeed}`,
      `material_profile = ${JSON.stringify(materialProfile, null, 4)}`,
      `mic_type = "${micType}"   # mono / hrtf / linear / circular`,
      `mic_params = ${JSON.stringify(micParams, null, 4)}`,
      `source_directivity = ${JSON.stringify(sourceModel, null, 4)}`,
      `acoustic_geometry = ${JSON.stringify(acousticGeometry, null, 4)}`,
      `quality = "${quality}"    # preview / simulation / fine / reference`,
      `rir_length = ${rirLength}  # seconds`,
      `sample_rate = ${sampleRate} # Hz`,
      "",
      "agent = AcousticAgent.from_floorplan(",
      "    idx=idx,",
      "    source=source, receiver=mic,",
      "    material_seed=material_seed,",
      "    material_profile=material_profile,",
      '    receiver_model={"type": mic_type, **mic_params},',
      "    source_model=source_directivity,",
      "    acoustic_geometry=acoustic_geometry,",
      "    quality=quality, duration_s=rir_length, fs=sample_rate,",
      ")",
      "",
      "print(agent.rooms)      # available room list",
      "print(agent.placement)  # sampled rooms and [x, y, z] positions",
      ...dynamicRunLines,
    ].join("\n");
  }

  const geometry = payload.geometry || {};
  const room = {
    shape,
    size: (payload.size || [6, 4, 2.8]).map(Number),
    material_profile: payload.material_profile || { wall: "auto", floor: "auto", ceiling: "auto" },
    material_seed: Number(payload.material_seed ?? 42),
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

  const source = JSON.stringify((payload.source || [1.2, 1.1, 1.5]).map(Number));
  const mic = JSON.stringify((payload.receiver || [4.7, 2.8, 1.4]).map(Number));
  const geometryRunLines = dynamicMotion ? [
    "motion = agent.sample_motion(",
    "    source=source, receiver=mic,",
    `    mode="${motion.mode}", moving="${motion.moving || "source"}",`,
    `    distance_m=${Number(motion.requested_distance_m || motion.distance_m || 0.8)},`,
    `    keyframe_spacing_m=${Number(state.motion?.keyframe_spacing_m || 0.25)},`,
    ...(motion.mode === "random" ? [`    seed=${Number(motion.random_seed ?? state.motion?.random_seed ?? 42)},`] : []),
    ")",
    "result = agent.run_dynamic(motion)",
    "rir_frames = result.rirs",
  ] : ["rir = agent.run(source=source, receiver=mic).rir"];
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
    ...geometryRunLines,
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
  const sourceCorners = floorplanRoomCorners(state.floorplan.roomId) || corners;
  const receiverCorners = floorplanRoomCorners(state.floorplan.receiverRoomId) || corners;
  state.source = safeRoomPoint(state.source, sourceCorners);
  state.receiver = safeRoomPoint(state.receiver, receiverCorners);
  state.source[2] = clamp(state.source[2], 0.05, Math.max(0.05, state.size[2] - 0.05));
  state.receiver[2] = clamp(state.receiver[2], 0.05, Math.max(0.05, state.size[2] - 0.05));
  normalizeAllObjectPlacements();
}

function randomizePositions() {
  const corners = cornersFor(state.shape, state.size, state.geometry);
  const sourceCorners = floorplanRoomCorners(state.floorplan.roomId) || corners;
  const receiverCorners = floorplanRoomCorners(state.floorplan.receiverRoomId) || corners;
  const bounds = getBounds(corners);
  const minDistance = Math.max(0.65, Math.min(bounds.w, bounds.h) * 0.22);
  const source = randomRoomPoint(sourceCorners, state.size[2]);
  let receiver = randomRoomPoint(receiverCorners, state.size[2]);
  for (let attempt = 0; attempt < 80 && distance2D(source, receiver) < minDistance; attempt += 1) {
    receiver = randomRoomPoint(receiverCorners, state.size[2]);
  }
  state.source = source;
  state.receiver = receiver;
  updateControls();
  markSimulationPending();
}

function resampleRandomMotionPath() {
  readControls();
  state.motion.mode = "random";
  state.motion.random_seed = Math.floor(Math.random() * 2147483647);
  setValue("motionMode", "random");
  markSimulationPending(`Random travel resampled · ${state.motion.distance_m.toFixed(1)} m.`);
}

function sampleMotionState() {
  const motion = state.motion || defaultState.motion;
  const source = state.source.map(Number);
  const receiver = state.receiver.map(Number);
  if (motion.mode === "static") {
    return {
      mode: "static",
      moving: motion.moving,
      requested_distance_m: 0,
      distance_m: 0,
      keyframes: 1,
      keyframe_spacing_m: 0,
      frames: [{ phase: 0, source, receiver }],
    };
  }

  const movingSource = motion.moving !== "receiver";
  const movingStart = movingSource ? source : receiver;
  const target = movingSource ? receiver : source;
  const roomId = movingSource ? state.floorplan.roomId : state.floorplan.receiverRoomId;
  const corners = floorplanRoomCorners(roomId) || cornersFor(state.shape, state.size, state.geometry);
  const dx = target[0] - movingStart[0];
  const dy = target[1] - movingStart[1];
  const separation = Math.max(Math.hypot(dx, dy), 1e-9);
  const portalRoute = acousticMotionRoute(movingSource);
  const routePortalIds = state.floorplan.roomMetadata?.multi_room?.route_portal_ids || [];
  const followsPortalRoute = multiRoomMode && routePortalIds.length > 0 && portalRoute.length >= 2;
  let requested = clamp(motion.distance_m, 0.2, 6.0);
  if (motion.mode === "random") {
    const sampledRoute = randomGeometryRoute(
      movingStart,
      corners,
      requested,
      motion.random_seed ?? 42,
    );
    const actual = sampledRoute.actual;
    const keyframes = motionKeyframeCount(actual, motion);
    const positions = samplePolyline3D(sampledRoute.route, actual, keyframes, false).reverse();
    const frames = positions.map((position, index) => ({
      phase: Number((index / Math.max(keyframes - 1, 1)).toFixed(6)),
      source: (movingSource ? position : source).map((value) => Number(value.toFixed(6))),
      receiver: (movingSource ? receiver : position).map((value) => Number(value.toFixed(6))),
    }));
    return {
      mode: "random",
      moving: movingSource ? "source" : "receiver",
      requested_distance_m: Number(requested.toFixed(4)),
      distance_m: Number(actual.toFixed(4)),
      keyframes,
      keyframe_spacing_m: Number((actual / Math.max(keyframes - 1, 1)).toFixed(4)),
      random_seed: Math.max(0, Math.round(Number(motion.random_seed ?? 42))),
      path_model: "random_room_route",
      frames,
    };
  }
  if (motion.mode === "approach" && !followsPortalRoute) {
    const route = geometryApproachRoute(movingStart, target, corners);
    const routeLength = polylineLength3D(route);
    const actual = Math.min(requested, Math.max(0.05, routeLength - 0.3));
    const keyframes = motionKeyframeCount(actual, motion);
    const positions = samplePolyline3D(route, actual, keyframes, false);
    const frames = positions.map((position, index) => ({
      phase: Number((index / Math.max(keyframes - 1, 1)).toFixed(6)),
      source: (movingSource ? position : source).map((value) => Number(value.toFixed(6))),
      receiver: (movingSource ? receiver : position).map((value) => Number(value.toFixed(6))),
    }));
    return {
      mode: "approach",
      moving: movingSource ? "source" : "receiver",
      requested_distance_m: Number(requested.toFixed(4)),
      distance_m: Number(actual.toFixed(4)),
      keyframes,
      keyframe_spacing_m: Number((actual / Math.max(keyframes - 1, 1)).toFixed(4)),
      path_model: "room_shortest_path",
      frames,
    };
  }
  let keyframes = motionKeyframeCount(requested, motion);
  if (motion.mode === "approach" && followsPortalRoute) {
    const routeLength = polylineLength3D(portalRoute);
    const actual = Math.min(requested, Math.max(0.05, routeLength - 0.3));
    keyframes = motionKeyframeCount(actual, motion);
    const positions = snapMotionPositionsToRooms(samplePolyline3D(portalRoute, actual, keyframes));
    const frames = positions.map((position, index) => ({
      phase: Number((index / Math.max(keyframes - 1, 1)).toFixed(6)),
      source: (movingSource ? position : source).map((value) => Number(value.toFixed(6))),
      receiver: (movingSource ? receiver : position).map((value) => Number(value.toFixed(6))),
    }));
    return {
      mode: motion.mode,
      moving: movingSource ? "source" : "receiver",
      requested_distance_m: Number(requested.toFixed(4)),
      distance_m: Number(actual.toFixed(4)),
      keyframes,
      keyframe_spacing_m: Number((actual / Math.max(keyframes - 1, 1)).toFixed(4)),
      path_model: "portal_route_smoothstep",
      frames,
    };
  }

  const directionTarget = portalRoute[1] || target;
  const routeDx = directionTarget[0] - movingStart[0];
  const routeDy = directionTarget[1] - movingStart[1];
  const routeLength = Math.max(Math.hypot(routeDx, routeDy), 1e-9);
  let direction = [routeDx / routeLength, routeDy / routeLength];
  if (motion.mode === "recede") direction = [-direction[0], -direction[1]];
  if (motion.mode === "pass_by") direction = [-direction[1], direction[0]];
  if (motion.mode === "approach") requested = Math.min(requested, Math.max(0.05, separation - 0.35));

  const positionsFor = (distance) => Array.from({ length: keyframes }, (_, index) => {
    const phase = index / Math.max(keyframes - 1, 1);
    const eased = smootherstep(phase);
    const offset = motion.mode === "pass_by" ? (eased - 0.5) * distance : eased * distance;
    return [
      movingStart[0] + direction[0] * offset,
      movingStart[1] + direction[1] * offset,
      movingStart[2],
    ];
  });
  const isSafe = (distance) => positionsFor(distance).every((position) => {
    if (!pointIsSafelyInsideRoom(position, corners)) return false;
    const dynamicSource = movingSource ? position : source;
    const dynamicReceiver = movingSource ? receiver : position;
    return distance2D(dynamicSource, dynamicReceiver) >= 0.3;
  });
  let actual = requested;
  if (!isSafe(actual)) {
    let low = 0;
    let high = requested;
    for (let iteration = 0; iteration < 20; iteration += 1) {
      const middle = (low + high) * 0.5;
      if (isSafe(middle)) low = middle;
      else high = middle;
    }
    actual = low;
  }
  keyframes = motionKeyframeCount(actual, motion);
  const frames = positionsFor(actual).map((position, index) => ({
    phase: Number((index / Math.max(keyframes - 1, 1)).toFixed(6)),
    source: (movingSource ? position : source).map((value) => Number(value.toFixed(6))),
    receiver: (movingSource ? receiver : position).map((value) => Number(value.toFixed(6))),
  }));
  return {
    mode: motion.mode,
    moving: movingSource ? "source" : "receiver",
    requested_distance_m: Number(requested.toFixed(4)),
    distance_m: Number(actual.toFixed(4)),
    keyframes,
    keyframe_spacing_m: Number((actual / Math.max(keyframes - 1, 1)).toFixed(4)),
    path_model: "local_smoothstep",
    frames,
  };
}

function motionKeyframeCount(distance, motion = state.motion) {
  const spacing = clamp(Number(motion.keyframe_spacing_m || 0.25), 0.1, 1.0);
  return clamp(Math.ceil(Math.max(0, Number(distance)) / spacing) + 1, 3, 65);
}

function geometryApproachRoute(start, end, corners) {
  const inset = insetPolygon2D(corners, MIN_WALL_DISTANCE_M);
  return visibilityPath2D(start, end, inset).map((point) => [
    Number(point[0]),
    Number(point[1]),
    Number(start[2]),
  ]);
}

function randomGeometryRoute(anchor, corners, requestedDistance, seed) {
  const signature = JSON.stringify({
    anchor: anchor.map((value) => Number(value).toFixed(4)),
    corners: corners.map((point) => point.slice(0, 2).map((value) => Number(value).toFixed(4))),
    requestedDistance: Number(requestedDistance).toFixed(4),
    seed: Math.max(0, Math.round(Number(seed) || 0)),
  });
  if (randomMotionRouteCache.signature === signature && randomMotionRouteCache.value) {
    return randomMotionRouteCache.value;
  }
  const domain = insetPolygon2D(corners, MIN_WALL_DISTANCE_M);
  const bounds = getBounds(domain);
  const random = seededRandom(seed);
  const qualified = new Map();
  let bestRoute = null;
  let bestLength = 0;
  for (let attempt = 0; attempt < 32; attempt += 1) {
    const candidate = [
      bounds.x0 + random() * bounds.w,
      bounds.y0 + random() * bounds.h,
      Number(anchor[2]),
    ];
    if (!pointInPolygon2D(candidate, domain)) continue;
    const route = visibilityPath2D(anchor, candidate, domain).map((point) => [
      Number(point[0]),
      Number(point[1]),
      Number(anchor[2]),
    ]);
    const routeLength = polylineLength3D(route);
    if (routeLength > bestLength) {
      bestRoute = route;
      bestLength = routeLength;
    }
    if (routeLength + 1e-9 < requestedDistance) continue;
    const sampledStart = samplePolyline3D(route, requestedDistance, 2, false).at(-1);
    const key = `${sampledStart[0].toFixed(3)},${sampledStart[1].toFixed(3)}`;
    if (!qualified.has(key)) qualified.set(key, route);
  }
  if (qualified.size > 0) {
    const routes = [...qualified.values()];
    const value = {
      route: routes[Math.min(routes.length - 1, Math.floor(random() * routes.length))],
      actual: requestedDistance,
    };
    randomMotionRouteCache = { signature, value };
    return value;
  }
  const value = {
    route: bestRoute || [anchor, anchor],
    actual: Math.min(requestedDistance, bestLength),
  };
  randomMotionRouteCache = { signature, value };
  return value;
}

function seededRandom(seed) {
  let value = Math.max(0, Math.round(Number(seed) || 0)) >>> 0;
  return () => {
    value = (Math.imul(1664525, value) + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function insetPolygon2D(corners, margin) {
  if (!Array.isArray(corners) || corners.length < 3 || margin <= 0) return corners;
  const signedArea = corners.reduce((sum, point, index) => {
    const next = corners[(index + 1) % corners.length];
    return sum + Number(point[0]) * Number(next[1]) - Number(next[0]) * Number(point[1]);
  }, 0) * 0.5;
  const orientation = signedArea >= 0 ? 1 : -1;
  const edges = corners.map((point, index) => {
    const next = corners[(index + 1) % corners.length];
    const dx = Number(next[0]) - Number(point[0]);
    const dy = Number(next[1]) - Number(point[1]);
    const length = Math.max(Math.hypot(dx, dy), 1e-9);
    const direction = [dx / length, dy / length];
    const normal = orientation > 0
      ? [-direction[1], direction[0]]
      : [direction[1], -direction[0]];
    return {
      point: [Number(point[0]) + normal[0] * margin, Number(point[1]) + normal[1] * margin],
      direction,
    };
  });
  const inset = corners.map((_, index) => lineIntersection2D(
    edges[(index - 1 + edges.length) % edges.length],
    edges[index],
  ));
  const valid = inset.every((point) => (
    Array.isArray(point)
    && point.every(Number.isFinite)
    && pointInPolygon2D(point, corners)
    && distanceToRoomBoundary(point, corners) >= margin * 0.8
  ));
  return valid ? inset : corners;
}

function lineIntersection2D(first, second) {
  const cross = first.direction[0] * second.direction[1] - first.direction[1] * second.direction[0];
  if (Math.abs(cross) < 1e-9) {
    return [
      (first.point[0] + second.point[0]) * 0.5,
      (first.point[1] + second.point[1]) * 0.5,
    ];
  }
  const dx = second.point[0] - first.point[0];
  const dy = second.point[1] - first.point[1];
  const along = (dx * second.direction[1] - dy * second.direction[0]) / cross;
  return [
    first.point[0] + first.direction[0] * along,
    first.point[1] + first.direction[1] * along,
  ];
}

function acousticMotionRoute(movingSource) {
  const metadataRoute = metadataPortalMotionRoute();
  if (metadataRoute.length >= 2) {
    return movingSource
      ? metadataRoute.map((point) => [point[0], point[1], Number(state.source[2])])
      : metadataRoute.slice().reverse().map((point) => [point[0], point[1], Number(state.receiver[2])]);
  }
  const paths = simData.paths || [];
  const portal = paths.find((path) => path.kind === "portal_path" && Array.isArray(path.points) && path.points.length >= 2);
  const direct = paths.find((path) => path.kind === "direct" && Array.isArray(path.points) && path.points.length >= 2);
  const path = portal || direct;
  const points = path ? path.points.map((point) => point.slice(0, 3).map(Number)) : [];
  if (points.length < 2) return [];
  return movingSource
    ? points.map((point) => [point[0], point[1], Number(state.source[2])])
    : points.reverse().map((point) => [point[0], point[1], Number(state.receiver[2])]);
}

function metadataPortalMotionRoute() {
  const multiRoom = state.floorplan.roomMetadata?.multi_room || {};
  const roomIds = Array.isArray(multiRoom.route_room_ids) ? multiRoom.route_room_ids : [];
  const portalIds = Array.isArray(multiRoom.route_portal_ids) ? multiRoom.route_portal_ids : [];
  if (!portalIds.length || roomIds.length !== portalIds.length + 1) return [];
  const roomById = new Map((multiRoom.rooms || []).map((room) => [room.id, room]));
  const portalById = new Map((multiRoom.portals || []).map((portal) => [portal.id, portal]));
  const height = Number(state.source[2]);
  const route = [state.source.map(Number)];
  let current = state.source.slice(0, 2).map(Number);
  for (let index = 0; index < portalIds.length; index += 1) {
    const currentRoom = roomIds[index];
    const nextRoom = roomIds[index + 1];
    const portal = portalById.get(portalIds[index]);
    const currentSide = portal?.room_points?.[currentRoom] || portal?.center;
    const nextSide = portal?.room_points?.[nextRoom] || portal?.center;
    const corners = roomById.get(currentRoom)?.corners || [];
    if (!Array.isArray(currentSide) || !Array.isArray(nextSide)) return [];
    const segment = visibilityPath2D(current, currentSide, corners);
    segment.slice(1).forEach((point) => route.push([Number(point[0]), Number(point[1]), height]));
    route.push([Number(nextSide[0]), Number(nextSide[1]), height]);
    current = nextSide.slice(0, 2).map(Number);
  }
  const finalCorners = roomById.get(roomIds[roomIds.length - 1])?.corners || [];
  visibilityPath2D(current, state.receiver, finalCorners).slice(1).forEach((point) => {
    route.push([Number(point[0]), Number(point[1]), height]);
  });
  return route.filter((point, index) => index === 0 || Math.hypot(
    point[0] - route[index - 1][0],
    point[1] - route[index - 1][1],
  ) > 1e-6);
}

function visibilityPath2D(start, end, corners) {
  const first = start.slice(0, 2).map(Number);
  const last = end.slice(0, 2).map(Number);
  if (!Array.isArray(corners) || corners.length < 3 || segmentInsidePolygon2D(first, last, corners)) return [first, last];
  const nodes = [first, last, ...corners.map((point) => point.slice(0, 2).map(Number))];
  const distance = Array(nodes.length).fill(Number.POSITIVE_INFINITY);
  const previous = Array(nodes.length).fill(-1);
  const visited = Array(nodes.length).fill(false);
  distance[0] = 0;
  for (let iteration = 0; iteration < nodes.length; iteration += 1) {
    let current = -1;
    for (let index = 0; index < nodes.length; index += 1) {
      if (!visited[index] && (current < 0 || distance[index] < distance[current])) current = index;
    }
    if (current < 0 || !Number.isFinite(distance[current]) || current === 1) break;
    visited[current] = true;
    for (let neighbor = 0; neighbor < nodes.length; neighbor += 1) {
      if (neighbor === current || visited[neighbor] || !segmentInsidePolygon2D(nodes[current], nodes[neighbor], corners)) continue;
      const candidate = distance[current] + Math.hypot(
        nodes[neighbor][0] - nodes[current][0],
        nodes[neighbor][1] - nodes[current][1],
      );
      if (candidate < distance[neighbor]) {
        distance[neighbor] = candidate;
        previous[neighbor] = current;
      }
    }
  }
  if (!Number.isFinite(distance[1])) return [first, last];
  const indices = [1];
  while (indices[indices.length - 1] !== 0) indices.push(previous[indices[indices.length - 1]]);
  return indices.reverse().map((index) => nodes[index]);
}

function polylineLength3D(points) {
  let total = 0;
  for (let index = 0; index < points.length - 1; index += 1) {
    total += Math.hypot(
      points[index + 1][0] - points[index][0],
      points[index + 1][1] - points[index][1],
      points[index + 1][2] - points[index][2],
    );
  }
  return total;
}

function samplePolyline3D(points, distance, count, eased = true) {
  const segmentLengths = points.slice(0, -1).map((point, index) => Math.hypot(
    points[index + 1][0] - point[0],
    points[index + 1][1] - point[1],
    points[index + 1][2] - point[2],
  ));
  return Array.from({ length: count }, (_, index) => {
    const phase = index / Math.max(count - 1, 1);
    let travel = (eased ? smootherstep(phase) : phase) * distance;
    let segment = 0;
    while (segment < segmentLengths.length - 1 && travel > segmentLengths[segment]) {
      travel -= segmentLengths[segment];
      segment += 1;
    }
    const mix = clamp(travel / Math.max(segmentLengths[segment], 1e-9), 0, 1);
    return points[segment].map((value, axis) => Number(value) + (Number(points[segment + 1][axis]) - Number(value)) * mix);
  });
}

function snapMotionPositionsToRooms(positions) {
  const multiRoom = state.floorplan.roomMetadata?.multi_room || {};
  const rooms = multiRoom.rooms || [];
  const portalPoints = (multiRoom.portals || [])
    .filter((portal) => portal.open)
    .flatMap((portal) => Object.values(portal.room_points || {}))
    .filter((point) => Array.isArray(point) && point.length >= 2);
  if (!rooms.length || !portalPoints.length) return positions;
  return positions.map((position) => {
    if (rooms.some((room) => Array.isArray(room.corners) && pointInPolygon2D(position, room.corners))) return position;
    const nearest = portalPoints.reduce((best, point) => {
      const distance = Math.hypot(position[0] - Number(point[0]), position[1] - Number(point[1]));
      return !best || distance < best.distance ? { point, distance } : best;
    }, null)?.point;
    return nearest ? [Number(nearest[0]), Number(nearest[1]), Number(position[2])] : position;
  });
}

function smootherstep(value) {
  const bounded = clamp(value, 0, 1);
  return bounded ** 3 * (bounded * (bounded * 6 - 15) + 10);
}

function floorplanRoomCorners(roomId) {
  if (!multiRoomMode || !roomId) return null;
  const rooms = state.floorplan.roomMetadata?.multi_room?.rooms || [];
  const room = rooms.find((item) => item.id === roomId);
  return Array.isArray(room?.corners) && room.corners.length >= 3 ? room.corners : null;
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
    semantic: spec.semantic || type,
    position: draft.position,
    rotation: draft.rotation,
    size: draft.size,
    z: draft.z,
    absorption_class: draft.absorption_class,
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
    absorption_class: value("objectAbsorption") || "auto",
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
  setValue("objectAbsorption", "auto");
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
  return ["panel", "rug", "curtain", "tv_mirror", "fridge", "washing_machine", "acoustic_panel", "tile_surface", "sanitary_fixture", "structural_element", "person"].includes(type) ? 0 : 12;
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
    if (addButton) addButton.textContent = "Add object";
    if (addButton) addButton.disabled = false;
    if (editHint) editHint.textContent = "Choose a furniture card, add it, then place it in the scene.";
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
  setValue("objectAbsorption", object.absorption_class || "auto");
  const needsConfirm = object.id === pendingObjectId || object.id === dirtyObjectId;
  if (confirmButton) confirmButton.hidden = !needsConfirm;
  if (commandRow) commandRow.classList.toggle("hasConfirm", needsConfirm);
  if (addButton) addButton.textContent = needsConfirm ? "Add later" : "Add object";
  if (addButton) addButton.disabled = needsConfirm;
  if (editHint) {
    editHint.textContent = needsConfirm
      ? "Preview is live. Press Update simulation to commit acoustics."
      : "Selected object. Drag in WebGL to move; edit dimensions here.";
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
  object.semantic = spec.semantic || nextType;
  object.size = [...spec.size];
  object.z = spec.z;
  object.absorption_class = "auto";
  delete object.material;
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
  object.absorption_class = draft.absorption_class;
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
  clearObjectSelection();
  setTimeout(clearObjectSelection, 0);
  setTimeout(clearObjectSelection, 120);
  markSimulationPending(`${title} confirmed · run simulation to update RIR.`);
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
  markSimulationPending();
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

function value(id) { return document.getElementById(id)?.value ?? ""; }
function setValue(id, v) {
  const element = document.getElementById(id);
  if (element) element.value = v;
}
function number(id) { return Number(document.getElementById(id)?.value ?? 0); }
function controlNumber(id, fallback = 0) {
  const element = document.getElementById(id);
  if (!element || element.value === "") return fallback;
  const numeric = Number(element.value);
  return Number.isFinite(numeric) ? numeric : fallback;
}
function roundControl(value) { return Number(value || 0).toFixed(2); }
function presetTitle(id) {
  if (id === "floorplan") return customMode
    ? `Custom ${state.floorplan.roomType || "room"}`
    : `Floorplan ${state.floorplan.roomType || "room"} #${state.floorplan.index}`;
  return presets.find((preset) => preset.id === id)?.title || id;
}
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

function sceneDisplayBounds() {
  return getBounds(sceneDisplayPoints());
}

function sceneDisplayPoints() {
  const points = [];
  const append = (point) => {
    if (!Array.isArray(point) || point.length < 2) return;
    const x = Number(point[0]);
    const y = Number(point[1]);
    if (Number.isFinite(x) && Number.isFinite(y)) points.push([x, y]);
  };
  (simData.room?.corners || []).forEach(append);
  const metadata = simData.room?.metadata || {};
  (metadata.multi_room?.rooms || []).forEach((room) => (room.corners || []).forEach(append));
  (metadata.surface_segments || []).forEach((segment) => {
    append(segment.a);
    append(segment.b);
  });
  append(state.source);
  append(state.receiver);
  if (state.motion?.mode !== "static") {
    motionFramesForDisplay().forEach((frame) => {
      append(frame.source);
      append(frame.receiver);
    });
  }
  return points.length >= 2 ? points : [[0, 0], [1, 1]];
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
