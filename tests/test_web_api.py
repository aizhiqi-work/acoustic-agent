import io
import struct

import numpy as np

from acoustic_agent.floorplan_web_server import _floorplan_viewer_html
from acoustic_agent.custom_floorplan_web import custom_viewer_html
from acoustic_agent.web_server import (
    WEB_ROOT,
    _get_stored_result,
    _simulate_payload,
    simulate_api_from_payload,
    simulate_dynamic_workbench_from_payload,
    simulate_workbench_from_payload,
)


def test_http_api_is_headless_while_workbench_keeps_visual_paths():
    payload = {
        "shape": "rectangle",
        "size": [3.0, 2.5, 2.4],
        "source": [0.7, 0.8, 1.2],
        "receiver": [2.2, 1.7, 1.2],
        "config": {
            "fs": 8000,
            "duration_s": 0.08,
            "rt_num_rays": 256,
            "rt_num_bounces": 2,
            "rt_duration_s": 0.08,
            "late_tail": False,
            "diffraction_enabled": False,
        },
    }

    api = _simulate_payload(payload, visualization=False).result
    workbench = _simulate_payload(payload, visualization=True).result

    np.testing.assert_array_equal(api.rir, workbench.rir)
    assert api.metadata["steam_audio"]["rt_visual"]["enabled"] is False
    assert workbench.metadata["steam_audio"]["rt_visual"]["enabled"] is True
    assert not any(path.kind == "rt_reflection" for path in api.paths)
    assert any(path.kind == "rt_reflection" for path in workbench.paths)


def test_geometry_and_floorplan_share_the_unified_workbench_shell():
    geometry_html = (WEB_ROOT / "viewer.html").read_text(encoding="utf-8")
    floorplan_html = _floorplan_viewer_html()
    custom_html = custom_viewer_html()

    for html in (geometry_html, floorplan_html, custom_html):
        assert 'href="/geometry"' in html
        assert 'href="/floorplan"' in html
        assert 'href="/custom"' in html
        assert 'id="materialsSection"' in html
        assert 'id="placementSection"' in html
        assert 'id="objectsSection"' in html
        assert 'id="solverSection"' in html
        assert 'id="motionMode"' in html
        assert 'id="motionMoving"' in html
        assert 'id="motionFrameSpacing"' in html
        assert 'id="intersectionBackend"' in html
        assert 'id="audioSection"' in html
        assert 'id="foregroundAudio"' in html
        assert 'id="backgroundEnabled"' in html
        assert 'id="backgroundAudio"' in html
        assert 'id="backgroundSnr"' in html
        assert 'id="backgroundGain"' not in html
        assert 'id="backgroundX"' in html
        assert 'id="foregroundAudioFile"' in html
        assert 'id="backgroundAudioFile"' in html
        assert 'id="resampleMotionPath"' in html
        assert 'id="viewToolbar"' in html
        assert 'id="stageDistance"' in html
        assert 'id="resultStatus"' in html
        assert 'class="resultDirectory"' in html
        assert html.count('data-result-link=') == 5
        assert 'id="motionKeyframes"' not in html
        assert 'id="motionPlay"' not in html
        assert '<option value="approach">Approach</option>' in html
        assert '<option value="random">Random</option>' in html
        assert html.count('id="setupScroll"') == 1
        assert html.count('id="status"') == 1
        assert html.count('id="sceneSection"') == 1

    assert 'data-scene-source="geometry"' in geometry_html
    assert 'data-scene-source="floorplan"' in floorplan_html
    assert 'data-scene-source="custom"' in custom_html
    assert "Floorplan scene" in floorplan_html
    assert "ResPlan scene" not in floorplan_html
    assert 'id="shape"' in geometry_html
    assert 'id="floorplanIdx"' in floorplan_html
    assert 'id="customDescription"' in custom_html
    assert 'id="customTextTab"' not in custom_html
    assert 'id="customImageTab"' not in custom_html
    assert 'name="customInputMode" value="image" checked' in custom_html
    assert 'name="customInputMode" value="text"' in custom_html
    assert 'id="customImageFile"' in custom_html
    assert 'id="customVlmPrompt"' in custom_html
    assert "Copy ChatGPT prompt" in custom_html
    assert 'id="customSpecJson"' in custom_html
    assert 'id="customGenerate"' not in custom_html
    assert 'id="customVariant"' not in custom_html
    assert 'id="customDownloadJson"' not in custom_html
    assert 'id="customImageOpacity"' not in custom_html
    assert "Overlay" not in custom_html
    assert 'id="floorplanRoom"' in custom_html
    assert 'id="furnitureCompactness"' in custom_html
    assert 'id="furnitureSeed"' in custom_html
    assert 'id="autoPlaceFurniture"' in custom_html
    app_js = (WEB_ROOT / "app.js").read_text(encoding="utf-8")
    assert "Broadband EDC RT60" in app_js
    assert "Early / late equivalent RT" not in app_js
    assert "Dominant path" in app_js
    assert "RIR max sample" in app_js
    assert "Source-room material band RT" not in app_js
    assert "Path-traced band RT" not in app_js
    assert "Coupled-space band RT" not in app_js
    assert "requestMotionSimulation" in app_js
    assert "motionSourceRoomCorners" not in app_js
    assert 'fetch("/api/v1/workbench"' in app_js
    assert "convolveDynamicChannels" in app_js
    assert "loadAudioCatalog" in app_js
    assert "loadProgramAudio" in app_js
    assert "mixRenderedTracks" in app_js
    assert "receiverSnrBackgroundGain" in app_js
    assert "backgroundSourcePoint3D" in app_js
    assert "agent.run_sources" in app_js
    assert "decodeFloat32WavFirstChannel" not in app_js
    assert "monitorRirChannels" in app_js
    assert "function sceneDisplayBounds()" in app_js
    assert 'fetch("/api/v1/furniture/auto-layout"' in app_js
    assert "const bounds = sceneDisplayBounds();" in app_js
    assert "camera.zoom = 1;" in app_js
    assert "controls.minZoom = 0.55;" in app_js
    assert "const RIR_DECAY_MIN_DB = -60;" in app_js
    assert 'const RIR_DECAY_DB_TICKS = ["0", "-20", "-40", "-60"];' in app_js
    assert 'keyframe_spacing_m: 0.25' in app_js
    assert 'intersection_backend: "auto"' in app_js
    assert '["intersection_backend", intersectionBackend]' in app_js
    assert 'keyframes=${Number(motion.keyframes' not in app_js
    assert "keyframe_spacing_m: Number(state.motion?.keyframe_spacing_m || 0.25)" in app_js
    assert '"agent = AcousticAgent.create("' in app_js
    assert "agent.run(motion=" in app_js
    assert 'path_model: "room_shortest_path"' in app_js
    assert 'path_model: "random_room_route"' in app_js
    assert 'randomOption.textContent = "Random"' in app_js
    assert "const metadataRoute = metadataPortalMotionRoute();" in app_js
    assert '["source", source]' in app_js
    assert '["receiver", mic]' in app_js
    assert '"    placement=placement,"' not in app_js


def test_workbench_background_source_returns_an_independent_rir():
    response = simulate_workbench_from_payload({
        "shape": "rectangle",
        "size": [4.0, 3.0, 2.5],
        "source": [0.7, 0.8, 1.2],
        "receiver": [2.0, 1.4, 1.2],
        "config": {
            "fs": 8000,
            "duration_s": 0.03,
            "reflections_enabled": False,
            "diffraction_enabled": False,
        },
        "auralization": {
            "foreground": {"audio_id": "voice", "gain_db": 0.0},
            "background": {
                "enabled": True,
                "audio_id": "pink_noise",
                "snr_db": 10.0,
                "source": [2.8, 2.2, 1.2],
                "source_model": {"type": "omni"},
            },
        },
    })

    auralization = response["auralization"]
    assert auralization["model"] == "independent_source_rirs_then_sum"
    assert auralization["foreground"]["result_id"] == response["result_id"]
    assert auralization["background"]["enabled"] is True
    assert auralization["background"]["audio_id"] == "pink_noise"
    assert auralization["background"]["snr_db"] == 10.0
    assert auralization["background"]["source"] == [2.8, 2.2, 1.2]
    assert auralization["background"]["result_id"] != response["result_id"]
    assert auralization["background"]["rir"]["wav_url"].endswith("/rir.wav")
    assert _get_stored_result(auralization["background"]["result_id"]) is not None


def test_simple_api_returns_exact_binary_artifacts_without_base64():
    response = simulate_api_from_payload({
        "shape": "rectangle",
        "size": [3.0, 2.5, 2.4],
        "source": [0.7, 0.8, 1.2],
        "receiver": [2.2, 1.7, 1.2],
        "source_model": {"type": "cardioid", "orientation_deg": 20.0},
        "config": {
            "fs": 8000,
            "duration_s": 0.04,
            "rt_num_rays": 1,
            "rt_num_bounces": 1,
            "rt_duration_s": 0.04,
            "diffraction_enabled": False,
        },
    })

    assert "f32_base64" not in response
    assert response["files"]["wav"].endswith("/rir.wav")
    assert response["files"]["npy"].endswith("/rir.npy")
    assert response["source_model"]["pattern"] == "cardioid"
    assert response["source_model"]["orientation_deg"] == 20.0

    stored = _get_stored_result(response["id"])
    assert stored is not None
    values = np.load(io.BytesIO(stored.npy), allow_pickle=False)
    assert values.dtype == np.float32
    assert list(values.shape) == response["shape"]

    assert stored.wav[:4] == b"RIFF"
    assert stored.wav[8:12] == b"WAVE"
    fmt_offset = stored.wav.index(b"fmt ") + 8
    audio_format, channels, fs, _, block_align, bits = struct.unpack_from("<HHIIHH", stored.wav, fmt_offset)
    assert (audio_format, channels, fs, block_align, bits) == (3, values.shape[0], 8000, values.shape[0] * 4, 32)
    data_offset = stored.wav.index(b"data")
    data_length = struct.unpack_from("<I", stored.wav, data_offset + 4)[0]
    interleaved = np.frombuffer(stored.wav, dtype="<f4", count=data_length // 4, offset=data_offset + 8)
    np.testing.assert_array_equal(interleaved.reshape(-1, channels).T, values)


def test_dynamic_workbench_returns_downloadable_rir_for_each_motion_frame():
    response = simulate_dynamic_workbench_from_payload({
        "shape": "rectangle",
        "size": [3.0, 2.5, 2.4],
        "source": [0.7, 0.8, 1.2],
        "receiver": [2.2, 1.7, 1.2],
        "config": {
            "fs": 8000,
            "duration_s": 0.02,
            "reflections_enabled": False,
            "diffraction_enabled": False,
        },
        "motion": {
            "mode": "approach",
            "moving": "source",
            "distance_m": 0.4,
            "frames": [
                {"phase": 0.0, "source": [0.7, 0.8, 1.2], "receiver": [2.2, 1.7, 1.2]},
                {"phase": 0.5, "source": [0.9, 0.9, 1.2], "receiver": [2.2, 1.7, 1.2]},
                {"phase": 1.0, "source": [1.1, 1.0, 1.2], "receiver": [2.2, 1.7, 1.2]},
            ],
        },
    })

    assert response["dynamic"]["keyframes"] == 3
    assert response["dynamic"]["renderer"] == "time_varying_rir_snapshot_interpolation"
    for frame in response["dynamic"]["frames"]:
        assert frame["rir"]["wav_url"].endswith("/rir.wav")
        assert _get_stored_result(frame["result_id"]) is not None


def test_geometry_dynamic_workbench_accepts_more_than_floorplan_frame_limit():
    frames = [
        {
            "phase": index / 17.0,
            "source": [0.7 + 0.02 * index, 0.8, 1.2],
            "receiver": [2.2, 1.7, 1.2],
        }
        for index in range(18)
    ]
    response = simulate_dynamic_workbench_from_payload({
        "shape": "rectangle",
        "size": [3.0, 2.5, 2.4],
        "source": frames[0]["source"],
        "receiver": frames[0]["receiver"],
        "config": {
            "fs": 8000,
            "duration_s": 0.01,
            "reflections_enabled": False,
            "diffraction_enabled": False,
        },
        "motion": {
            "mode": "approach",
            "moving": "source",
            "distance_m": 0.34,
            "frames": frames,
        },
    })

    assert response["dynamic"]["keyframes"] == 18
