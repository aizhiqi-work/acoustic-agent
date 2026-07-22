from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from acoustic_agent import SimConfig, make_room, simulate_rir
from acoustic_agent.steam_rt import RoomRayScene, _trace_energy_field_numba, trace_energy_field
from acoustic_agent.steam_rt_cuda import cuda_available


def _small_trace_config(**overrides) -> SimConfig:
    base = SimConfig(
        duration_s=0.3,
        rt_duration_s=0.3,
        rt_num_rays=256,
        rt_num_bounces=6,
        late_tail=False,
        collect_visual_paths=False,
        render_ambisonics=False,
    )
    return replace(base, **overrides)


def _trace_inputs():
    room = make_room("rectangle", size=(6.0, 4.0, 2.8))
    return RoomRayScene(room), np.asarray((1.0, 1.0, 1.4)), np.asarray((4.5, 2.7, 1.4))


def test_numba_precision_selects_compute_array_dtype():
    scene, source, listener = _trace_inputs()
    fp64 = _trace_energy_field_numba(
        scene,
        source,
        listener,
        _small_trace_config(rt_precision="float64"),
        render_ambisonics=False,
    )
    fp32 = _trace_energy_field_numba(
        scene,
        source,
        listener,
        _small_trace_config(rt_precision="fp32"),
        render_ambisonics=False,
    )

    assert fp64["echogram"].dtype == np.float64
    assert fp32["echogram"].dtype == np.float32
    assert fp64["precision"] == "float64"
    assert fp32["precision"] == "float32"
    np.testing.assert_allclose(fp32["echogram"], fp64["echogram"], rtol=2e-4, atol=1e-8)
    assert fp32["surface_hit_count"] == fp64["surface_hit_count"]


@pytest.mark.skipif(not cuda_available(0), reason="CUDA device is not available")
def test_cuda_fp32_matches_numba_fp32_energy_field():
    scene, source, listener = _trace_inputs()
    config = _small_trace_config(rt_precision="float32")
    numba_field = _trace_energy_field_numba(scene, source, listener, config, render_ambisonics=False)
    cuda_field = trace_energy_field(
        scene,
        source,
        listener,
        replace(config, rt_accelerator="cuda", rt_cuda_device=0),
        render_ambisonics=False,
    )

    assert cuda_field["accelerator"] == "cuda"
    assert cuda_field["precision"] == "float32"
    assert cuda_field["echogram"].dtype == np.float32
    np.testing.assert_allclose(cuda_field["echogram"], numba_field["echogram"], rtol=3e-3, atol=2e-8)
    assert cuda_field["surface_hit_count"] == numba_field["surface_hit_count"]


@pytest.mark.skipif(not cuda_available(0), reason="CUDA device is not available")
def test_cuda_bvh_matches_numba_fp32_energy_field():
    room = make_room("circle", size=(8.0, 6.0, 2.8), circle_segments=24)
    scene = RoomRayScene(room)
    source = np.asarray((1.0, 3.0, 1.4))
    listener = np.asarray((7.0, 3.0, 1.4))
    config = _small_trace_config(rt_precision="float32", intersection_backend="bvh")
    numba_field = _trace_energy_field_numba(scene, source, listener, config, render_ambisonics=False)
    cuda_field = trace_energy_field(
        scene,
        source,
        listener,
        replace(config, rt_accelerator="cuda", rt_cuda_device=0),
        render_ambisonics=False,
    )

    assert cuda_field["intersection_backend"] == "bvh"
    np.testing.assert_allclose(cuda_field["echogram"], numba_field["echogram"], rtol=4e-3, atol=2e-8)
    assert cuda_field["surface_hit_count"] == numba_field["surface_hit_count"]


@pytest.mark.skipif(not cuda_available(0), reason="CUDA device is not available")
def test_cuda_end_to_end_supports_visual_paths_and_ambisonics():
    room = make_room("rectangle", size=(6.0, 4.0, 2.8))
    config = _small_trace_config(
        rt_accelerator="cuda",
        rt_precision="float32",
        collect_visual_paths=True,
        render_ambisonics=True,
    )
    result = simulate_rir(room, (1.0, 1.0, 1.4), (4.5, 2.7, 1.4), config=config)

    reflections = result.metadata["steam_audio"]["reflections"]
    assert result.rir.shape == (1, int(config.fs * config.duration_s))
    assert result.ambisonic_rir is not None
    assert result.ambisonic_rir.shape == (4, int(config.fs * config.duration_s))
    assert reflections["accelerator"] == "cuda"
    assert reflections["precision"] == "float32"
    assert reflections["cuda"]["name"]


def test_cuda_rejects_fp64_before_launch(monkeypatch):
    scene, source, listener = _trace_inputs()
    monkeypatch.setattr("acoustic_agent.steam_rt_cuda.cuda_available", lambda _device_id=0: True)
    with pytest.raises(ValueError, match="float32"):
        trace_energy_field(
            scene,
            source,
            listener,
            _small_trace_config(rt_accelerator="cuda", rt_precision="float64"),
            render_ambisonics=False,
        )


def test_auto_fp64_falls_back_to_numba(monkeypatch):
    scene, source, listener = _trace_inputs()
    monkeypatch.setattr("acoustic_agent.steam_rt_cuda.cuda_available", lambda _device_id=0: True)
    field = trace_energy_field(
        scene,
        source,
        listener,
        _small_trace_config(rt_accelerator="auto", rt_precision="float64"),
        render_ambisonics=False,
    )

    assert field["accelerator"] == "numba"
    assert field["precision"] == "float64"


@pytest.mark.parametrize("field,value", [("rt_accelerator", "metal"), ("rt_precision", "float16")])
def test_invalid_accelerator_configuration_is_rejected(field, value):
    scene, source, listener = _trace_inputs()
    with pytest.raises(ValueError):
        trace_energy_field(
            scene,
            source,
            listener,
            replace(_small_trace_config(), **{field: value}),
            render_ambisonics=False,
        )
