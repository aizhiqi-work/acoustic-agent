import io
import struct

import numpy as np

from acoustic_agent.web_server import _get_stored_result, simulate_api_from_payload


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
