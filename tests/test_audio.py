import io
import wave

import numpy as np
import pytest

from acoustic_agent import AcousticAgent, SimConfig, mix_audio, mix_audio_at_snr, render_audio, resample_audio
from acoustic_agent.web_server import _audio_catalog, _audio_source_bytes, _generated_noise_wav


def test_render_and_mix_audio_preserve_relative_source_gain():
    dry = np.asarray([1.0, 0.0], dtype=np.float32)
    rir = np.asarray([[1.0, 0.5], [0.25, -0.25]], dtype=np.float32)

    foreground = render_audio(dry, rir)
    background = render_audio(dry, rir, gain_db=-6.020599913279624)
    mixed = mix_audio([foreground, background])

    np.testing.assert_allclose(foreground, [[1.0, 0.5, 0.0], [0.25, -0.25, 0.0]], atol=1e-6)
    np.testing.assert_allclose(background, foreground * 0.5, atol=1e-6)
    np.testing.assert_allclose(mixed, foreground * 1.5, atol=1e-6)


def test_render_audio_downmixes_source_channels_and_keeps_receiver_channels():
    dry = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    rir = np.asarray([[1.0], [0.5]], dtype=np.float32)

    rendered = render_audio(dry, rir)

    assert rendered.shape == (2, 2)
    np.testing.assert_allclose(rendered, [[0.5, 0.5], [0.25, 0.25]], atol=1e-6)


def test_mix_audio_at_snr_sets_receiver_domain_rms_ratio():
    foreground = np.full((2, 1000), 0.2, dtype=np.float32)
    background = np.full((2, 1000), 0.5, dtype=np.float32)

    mixed = mix_audio_at_snr(foreground, background, snr_db=10.0)
    scaled_background = mixed - foreground
    foreground_rms = float(np.sqrt(np.mean(foreground.astype(np.float64) ** 2)))
    background_rms = float(np.sqrt(np.mean(scaled_background.astype(np.float64) ** 2)))

    assert 20.0 * np.log10(foreground_rms / background_rms) == pytest.approx(10.0, abs=1e-5)


def test_resample_audio_preserves_channel_first_shape():
    values = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, -1.0]], dtype=np.float32)
    output = resample_audio(values, 3, 6)

    assert output.shape == (2, 6)
    np.testing.assert_allclose(output[:, 0], values[:, 0])


def test_run_sources_returns_independent_named_rirs():
    config = SimConfig(
        fs=8000,
        duration_s=0.03,
        reflections_enabled=False,
        diffraction_enabled=False,
    )
    agent = AcousticAgent(
        room=(4.0, 3.0, 2.5),
        source_model="omni",
        receiver_model="mono",
        config=config,
    )

    result = agent.run_sources(
        {
            "voice": [0.7, 0.8, 1.2],
            "background": {"position": [2.8, 2.2, 1.2], "source_model": "cardioid"},
        },
        receiver=[2.0, 1.4, 1.2],
    )

    assert set(result.items) == {"voice", "background"}
    assert set(result.rirs) == {"voice", "background"}
    assert result["voice"].rir.shape == result["background"].rir.shape == (1, 240)
    assert result["background"].source_model["pattern"] == "cardioid"
    assert not np.array_equal(result["voice"].rir, result["background"].rir)


@pytest.mark.parametrize("source_id", ["white_noise", "pink_noise", "brown_noise"])
def test_generated_noise_is_deterministic_finite_pcm16(source_id):
    first = _generated_noise_wav(source_id, 8000, 0.5, 42)
    second = _generated_noise_wav(source_id, 8000, 0.5, 42)

    assert first == second
    with wave.open(io.BytesIO(first), "rb") as wav_file:
        assert wav_file.getframerate() == 8000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        values = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    assert values.size == 4000
    assert np.isfinite(values).all()
    assert 1000.0 < float(np.sqrt(np.mean(values.astype(np.float64) ** 2))) < 8000.0


def test_audio_catalog_and_source_endpoint_include_generated_noise():
    catalog = {item["id"]: item for item in _audio_catalog()}
    assert catalog["voice"]["title"] == "Main voice"
    assert catalog["voice"]["bundled"] is True
    assert catalog["background_speech"]["available"] is True
    assert catalog["piano_1"]["filename"] == "piano_1.mp3"
    assert catalog["piano_2"]["recommended_role"] == "background"
    assert catalog["pink_noise_bed"]["available"] is True
    assert catalog["pink_noise"]["available"] is True
    assert catalog["white_noise"]["kind"] == "generated_noise"

    data, content_type, filename = _audio_source_bytes("pink_noise", 8000, 1.0, 7)
    assert data[:4] == b"RIFF"
    assert content_type == "audio/wav"
    assert filename == "pink_noise.wav"

    bundled, bundled_type, bundled_name = _audio_source_bytes("pink_noise_bed", 8000, 1.0, 7)
    assert bundled[:4] == b"RIFF"
    assert bundled_type == "audio/wav"
    assert bundled_name == "pink_noise_bed.wav"

    legacy_piano, legacy_type, legacy_name = _audio_source_bytes("piano", 8000, 1.0, 7)
    assert legacy_piano
    assert legacy_type == "audio/mpeg"
    assert legacy_name == "piano_1.mp3"
