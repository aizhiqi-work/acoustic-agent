from pathlib import Path
import wave

import numpy as np

from acoustic_agent import AcousticAgent, mix_audio, render_audio


OUT = Path(__file__).resolve().parent / "output" / "multi_source_mix.wav"


def main() -> None:
    fs = 16000
    agent = AcousticAgent.create(
        scene="geometry",
        room={"shape": "rectangle", "size": [6.0, 4.0, 2.8]},
        source=[1.2, 1.1, 1.5],
        receiver=[4.7, 2.8, 1.4],
        quality="simulation",
        fs=fs,
        duration_s=2.0,
    )
    sources = agent.run_sources({
        "foreground": {"position": [1.2, 1.1, 1.5], "source_model": "cardioid"},
        "background": {"position": [4.8, 1.0, 1.2], "source_model": "omni"},
    })

    # Replace these arrays with decoded speech, music, or noise at `fs`.
    time = np.arange(fs * 4, dtype=np.float32) / fs
    foreground = 0.18 * np.sin(2.0 * np.pi * (220.0 + 40.0 * time) * time)
    rng = np.random.default_rng(42)
    background = rng.normal(0.0, 0.08, time.size).astype(np.float32)

    foreground_wet = render_audio(foreground, sources["foreground"].rir)
    background_wet = render_audio(background, sources["background"].rir, gain_db=-18.0)
    room_mix = mix_audio([foreground_wet, background_wet], normalize=True)
    write_pcm16_wav(OUT, room_mix, fs)
    print(OUT)


def write_pcm16_wav(path: Path, channels: np.ndarray, fs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(channels, dtype=np.float32)
    pcm = np.rint(np.clip(values.T, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(int(values.shape[0]))
        output.setsampwidth(2)
        output.setframerate(int(fs))
        output.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
