# Bundled Demo Audio

These files provide reproducible auralization programs for the WebGL
workbench. They are source signals, not acoustic responses; each selected
signal is convolved with the RIR computed for its source position.

| File | UI label | Intended role |
| --- | --- | --- |
| `main_voice.wav` | Main voice | Default foreground speech |
| `background_speech.wav` | Background speech | Competing speech |
| `piano_1.mp3` | Piano 1 | Background music |
| `piano_2.mp3` | Piano 2 | Background music |
| `pink_noise_bed.wav` | Pink-noise bed | Stationary broadband noise bed |

The workbench also synthesizes deterministic white, pink, and brown noise at
runtime. A target SNR is applied after each source has been convolved with its
own RIR, so the setting describes the signals at the receiver.

See `DATA_LICENSE.md` before redistributing these recordings.
