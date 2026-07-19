from pathlib import Path

import numpy as np

from acoustic_agent import AcousticAgent


OUTPUT = Path(__file__).resolve().parent / "output" / "dynamic_rirs.npz"


def main() -> None:
    source = [1.2, 1.1, 1.5]
    receiver = [4.7, 2.8, 1.4]
    agent = AcousticAgent(
        room={
            "shape": "rectangle",
            "size": [6.0, 4.0, 2.8],
            "material_profile": {"wall": "auto", "floor": "auto", "ceiling": "auto"},
            "material_seed": 42,
        },
        quality="preview",
        duration_s=1.0,
        fs=16000,
    )
    motion = agent.sample_motion(
        source=source,
        receiver=receiver,
        mode="approach",
        moving="receiver",
        distance_m=1.0,
        keyframe_spacing_m=0.25,
        seed=42,
    )
    result = agent.run_dynamic(motion)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        **{f"rir_{index:03d}": rir for index, rir in enumerate(result.rirs)},
    )
    print("frames:", len(result.frames), "output:", OUTPUT)


if __name__ == "__main__":
    main()
