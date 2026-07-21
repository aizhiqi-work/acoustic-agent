from pathlib import Path

import numpy as np

from acoustic_agent import AcousticAgent


OUTPUT = Path(__file__).resolve().parent / "output" / "floorplan_rir.npy"


def main() -> None:
    agent = AcousticAgent.create(
        scene="floorplan",
        idx=0,
        placement="same_room",
        seed=42,
        material_seed=2026,
        material_profile={
            "wall": "auto",
            "floor": "auto",
            "ceiling": "auto",
            "door": "auto",
            "window": "auto",
        },
        source_model={"type": "omni"},
        receiver_model={"type": "mono"},
        furnishing={"mode": "auto", "compactness": "balanced", "seed": 42},
        quality="preview",
        duration_s=1.0,
        fs=16000,
    )
    result = agent.run()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT, result.rir)
    print("placement:", agent.placement)
    print("RIR:", result.rir.shape, "RT60:", result.rt60["rt60_s"], "output:", OUTPUT)


if __name__ == "__main__":
    main()
