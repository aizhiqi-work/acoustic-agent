from pathlib import Path

import numpy as np

from acoustic_agent import AcousticAgent, FloorplanBuilder


OUTPUT = Path(__file__).resolve().parent / "output" / "custom_floorplan_rir.npy"


def main() -> None:
    spec = FloorplanBuilder.from_text(
        "12m x 9m，三室两厅一厨两卫，一个储物间",
        seed=42,
    )
    agent = AcousticAgent.from_floorplan_spec(
        spec,
        source_room="living_0",
        receiver_room="bedroom_2",
        seed=42,
        material_seed=2026,
        receiver_model={"type": "mono"},
        source_model={"type": "omni"},
        quality="preview",
        duration_s=1.0,
        fs=16000,
    )
    result = agent.run()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT, result.rir)
    print("rooms:", [room["id"] for room in agent.rooms])
    print("placement:", agent.placement)
    print("RIR:", result.rir.shape, "RT60:", result.rt60["rt60_s"], "output:", OUTPUT)


if __name__ == "__main__":
    main()
