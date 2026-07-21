from pathlib import Path

import numpy as np

from acoustic_agent import AcousticAgent


OUT = Path(__file__).resolve().parent / "output"


def main() -> None:
    agent = AcousticAgent.create(
        scene="geometry",
        room={
            "shape": "l_shape",
            "size": [7.0, 5.0, 2.9],
            "material_profile": {"wall": "auto", "floor": "auto", "ceiling": "auto"},
        },
        source=[1.2, 1.4, 1.5],
        receiver=[2.7, 3.6, 1.4],
        receiver_model="mono",
        quality="simulation",
        duration_s=1.2,
        fs=16000,
        seed=7,
    )

    mono = agent.run()
    hrtf = agent.run(receiver_model="hrtf")
    batch = agent.run_batch([
        ([1.0, 1.0, 1.5], [3.5, 1.6, 1.4]),
        {"id": "pair_b", "source": [1.8, 3.8, 1.5], "receiver": [2.8, 3.4, 1.4]},
    ])

    OUT.mkdir(exist_ok=True)
    np.save(OUT / "mono_rir.npy", mono.rir)
    np.save(OUT / "hrtf_rir.npy", hrtf.rir)
    batch.save_npz(OUT / "batch_rirs.npz")
    print("mono", mono.rir.shape, "hrtf", hrtf.rir.shape, "batch", len(batch.items))


if __name__ == "__main__":
    main()
