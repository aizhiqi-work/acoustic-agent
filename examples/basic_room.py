from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from acoustic_agent import SimConfig, make_room, microphone_array, simulate_batch, simulate_rir, SimulationPair
from acoustic_agent.web_export import export_scene_json


OUT = Path(__file__).resolve().parent / "output"


def main() -> None:
    room = make_room(
        "l_shape",
        size=(7.0, 5.0, 2.9),
        materials={"wall": "wall", "floor": "carpet", "ceiling": "ceiling"},
    )
    source = (1.2, 1.4, 1.5)
    receiver = (2.7, 3.6, 1.4)
    config = SimConfig(
        fs=16000,
        duration_s=1.2,
        seed=7,
        rt_num_rays=32768,
        rt_num_bounces=24,
        rt_duration_s=1.5,
    )

    mono = simulate_rir(room, source, receiver, config=config)
    hrtf = simulate_rir(room, source, receiver, config=config, receiver_model=microphone_array("hrtf"))
    circular = simulate_rir(room, source, receiver, config=config, receiver_model=microphone_array("circular", count=6, radius_m=0.10))

    pairs = [
        SimulationPair(source=(1.0, 1.0, 1.5), receiver=(3.5, 1.6, 1.4), id="pair_a"),
        SimulationPair(source=(1.8, 3.8, 1.5), receiver=(2.8, 3.4, 1.4), id="pair_b"),
    ]
    batch = simulate_batch(room, pairs, config=config, receiver_model=microphone_array("linear", count=4, spacing_m=0.06))

    OUT.mkdir(exist_ok=True)
    np.save(OUT / "mono_rir.npy", mono.rir)
    np.save(OUT / "hrtf_rir.npy", hrtf.rir)
    np.save(OUT / "circular_rir.npy", circular.rir)
    batch.save_npz(OUT / "batch_linear_array.npz")
    export_scene_json(room, OUT / "scene.json", sources=[source], receivers=[receiver], result=mono)

    print("mono", mono.rir.shape, "hrtf", hrtf.rir.shape, "circular", circular.rir.shape)
    print("batch items", len(batch.items), "scene", OUT / "scene.json")


if __name__ == "__main__":
    main()
