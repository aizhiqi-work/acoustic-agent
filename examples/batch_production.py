from pathlib import Path

from acoustic_agent import AcousticAgent


OUTPUT = Path(__file__).resolve().parent / "output" / "floorplan_dataset.npz"


def main() -> None:
    jobs = [
        {
            "id": f"floorplan_{idx}_repeat_{repeat}",
            "scene": "floorplan",
            "idx": idx,
            "placement": "random",
            "seed": 1000 + idx * 10 + repeat,
            "quality": "preview",
            "duration_s": 1.0,
            "fs": 16000,
        }
        for idx in range(4)
        for repeat in range(2)
    ]

    dataset = AcousticAgent.run_many(jobs, workers=4)
    dataset.save_npz(OUTPUT)
    print("RIR jobs:", len(dataset.items), "output:", OUTPUT)


if __name__ == "__main__":
    main()
