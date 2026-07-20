from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acoustic_agent.floorplan_web_server import main


if __name__ == "__main__":
    main()
