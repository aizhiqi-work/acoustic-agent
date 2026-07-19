"""Acoustic Agent: compact indoor sound-field simulation primitives."""

__version__ = "0.1.0"

from .api import AcousticAgent, DynamicSimulationResult
from .batch import BatchResult, SimulationPair, simulate_batch
from .directivity import source_directivity
from .engine import SimulationResult, simulate_rir
from .geometry import make_room
from .materials import MaterialLibrary
from .mic import microphone_array
from .models import Material, Room, SimConfig
from .resplan_resource import ResPlanResource

__all__ = [
    "__version__",
    "AcousticAgent",
    "DynamicSimulationResult",
    "BatchResult",
    "Material",
    "MaterialLibrary",
    "Room",
    "ResPlanResource",
    "SimConfig",
    "SimulationPair",
    "SimulationResult",
    "make_room",
    "microphone_array",
    "simulate_batch",
    "simulate_rir",
    "source_directivity",
]
