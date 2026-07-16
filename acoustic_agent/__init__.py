"""Acoustic Agent: compact indoor sound-field simulation primitives."""

from .api import AcousticAgent
from .batch import BatchResult, SimulationPair, simulate_batch
from .directivity import source_directivity
from .engine import SimulationResult, simulate_rir
from .geometry import make_room
from .materials import MaterialLibrary
from .mic import microphone_array
from .models import Material, Room, SimConfig

__all__ = [
    "AcousticAgent",
    "BatchResult",
    "Material",
    "MaterialLibrary",
    "Room",
    "SimConfig",
    "SimulationPair",
    "SimulationResult",
    "make_room",
    "microphone_array",
    "simulate_batch",
    "simulate_rir",
    "source_directivity",
]
