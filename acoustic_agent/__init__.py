"""Acoustic Agent: compact indoor sound-field simulation primitives."""

__version__ = "0.1.0"

from .api import AcousticAgent, DynamicSimulationResult
from .batch import BatchResult, SimulationPair, simulate_batch
from .custom_floorplan import FloorplanBuilder
from .directivity import source_directivity
from .engine import SimulationResult, simulate_rir
from .geometry import make_room
from .materials import MaterialLibrary
from .mic import microphone_array
from .models import Material, Room, SimConfig
from .floorplan_resource import FloorplanResource
from .furnishing import FURNITURE_CATALOG, generate_floorplan_furniture

# Backward compatibility for the v0.1 ResPlan-named API.
ResPlanResource = FloorplanResource

__all__ = [
    "__version__",
    "AcousticAgent",
    "DynamicSimulationResult",
    "BatchResult",
    "Material",
    "MaterialLibrary",
    "Room",
    "FloorplanResource",
    "FloorplanBuilder",
    "FURNITURE_CATALOG",
    "ResPlanResource",
    "SimConfig",
    "SimulationPair",
    "SimulationResult",
    "make_room",
    "microphone_array",
    "simulate_batch",
    "simulate_rir",
    "generate_floorplan_furniture",
    "source_directivity",
]
