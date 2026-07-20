"""Compatibility aliases for the pre-Floorplan resource module name."""

from .floorplan_resource import (
    DEFAULT_FLOORPLAN_RESOURCE,
    RESOURCE_SCHEMA_VERSION,
    FloorplanResource,
)

DEFAULT_RESPLAN_RESOURCE = DEFAULT_FLOORPLAN_RESOURCE
ResPlanResource = FloorplanResource

__all__ = [
    "DEFAULT_RESPLAN_RESOURCE",
    "RESOURCE_SCHEMA_VERSION",
    "ResPlanResource",
]
