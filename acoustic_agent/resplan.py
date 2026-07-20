"""Compatibility aliases for the pre-Floorplan module name."""

from . import floorplan as _floorplan

DEFAULT_RESPLAN_PATH = _floorplan.DEFAULT_FLOORPLAN_PATH
ResPlanDataset = _floorplan.FloorplanDataset
scene_from_record = _floorplan.scene_from_record
_metric_scale = _floorplan._metric_scale
_plan_profile = _floorplan._plan_profile


def __getattr__(name: str):
    return getattr(_floorplan, name)
