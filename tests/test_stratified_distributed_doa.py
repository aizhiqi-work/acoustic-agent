from __future__ import annotations

from acoustic_agent.floorplan_resource import FloorplanResource
from research.doa.stratified import build_stratified_split, population_summary, scan_floorplan_population


def test_population_scan_matches_compiled_resource() -> None:
    resource = FloorplanResource()
    profiles = scan_floorplan_population(resource)
    assert len(profiles) == len(resource) == 15376
    summary = {row["room_count"]: row for row in population_summary(profiles)}
    assert summary[4]["records"] == 108
    assert summary[14]["records"] == 103
    assert summary[14]["eligible_records"] >= 15


def test_room_count_area_split_is_balanced_disjoint_and_deterministic() -> None:
    profiles = scan_floorplan_population(FloorplanResource())
    first = build_stratified_split(
        profiles,
        room_counts=(4, 8, 14),
        calibration_per_count=5,
        validation_per_count=10,
    )
    second = build_stratified_split(
        profiles,
        room_counts=(4, 8, 14),
        calibration_per_count=5,
        validation_per_count=10,
    )
    assert first == second
    calibration = {int(row["index"]) for row in first if row["split"] == "calibration"}
    validation = {int(row["index"]) for row in first if row["split"] == "validation"}
    assert calibration.isdisjoint(validation)
    for room_count in (4, 8, 14):
        rows = [row for row in first if int(row["room_count"]) == room_count]
        assert sum(row["split"] == "calibration" for row in rows) == 5
        assert sum(row["split"] == "validation" for row in rows) == 10
        assert all(row["connected"] and row["geometry_valid"] for row in rows)
        assert {row["relative_area_bin"] for row in rows if row["split"] == "validation"} == {
            "small",
            "medium",
            "large",
        }
