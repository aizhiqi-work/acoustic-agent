from __future__ import annotations

import numpy as np

from acoustic_agent.floorplan_resource import FloorplanResource
from research.doa.distributed import (
    DOAMeasurement,
    TOAMeasurement,
    _kalman_constant_velocity,
    _portal_trajectory,
    load_model,
    localization_grid,
    localize_doa,
    localize_tdoa,
    place_nodes,
    sample_target_points,
)


def test_topology_placement_is_deterministic_and_position_independent() -> None:
    model = load_model(0, FloorplanResource())
    first = place_nodes(model, 3, mode="array", risk_quantile=0.2)
    second = place_nodes(model, 3, mode="array", risk_quantile=0.2)
    assert first == second
    assert len({node.room_id for node in first}) == 3


def test_ideal_doa_and_tdoa_fusion_recover_floorplan_grid_location() -> None:
    model = load_model(0, FloorplanResource())
    target = sample_target_points(model, points_per_room=1, seed=123)[0]
    grid = localization_grid(model, spacing_m=0.35)

    arrays = place_nodes(model, 4, mode="array", risk_quantile=0.2)
    doa = []
    for node in arrays:
        bearing, _, _ = model.propagation(target.position, target.room_id, node.position, node.room_id)
        doa.append(DOAMeasurement(node.id, bearing, 1.0, 2.0))
    doa_position, doa_room, _ = localize_doa(model, arrays, doa, grid)
    assert doa_room == target.room_id
    assert np.linalg.norm(doa_position - np.asarray(target.position[:2])) < 0.8

    singles = place_nodes(model, min(8, len(model.rooms)), mode="single", risk_quantile=0.2)
    toa = []
    for node in singles:
        _, distance, _ = model.propagation(target.position, target.room_id, node.position, node.room_id)
        toa.append(TOAMeasurement(node.id, distance / 343.0, 1.0))
    toa_position, toa_room, _ = localize_tdoa(model, singles, toa, grid)
    assert toa_room == target.room_id
    assert np.linalg.norm(toa_position - np.asarray(target.position[:2])) < 0.8


def test_portal_trajectory_and_kalman_tracker_are_well_formed() -> None:
    model = load_model(0, FloorplanResource())
    trajectory, destination_room = _portal_trajectory(model)
    assert len(trajectory) == 6
    assert trajectory[0].room_id != destination_room
    assert trajectory[-1].room_id == destination_room
    observations = np.asarray([target.position[:2] for target in trajectory], dtype=float)
    positions, velocities = _kalman_constant_velocity(observations)
    assert positions.shape == observations.shape
    assert velocities.shape == observations.shape
    assert np.all(np.isfinite(positions))
