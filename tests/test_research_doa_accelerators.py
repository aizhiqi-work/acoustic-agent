from __future__ import annotations

from types import SimpleNamespace

from acoustic_agent import SimConfig
from shapely.geometry import Polygon
from research.doa import distributed
from research.doa.distributed import AcousticMeasurementGenerator, candidate_nodes
from research.doa.experiment import _condition_config
from research.doa.static_scaling import _adaptive_sample_counts, _area_stratified_rows, _recommend
from research.doa.stratified import _append_area_metric_rows


def test_los_condition_config_propagates_cuda_settings() -> None:
    config = _condition_config(
        SimConfig(),
        "room",
        rt_accelerator="cuda",
        rt_precision="float32",
        rt_cuda_device=2,
    )

    assert config.rt_accelerator == "cuda"
    assert config.rt_precision == "float32"
    assert config.rt_cuda_device == 2
    assert config.collect_visual_paths is False


def test_distributed_cache_key_separates_accelerators(tmp_path) -> None:
    model = SimpleNamespace(index=20)
    target = SimpleNamespace(position=(1.0, 2.0, 1.4), room_id="room_a")
    node = SimpleNamespace(position=(4.0, 3.0, 1.4), room_id="room_b")
    cpu = AcousticMeasurementGenerator(tmp_path / "cpu")
    gpu = AcousticMeasurementGenerator(
        tmp_path / "gpu",
        rt_accelerator="cuda",
        rt_precision="float32",
        rt_cuda_device=0,
    )

    assert cpu._key("single", model, target, node, 1) != gpu._key("single", model, target, node, 1)


def test_distributed_simulation_propagates_cuda_settings(tmp_path, monkeypatch) -> None:
    captured = {}

    class FakeAgent:
        config = SimConfig()

        def run(self, *, config):
            captured["config"] = config
            return object()

    monkeypatch.setattr(distributed.AcousticAgent, "create", staticmethod(lambda **_options: FakeAgent()))
    generator = AcousticMeasurementGenerator(
        tmp_path,
        rt_accelerator="cuda",
        rt_precision="float32",
        rt_cuda_device=1,
    )
    model = SimpleNamespace(index=20)
    target = SimpleNamespace(position=(1.0, 2.0, 1.4), room_id="room_a")
    node = SimpleNamespace(position=(4.0, 3.0, 1.4), room_id="room_b")

    generator._simulate(model, target, node, {"type": "mono"})

    config = captured["config"]
    assert config.rt_accelerator == "cuda"
    assert config.rt_precision == "float32"
    assert config.rt_cuda_device == 1
    assert config.collect_visual_paths is False
    assert config.render_ambisonics is False


def test_stratified_quick_report_handles_missing_area_bins() -> None:
    lines = []
    _append_area_metric_rows(
        lines,
        {
            "small": {
                "floorplans": 2,
                "cases": 8,
                "median_error_m": 0.5,
                "p90_error_m": 1.2,
                "room_accuracy": 0.875,
            }
        },
        ("small", "medium", "large"),
    )

    assert lines[0] == "| small | 2 | 8 | 0.50 m | 1.20 m | 87.5% |"
    assert lines[1] == "| medium | 0 | 0 | n/a | n/a | n/a |"
    assert lines[2] == "| large | 0 | 0 | n/a | n/a | n/a |"


def test_candidate_nodes_support_multiple_positions_per_room() -> None:
    model = SimpleNamespace(
        rooms={"a": {"area_m2": 16.0}, "b": {"area_m2": 12.0}},
        polygons={
            "a": Polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0))),
            "b": Polygon(((4.0, 0.0), (7.0, 0.0), (7.0, 4.0), (4.0, 4.0))),
        },
    )

    nodes = candidate_nodes(model, positions_per_room=2)

    assert len(nodes) == 4
    assert len({node.id for node in nodes}) == 4
    assert {node.room_id for node in nodes} == {"a", "b"}


def test_static_scaling_adapts_only_sparse_room_counts() -> None:
    assert _adaptive_sample_counts(3, calibration_per_count=5, validation_per_count=10) == (1, 2)
    assert _adaptive_sample_counts(5, calibration_per_count=5, validation_per_count=10) == (1, 4)
    assert _adaptive_sample_counts(100, calibration_per_count=5, validation_per_count=10) == (5, 10)


def test_static_scaling_filters_five_area_strata_without_duplicates() -> None:
    rows = [{"index": index, "area_m2": float(index)} for index in range(10)]

    selected = _area_stratified_rows(rows, 5, seed=3)

    assert len({row["index"] for row in selected}) == 5
    assert [row["index"] // 2 for row in selected] == [0, 1, 2, 3, 4]


def test_static_scaling_recommendation_rejects_single_array_baseline() -> None:
    rows = []
    for sensor_kind, node_counts in (("single", (3,)), ("array_4ch", (1, 2))):
        for node_count in node_counts:
            rows.append(
                {
                    "floorplan_idx": 1,
                    "room_count": 2,
                    "sensor_kind": sensor_kind,
                    "node_count": node_count,
                    "physical_microphones": node_count if sensor_kind == "single" else node_count * 4,
                    "position_error_m": 0.2,
                    "room_correct": True,
                    "local_sensor": True,
                    "evidence_tier": "exploratory",
                }
            )

    recommendations = _recommend(rows, seed=7)
    array = next(row for row in recommendations if row["sensor_kind"] == "array_4ch")

    assert array["recommended_nodes"] == 2
    assert array["physical_microphones"] == 8
    assert array["meets_thresholds"] is True
