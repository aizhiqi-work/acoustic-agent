from __future__ import annotations

from types import SimpleNamespace

from acoustic_agent import SimConfig
from research.doa import distributed
from research.doa.distributed import AcousticMeasurementGenerator
from research.doa.experiment import _condition_config
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
