import numpy as np

from acoustic_agent import AcousticAgent, MaterialLibrary, SimConfig
from acoustic_agent.materials import ABSORPTION_CLASSES
from acoustic_agent.models import FREQUENCY_BANDS
from acoustic_agent.steam_rt import _effective_object_absorption, _object_proxy_boxes


def test_v3_runtime_resource_is_complete_and_cached():
    first = MaterialLibrary.load()
    second = MaterialLibrary.load()
    stats = first.stats()

    assert first is second
    assert stats["source_schema_version"] == "3.0.0"
    assert stats["material_count"] == 3741
    assert stats["material_type_count"] == 16
    assert stats["semantic_count"] == 20


def test_semantic_sampling_is_deterministic_and_six_band():
    library = MaterialLibrary.load()

    for level in ABSORPTION_CLASSES:
        first = library.sample_semantic("wall", absorption_class=level, seed=73)
        second = library.sample_semantic("wall", absorption_class=level, seed=73)
        assert first is second
        assert first.metadata["resolved_absorption_class"] == level
        assert tuple(first.absorption) == FREQUENCY_BANDS
        assert all(0.0 <= value <= 1.0 for value in first.absorption.values())


def test_semantic_sampler_reports_physical_class_fallback():
    material = MaterialLibrary.load().sample_semantic(
        "window_glass",
        absorption_class="highly_absorptive",
        seed=9,
    )

    assert material.metadata["fallback_from"] == "highly_absorptive"
    assert material.metadata["resolved_absorption_class"] == "semi_reflective"
    assert material.metadata["material_type"] == "glass_reflective"


def test_surface_sampler_covers_resplan_boundary_semantics():
    materials = MaterialLibrary.load().sample_surface_set(
        {"wall": "absorptive", "floor": "semi_reflective", "window": "reflective"},
        seed=41,
    )

    assert set(materials) == {"wall", "floor", "ceiling", "door", "window"}
    assert materials["wall"].semantic == "wall"
    assert materials["window"].semantic == "window_glass"
    assert materials["wall"].metadata["resolved_absorption_class"] == "absorptive"


def test_equivalent_absorption_area_is_normalized_by_object_surface_area():
    material = next(
        MaterialLibrary.load().sample_geometry(
            {"semantic": "human_person", "absorption_class": "highly_absorptive"},
            seed=seed,
        )
        for seed in range(1000)
        if MaterialLibrary.load().sample_geometry(
            {"semantic": "human_person", "absorption_class": "highly_absorptive"},
            seed=seed,
        ).metadata["coefficient_kind"] == "equivalent_absorption_area_m2"
    )
    boxes = [{"size": np.asarray([0.4, 0.3, 1.7])}]
    effective = _effective_object_absorption(material, boxes)
    area = 2.0 * (0.4 * 0.3 + 0.4 * 1.7 + 0.3 * 1.7)

    expected = np.clip(np.asarray(list(material.absorption.values())) / area, 0.0, 0.99)
    assert np.allclose(effective, expected)
    assert np.all((effective >= 0.0) & (effective <= 0.99))


def test_new_semantic_furniture_uses_matching_reflection_proxies():
    common = {"position": [2.0, 1.5], "rotation": 30.0}
    tile = _object_proxy_boxes({
        **common,
        "type": "tile_surface",
        "size": [1.6, 1.2, 0.05],
        "z": 0.025,
    }, 2.8)
    sanitary = _object_proxy_boxes({
        **common,
        "type": "sanitary_fixture",
        "size": [1.55, 0.76, 0.62],
        "z": 0.31,
    }, 2.8)
    structural = _object_proxy_boxes({
        **common,
        "type": "structural_element",
        "size": [0.46, 0.46, 2.45],
        "z": 1.225,
    }, 2.8)

    assert [box["part"] for box in tile] == ["body"]
    assert [box["part"] for box in sanitary] == ["base", "back", "front", "left", "right"]
    assert [box["part"] for box in structural] == ["base", "shaft", "capital"]
    assert all(np.all(np.asarray(box["size"]) > 0.0) for box in tile + sanitary + structural)


def test_resplan_api_exposes_material_selection_and_semantic_furniture():
    agent = AcousticAgent.from_resplan(
        0,
        seed=42,
        material_seed=2026,
        material_profile={
            "wall": "absorptive",
            "floor": "semi_reflective",
            "ceiling": "reflective",
            "door": "reflective",
            "window": "highly_absorptive",
        },
        acoustic_geometry=[{
            "type": "sofa",
            "semantic": "sofa_couch",
            "absorption_class": "absorptive",
            "position": [2.5, 2.0],
            "size": [1.8, 0.8, 0.7],
        }],
        config=SimConfig(
            fs=8000,
            duration_s=0.02,
            reflections_enabled=False,
            diffraction_enabled=False,
        ),
    )

    result = agent.run()
    selected = agent.room.metadata["material_selection"]
    furniture = agent.room.metadata["objects"][0]["material_selection"]

    assert result.rir.shape == (1, 160)
    assert selected["wall"]["resolved_absorption_class"] == "absorptive"
    assert selected["window"]["fallback_from"] == "highly_absorptive"
    assert furniture["semantic"] == "sofa_couch"
    assert furniture["resolved_absorption_class"] == "absorptive"
    assert tuple(furniture["absorption"]) == FREQUENCY_BANDS
