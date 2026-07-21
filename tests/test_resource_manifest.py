from acoustic_agent.cli import build_parser, main
from acoustic_agent.resource_manifest import load_resource_manifest, verify_packaged_resources


def test_required_binary_resources_are_packaged_and_readable() -> None:
    report = verify_packaged_resources(hashes=False)
    assert report["ok"], report["errors"]
    assert {check["id"] for check in report["checks"]} == {
        "cipic_124",
        "sadie_h12",
        "acoustic_materials_v3",
        "floorplan_v1",
        "audio_main_voice",
        "audio_background_speech",
        "audio_piano_1",
        "audio_piano_2",
        "audio_pink_noise_bed",
    }


def test_resource_manifest_has_integrity_metadata() -> None:
    manifest = load_resource_manifest()
    assert manifest["schema_version"] == 1
    for resource in manifest["resources"]:
        assert resource["size_bytes"] > 0
        assert len(resource["sha256"]) == 64
        int(resource["sha256"], 16)
    floorplan = next(row for row in manifest["resources"] if row["id"] == "floorplan_v1")
    assert floorplan["license"].startswith("CC-BY-NC-SA-4.0")


def test_legacy_resplan_cli_flags_map_to_floorplan_options() -> None:
    args = build_parser().parse_args([
        "web",
        "--resplan-resource", "/tmp/scenes.sqlite3",
        "--resplan-dataset", "/tmp/scenes.pkl",
    ])
    assert str(args.floorplan_resource) == "/tmp/scenes.sqlite3"
    assert str(args.floorplan_dataset) == "/tmp/scenes.pkl"


def test_cli_info_and_resource_verification(capsys) -> None:
    assert main(["info"]) == 0
    assert "Acoustic Agent" in capsys.readouterr().out
    assert main(["verify-resources"]) == 0
    assert "Resource verification passed" in capsys.readouterr().out
