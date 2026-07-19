from acoustic_agent.cli import main
from acoustic_agent.resource_manifest import load_resource_manifest, verify_packaged_resources


def test_required_binary_resources_are_packaged_and_readable() -> None:
    report = verify_packaged_resources(hashes=False)
    assert report["ok"], report["errors"]
    assert {check["id"] for check in report["checks"]} == {
        "cipic_124",
        "sadie_h12",
        "acoustic_materials_v3",
        "resplan_v1",
    }


def test_resource_manifest_has_integrity_metadata() -> None:
    manifest = load_resource_manifest()
    assert manifest["schema_version"] == 1
    for resource in manifest["resources"]:
        assert resource["size_bytes"] > 0
        assert len(resource["sha256"]) == 64
        int(resource["sha256"], 16)


def test_cli_info_and_resource_verification(capsys) -> None:
    assert main(["info"]) == 0
    assert "Acoustic Agent" in capsys.readouterr().out
    assert main(["verify-resources"]) == 0
    assert "Resource verification passed" in capsys.readouterr().out
