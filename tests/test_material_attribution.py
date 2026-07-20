import json
import sqlite3
from importlib.resources import files


def test_material_source_manifest_matches_runtime_database() -> None:
    resource_dir = files("acoustic_agent").joinpath("resources", "acoustic_materials")
    sources = json.loads(resource_dir.joinpath("sources.json").read_text(encoding="utf-8"))

    with resource_dir.joinpath("acoustic_materials_v3.sqlite3").open("rb") as stream:
        database_path = stream.name

    connection = sqlite3.connect(database_path)
    try:
        actual = dict(connection.execute(
            "SELECT source_group, COUNT(*) FROM materials GROUP BY source_group"
        ))
    finally:
        connection.close()

    declared = {row["id"]: row["record_count"] for row in sources["sources"]}
    assert sources["license_expression"] == "NOASSERTION"
    assert sources["record_count"] == sum(declared.values())
    assert declared == actual


def test_every_material_source_has_release_status_and_url() -> None:
    source_path = files("acoustic_agent").joinpath(
        "resources", "acoustic_materials", "sources.json"
    )
    sources = json.loads(source_path.read_text(encoding="utf-8"))["sources"]

    for source in sources:
        assert source["url"].startswith("https://")
        assert source["license"]
        assert source["redistribution_status"] in {
            "website_permission",
            "permission_required",
            "redistributable_with_notice",
            "redistributable_with_attribution_and_change_notice",
        }
