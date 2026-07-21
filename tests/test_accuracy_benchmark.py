from __future__ import annotations

import json

import numpy as np

from acoustic_agent.benchmark import run_accuracy_benchmark, write_accuracy_report
from acoustic_agent.cli import build_parser
from acoustic_agent.rir import render_impulses


def test_direct_reference_cases_pass_and_write_all_report_formats(tmp_path) -> None:
    report = run_accuracy_benchmark(
        profile="quick",
        case_ids=["direct_arrival", "distance_attenuation", "fdn_isolation"],
    )
    paths = write_accuracy_report(report, tmp_path)

    assert report.summary["pass"] == 3
    assert report.summary["required_passed"] is True
    assert set(paths) == {"json", "markdown", "html"}
    assert all(path.is_file() for path in paths.values())
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert [case["id"] for case in payload["cases"]] == [
        "direct_arrival",
        "distance_attenuation",
        "fdn_isolation",
    ]
    assert "Acoustic Accuracy Benchmark" in paths["markdown"].read_text(encoding="utf-8")
    assert "<!doctype html>" in paths["html"].read_text(encoding="utf-8")


def test_external_reference_is_explicitly_skipped_without_sdk(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("STEAM_AUDIO_ROOT", raising=False)
    report = run_accuracy_benchmark(
        profile="quick",
        output_dir=tmp_path,
        steam_audio_root=tmp_path / "missing-sdk",
        case_ids=["steam_audio_native"],
    )

    assert report.cases[0].status == "skip"
    assert report.summary["skip"] == 1
    assert report.summary["required_passed"] is True


def test_benchmark_cli_contract() -> None:
    args = build_parser().parse_args([
        "benchmark",
        "--profile",
        "full",
        "--output",
        "evidence",
        "--case",
        "direct_arrival",
        "--allow-failures",
    ])

    assert args.command == "benchmark"
    assert args.profile == "full"
    assert args.cases == ["direct_arrival"]
    assert args.allow_failures is True


def test_fractional_delay_sweep_preserves_impulse_energy() -> None:
    fs = 16000
    fractions = np.linspace(0.0, 0.95, 20)
    energies = []
    for fraction in fractions:
        impulse = render_impulses(
            np.asarray([0.02 + fraction / fs]),
            np.asarray([0.37]),
            fs=fs,
            duration_s=0.05,
            fractional=True,
            sinc_half_width=8,
        )
        energies.append(float(np.sum(impulse.astype(np.float64) ** 2)))

    np.testing.assert_allclose(energies, 0.37 ** 2, rtol=2e-7, atol=2e-9)


def test_dynamic_benchmark_uses_physical_energy_continuity() -> None:
    report = run_accuracy_benchmark(profile="quick", case_ids=["dynamic_continuity"])
    case = report.cases[0]
    metrics = {metric.name: metric for metric in case.metrics}

    assert case.status == "pass"
    assert metrics["maximum direct-level step error"].measured < 0.01
    assert metrics["raw maximum-sample jump"].passed is None
