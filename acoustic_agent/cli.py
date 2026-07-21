from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import __version__
from .api import QUALITY_PRESETS
from .resource_manifest import (
    format_resource_report,
    load_resource_manifest,
    verify_packaged_resources,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acoustic-agent",
        description="Acoustic Agent indoor RIR simulation tools.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    web = commands.add_parser("web", help="Run the unified Geometry and Floorplan workbench.")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", default=8765, type=int)
    web.add_argument("--floorplan-resource", "--resplan-resource", dest="floorplan_resource", type=Path, default=None)
    web.add_argument("--floorplan-dataset", "--resplan-dataset", dest="floorplan_dataset", type=Path, default=None)
    web.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the small startup JIT warmup; the first simulation will compile kernels.",
    )

    verify = commands.add_parser("verify-resources", help="Validate all bundled runtime resources.")
    verify.add_argument(
        "--hashes",
        action="store_true",
        help="Also compute and compare full SHA-256 hashes.",
    )
    verify.add_argument("--json", action="store_true", help="Print the report as JSON.")

    info = commands.add_parser("info", help="Print version, quality presets, and resource inventory.")
    info.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    benchmark = commands.add_parser("benchmark", help="Run the acoustic-accuracy benchmark suite.")
    benchmark.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="quick",
        help="quick is suitable for CI; full uses the reference ray budget.",
    )
    benchmark.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results"),
        help="Directory for JSON, Markdown, and self-contained HTML reports.",
    )
    benchmark.add_argument(
        "--steam-audio-root",
        type=Path,
        default=None,
        help="Optional Steam Audio repository/SDK root for the native same-scene comparison.",
    )
    benchmark.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run one case id; repeat to select multiple cases.",
    )
    benchmark.add_argument(
        "--allow-failures",
        action="store_true",
        help="Always exit with status 0 after writing the report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "web":
        from .web_server import serve

        serve(
            host=args.host,
            port=args.port,
            floorplan_resource=args.floorplan_resource,
            floorplan_dataset=args.floorplan_dataset,
            warmup=not args.no_warmup,
        )
        return 0
    if args.command == "verify-resources":
        report = verify_packaged_resources(hashes=args.hashes)
        print(json.dumps(report, indent=2) if args.json else format_resource_report(report))
        return 0 if report["ok"] else 1
    if args.command == "info":
        payload = {
            "name": "acoustic-agent",
            "version": __version__,
            "quality_presets": QUALITY_PRESETS,
            "resources": load_resource_manifest()["resources"],
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"Acoustic Agent {__version__}")
            print("Quality presets:")
            for name, preset in QUALITY_PRESETS.items():
                print(
                    f"  {name:10s} {preset['rt_num_rays']:>6} rays, "
                    f"{preset['rt_num_bounces']:>3} bounces"
                )
            print(f"Bundled resources: {len(payload['resources'])}")
        return 0
    if args.command == "benchmark":
        from .benchmark import run_accuracy_benchmark

        report = run_accuracy_benchmark(
            profile=args.profile,
            output_dir=args.output,
            steam_audio_root=args.steam_audio_root,
            case_ids=args.cases,
        )
        paths = {
            "json": args.output.resolve() / "accuracy-benchmark.json",
            "markdown": args.output.resolve() / "accuracy-benchmark.md",
            "html": args.output.resolve() / "accuracy-benchmark.html",
        }
        summary = report.summary
        print(
            f"Accuracy benchmark: {summary['pass']} passed, {summary['fail']} failed, "
            f"{summary['error']} errors, {summary['skip']} skipped ({summary['duration_s']:.2f} s)"
        )
        for case in report.cases:
            print(f"  {case.status.upper():5s} {case.id:24s} {case.summary}")
        print(f"HTML: {paths['html']}")
        print(f"Markdown: {paths['markdown']}")
        print(f"JSON: {paths['json']}")
        return 0 if args.allow_failures or summary["required_passed"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
