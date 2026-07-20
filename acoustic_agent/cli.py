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

    verify = commands.add_parser("verify-resources", help="Validate bundled SOFA and SQLite files.")
    verify.add_argument(
        "--hashes",
        action="store_true",
        help="Also compute and compare full SHA-256 hashes.",
    )
    verify.add_argument("--json", action="store_true", help="Print the report as JSON.")

    info = commands.add_parser("info", help="Print version, quality presets, and resource inventory.")
    info.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
