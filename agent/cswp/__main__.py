"""Production CSWP compiler CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.cswp.compiler import compile_and_write
from agent.cswp.constants import PRODUCTION_INDEX_DIR, ROOT
from agent.cswp.offline import install_offline_guard


def main(argv: list[str] | None = None) -> int:
    install_offline_guard()
    parser = argparse.ArgumentParser(
        description="Compile kb/seed_docs into production-owned CSWP-v1 index artifacts."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "kb/seed_docs",
        help="Frozen seed corpus root (must be kb/seed_docs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PRODUCTION_INDEX_DIR,
        help="Production index output directory",
    )
    args = parser.parse_args(argv)
    manifest, report, units = compile_and_write(output_dir=args.output, root=ROOT)
    print(
        json.dumps(
            {
                "compiler_version": manifest["compiler_version"],
                "unit_count": manifest["unit_count"],
                "representation_fingerprint": manifest["representation_fingerprint"],
                "baseline_corpus_fingerprint": manifest["baseline_corpus_fingerprint"],
                "frozen_B_fingerprint": manifest["frozen_B_fingerprint"],
                "output_dir": str(args.output),
                "max_content_tokens": report["token_distribution"]["max"],
                "children_above_254": report["children_above_254"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
