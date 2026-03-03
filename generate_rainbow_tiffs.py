from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from img_timeline.core import generate_rainbow_tiffs  # noqa: E402


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TIFF files that cycle evenly through rainbow colors."
    )
    parser.add_argument("output_dir", type=Path, help="Directory where TIFF files are written")
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of TIFF files to generate (default: 500)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2,
        help="Square pixel size for each TIFF image (default: 2)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    created = generate_rainbow_tiffs(args.output_dir, count=args.count, size=args.size)
    print(f"Generated {created} TIFF file(s) in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
