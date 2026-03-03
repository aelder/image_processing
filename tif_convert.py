from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from img_timeline.core import convert_to_strips  # noqa: E402


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert TIFF frames into 1-pixel-high TIFF strips, each filled with the frame's "
            "average color."
        )
    )
    parser.add_argument("input_folder", type=Path, help="Directory containing source TIFF files")
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Directory where 1px TIFF files are written",
    )
    return parser.parse_args(argv)


def process_images(input_folder: Path, output_folder: Path) -> int:
    return convert_to_strips(input_folder, output_folder, mode="average")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    count = process_images(args.input_folder, args.output_folder)
    print(f"Processed {count} TIFF file(s) into {args.output_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
