from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from pipeline_core import convert_to_strips


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert TIFF frames into 1-pixel-high TIFF strips, each filled with the frame's "
            "average color."
        )
    )
    parser.add_argument("input_folder", type=Path, help="Directory containing source TIFF files")
    parser.add_argument("output_folder", type=Path, help="Directory where 1px TIFF files are written")
    return parser.parse_args(argv)


def process_images(input_folder: Path, output_folder: Path) -> int:
    return convert_to_strips(input_folder, output_folder)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    count = process_images(args.input_folder, args.output_folder)
    print(f"Processed {count} TIFF file(s) into {args.output_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
