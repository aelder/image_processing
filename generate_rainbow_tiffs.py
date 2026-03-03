from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
from typing import Iterable

from PIL import Image


def generate_rainbow_tiffs(output_dir: Path, count: int = 500, size: int = 2) -> int:
    if count <= 0:
        raise ValueError("count must be greater than 0")
    if size <= 0:
        raise ValueError("size must be greater than 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    pad = max(3, len(str(count - 1)))

    for index in range(count):
        hue = index / count
        r_float, g_float, b_float = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        color = (
            int(round(r_float * 255)),
            int(round(g_float * 255)),
            int(round(b_float * 255)),
        )

        image = Image.new("RGB", (size, size), color)
        filename = f"{index:0{pad}d}.tif"
        image.save(output_dir / filename)

    return count


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
