from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from pipeline_core import stack_tiff_images


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vertically stack TIFF files in lexical filename order."
    )
    parser.add_argument("input_folder", type=Path, help="Directory containing TIFF files to stack")
    parser.add_argument("output_file", type=Path, help="Output TIFF file path")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars (requires tqdm)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    count = stack_tiff_images(args.input_folder, args.output_file, show_progress=args.progress)
    print(f"Stacked {count} TIFF file(s) into {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
