from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from pipeline_core import build_timeline_from_frames


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a TIFF color timeline directly from source TIFF frames in one pass, "
            "with optional intermediate 1px strips."
        )
    )
    parser.add_argument("input_folder", type=Path, help="Directory containing source TIFF frames")
    parser.add_argument("output_file", type=Path, help="Output TIFF timeline path")
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=None,
        help="Optional directory to write intermediate 1px TIFF strips",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars (requires tqdm)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    count = build_timeline_from_frames(
        args.input_folder,
        args.output_file,
        intermediate_dir=args.intermediate_dir,
        show_progress=args.progress,
    )
    print(f"Processed {count} TIFF frame(s) into {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
