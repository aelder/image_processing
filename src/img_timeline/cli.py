from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from .core import (
    build_timeline_from_frames,
    convert_to_strips,
    generate_rainbow_tiffs,
    stack_tiff_images,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="img-timeline",
        description="Create color timeline images from TIFF frame sequences.",
    )
    parser.add_argument("--version", action="version", version="img-timeline 0.1.0")

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Build a timeline directly from source frames.",
    )
    build_parser.add_argument(
        "input_folder",
        type=Path,
        help="Directory containing source TIFF frames",
    )
    build_parser.add_argument("output_file", type=Path, help="Output TIFF timeline path")
    build_parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=None,
        help="Optional directory to write intermediate 1px TIFF strips",
    )
    build_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars (requires tqdm)",
    )

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert source frames to 1px strips.",
    )
    convert_parser.add_argument(
        "input_folder",
        type=Path,
        help="Directory containing source TIFF files",
    )
    convert_parser.add_argument(
        "output_folder",
        type=Path,
        help="Directory where 1px TIFF files are written",
    )

    stack_parser = subparsers.add_parser(
        "stack",
        help="Stack strips into a final timeline image.",
    )
    stack_parser.add_argument(
        "input_folder",
        type=Path,
        help="Directory containing TIFF files to stack",
    )
    stack_parser.add_argument("output_file", type=Path, help="Output TIFF file path")
    stack_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars (requires tqdm)",
    )

    demo_parser = subparsers.add_parser(
        "generate-demo",
        help="Generate rainbow TIFF frames for testing/demo.",
    )
    demo_parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where TIFF files are written",
    )
    demo_parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of TIFF files to generate",
    )
    demo_parser.add_argument("--size", type=int, default=2, help="Square pixel size for each image")

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        count = build_timeline_from_frames(
            args.input_folder,
            args.output_file,
            intermediate_dir=args.intermediate_dir,
            show_progress=args.progress,
        )
        print(f"Processed {count} TIFF frame(s) into {args.output_file}")
        return 0

    if args.command == "convert":
        count = convert_to_strips(args.input_folder, args.output_folder)
        print(f"Processed {count} TIFF file(s) into {args.output_folder}")
        return 0

    if args.command == "stack":
        count = stack_tiff_images(args.input_folder, args.output_file, show_progress=args.progress)
        print(f"Stacked {count} TIFF file(s) into {args.output_file}")
        return 0

    if args.command == "generate-demo":
        count = generate_rainbow_tiffs(args.output_dir, count=args.count, size=args.size)
        print(f"Generated {count} TIFF file(s) in {args.output_dir}")
        return 0

    parser.error("Unknown command")
    return 2
