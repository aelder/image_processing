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

OUTPUT_FORMAT_CHOICES = ("tiff", "png")
MODE_CHOICES = ("average", "flow")
DITHER_CHOICES = ("none", "floyd-steinberg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="img-timeline",
        description="Create color timeline images from image-frame folders or video files.",
    )
    parser.add_argument("--version", action="version", version="img-timeline 0.1.2")

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Build a timeline directly from source frames or a video file.",
    )
    build_parser.add_argument(
        "input_path",
        type=Path,
        help="Directory of source frames, or a supported video file (e.g. .mp4)",
    )
    build_parser.add_argument(
        "output_file",
        type=Path,
        help="Output timeline path (.tif/.tiff for TIFF or .png for PNG)",
    )
    build_parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=None,
        help="Optional directory to write intermediate 1px strips",
    )
    build_parser.add_argument(
        "--output-format",
        choices=OUTPUT_FORMAT_CHOICES,
        default=None,
        help="Output image format. Defaults to TIFF unless output file ends with .png.",
    )
    build_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars (requires tqdm)",
    )
    build_parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="average",
        help="Row extraction mode: average (default) or flow.",
    )
    build_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes for frame processing "
            "(flow mode auto-parallelizes when omitted)."
        ),
    )
    build_parser.add_argument(
        "--dither",
        choices=DITHER_CHOICES,
        default="none",
        help="Apply dithering to final output image.",
    )
    build_parser.add_argument(
        "--palette-color",
        action="append",
        default=None,
        help="Repeat exactly 16 times with #RRGGBB to provide a custom dither palette.",
    )

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert source frames to 1px strips.",
    )
    convert_parser.add_argument(
        "input_folder",
        type=Path,
        help="Directory containing source image files",
    )
    convert_parser.add_argument(
        "output_folder",
        type=Path,
        help="Directory where 1px strip files are written",
    )
    convert_parser.add_argument(
        "--output-format",
        choices=OUTPUT_FORMAT_CHOICES,
        default="tiff",
        help="Output image format for strips (default: tiff)",
    )
    convert_parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="average",
        help="Strip generation mode: average (default) or flow.",
    )
    convert_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes for strip generation "
            "(flow mode auto-parallelizes when omitted)."
        ),
    )

    stack_parser = subparsers.add_parser(
        "stack",
        help="Stack strips into a final timeline image.",
    )
    stack_parser.add_argument(
        "input_folder",
        type=Path,
        help="Directory containing image strips to stack",
    )
    stack_parser.add_argument(
        "output_file",
        type=Path,
        help="Output timeline path (.tif/.tiff for TIFF or .png for PNG)",
    )
    stack_parser.add_argument(
        "--output-format",
        choices=OUTPUT_FORMAT_CHOICES,
        default=None,
        help="Output image format. Defaults to TIFF unless output file ends with .png.",
    )
    stack_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars (requires tqdm)",
    )
    stack_parser.add_argument(
        "--dither",
        choices=DITHER_CHOICES,
        default="none",
        help="Apply dithering to final output image.",
    )
    stack_parser.add_argument(
        "--palette-color",
        action="append",
        default=None,
        help="Repeat exactly 16 times with #RRGGBB to provide a custom dither palette.",
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
            args.input_path,
            args.output_file,
            intermediate_dir=args.intermediate_dir,
            show_progress=args.progress,
            output_format=args.output_format,
            mode=args.mode,
            workers=args.workers,
            dither=args.dither,
            palette_colors=args.palette_color,
        )
        print(f"Processed {count} frame(s) into {args.output_file}")
        return 0

    if args.command == "convert":
        count = convert_to_strips(
            args.input_folder,
            args.output_folder,
            output_format=args.output_format,
            mode=args.mode,
            workers=args.workers,
        )
        print(f"Processed {count} image file(s) into {args.output_folder}")
        return 0

    if args.command == "stack":
        count = stack_tiff_images(
            args.input_folder,
            args.output_file,
            show_progress=args.progress,
            output_format=args.output_format,
            dither=args.dither,
            palette_colors=args.palette_color,
        )
        print(f"Stacked {count} image file(s) into {args.output_file}")
        return 0

    if args.command == "generate-demo":
        count = generate_rainbow_tiffs(args.output_dir, count=args.count, size=args.size)
        print(f"Generated {count} TIFF file(s) in {args.output_dir}")
        return 0

    parser.error("Unknown command")
    return 2
