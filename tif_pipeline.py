from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from img_timeline.core import build_timeline_from_frames  # noqa: E402


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
        mode="average",
    )
    print(f"Processed {count} TIFF frame(s) into {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
