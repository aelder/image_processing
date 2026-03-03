from __future__ import annotations

import sys
from pathlib import Path

# Local editable compatibility: allow imports without installation.
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from img_timeline.core import (  # noqa: E402
    DEFAULT_FILMIC_16_PALETTE,
    DITHER_MODES,
    PALETTE_COLOR_COUNT,
    TIMELINE_MODES,
    average_image_color,
    build_timeline_from_frames,
    convert_to_strips,
    generate_rainbow_tiffs,
    iter_image_files,
    iter_tiff_files,
    stack_tiff_images,
)

__all__ = [
    "iter_image_files",
    "iter_tiff_files",
    "DITHER_MODES",
    "TIMELINE_MODES",
    "PALETTE_COLOR_COUNT",
    "DEFAULT_FILMIC_16_PALETTE",
    "average_image_color",
    "convert_to_strips",
    "stack_tiff_images",
    "build_timeline_from_frames",
    "generate_rainbow_tiffs",
]
