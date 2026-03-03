"""img_timeline package."""

from .core import (
    average_image_color,
    build_timeline_from_frames,
    convert_to_strips,
    generate_rainbow_tiffs,
    iter_tiff_files,
    stack_tiff_images,
)

__all__ = [
    "iter_tiff_files",
    "average_image_color",
    "convert_to_strips",
    "stack_tiff_images",
    "build_timeline_from_frames",
    "generate_rainbow_tiffs",
]
