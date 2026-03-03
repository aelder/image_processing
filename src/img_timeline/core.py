from __future__ import annotations

import colorsys
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageStat

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def iter_tiff_files(input_folder: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in input_folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ],
        key=lambda p: p.name,
    )


def average_image_color(image: Image.Image) -> tuple[int, int, int]:
    rgb_image = image.convert("RGB")
    mean_channels = ImageStat.Stat(rgb_image).mean[:3]
    return tuple(int(round(channel)) for channel in mean_channels)


def _validate_input_folder(input_folder: Path) -> None:
    if not input_folder.exists() or not input_folder.is_dir():
        raise FileNotFoundError(
            f"Input folder does not exist or is not a directory: {input_folder}"
        )


def _progress(iterable: Iterable[Path], enabled: bool, desc: str):
    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc)
    return iterable


def convert_to_strips(input_folder: Path, output_folder: Path) -> int:
    _validate_input_folder(input_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    tiff_files = iter_tiff_files(input_folder)
    if not tiff_files:
        raise ValueError(f"No TIFF files found in input folder: {input_folder}")

    for input_path in tiff_files:
        with Image.open(input_path) as image:
            avg_color = average_image_color(image)
            strip = Image.new("RGB", (image.width, 1), avg_color)
            strip.save(output_folder / input_path.name)

    return len(tiff_files)


def stack_tiff_images(input_folder: Path, output_file: Path, show_progress: bool = False) -> int:
    _validate_input_folder(input_folder)

    tiff_files = iter_tiff_files(input_folder)
    if not tiff_files:
        raise ValueError(f"No TIFF files found in input folder: {input_folder}")

    total_height = 0
    max_width = 0

    for path in _progress(tiff_files, show_progress, desc="Scanning images"):
        with Image.open(path) as image:
            width, height = image.size
            total_height += height
            if width > max_width:
                max_width = width

    output_file.parent.mkdir(parents=True, exist_ok=True)
    stacked_image = Image.new("RGB", (max_width, total_height))

    current_height = 0
    for path in _progress(tiff_files, show_progress, desc="Stacking images"):
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            stacked_image.paste(rgb_image, (0, current_height))
            current_height += rgb_image.size[1]

    stacked_image.save(output_file)
    return len(tiff_files)


def build_timeline_from_frames(
    input_folder: Path,
    output_file: Path,
    intermediate_dir: Path | None = None,
    show_progress: bool = False,
) -> int:
    _validate_input_folder(input_folder)

    tiff_files = iter_tiff_files(input_folder)
    if not tiff_files:
        raise ValueError(f"No TIFF files found in input folder: {input_folder}")

    if intermediate_dir is not None:
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[int, tuple[int, int, int], str]] = []
    max_width = 0

    for path in _progress(tiff_files, show_progress, desc="Processing frames"):
        with Image.open(path) as image:
            avg_color = average_image_color(image)
            width = image.size[0]
            if width > max_width:
                max_width = width

        rows.append((width, avg_color, path.name))

        if intermediate_dir is not None:
            strip = Image.new("RGB", (width, 1), avg_color)
            strip.save(intermediate_dir / path.name)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    timeline = Image.new("RGB", (max_width, len(rows)))

    for row_index, (_, avg_color, _) in enumerate(rows):
        row = Image.new("RGB", (max_width, 1), avg_color)
        timeline.paste(row, (0, row_index))

    timeline.save(output_file)
    return len(rows)


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
