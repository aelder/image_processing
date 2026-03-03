from __future__ import annotations

import colorsys
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageStat

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

SUPPORTED_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
SUPPORTED_VIDEO_SUFFIXES = {
    ".mp4",
    ".m4v",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".flv",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".m2ts",
    ".ts",
    ".3gp",
}
OUTPUT_FORMATS = {"tiff", "png"}


def iter_image_files(input_folder: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in input_folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ],
        key=lambda p: p.name,
    )


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


def _normalize_output_format(output_format: str | None, output_file: Path | None = None) -> str:
    if output_format is None:
        if output_file is not None:
            suffix = output_file.suffix.lower()
            if suffix == ".png":
                return "png"
            if suffix in {"", ".tif", ".tiff"}:
                return "tiff"
            raise ValueError(
                "Output file extension must be .tif/.tiff or .png unless --output-format is provided."
            )
        return "tiff"

    normalized = output_format.lower()
    if normalized not in OUTPUT_FORMATS:
        allowed = ", ".join(sorted(OUTPUT_FORMATS))
        raise ValueError(f"Unsupported output format: {output_format}. Choose one of: {allowed}")

    if output_file is not None:
        suffix = output_file.suffix.lower()
        if suffix:
            if normalized == "png" and suffix != ".png":
                raise ValueError(
                    f"Output file extension '{suffix}' does not match output format 'png'."
                )
            if normalized == "tiff" and suffix not in {".tif", ".tiff"}:
                raise ValueError(
                    f"Output file extension '{suffix}' does not match output format 'tiff'."
                )

    return normalized


def _intermediate_path(output_folder: Path, input_path: Path, output_format: str) -> Path:
    if output_format == "tiff":
        return output_folder / input_path.name
    return output_folder / f"{input_path.stem}.png"


def _is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES


def _extract_video_frames(video_file: Path, frame_dir: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required to process video files. Install ffmpeg or provide a directory of image frames."
        )

    output_pattern = frame_dir / "%09d.png"
    result = subprocess.run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_file),
            "-vsync",
            "0",
            str(output_pattern),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error_details = result.stderr.strip() or "Unknown ffmpeg error"
        raise RuntimeError(f"Failed to extract frames from video '{video_file}': {error_details}")


def _collect_source_images(input_path: Path) -> tuple[list[Path], tempfile.TemporaryDirectory[str] | None]:
    if input_path.is_dir():
        image_files = iter_image_files(input_path)
        if not image_files:
            raise ValueError(f"No supported image files found in input folder: {input_path}")
        return image_files, None

    if _is_video_file(input_path):
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = Path(temp_dir_obj.name)
        _extract_video_frames(input_path, temp_dir)
        image_files = iter_image_files(temp_dir)
        if not image_files:
            temp_dir_obj.cleanup()
            raise ValueError(f"No frames extracted from video file: {input_path}")
        return image_files, temp_dir_obj

    if input_path.is_file() and input_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
        return [input_path], None

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    raise ValueError(
        "Input path must be a directory of supported image files or a supported video file."
    )


def convert_to_strips(
    input_folder: Path,
    output_folder: Path,
    output_format: str = "tiff",
) -> int:
    _validate_input_folder(input_folder)
    normalized_format = _normalize_output_format(output_format)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_files = iter_image_files(input_folder)
    if not image_files:
        raise ValueError(f"No supported image files found in input folder: {input_folder}")

    for input_path in image_files:
        with Image.open(input_path) as image:
            avg_color = average_image_color(image)
            strip = Image.new("RGB", (image.width, 1), avg_color)
            strip.save(_intermediate_path(output_folder, input_path, normalized_format))

    return len(image_files)


def stack_tiff_images(
    input_folder: Path,
    output_file: Path,
    show_progress: bool = False,
    output_format: str | None = None,
) -> int:
    _validate_input_folder(input_folder)

    normalized_format = _normalize_output_format(output_format, output_file=output_file)
    image_files = iter_image_files(input_folder)
    if not image_files:
        raise ValueError(f"No supported image files found in input folder: {input_folder}")

    total_height = 0
    max_width = 0

    for path in _progress(image_files, show_progress, desc="Scanning images"):
        with Image.open(path) as image:
            width, height = image.size
            total_height += height
            if width > max_width:
                max_width = width

    output_file.parent.mkdir(parents=True, exist_ok=True)
    stacked_image = Image.new("RGB", (max_width, total_height))

    current_height = 0
    for path in _progress(image_files, show_progress, desc="Stacking images"):
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            stacked_image.paste(rgb_image, (0, current_height))
            current_height += rgb_image.size[1]

    if normalized_format == "png":
        stacked_image.save(output_file, format="PNG")
    else:
        stacked_image.save(output_file, format="TIFF")
    return len(image_files)


def build_timeline_from_frames(
    input_folder: Path,
    output_file: Path,
    intermediate_dir: Path | None = None,
    show_progress: bool = False,
    output_format: str | None = None,
) -> int:
    normalized_format = _normalize_output_format(output_format, output_file=output_file)
    image_files, temp_dir_obj = _collect_source_images(input_folder)

    try:
        if intermediate_dir is not None:
            intermediate_dir.mkdir(parents=True, exist_ok=True)

        rows: list[tuple[int, tuple[int, int, int], Path]] = []
        max_width = 0

        for path in _progress(image_files, show_progress, desc="Processing frames"):
            with Image.open(path) as image:
                avg_color = average_image_color(image)
                width = image.size[0]
                if width > max_width:
                    max_width = width

            rows.append((width, avg_color, path))

            if intermediate_dir is not None:
                strip = Image.new("RGB", (width, 1), avg_color)
                strip.save(_intermediate_path(intermediate_dir, path, normalized_format))

        output_file.parent.mkdir(parents=True, exist_ok=True)
        timeline = Image.new("RGB", (max_width, len(rows)))

        for row_index, (_, avg_color, _) in enumerate(rows):
            row = Image.new("RGB", (max_width, 1), avg_color)
            timeline.paste(row, (0, row_index))

        if normalized_format == "png":
            timeline.save(output_file, format="PNG")
        else:
            timeline.save(output_file, format="TIFF")
        return len(rows)
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


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
