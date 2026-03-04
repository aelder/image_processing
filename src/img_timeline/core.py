from __future__ import annotations

import colorsys
import concurrent.futures
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageStat

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

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
TIMELINE_MODES = {"average", "flow"}
DITHER_MODES = {"none", "floyd-steinberg"}
PALETTE_COLOR_COUNT = 16
PALETTE_MIN_DISTANCE = 24
DEFAULT_FILMIC_16_PALETTE: tuple[tuple[int, int, int], ...] = (
    (6, 8, 12),
    (27, 20, 16),
    (39, 52, 69),
    (68, 45, 37),
    (82, 92, 71),
    (104, 70, 53),
    (78, 108, 132),
    (124, 96, 73),
    (92, 128, 97),
    (152, 102, 68),
    (132, 124, 97),
    (110, 150, 166),
    (181, 132, 84),
    (164, 168, 124),
    (201, 178, 122),
    (241, 229, 190),
)
FLOW_QUANTIZATION_SIZE = 16
FLOW_BASE_COLOR_WEIGHT = 0.10
FLOW_VIBRANCE_WEIGHT = 0.60
FLOW_LUMINANCE_WEIGHT = 0.30
FLOW_NEAR_BLACK_LUMA_THRESHOLD = 0.08
FLOW_NEAR_BLACK_MAX_CHANNEL_THRESHOLD = 0.30
FLOW_NEAR_BLACK_FRAME_DOMINANCE_THRESHOLD = 0.50
FLOW_NEAR_BLACK_COLUMN_DOMINANCE_THRESHOLD = 0.50
FLOW_NEAR_BLACK_PENALTY_MULTIPLIER = 0.15
FLOW_NEAR_BLACK_DOMINANCE_BOOST = 0.20
DEFAULT_AUTO_MIN_FRAMES = 8
VIDEO_POLL_INTERVAL_SECONDS = 0.05
VIDEO_INFLIGHT_MULTIPLIER = 2
FLOW_BIN_COUNT = 4096
FLOW_CUDA_BATCH_SIZE = 4
VIDEO_FRAME_COUNT_PROBE_TIMEOUT_SECONDS = 1.5

_FLOW_CPU_CACHE: dict[tuple[int, int], dict[str, np.ndarray]] = {}
_FLOW_CUDA_CACHE: dict[tuple[int, int], dict[str, cp.ndarray]] = {}
_FLOW_CUDA_HISTOGRAM_KERNEL = None
_FLOW_CUDA_HISTOGRAM_KERNEL_SOURCE = r"""
extern "C" __global__
void flow_histogram_kernel(
    const unsigned char* rgb,
    const int* packed_bins,
    const int width,
    const int frame_pixel_count,
    const int pixel_count,
    int* counts,
    float* sum_r,
    float* sum_g,
    float* sum_b
) {
    int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (idx >= pixel_count) {
        return;
    }

    int pixel_in_frame = idx % frame_pixel_count;
    int frame = idx / frame_pixel_count;
    int col = pixel_in_frame % width;
    int bin = packed_bins[idx];
    int offset = ((frame * width + col) * 4096) + bin;
    int rgb_offset = idx * 3;

    atomicAdd(&counts[offset], 1);
    atomicAdd(&sum_r[offset], (float)rgb[rgb_offset]);
    atomicAdd(&sum_g[offset], (float)rgb[rgb_offset + 1]);
    atomicAdd(&sum_b[offset], (float)rgb[rgb_offset + 2]);
}
"""


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
                "Output file extension must be .tif/.tiff or .png unless "
                "--output-format is provided."
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


def _normalize_mode(mode: str | None) -> str:
    normalized = "average" if mode is None else mode.lower()
    if normalized not in TIMELINE_MODES:
        allowed = ", ".join(sorted(TIMELINE_MODES))
        raise ValueError(f"Unsupported mode: {mode}. Choose one of: {allowed}")
    return normalized


def _normalize_dither(dither: str | None) -> str:
    normalized = "none" if dither is None else dither.lower()
    if normalized not in DITHER_MODES:
        allowed = ", ".join(sorted(DITHER_MODES))
        raise ValueError(f"Unsupported dither mode: {dither}. Choose one of: {allowed}")
    return normalized


def _parse_hex_color(value: str) -> tuple[int, int, int]:
    if not re.fullmatch(r"#?[0-9a-fA-F]{6}", value):
        raise ValueError(f"Invalid palette color '{value}'. Expected #RRGGBB.")
    hex_value = value[1:] if value.startswith("#") else value
    return (
        int(hex_value[0:2], 16),
        int(hex_value[2:4], 16),
        int(hex_value[4:6], 16),
    )


def _resolve_palette(palette_colors: list[str] | None) -> list[tuple[int, int, int]]:
    if palette_colors is None:
        palette = list(DEFAULT_FILMIC_16_PALETTE)
    else:
        if len(palette_colors) != PALETTE_COLOR_COUNT:
            raise ValueError(
                f"Expected exactly {PALETTE_COLOR_COUNT} --palette-color values, got "
                f"{len(palette_colors)}."
            )
        palette = [_parse_hex_color(color) for color in palette_colors]

    if len(set(palette)) != PALETTE_COLOR_COUNT:
        raise ValueError(f"Palette colors must be unique ({PALETTE_COLOR_COUNT} distinct colors).")

    min_sq = PALETTE_MIN_DISTANCE * PALETTE_MIN_DISTANCE
    for index, color_a in enumerate(palette):
        for compare_index in range(index + 1, len(palette)):
            color_b = palette[compare_index]
            dr = color_a[0] - color_b[0]
            dg = color_a[1] - color_b[1]
            db = color_a[2] - color_b[2]
            if dr * dr + dg * dg + db * db < min_sq:
                raise ValueError(
                    f"Palette colors at indexes {index} and {compare_index} are too similar. "
                    f"Minimum RGB distance is {PALETTE_MIN_DISTANCE}."
                )
    return palette


def _build_palette_image(palette: list[tuple[int, int, int]]) -> Image.Image:
    palette_image = Image.new("P", (1, 1))
    palette_values: list[int] = []
    for red, green, blue in palette:
        palette_values.extend([red, green, blue])
    palette_values.extend([0] * (768 - len(palette_values)))
    palette_image.putpalette(palette_values)
    return palette_image


def _apply_palette_dither(
    image: Image.Image,
    dither: str | None = None,
    palette_colors: list[str] | None = None,
) -> Image.Image:
    normalized_dither = _normalize_dither(dither)
    if normalized_dither == "none":
        if palette_colors is not None:
            raise ValueError("--palette-color requires --dither floyd-steinberg.")
        return image

    palette = _resolve_palette(palette_colors)
    palette_image = _build_palette_image(palette)
    rgb_image = image.convert("RGB")
    return rgb_image.quantize(
        colors=PALETTE_COLOR_COUNT,
        palette=palette_image,
        dither=Image.Dither.FLOYDSTEINBERG,
    )


def _normalize_workers(workers: int | None, mode: str, frame_count: int) -> int:
    if workers is not None:
        if workers <= 0:
            raise ValueError("workers must be greater than 0")
        return min(workers, frame_count)

    if frame_count <= 1:
        return 1

    if mode != "flow" or frame_count < DEFAULT_AUTO_MIN_FRAMES:
        return 1

    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, frame_count))


def _normalize_cuda(use_cuda: bool, mode: str) -> bool:
    if not use_cuda:
        return False
    if mode != "flow":
        return False
    return True


def _normalize_flow_worker_count(
    workers: int | None,
    mode: str,
    frame_count: int,
    use_cuda: bool,
) -> int:
    worker_count = _normalize_workers(workers, mode, frame_count)
    if _normalize_cuda(use_cuda, mode):
        # Keep a single CUDA context in-process to avoid GPU contention across child processes.
        return 1
    return worker_count


def _intermediate_path(output_folder: Path, input_path: Path, output_format: str) -> Path:
    if output_format == "tiff":
        return output_folder / input_path.name
    return output_folder / f"{input_path.stem}.png"


def _is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES


def _extract_video_frames(video_file: Path, frame_dir: Path, use_cuda: bool = False) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required to process video files. Install ffmpeg or "
            "provide a directory of image frames."
        )

    output_pattern = frame_dir / "%09d.png"
    command: list[str] = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if use_cuda:
        command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
    command.extend(
        [
            "-i",
            str(video_file),
            "-vsync",
            "0",
        ]
    )
    if use_cuda:
        command.extend(["-vf", "hwdownload,format=rgb24"])
    command.append(str(output_pattern))

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error_details = result.stderr.strip() or "Unknown ffmpeg error"
        raise RuntimeError(f"Failed to extract frames from video '{video_file}': {error_details}")


def _start_video_frame_extractor(
    video_file: Path,
    frame_dir: Path,
    use_cuda: bool = False,
) -> subprocess.Popen[str]:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required to process video files. Install ffmpeg or "
            "provide a directory of image frames."
        )

    output_pattern = frame_dir / "%09d.png"
    command: list[str] = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if use_cuda:
        command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
    command.extend(
        [
            "-i",
            str(video_file),
            "-vsync",
            "0",
        ]
    )
    if use_cuda:
        command.extend(["-vf", "hwdownload,format=rgb24"])
    command.append(str(output_pattern))

    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _parse_video_dimensions(raw_output: str, video_file: Path) -> tuple[int, int]:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if len(lines) >= 2 and lines[0].isdigit() and lines[1].isdigit():
        width = int(lines[0])
        height = int(lines[1])
    else:
        match = re.search(r"(\d+)\s*x\s*(\d+)", raw_output)
        if not match:
            raise RuntimeError(
                f"Unable to parse video dimensions for '{video_file}' from ffprobe output: "
                f"{raw_output.strip()!r}"
            )
        width = int(match.group(1))
        height = int(match.group(2))

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid probed dimensions for '{video_file}': {width}x{height}")
    return width, height


def _probe_video_dimensions(video_file: Path) -> tuple[int, int]:
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        raise RuntimeError("ffprobe is required for in-memory video processing.")

    result = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or "Unknown ffprobe error"
        raise RuntimeError(f"Failed to probe video stream metadata for '{video_file}': {details}")

    return _parse_video_dimensions(result.stdout, video_file)


def _probe_video_frame_count(video_file: Path) -> int | None:
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        return None

    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_file),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=VIDEO_FRAME_COUNT_PROBE_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None

    value = lines[0]
    if not value.isdigit():
        return None

    frame_count = int(value)
    if frame_count <= 0:
        return None
    return frame_count


def _start_video_raw_extractor(video_file: Path, use_cuda: bool = False) -> subprocess.Popen[bytes]:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required to process video files. Install ffmpeg or "
            "provide a directory of image frames."
        )

    command: list[str] = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if use_cuda:
        command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
    command.extend(
        [
            "-i",
            str(video_file),
            "-vsync",
            "0",
        ]
    )
    if use_cuda:
        command.extend(["-vf", "hwdownload,format=rgb24"])
    command.extend(
        [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )

    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _read_exact_bytes(stream: BinaryIO, byte_count: int) -> bytes:
    buffer = bytearray()
    while len(buffer) < byte_count:
        chunk = stream.read(byte_count - len(buffer))
        if not chunk:
            break
        buffer.extend(chunk)
    return bytes(buffer)


def _read_raw_video_frame(stream: BinaryIO, frame_byte_count: int) -> bytes | None:
    frame = _read_exact_bytes(stream, frame_byte_count)
    if not frame:
        return None
    if len(frame) != frame_byte_count:
        raise RuntimeError("Incomplete raw frame read from ffmpeg stream.")
    return frame


def _normalize_video_workers(workers: int | None, mode: str, use_cuda: bool = False) -> int:
    if workers is not None:
        if workers <= 0:
            raise ValueError("workers must be greater than 0")
        return workers
    if mode != "flow":
        return 1
    if _normalize_cuda(use_cuda, mode):
        # Single process avoids multiple CUDA contexts contending for one device.
        return 1
    return max(1, os.cpu_count() or 1)


def _video_frame_path(frame_dir: Path, frame_number: int) -> Path:
    return frame_dir / f"{frame_number:09d}.png"


def _video_frame_ready(frame_dir: Path, frame_number: int, extractor_done: bool) -> bool:
    frame_path = _video_frame_path(frame_dir, frame_number)
    if not frame_path.exists():
        return False
    if extractor_done:
        return True
    return _video_frame_path(frame_dir, frame_number + 1).exists()


def _process_video_frames_streaming_disk(
    input_video: Path,
    output_file: Path,
    mode: str,
    workers: int | None,
    show_progress: bool,
    intermediate_dir: Path | None,
    output_format: str,
    use_cuda_compute: bool = False,
    use_cuda_decode: bool = False,
) -> tuple[list[tuple[int, bytes]], int]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_dir_obj = tempfile.TemporaryDirectory(dir=str(output_file.parent))
    temp_dir = Path(temp_dir_obj.name)
    extractor = _start_video_frame_extractor(input_video, temp_dir, use_cuda=use_cuda_decode)
    worker_count = _normalize_video_workers(workers, mode, use_cuda=use_cuda_compute)
    max_inflight = max(1, worker_count * VIDEO_INFLIGHT_MULTIPLIER)
    rows: list[tuple[int, bytes] | None] = []
    max_width = 0
    next_frame_number = 1
    inflight: dict[concurrent.futures.Future[tuple[int, bytes]], tuple[int, Path]] = {}
    total_frames = _probe_video_frame_count(input_video) if show_progress and tqdm else None
    progress_bar = (
        tqdm(desc="Processing frames", unit="frame", total=total_frames)
        if show_progress and tqdm
        else None
    )

    try:
        if worker_count == 1:
            while True:
                extractor_done = extractor.poll() is not None
                if _video_frame_ready(temp_dir, next_frame_number, extractor_done):
                    frame_path = _video_frame_path(temp_dir, next_frame_number)
                    row_index = next_frame_number - 1
                    width, data = _build_strip_from_path(
                        frame_path,
                        mode,
                        use_cuda=use_cuda_compute,
                    )
                    while len(rows) <= row_index:
                        rows.append(None)
                    rows[row_index] = (width, data)
                    if width > max_width:
                        max_width = width
                    if intermediate_dir is not None:
                        strip = Image.frombytes("RGB", (width, 1), data)
                        strip.save(_intermediate_path(intermediate_dir, frame_path, output_format))
                    frame_path.unlink(missing_ok=True)
                    if progress_bar is not None:
                        progress_bar.update(1)
                    next_frame_number += 1
                    continue

                if extractor_done:
                    break
                time.sleep(VIDEO_POLL_INTERVAL_SECONDS)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
                while True:
                    extractor_done = extractor.poll() is not None
                    while len(inflight) < max_inflight and _video_frame_ready(
                        temp_dir, next_frame_number, extractor_done
                    ):
                        frame_path = _video_frame_path(temp_dir, next_frame_number)
                        row_index = next_frame_number - 1
                        future = executor.submit(
                            _build_strip_from_path,
                            frame_path,
                            mode,
                            use_cuda_compute,
                        )
                        inflight[future] = (row_index, frame_path)
                        next_frame_number += 1

                    if inflight:
                        done, _ = concurrent.futures.wait(
                            inflight.keys(),
                            timeout=VIDEO_POLL_INTERVAL_SECONDS,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            row_index, frame_path = inflight.pop(future)
                            width, data = future.result()
                            while len(rows) <= row_index:
                                rows.append(None)
                            rows[row_index] = (width, data)
                            if width > max_width:
                                max_width = width

                            if intermediate_dir is not None:
                                strip = Image.frombytes("RGB", (width, 1), data)
                                strip.save(
                                    _intermediate_path(intermediate_dir, frame_path, output_format)
                                )

                            frame_path.unlink(missing_ok=True)
                            if progress_bar is not None:
                                progress_bar.update(1)
                    else:
                        if extractor_done:
                            break
                        time.sleep(VIDEO_POLL_INTERVAL_SECONDS)

        stderr_output = extractor.stderr.read().strip() if extractor.stderr is not None else ""
        if extractor.returncode and extractor.returncode != 0:
            details = stderr_output or "Unknown ffmpeg error"
            raise RuntimeError(f"Failed to extract frames from video '{input_video}': {details}")

        if any(row is None for row in rows):
            raise RuntimeError(
                f"Non-contiguous frame sequence extracted from video '{input_video}'."
            )

        dense_rows = [row for row in rows if row is not None]
        if not dense_rows:
            raise ValueError(f"No frames extracted from video file: {input_video}")
        return dense_rows, max_width
    finally:
        if extractor.poll() is None:
            extractor.terminate()
            try:
                extractor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                extractor.kill()
                extractor.wait(timeout=5)
        if extractor.stderr is not None:
            extractor.stderr.close()
        if progress_bar is not None:
            progress_bar.close()
        temp_dir_obj.cleanup()


def _build_strip_from_frame_bytes(
    frame_data: bytes,
    width: int,
    height: int,
    mode: str,
    use_cuda: bool = False,
) -> tuple[int, bytes]:
    image = Image.frombytes("RGB", (width, height), frame_data)
    strip = _build_strip(image, mode, use_cuda=use_cuda)
    return strip.size[0], strip.tobytes()


def _process_video_frames_streaming_in_memory(
    input_video: Path,
    mode: str,
    workers: int | None,
    show_progress: bool,
    intermediate_dir: Path | None,
    output_format: str,
    use_cuda_compute: bool = False,
    use_cuda_decode: bool = False,
) -> tuple[list[tuple[int, bytes]], int]:
    frame_width, frame_height = _probe_video_dimensions(input_video)
    frame_byte_count = frame_width * frame_height * 3
    extractor = _start_video_raw_extractor(input_video, use_cuda=use_cuda_decode)
    worker_count = _normalize_video_workers(workers, mode, use_cuda=use_cuda_compute)
    max_inflight = max(1, worker_count * VIDEO_INFLIGHT_MULTIPLIER)
    rows: list[tuple[int, bytes] | None] = []
    max_width = 0
    total_frames = _probe_video_frame_count(input_video) if show_progress and tqdm else None
    progress_bar = (
        tqdm(desc="Processing frames", unit="frame", total=total_frames)
        if show_progress and tqdm
        else None
    )

    try:
        stdout_stream = extractor.stdout
        if stdout_stream is None:
            raise RuntimeError("Failed to initialize ffmpeg frame stream.")

        if worker_count == 1:
            frame_queue: queue.Queue[tuple[int, bytes] | None] = queue.Queue(maxsize=max_inflight)
            producer_error: list[Exception] = []

            def _producer() -> None:
                frame_number = 1
                try:
                    while True:
                        frame_data = _read_raw_video_frame(stdout_stream, frame_byte_count)
                        if frame_data is None:
                            break
                        frame_queue.put((frame_number, frame_data))
                        frame_number += 1
                except Exception as error:  # pragma: no cover - protective path
                    producer_error.append(error)
                finally:
                    frame_queue.put(None)

            producer = threading.Thread(target=_producer, daemon=True)
            producer.start()

            stream_done = False
            while True:
                if stream_done:
                    break

                item = frame_queue.get()
                if item is None:
                    break

                frame_numbers: list[int] = [item[0]]
                frame_batch: list[bytes] = [item[1]]
                target_batch = (
                    FLOW_CUDA_BATCH_SIZE if use_cuda_compute and mode == "flow" else 1
                )

                while len(frame_batch) < target_batch:
                    try:
                        queued_item = frame_queue.get_nowait()
                    except queue.Empty:
                        break
                    if queued_item is None:
                        stream_done = True
                        break
                    frame_numbers.append(queued_item[0])
                    frame_batch.append(queued_item[1])

                if use_cuda_compute and mode == "flow":
                    batch_rows = _build_flow_strips_from_frame_batch_bytes(
                        frame_batch,
                        frame_width,
                        frame_height,
                        use_cuda=use_cuda_compute,
                    )
                else:
                    batch_rows = [
                        _build_strip_from_frame_bytes(
                            frame_data,
                            frame_width,
                            frame_height,
                            mode,
                            use_cuda=use_cuda_compute,
                        )
                        for frame_data in frame_batch
                    ]

                for frame_number, (width, data) in zip(frame_numbers, batch_rows, strict=True):
                    row_index = frame_number - 1
                    while len(rows) <= row_index:
                        rows.append(None)
                    rows[row_index] = (width, data)
                    if width > max_width:
                        max_width = width
                    if intermediate_dir is not None:
                        strip = Image.frombytes("RGB", (width, 1), data)
                        frame_name = Path(f"{frame_number:09d}.png")
                        strip.save(_intermediate_path(intermediate_dir, frame_name, output_format))
                    if progress_bar is not None:
                        progress_bar.update(1)

            producer.join(timeout=5)
            if producer_error:
                raise producer_error[0]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
                inflight: dict[concurrent.futures.Future[tuple[int, bytes]], tuple[int, int]] = {}
                next_frame_number = 1
                reached_end = False

                while True:
                    while not reached_end and len(inflight) < max_inflight:
                        frame_data = _read_raw_video_frame(stdout_stream, frame_byte_count)
                        if frame_data is None:
                            reached_end = True
                            break
                        row_index = next_frame_number - 1
                        future = executor.submit(
                            _build_strip_from_frame_bytes,
                            frame_data,
                            frame_width,
                            frame_height,
                            mode,
                            use_cuda_compute,
                        )
                        inflight[future] = (row_index, next_frame_number)
                        next_frame_number += 1

                    if not inflight:
                        if reached_end:
                            break
                        continue

                    done, _ = concurrent.futures.wait(
                        inflight.keys(),
                        timeout=VIDEO_POLL_INTERVAL_SECONDS,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        continue

                    for future in done:
                        row_index, frame_number = inflight.pop(future)
                        width, data = future.result()
                        while len(rows) <= row_index:
                            rows.append(None)
                        rows[row_index] = (width, data)
                        if width > max_width:
                            max_width = width
                        if intermediate_dir is not None:
                            strip = Image.frombytes("RGB", (width, 1), data)
                            frame_name = Path(f"{frame_number:09d}.png")
                            strip.save(
                                _intermediate_path(intermediate_dir, frame_name, output_format)
                            )
                        if progress_bar is not None:
                            progress_bar.update(1)

        extractor.wait(timeout=5)
        stderr_output = (
            extractor.stderr.read().decode("utf-8", errors="replace").strip()
            if extractor.stderr is not None
            else ""
        )
        if extractor.returncode and extractor.returncode != 0:
            details = stderr_output or "Unknown ffmpeg error"
            raise RuntimeError(f"Failed to extract frames from video '{input_video}': {details}")

        if any(row is None for row in rows):
            raise RuntimeError(
                f"Non-contiguous frame sequence extracted from video '{input_video}'."
            )

        dense_rows = [row for row in rows if row is not None]
        if not dense_rows:
            raise ValueError(f"No frames extracted from video file: {input_video}")
        return dense_rows, max_width
    finally:
        if extractor.poll() is None:
            extractor.terminate()
            try:
                extractor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                extractor.kill()
                extractor.wait(timeout=5)
        if extractor.stdout is not None:
            extractor.stdout.close()
        if extractor.stderr is not None:
            extractor.stderr.close()
        if progress_bar is not None:
            progress_bar.close()


def _process_video_frames_streaming(
    input_video: Path,
    output_file: Path,
    mode: str,
    workers: int | None,
    show_progress: bool,
    intermediate_dir: Path | None,
    output_format: str,
    use_cuda: bool = False,
) -> tuple[list[tuple[int, bytes]], int]:
    try:
        return _process_video_frames_streaming_in_memory(
            input_video=input_video,
            mode=mode,
            workers=workers,
            show_progress=show_progress,
            intermediate_dir=intermediate_dir,
            output_format=output_format,
            use_cuda_compute=use_cuda,
            use_cuda_decode=use_cuda,
        )
    except RuntimeError as error:
        if use_cuda and (
            "Failed to extract frames from video" in str(error)
            or "ffmpeg is required to process video files." in str(error)
        ):
            # Retry with software decode if NVDEC path is unavailable, while keeping
            # CUDA strip compute enabled when requested.
            return _process_video_frames_streaming_in_memory(
                input_video=input_video,
                mode=mode,
                workers=workers,
                show_progress=show_progress,
                intermediate_dir=intermediate_dir,
                output_format=output_format,
                use_cuda_compute=use_cuda,
                use_cuda_decode=False,
            )
        if "ffprobe is required for in-memory video processing." not in str(error):
            raise
        return _process_video_frames_streaming_disk(
            input_video=input_video,
            output_file=output_file,
            mode=mode,
            workers=workers,
            show_progress=show_progress,
            intermediate_dir=intermediate_dir,
            output_format=output_format,
            use_cuda_compute=use_cuda,
            use_cuda_decode=use_cuda,
        )


def _collect_source_images(
    input_path: Path,
    temp_parent_dir: Path | None = None,
    use_cuda: bool = False,
) -> tuple[list[Path], tempfile.TemporaryDirectory[str] | None]:
    if input_path.is_dir():
        image_files = iter_image_files(input_path)
        if not image_files:
            raise ValueError(f"No supported image files found in input folder: {input_path}")
        return image_files, None

    if _is_video_file(input_path):
        if temp_parent_dir is not None:
            temp_parent_dir.mkdir(parents=True, exist_ok=True)
            temp_dir_obj = tempfile.TemporaryDirectory(dir=str(temp_parent_dir))
        else:
            temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = Path(temp_dir_obj.name)
        _extract_video_frames(input_path, temp_dir, use_cuda=use_cuda)
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


def _average_strip(image: Image.Image) -> Image.Image:
    avg_color = average_image_color(image)
    return Image.new("RGB", (image.width, 1), avg_color)


def _image_to_rgb_uint8_array(image: Image.Image) -> np.ndarray:
    if image.mode == "RGB":
        return np.asarray(image, dtype=np.uint8)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _flow_cpu_cache_for_frame(height: int, width: int) -> dict[str, np.ndarray]:
    cache_key = (height, width)
    cached = _FLOW_CPU_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cached = {
        "counts": np.zeros(FLOW_BIN_COUNT, dtype=np.int32),
        "sum_r": np.zeros(FLOW_BIN_COUNT, dtype=np.float64),
        "sum_g": np.zeros(FLOW_BIN_COUNT, dtype=np.float64),
        "sum_b": np.zeros(FLOW_BIN_COUNT, dtype=np.float64),
    }
    _FLOW_CPU_CACHE[cache_key] = cached
    return cached


def _flow_strip_cpu(image: Image.Image) -> Image.Image:
    rgb = _image_to_rgb_uint8_array(image)
    height, width, _ = rgb.shape
    cache = _flow_cpu_cache_for_frame(height, width)
    counts_cache = cache["counts"]
    sum_r_cache = cache["sum_r"]
    sum_g_cache = cache["sum_g"]
    sum_b_cache = cache["sum_b"]

    red = rgb[:, :, 0].astype(np.float64)
    green = rgb[:, :, 1].astype(np.float64)
    blue = rgb[:, :, 2].astype(np.float64)
    frame_max_channel = np.maximum(np.maximum(red, green), blue) / 255.0
    frame_luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0
    frame_near_black_ratio = float(
        np.mean(
            (frame_luminance <= FLOW_NEAR_BLACK_LUMA_THRESHOLD)
            & (frame_max_channel <= FLOW_NEAR_BLACK_MAX_CHANNEL_THRESHOLD)
        )
    )

    quantized = rgb // FLOW_QUANTIZATION_SIZE
    packed_bins = (
        (quantized[:, :, 0].astype(np.uint16) << 8)
        | (quantized[:, :, 1].astype(np.uint16) << 4)
        | quantized[:, :, 2].astype(np.uint16)
    )

    row = np.zeros((width, 3), dtype=np.uint8)
    denominator = float(height)

    for x in range(width):
        col_bins = packed_bins[:, x]
        counts_cache.fill(0)
        counts_raw = np.bincount(col_bins)
        counts_cache[: counts_raw.size] = counts_raw.astype(np.int32, copy=False)
        used = np.nonzero(counts_cache)[0]
        used_counts = counts_cache[used].astype(np.float64, copy=False)

        col_r = red[:, x]
        col_g = green[:, x]
        col_b = blue[:, x]

        sum_r_cache.fill(0.0)
        sum_g_cache.fill(0.0)
        sum_b_cache.fill(0.0)
        sum_r_raw = np.bincount(col_bins, weights=col_r)
        sum_g_raw = np.bincount(col_bins, weights=col_g)
        sum_b_raw = np.bincount(col_bins, weights=col_b)
        sum_r_cache[: sum_r_raw.size] = sum_r_raw
        sum_g_cache[: sum_g_raw.size] = sum_g_raw
        sum_b_cache[: sum_b_raw.size] = sum_b_raw
        sum_r = sum_r_cache[used]
        sum_g = sum_g_cache[used]
        sum_b = sum_b_cache[used]

        centroid_r = np.rint(sum_r / used_counts).astype(np.uint8)
        centroid_g = np.rint(sum_g / used_counts).astype(np.uint8)
        centroid_b = np.rint(sum_b / used_counts).astype(np.uint8)

        r_norm = centroid_r.astype(np.float64) / 255.0
        g_norm = centroid_g.astype(np.float64) / 255.0
        b_norm = centroid_b.astype(np.float64) / 255.0

        maxc = np.maximum.reduce([r_norm, g_norm, b_norm])
        minc = np.minimum.reduce([r_norm, g_norm, b_norm])
        saturation = np.divide(
            maxc - minc,
            maxc,
            out=np.zeros_like(maxc),
            where=maxc != 0.0,
        )
        vibrance = saturation * maxc
        luminance = 0.2126 * r_norm + 0.7152 * g_norm + 0.0722 * b_norm

        freq = used_counts / denominator
        score = freq * (
            FLOW_BASE_COLOR_WEIGHT
            + FLOW_VIBRANCE_WEIGHT * vibrance
            + FLOW_LUMINANCE_WEIGHT * luminance
        )
        near_black_candidates = (luminance <= FLOW_NEAR_BLACK_LUMA_THRESHOLD) & (
            maxc <= FLOW_NEAR_BLACK_MAX_CHANNEL_THRESHOLD
        )
        score = np.where(
            near_black_candidates & (freq >= FLOW_NEAR_BLACK_COLUMN_DOMINANCE_THRESHOLD),
            score + FLOW_NEAR_BLACK_DOMINANCE_BOOST * freq,
            score,
        )
        if frame_near_black_ratio < FLOW_NEAR_BLACK_FRAME_DOMINANCE_THRESHOLD:
            score = np.where(
                near_black_candidates & (freq < FLOW_NEAR_BLACK_COLUMN_DOMINANCE_THRESHOLD),
                score * FLOW_NEAR_BLACK_PENALTY_MULTIPLIER,
                score,
            )

        best_score = np.max(score)
        mask_score = score == best_score
        best_count = np.max(used_counts[mask_score])
        mask_count = mask_score & (used_counts == best_count)
        best_vibrance = np.max(vibrance[mask_count])
        mask_vibrance = mask_count & (vibrance == best_vibrance)
        candidate_idx = np.nonzero(mask_vibrance)[0]
        best = candidate_idx[np.argmin(used[candidate_idx])]
        row[x, 0] = centroid_r[best]
        row[x, 1] = centroid_g[best]
        row[x, 2] = centroid_b[best]

    return Image.fromarray(row[np.newaxis, :, :], mode="RGB")


def _flow_cuda_cache_for_shape(batch_size: int, width: int):
    cache_key = (batch_size, width)
    cached = _FLOW_CUDA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if cp is None:  # pragma: no cover
        raise RuntimeError("CUDA cache requested without CuPy installed.")

    cached = {
        "bin_indices": cp.arange(FLOW_BIN_COUNT, dtype=cp.int32)[cp.newaxis, cp.newaxis, :],
        "frame_indices": cp.arange(batch_size, dtype=cp.int32)[:, cp.newaxis],
        "row_indices": cp.arange(width, dtype=cp.int32)[cp.newaxis, :],
        "counts": cp.zeros((batch_size, width, FLOW_BIN_COUNT), dtype=cp.int32),
        "sum_r": cp.zeros((batch_size, width, FLOW_BIN_COUNT), dtype=cp.float32),
        "sum_g": cp.zeros((batch_size, width, FLOW_BIN_COUNT), dtype=cp.float32),
        "sum_b": cp.zeros((batch_size, width, FLOW_BIN_COUNT), dtype=cp.float32),
    }
    _FLOW_CUDA_CACHE[cache_key] = cached
    return cached


def _flow_cuda_histogram_kernel():
    global _FLOW_CUDA_HISTOGRAM_KERNEL
    if _FLOW_CUDA_HISTOGRAM_KERNEL is not None:
        return _FLOW_CUDA_HISTOGRAM_KERNEL
    if cp is None:  # pragma: no cover
        raise RuntimeError("CUDA histogram kernel requested without CuPy installed.")
    _FLOW_CUDA_HISTOGRAM_KERNEL = cp.RawKernel(
        _FLOW_CUDA_HISTOGRAM_KERNEL_SOURCE,
        "flow_histogram_kernel",
    )
    return _FLOW_CUDA_HISTOGRAM_KERNEL


def _flow_rows_cuda_from_rgb_batch(rgb_batch: cp.ndarray) -> cp.ndarray:
    batch_size, height, width, _ = rgb_batch.shape
    denominator = float(height)
    cache = _flow_cuda_cache_for_shape(batch_size, width)

    frame_max_channel = cp.max(rgb_batch.astype(cp.float32), axis=3) / 255.0
    frame_luminance = (
        0.2126 * rgb_batch[:, :, :, 0].astype(cp.float32)
        + 0.7152 * rgb_batch[:, :, :, 1].astype(cp.float32)
        + 0.0722 * rgb_batch[:, :, :, 2].astype(cp.float32)
    ) / 255.0
    frame_near_black_ratio = cp.mean(
        (frame_luminance <= FLOW_NEAR_BLACK_LUMA_THRESHOLD)
        & (frame_max_channel <= FLOW_NEAR_BLACK_MAX_CHANNEL_THRESHOLD),
        axis=(1, 2),
    )

    quantized = rgb_batch // FLOW_QUANTIZATION_SIZE
    packed_bins = (
        (quantized[:, :, :, 0].astype(cp.int32) << 8)
        | (quantized[:, :, :, 1].astype(cp.int32) << 4)
        | quantized[:, :, :, 2].astype(cp.int32)
    )

    bin_indices_flat = packed_bins.ravel()
    counts = cache["counts"]
    sum_r = cache["sum_r"]
    sum_g = cache["sum_g"]
    sum_b = cache["sum_b"]
    counts.fill(0)
    sum_r.fill(0)
    sum_g.fill(0)
    sum_b.fill(0)

    kernel = _flow_cuda_histogram_kernel()
    frame_pixel_count = int(height * width)
    pixel_count = int(batch_size * frame_pixel_count)
    threads_per_block = 256
    blocks = (pixel_count + threads_per_block - 1) // threads_per_block
    kernel(
        (blocks,),
        (threads_per_block,),
        (
            rgb_batch.ravel(),
            bin_indices_flat,
            np.int32(width),
            np.int32(frame_pixel_count),
            np.int32(pixel_count),
            counts.ravel(),
            sum_r.ravel(),
            sum_g.ravel(),
            sum_b.ravel(),
        ),
    )

    used = counts > 0
    counts_f = counts.astype(cp.float32)
    safe_counts = cp.where(used, counts_f, 1.0)

    centroid_r = cp.rint(sum_r / safe_counts).astype(cp.uint8)
    centroid_g = cp.rint(sum_g / safe_counts).astype(cp.uint8)
    centroid_b = cp.rint(sum_b / safe_counts).astype(cp.uint8)

    r_norm = centroid_r.astype(cp.float32) / 255.0
    g_norm = centroid_g.astype(cp.float32) / 255.0
    b_norm = centroid_b.astype(cp.float32) / 255.0

    maxc = cp.maximum(cp.maximum(r_norm, g_norm), b_norm)
    minc = cp.minimum(cp.minimum(r_norm, g_norm), b_norm)
    saturation = cp.zeros_like(maxc)
    nonzero_mask = maxc != 0.0
    saturation[nonzero_mask] = (maxc[nonzero_mask] - minc[nonzero_mask]) / maxc[nonzero_mask]
    vibrance = saturation * maxc
    luminance = 0.2126 * r_norm + 0.7152 * g_norm + 0.0722 * b_norm

    freq = counts_f / denominator
    score = freq * (
        FLOW_BASE_COLOR_WEIGHT
        + FLOW_VIBRANCE_WEIGHT * vibrance
        + FLOW_LUMINANCE_WEIGHT * luminance
    )
    near_black_candidates = (luminance <= FLOW_NEAR_BLACK_LUMA_THRESHOLD) & (
        maxc <= FLOW_NEAR_BLACK_MAX_CHANNEL_THRESHOLD
    )
    score = cp.where(
        near_black_candidates & (freq >= FLOW_NEAR_BLACK_COLUMN_DOMINANCE_THRESHOLD),
        score + FLOW_NEAR_BLACK_DOMINANCE_BOOST * freq,
        score,
    )
    frame_is_dark = (
        frame_near_black_ratio[:, cp.newaxis, cp.newaxis]
        < FLOW_NEAR_BLACK_FRAME_DOMINANCE_THRESHOLD
    )
    score = cp.where(
        frame_is_dark & near_black_candidates & (freq < FLOW_NEAR_BLACK_COLUMN_DOMINANCE_THRESHOLD),
        score * FLOW_NEAR_BLACK_PENALTY_MULTIPLIER,
        score,
    )

    invalid_score = cp.asarray(-np.inf, dtype=cp.float32)
    candidate_score = cp.where(used, score, invalid_score)
    best_score = cp.max(candidate_score, axis=2, keepdims=True)
    mask_score = used & (candidate_score == best_score)

    candidate_count = cp.where(mask_score, counts_f, -1.0)
    best_count = cp.max(candidate_count, axis=2, keepdims=True)
    mask_count = mask_score & (counts_f == best_count)

    candidate_vibrance = cp.where(mask_count, vibrance, -1.0)
    best_vibrance = cp.max(candidate_vibrance, axis=2, keepdims=True)
    mask_vibrance = mask_count & (vibrance == best_vibrance)

    best_bins = cp.min(cp.where(mask_vibrance, cache["bin_indices"], FLOW_BIN_COUNT), axis=2)

    rows = cp.empty((batch_size, width, 3), dtype=cp.uint8)
    rows[:, :, 0] = centroid_r[cache["frame_indices"], cache["row_indices"], best_bins]
    rows[:, :, 1] = centroid_g[cache["frame_indices"], cache["row_indices"], best_bins]
    rows[:, :, 2] = centroid_b[cache["frame_indices"], cache["row_indices"], best_bins]
    return rows

def _flow_strip_cuda_from_rgb(rgb: cp.ndarray) -> Image.Image:
    rows = _flow_rows_cuda_from_rgb_batch(rgb[cp.newaxis, :, :, :])
    row_cpu = cp.asnumpy(rows[0])
    return Image.fromarray(row_cpu[np.newaxis, :, :], mode="RGB")


def _flow_strip_cuda(image: Image.Image) -> Image.Image:
    if cp is None:
        raise RuntimeError(
            "CUDA flow processing requested, but CuPy is not installed. "
            "Install a CUDA build such as cupy-cuda12x and retry with --cuda."
        )

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        raise RuntimeError(f"CUDA flow processing unavailable: {error}") from error
    if device_count < 1:
        raise RuntimeError("CUDA flow processing requested, but no CUDA devices were detected.")

    rgb = cp.asarray(_image_to_rgb_uint8_array(image))
    return _flow_strip_cuda_from_rgb(rgb)


def _build_flow_strips_from_frame_batch_bytes(
    frame_batch: list[bytes],
    width: int,
    height: int,
    use_cuda: bool,
) -> list[tuple[int, bytes]]:
    if not frame_batch:
        return []
    if not use_cuda or cp is None:
        return [
            _build_strip_from_frame_bytes(frame_data, width, height, "flow", use_cuda=False)
            for frame_data in frame_batch
        ]

    frame_array = np.frombuffer(b"".join(frame_batch), dtype=np.uint8).reshape(
        len(frame_batch), height, width, 3
    )
    rgb_batch = cp.asarray(frame_array)
    rows = _flow_rows_cuda_from_rgb_batch(rgb_batch)
    rows_cpu = cp.asnumpy(rows)
    return [(width, rows_cpu[index].tobytes()) for index in range(len(frame_batch))]


def _build_strip(image: Image.Image, mode: str, use_cuda: bool = False) -> Image.Image:
    if mode == "average":
        return _average_strip(image)
    if _normalize_cuda(use_cuda, mode):
        return _flow_strip_cuda(image)
    return _flow_strip_cpu(image)


def _build_strip_from_path(
    input_path: Path,
    mode: str,
    use_cuda: bool = False,
) -> tuple[int, bytes]:
    with Image.open(input_path) as image:
        strip = _build_strip(image, mode, use_cuda=use_cuda)
    return strip.size[0], strip.tobytes()


def convert_to_strips(
    input_folder: Path,
    output_folder: Path,
    output_format: str = "tiff",
    mode: str = "average",
    workers: int | None = None,
    use_cuda: bool = False,
) -> int:
    _validate_input_folder(input_folder)
    normalized_format = _normalize_output_format(output_format)
    normalized_mode = _normalize_mode(mode)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_files = iter_image_files(input_folder)
    if not image_files:
        raise ValueError(f"No supported image files found in input folder: {input_folder}")

    worker_count = _normalize_flow_worker_count(
        workers,
        normalized_mode,
        len(image_files),
        use_cuda=use_cuda,
    )

    if worker_count == 1:
        for input_path in image_files:
            with Image.open(input_path) as image:
                strip = _build_strip(image, normalized_mode, use_cuda=use_cuda)
                strip.save(_intermediate_path(output_folder, input_path, normalized_format))
        return len(image_files)

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _build_strip_from_path,
                input_path,
                normalized_mode,
                use_cuda,
            ): input_path
            for input_path in image_files
        }
        for future in concurrent.futures.as_completed(futures):
            input_path = futures[future]
            width, data = future.result()
            strip = Image.frombytes("RGB", (width, 1), data)
            strip.save(_intermediate_path(output_folder, input_path, normalized_format))

    return len(image_files)


def stack_tiff_images(
    input_folder: Path,
    output_file: Path,
    show_progress: bool = False,
    output_format: str | None = None,
    dither: str | None = None,
    palette_colors: list[str] | None = None,
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

    stacked_image = _apply_palette_dither(
        stacked_image,
        dither=dither,
        palette_colors=palette_colors,
    )

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
    mode: str = "average",
    workers: int | None = None,
    use_cuda: bool = False,
    dither: str | None = None,
    palette_colors: list[str] | None = None,
) -> int:
    normalized_format = _normalize_output_format(output_format, output_file=output_file)
    normalized_mode = _normalize_mode(mode)

    if intermediate_dir is not None:
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    if _is_video_file(input_folder):
        rows, max_width = _process_video_frames_streaming(
            input_video=input_folder,
            output_file=output_file,
            mode=normalized_mode,
            workers=workers,
            show_progress=show_progress,
            intermediate_dir=intermediate_dir,
            output_format=normalized_format,
            use_cuda=use_cuda,
        )
    else:
        image_files, temp_dir_obj = _collect_source_images(
            input_folder,
            temp_parent_dir=output_file.parent,
            use_cuda=use_cuda,
        )
        try:
            worker_count = _normalize_flow_worker_count(
                workers,
                normalized_mode,
                len(image_files),
                use_cuda=use_cuda,
            )
            rows: list[tuple[int, bytes]] = [(0, b"")] * len(image_files)
            max_width = 0

            if worker_count == 1:
                for row_index, path in enumerate(
                    _progress(image_files, show_progress, desc="Processing frames")
                ):
                    with Image.open(path) as image:
                        strip = _build_strip(image, normalized_mode, use_cuda=use_cuda)
                    width = strip.size[0]
                    data = strip.tobytes()
                    rows[row_index] = (width, data)
                    if width > max_width:
                        max_width = width
                    if intermediate_dir is not None:
                        strip.save(_intermediate_path(intermediate_dir, path, normalized_format))
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
                    futures = {
                        executor.submit(_build_strip_from_path, path, normalized_mode, use_cuda): (
                            row_index,
                            path,
                        )
                        for row_index, path in enumerate(image_files)
                    }
                    for future in _progress(
                        concurrent.futures.as_completed(futures),
                        show_progress,
                        desc="Processing frames",
                    ):
                        row_index, path = futures[future]
                        width, data = future.result()
                        rows[row_index] = (width, data)
                        if width > max_width:
                            max_width = width
                        if intermediate_dir is not None:
                            strip = Image.frombytes("RGB", (width, 1), data)
                            strip.save(
                                _intermediate_path(intermediate_dir, path, normalized_format)
                            )
        finally:
            if temp_dir_obj is not None:
                temp_dir_obj.cleanup()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    timeline = Image.new("RGB", (max_width, len(rows)))

    for row_index, (width, data) in enumerate(rows):
        row = Image.frombytes("RGB", (width, 1), data)
        timeline.paste(row, (0, row_index))

    timeline = _apply_palette_dither(
        timeline,
        dither=dither,
        palette_colors=palette_colors,
    )

    if normalized_format == "png":
        timeline.save(output_file, format="PNG")
    else:
        timeline.save(output_file, format="TIFF")
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
