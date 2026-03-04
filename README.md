# img-timeline

Create a color timeline from image frame sequences or directly from video files.

![CI](https://github.com/aelder/image_processing/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-GPLv3-green)

`img-timeline` reads source frames (`.tif/.tiff`, `.png`, `.jpg/.jpeg`, `.webp`, `.bmp`, `.gif`) or supported video files (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, and more), then writes one timeline row per frame.

The resulting image is usually narrow and very tall. Scale it vertically to produce poster-style color progression visuals.

## What It Does

- Converts each frame to a 1px-high strip in one of two modes:
  - `average`: one average RGB color for the full row width
  - `flow`: one color per column chosen from frequent/vibrant quantized colors
- Stacks strips in deterministic lexical frame order.
- Supports optional 16-color Floyd-Steinberg dithering on final output.
- Supports optional writing of intermediate strip images for inspection.

## Current Video Pipeline Behavior

For `build` with video input:

- Primary path: frames are decoded by `ffmpeg` as raw RGB to memory, then consumed by a bounded in-memory worker pipeline.
- Backpressure: in-flight frame processing is capped and scales with worker count, so decode cannot run far ahead of strip generation.
- Fallback path: if `ffprobe` is unavailable, the tool falls back to disk-backed temporary frame extraction.
- Progress totals: with `--progress`, the pipeline performs a fast metadata probe for total frame count and shows determinate progress when available; if probing is unavailable/slow, it falls back to indeterminate progress updates.
- With `--cuda` and `--mode flow`, decode and strip generation are pipelined in-memory for better overlap.
- If NVDEC cannot decode the source stream/profile, decode automatically falls back to software while keeping CUDA strip compute enabled.

This design reduces temporary full-resolution frame buildup and SSD write pressure in normal operation.

## Install (from source)

```bash
git clone https://github.com/aelder/image_processing.git
cd image_processing
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install package:

```bash
pip install -e .
```

Optional progress bar dependency:

```bash
pip install -e ".[progress]"
```

## Quickstart

Build from frame folder:

```bash
img-timeline build ./frames ./out/movie_timeline.tif --progress
```

Build directly from video:

```bash
img-timeline build ./movie.mp4 ./out/movie_flow.png --mode flow --output-format png --progress
```

Equivalent module invocation:

```bash
python -m img_timeline build ./movie.mp4 ./out/movie_flow.png --mode flow --output-format png --progress
```

## CLI Commands

| Command | Purpose |
|---|---|
| `img-timeline build <input_path> <output_file>` | Full pipeline from frame directory or video to final timeline |
| `img-timeline convert <input_folder> <output_folder>` | Legacy step: convert frames to 1px strips |
| `img-timeline stack <input_folder> <output_file>` | Legacy step: stack strip files into final timeline |
| `img-timeline generate-demo <output_dir>` | Generate synthetic demo TIFF frames |

### `build` Options

- `--mode {average,flow}` (default: `average`)
- `--workers <n>` (default: auto for larger `flow`; otherwise 1)
- `--cuda` (enable CUDA acceleration for `flow`: NVDEC decode + CuPy strip compute)
- `--output-format {tiff,png}` (default: inferred from output extension, TIFF if no extension)
- `--intermediate-dir <dir>` (write intermediate strips)
- `--progress` (show progress bars when `tqdm` is installed)
- `--dither {none,floyd-steinberg}` (default: `none`)
- `--palette-color <#RRGGBB>` (repeat exactly 16 times, requires `--dither floyd-steinberg`)

### `convert` Options

- `--mode {average,flow}` (default: `average`)
- `--workers <n>`
- `--cuda` (enable CUDA acceleration for `flow` strip compute)
- `--output-format {tiff,png}` (default: `tiff`)

### `stack` Options

- `--output-format {tiff,png}` (default: inferred from output extension, TIFF if no extension)
- `--progress`
- `--dither {none,floyd-steinberg}`
- `--palette-color <#RRGGBB>` (repeat exactly 16 times with dithering)

## Examples

Create flow timeline from `matrix.mkv`:

```powershell
python -m img_timeline build .\matrix.mkv .\out\matrix_flow.png --mode flow --output-format png --progress
```

Force worker count:

```bash
img-timeline build ./movie.mp4 ./out/movie_flow.png --mode flow --output-format png --workers 8 --progress
```

Use CUDA/NVIDIA acceleration:

```bash
img-timeline build ./movie.mp4 ./out/movie_flow.png --mode flow --output-format png --cuda --progress
```

Write intermediate strips:

```bash
img-timeline build ./frames ./out/movie_timeline.tif --intermediate-dir ./out/strips --progress
```

Dither final output with default palette:

```bash
img-timeline build ./movie.mp4 ./out/movie_flow_dithered.png --mode flow --output-format png --dither floyd-steinberg --progress
```

Custom 16-color palette:

```bash
img-timeline build ./frames ./out/timeline_custom_palette.png --output-format png --dither floyd-steinberg \
  --palette-color '#06080C' --palette-color '#1B1410' --palette-color '#273445' --palette-color '#442D25' \
  --palette-color '#525C47' --palette-color '#684635' --palette-color '#4E6C84' --palette-color '#7C6049' \
  --palette-color '#5C8061' --palette-color '#986644' --palette-color '#847C61' --palette-color '#6E96A6' \
  --palette-color '#B58454' --palette-color '#A4A87C' --palette-color '#C9B27A' --palette-color '#F1E5BE'
```

Legacy two-step workflow:

```bash
img-timeline convert ./frames ./out/strips --mode flow --output-format png
img-timeline stack ./out/strips ./out/movie_timeline.tif --progress
```

## Output Semantics

- Timeline width = max input frame width.
- Timeline height = number of source frames.
- Ordering is lexical sort by filename for folder inputs.
- Default output mode is RGB.
- With `--dither floyd-steinberg`, final image is paletted (`P`) with 16 colors.

## Recommended Frame Naming

Use zero-padded names so lexical sort matches chronological order:

- Good: `000001.tif`, `000002.tif`, ...
- Risky: `1.tif`, `10.tif`, `2.tif`

## Requirements

- Python 3.10+
- NumPy
- Pillow
- Optional for CUDA acceleration: a compatible CuPy package (for example `cupy-cuda12x`)
- `ffmpeg` (for video inputs)
- `ffprobe` (recommended for primary in-memory video path; without it, disk fallback is used)
- Optional: `tqdm` for progress bars

## Performance Notes

- `flow` + `--cuda` uses a fixed-bin (`4096`) GPU histogram/reduction implementation that preserves existing flow scoring semantics.
- CUDA histogram accumulation now uses a dedicated GPU kernel (replacing `cp.add.at`) for lower per-frame overhead.
- In single-worker video `flow` mode, CUDA processing uses bounded frame batching to reduce launch/transfer overhead while preserving row order.
- For most videos this is significantly faster than CPU flow mode, especially at 4K widths.
- Hardware decode acceleration depends on codec/profile support; unsupported streams transparently use software decode.
- On this repository's current implementation, a 1000-frame 4K flow CUDA run completed in about 39 seconds on an RTX 4080 during local validation.

## Troubleshooting

- `FileNotFoundError`: input path is invalid.
- `ValueError: No supported image files found`: input folder has no supported image files.
- `ValueError: Output file extension must be ...`: use `.tif/.tiff/.png` or provide `--output-format`.
- `RuntimeError: ffmpeg is required`: install ffmpeg or use an image folder input.
- `RuntimeError: Failed to extract frames from video ...`: ffmpeg/ffprobe could not decode/probe the video.
- `--palette-color` without `--dither floyd-steinberg` is rejected.

## Legacy Script Wrappers

These remain available for compatibility:

- `python tif_pipeline.py ...`
- `python tif_convert.py ...`
- `python tif_stacker.py ...`
- `python generate_rainbow_tiffs.py ...`

## License

Licensed under GNU GPL v3.0 as provided in [`LICENSE`](./LICENSE).
