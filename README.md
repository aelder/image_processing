# img-timeline

Create a color timeline from every frame of an exported movie.

![CI](https://github.com/aelder/image_processing/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-GPLv3-green)

`img-timeline` is a movie color analysis tool that reads image frame sequences (TIFF, PNG, JPEG, WebP, BMP, GIF) or supported video files (MP4/MOV/MKV/AVI/WebM and more), and writes a timeline image with one row per frame.  
Stretch that timeline vertically in an editor and you get a compact infographic of the movie's color progression over time.

## Sample Output

These examples are generated from the trailer for *The Red Balloon* (1956):

```bash
img-timeline build ./red.mkv ./out/red_flow.png --mode flow --output-format png --progress
img-timeline build ./red.mkv ./out/red_flow_dithered_20260303_154840.png --mode flow --output-format png --dither floyd-steinberg --progress
```

Flow timeline (non-dithered):

![Red Balloon flow timeline](./out/red_flow.png)

16-color Floyd-Steinberg dithered timeline:

![Red Balloon dithered timeline](./out/red_flow_dithered_20260303_154840.png)

## Why this tool

- Turns thousands of frames into one compact visual summary
- Deterministic output (sorted filename order)
- Works as a one-command pipeline or step-by-step workflow
- Supports optional intermediate strip output for debugging/inspection

## Install

### From source (current method)

First, download the repository:

```bash
git clone https://github.com/aelder/image_processing.git
cd image_processing
```

Then install in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[progress]"
```

### Command installed

```bash
img-timeline
```

### Planned PyPI install

Once published to PyPI, install without cloning:

```bash
pip install movie-color-timeline
```

## Quickstart (60 seconds)

1. Put movie frames (for example `.tif`, `.png`, `.jpg`) in a folder, e.g. `./frames` or pass a video file directly.
2. Run:

```bash
img-timeline build ./frames ./out/movie_timeline.tif --progress
```

3. Open `movie_timeline.tif` and scale it up vertically to create a poster-style graphic.

## Cookbook Recipes

### 1) Export TIFF frames from a movie (ffmpeg)

```bash
mkdir -p ./frames
ffmpeg -i ./movie.mp4 -vsync 0 ./frames/%06d.tif
```

### 2) Build a timeline in one command (from frames or directly from a movie)

```bash
mkdir -p ./out
img-timeline build ./frames ./out/movie_timeline.tif --progress
img-timeline build ./movie.mp4 ./out/movie_timeline.png --output-format png --progress
img-timeline build ./movie.mp4 ./out/movie_flow.png --mode flow --output-format png --progress
```

### 3) Keep intermediate strips for inspection

```bash
mkdir -p ./out/strips
img-timeline build ./frames ./out/movie_timeline.tif --intermediate-dir ./out/strips --progress
```

### 4) Legacy two-step workflow

```bash
img-timeline convert ./frames ./out/strips
img-timeline convert ./frames ./out/flow_strips --mode flow --output-format png
img-timeline stack ./out/strips ./out/movie_timeline.tif --progress
```

### 5) Make a poster-style image from the timeline

`movie_timeline.tif` is typically narrow (often 1-2px wide). Scale it up in an editor or with ImageMagick:

```bash
magick ./out/movie_timeline.tif -filter point -resize 2000x30000! ./out/movie_poster.png
```

Use `-filter point` to preserve hard row boundaries without blending.

## CLI Reference

| Command | Purpose | Example |
|---|---|---|
| `img-timeline build <input_path> <output_file>` | Full pipeline from source frames/video to final timeline | `img-timeline build ./frames ./out/timeline.tif` |
| `img-timeline convert <input_folder> <output_folder>` | Convert each frame to a 1px strip (average or flow mode) | `img-timeline convert ./frames ./out/strips --mode flow` |
| `img-timeline stack <input_folder> <output_file>` | Stack strips into final timeline (optional palette dithering) | `img-timeline stack ./out/strips ./out/timeline.tif` |
| `img-timeline generate-demo <output_dir>` | Create synthetic TIFF frames for testing | `img-timeline generate-demo ./demo_frames --count 500 --size 2` |

### Useful flags

- `--progress`: show progress bars (requires `tqdm`)
- `--intermediate-dir <dir>` (for `build`): also write 1px strip images (TIFF/PNG based on output format)
- `--output-format {tiff,png}` (for `build`, `convert`, `stack`): choose output encoding. Default is TIFF unless output filename ends with `.png`.
- `--mode {average,flow}` (for `build`, `convert`): choose row extraction mode. `average` is the default for compatibility.
- `--workers <n>` (for `build`, `convert`): number of worker processes for frame/strip generation. If omitted, `flow` mode auto-parallelizes on larger inputs.
- `--dither {none,floyd-steinberg}` (for `build`, `stack`): apply Floyd-Steinberg dithering to the final output image.
- `--palette-color <#RRGGBB>` (repeatable, for `build`, `stack`): provide exactly 16 custom palette colors for dithering.

## How output is computed

For each input frame:

1. Convert to RGB
2. Build a 1px-high row using selected mode:
   - `average`: one average RGB color for the full row
   - `flow`: one color per column chosen from the most frequent + vibrant quantized column color
3. Assign one output row to that frame

Final timeline dimensions:

- Width: max width across input frames
- Height: number of input frames

### red.mkv flow example

```bash
img-timeline build ./red.mkv ./out/red_flow.png --mode flow --output-format png --progress
```

### 16-color dithering examples

```bash
img-timeline build ./red.mkv ./out/red_flow_dithered.png --mode flow --output-format png --dither floyd-steinberg --progress
img-timeline stack ./out/strips ./out/timeline_dithered.png --output-format png --dither floyd-steinberg
```

Custom palette (exactly 16 entries):

```bash
img-timeline build ./frames ./out/timeline_custom_palette.png --output-format png --dither floyd-steinberg \
  --palette-color '#06080C' --palette-color '#1B1410' --palette-color '#273445' --palette-color '#442D25' \
  --palette-color '#525C47' --palette-color '#684635' --palette-color '#4E6C84' --palette-color '#7C6049' \
  --palette-color '#5C8061' --palette-color '#986644' --palette-color '#847C61' --palette-color '#6E96A6' \
  --palette-color '#B58454' --palette-color '#A4A87C' --palette-color '#C9B27A' --palette-color '#F1E5BE'
```

## Recommended frame naming

Use zero-padded names so lexical sort matches frame order:

- Good: `000001.tif`, `000002.tif`, ...
- Risky: `1.tif`, `10.tif`, `2.tif`

## Legacy script entrypoints (still supported)

- `python3 tif_pipeline.py ...`
- `python3 tif_convert.py ...`
- `python3 tif_stacker.py ...`
- `python3 generate_rainbow_tiffs.py ...`

## Requirements

- Python 3.10+
- NumPy
- Pillow
- Optional: `tqdm` for progress bars

## Troubleshooting

- `FileNotFoundError`: input path is wrong.
- `ValueError: No supported image files found`: input directory has no supported image files.
- `ValueError: Output file extension must be ...`: use `.tif`/`.tiff`/`.png`, or provide `--output-format`.
- `RuntimeError: ffmpeg is required`: install ffmpeg or provide a directory of image frames.
- If `pip install -e .` fails with an externally-managed Python error, install inside a virtual environment (see Install section).

## License

Licensed under GNU GPL v3.0 as provided in [`LICENSE`](./LICENSE).
