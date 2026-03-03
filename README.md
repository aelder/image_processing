# img-timeline

Create a color timeline from every frame of an exported movie.

![CI](https://github.com/aelder/image_processing/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-GPLv3-green)

`img-timeline` is a movie color analysis tool that reads a TIFF frame sequence, computes the average color per frame, and writes a timeline image with one row per frame.  
Stretch that timeline vertically in an editor and you get an average color film poster: a compact infographic of the movie's color palette progression over time.

## Sample Output

Generated from 500 synthetic rainbow frames using:

```bash
img-timeline generate-demo ./rainbow --count 500 --size 2
img-timeline build ./rainbow ./rainbow_timeline.tif
```

![Sample timeline output](./assets/sample_timeline_preview.png)

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

1. Put movie frames (`.tif` / `.tiff`) in a folder, e.g. `./frames`.
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

### 2) Build a timeline in one command

```bash
mkdir -p ./out
img-timeline build ./frames ./out/movie_timeline.tif --progress
```

### 3) Keep intermediate strips for inspection

```bash
mkdir -p ./out/strips
img-timeline build ./frames ./out/movie_timeline.tif --intermediate-dir ./out/strips --progress
```

### 4) Legacy two-step workflow

```bash
img-timeline convert ./frames ./out/strips
img-timeline stack ./out/strips ./out/movie_timeline.tif --progress
```

### 5) Generate demo data and run end-to-end

```bash
img-timeline generate-demo ./demo_frames --count 500 --size 2
img-timeline build ./demo_frames ./out/demo_timeline.tif
```

### 6) Make a poster-style image from the timeline

`movie_timeline.tif` is typically narrow (often 1-2px wide). Scale it up in an editor or with ImageMagick:

```bash
magick ./out/movie_timeline.tif -filter point -resize 2000x30000! ./out/movie_poster.png
```

Use `-filter point` to preserve hard row boundaries without blending.

## CLI Reference

| Command | Purpose | Example |
|---|---|---|
| `img-timeline build <input_folder> <output_file>` | Full pipeline from source frames to final timeline | `img-timeline build ./frames ./out/timeline.tif` |
| `img-timeline convert <input_folder> <output_folder>` | Convert each frame to a 1px strip of its average color | `img-timeline convert ./frames ./out/strips` |
| `img-timeline stack <input_folder> <output_file>` | Stack strips into final timeline | `img-timeline stack ./out/strips ./out/timeline.tif` |
| `img-timeline generate-demo <output_dir>` | Create synthetic rainbow TIFF frames for testing | `img-timeline generate-demo ./demo_frames --count 500 --size 2` |

### Useful flags

- `--progress`: show progress bars (requires `tqdm`)
- `--intermediate-dir <dir>` (for `build`): also write 1px strip TIFFs

## How output is computed

For each input frame:

1. Convert to RGB
2. Compute average RGB color
3. Assign one output row to that color

Final timeline dimensions:

- Width: max width across input frames
- Height: number of input frames

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
- Pillow
- Optional: `tqdm` for progress bars

## Troubleshooting

- `FileNotFoundError`: input folder path is wrong or not a directory.
- `ValueError: No TIFF files found`: input directory has no `.tif` / `.tiff`.
- If `pip install -e .` fails with an externally-managed Python error, install inside a virtual environment (see Install section).

## License

Licensed under GNU GPL v3.0 as provided in [`LICENSE`](./LICENSE).
