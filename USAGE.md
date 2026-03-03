# img-timeline Usage

`img-timeline` is an installable CLI for turning image frame sequences or videos into average-color timeline images.

## Install

```bash
pip install -e .
```

## Commands

### 1) Build timeline (single-step)

```bash
img-timeline build <input_path> <output_file>
```

Options:
- `--intermediate-dir <dir>`: also write 1px strip images.
- `--output-format {tiff,png}`: choose output format (default TIFF unless output filename is `.png`).
- `--progress`: show progress bars (requires `tqdm`).

### 2) Convert to strips (legacy step)

```bash
img-timeline convert <input_folder> <output_folder> [--output-format tiff|png]
```

### 3) Stack strips (legacy step)

```bash
img-timeline stack <input_folder> <output_file> [--output-format tiff|png] [--progress]
```

### 4) Generate demo rainbow frames

```bash
img-timeline generate-demo <output_dir> [--count 500] [--size 2]
```

## Typical workflow

```bash
img-timeline build ./frames ./out/timeline.tif --progress
img-timeline build ./movie.mp4 ./out/timeline.png --output-format png --progress
```

## Legacy script wrappers

These still work for compatibility:

- `python3 tif_pipeline.py ...`
- `python3 tif_convert.py ...`
- `python3 tif_stacker.py ...`
- `python3 generate_rainbow_tiffs.py ...`

## Error behavior

- Raises `FileNotFoundError` if the input path does not exist.
- Raises `ValueError` if no supported image files are found in an input folder.
- Raises `RuntimeError` for video input if `ffmpeg` is not installed.

## Output semantics

- Output format is RGB TIFF or PNG.
- Ordering is sorted lexical filename order.
- Timeline output has one row per source frame.
