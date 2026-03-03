# TIFF Color Timeline Tool

This project turns a folder of TIFF frames into a single timeline image where each row represents one frame's average color.

## What it does

- Reads source `.tif` / `.tiff` files from an input folder.
- Sorts files lexically by filename for deterministic ordering.
- Computes the average RGB color of each frame.
- Builds a final timeline TIFF where:
  - width = maximum frame width in the input set
  - height = number of input frames
  - each output row = average color of one input frame

## Main command (recommended)

Use the unified pipeline command:

```bash
python3 tif_pipeline.py <input_folder> <output_file>
```

Example:

```bash
python3 tif_pipeline.py ./frames ./out/timeline.tif
```

### Optional flags

- `--intermediate-dir <dir>`
  - Also writes 1-pixel-high TIFF strips for each frame.
- `--progress`
  - Shows progress bars (requires `tqdm` installed).

Example with both options:

```bash
python3 tif_pipeline.py ./frames ./out/timeline.tif --intermediate-dir ./out/strips --progress
```

## Legacy two-step commands (still supported)

Step 1: Convert frames to 1px strips:

```bash
python3 tif_convert.py <input_folder> <output_folder>
```

Step 2: Stack strips into timeline:

```bash
python3 tif_stacker.py <input_folder> <output_file> [--progress]
```

## Requirements

- Python 3.10+
- Pillow

Install dependencies:

```bash
pip install pillow
```

Optional progress bars:

```bash
pip install tqdm
```

## Error behavior

- Raises `FileNotFoundError` if input folder does not exist or is not a directory.
- Raises `ValueError` if no TIFF files are found in the input folder.

## Notes

- Output is always RGB TIFF.
- Ordering is by sorted filename; make sure filenames reflect frame order.
