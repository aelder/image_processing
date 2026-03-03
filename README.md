Create a color-based timeline from every frame of an exported movie.

This tool takes a TIFF frame sequence (for example, frames exported from a film), calculates the average color of each frame, and stacks those colors in order to produce a timeline image. The result is a compact color summary of the full movie.

If you stretch the 1px-per-frame timeline in an image editor, you can turn it into an average-color poster of the entire film: a color infographic that shows the movie's palette progression over time.

## Preferred workflow (single command)

Use `tif_pipeline.py` to process source frames directly into the final timeline image:

```bash
python3 tif_pipeline.py /path/to/input_frames /path/to/stacked_output.tif
```

Optionally write intermediate 1px strips while also producing the final timeline:

```bash
python3 tif_pipeline.py /path/to/input_frames /path/to/stacked_output.tif --intermediate-dir /path/to/output_strips
```

With progress bars (if `tqdm` is installed):

```bash
python3 tif_pipeline.py /path/to/input_frames /path/to/stacked_output.tif --progress
```

## Legacy two-step workflow (still supported)

Step 1: convert each source frame into a 1-pixel-high strip of the frame's average color.

```bash
python3 tif_convert.py /path/to/input_frames /path/to/output_strips
```

Step 2: stack strips top-to-bottom in filename order.

```bash
python3 tif_stacker.py /path/to/output_strips /path/to/stacked_output.tif
```

With progress bars in step 2:

```bash
python3 tif_stacker.py /path/to/output_strips /path/to/stacked_output.tif --progress
```

## Requirements

- Python 3.10+
- Pillow (`pip install pillow`)
- Optional: tqdm (`pip install tqdm`) for progress bars

## Notes

- Supported input extensions: `.tif`, `.tiff`
- Files are processed in sorted lexical filename order for deterministic output.
- The final timeline has one row per input frame.
