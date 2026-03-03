# img-timeline Usage

`img-timeline` is an installable CLI for turning TIFF frame sequences into average-color timeline images.

## Install

```bash
pip install -e .
```

## Commands

### 1) Build timeline (single-step)

```bash
img-timeline build <input_folder> <output_file>
```

Options:
- `--intermediate-dir <dir>`: also write 1px strip TIFFs.
- `--progress`: show progress bars (requires `tqdm`).

### 2) Convert to strips (legacy step)

```bash
img-timeline convert <input_folder> <output_folder>
```

### 3) Stack strips (legacy step)

```bash
img-timeline stack <input_folder> <output_file> [--progress]
```

### 4) Generate demo rainbow frames

```bash
img-timeline generate-demo <output_dir> [--count 500] [--size 2]
```

## Typical workflow

```bash
img-timeline build ./frames ./out/timeline.tif --progress
```

## Legacy script wrappers

These still work for compatibility:

- `python3 tif_pipeline.py ...`
- `python3 tif_convert.py ...`
- `python3 tif_stacker.py ...`
- `python3 generate_rainbow_tiffs.py ...`

## Error behavior

- Raises `FileNotFoundError` if input folder does not exist or is not a directory.
- Raises `ValueError` if no TIFF files are found in the input folder.

## Output semantics

- Output format is RGB TIFF.
- Ordering is sorted lexical filename order.
- Timeline output has one row per source frame.
