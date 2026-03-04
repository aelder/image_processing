# img-timeline Usage

`img-timeline` is an installable CLI for turning image frame sequences or videos into timeline images.

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
- `--mode {average,flow}`: choose row extraction mode (`average` default).
- `--workers <n>`: number of worker processes. If omitted, `flow` mode auto-parallelizes for larger inputs.
- `--cuda`: enable CUDA acceleration for `flow` mode (NVDEC decode + CuPy strip compute when available).
- `--dither {none,floyd-steinberg}`: apply 16-color Floyd-Steinberg dithering to final output.
- `--palette-color <#RRGGBB>` (repeat 16 times): custom palette for dithering.
- `--progress`: show progress bars (requires `tqdm`).

For video inputs, frames are decoded and consumed in memory through a bounded frame buffer before
timeline row generation. This introduces backpressure so extraction/decode cannot run far ahead of
processing. If `ffprobe` is unavailable, processing falls back to temporary on-disk frame extraction.
With `--cuda` in `flow` mode, decode and strip compute are pipelined for overlap. If NVDEC does not
support the source stream profile, decode falls back to software while CUDA strip computation stays on.

### 2) Convert to strips (legacy step)

```bash
img-timeline convert <input_folder> <output_folder> [--output-format tiff|png] [--mode average|flow] [--workers N] [--cuda]
```

### 3) Stack strips (legacy step)

```bash
img-timeline stack <input_folder> <output_file> [--output-format tiff|png] [--dither none|floyd-steinberg] [--palette-color #RRGGBB ...] [--progress]
```

### 4) Generate demo rainbow frames

```bash
img-timeline generate-demo <output_dir> [--count 500] [--size 2]
```

## Typical workflow

```bash
img-timeline build ./frames ./out/timeline.tif --progress
img-timeline build ./movie.mp4 ./out/timeline.png --output-format png --progress
img-timeline build ./red.mkv ./out/red_flow.png --mode flow --output-format png --progress
img-timeline build ./red.mkv ./out/red_flow_cuda.png --mode flow --output-format png --cuda --progress
img-timeline build ./red.mkv ./out/red_flow_dithered.png --mode flow --output-format png --dither floyd-steinberg --progress
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
- Raises `RuntimeError` if ffmpeg extraction fails.

## Output semantics

- Output format is RGB TIFF/PNG by default.
- With `--dither floyd-steinberg`, final `build`/`stack` output is indexed paletted image (`P`) using 16 colors.
- Ordering is sorted lexical filename order.
- Timeline output has one row per source frame.

## Performance notes

- `flow` + `--cuda` uses a fixed-bin GPU histogram/reduction path that preserves current flow scoring behavior.
- CUDA flow accumulation uses a dedicated kernel for histogram/sum passes (instead of scatter-add primitives).
- Single-worker CUDA video flow processes frames in bounded batches to improve throughput without changing output ordering.
- For unsupported hardware decode cases, fallback decode is automatic and does not disable CUDA strip compute.
