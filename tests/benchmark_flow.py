import argparse
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from img_timeline.core import build_timeline_from_frames  # noqa: E402


def _generate_frames(
    output_dir: Path,
    frame_count: int,
    width: int,
    height: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(frame_count):
        frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(frame, mode="RGB").save(output_dir / f"{index:06d}.png")


def _run_case(
    label: str,
    input_dir: Path,
    output_dir: Path,
    frame_count: int,
    workers: int,
    use_cuda: bool,
    warmup_runs: int,
    measured_runs: int,
) -> None:
    for warmup_index in range(warmup_runs):
        build_timeline_from_frames(
            input_dir,
            output_dir / f"{label}_warmup_{warmup_index}.png",
            output_format="png",
            mode="flow",
            workers=workers,
            use_cuda=use_cuda,
        )

    timings: list[float] = []
    for run_index in range(measured_runs):
        start = time.perf_counter()
        build_timeline_from_frames(
            input_dir,
            output_dir / f"{label}_run_{run_index}.png",
            output_format="png",
            mode="flow",
            workers=workers,
            use_cuda=use_cuda,
        )
        timings.append(time.perf_counter() - start)

    avg_seconds = sum(timings) / len(timings)
    fps = frame_count / avg_seconds
    ms_per_frame = (avg_seconds * 1000.0) / frame_count
    print(
        f"{label}: avg={avg_seconds:.4f}s "
        f"({fps:.2f} frames/s, {ms_per_frame:.3f} ms/frame) over {measured_runs} run(s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark flow timeline processing.")
    parser.add_argument("--frames", type=int, default=120, help="Number of input frames.")
    parser.add_argument("--width", type=int, default=640, help="Frame width.")
    parser.add_argument("--height", type=int, default=360, help="Frame height.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for synthetic frames.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup run count per case.")
    parser.add_argument("--runs", type=int, default=3, help="Measured run count per case.")
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Worker count for CPU multi-worker case.",
    )
    parser.add_argument(
        "--skip-cuda",
        action="store_true",
        help="Skip CUDA benchmark case.",
    )
    args = parser.parse_args()

    if args.frames <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("frames, width, and height must be greater than 0")
    if args.warmup_runs < 0 or args.runs <= 0:
        raise ValueError("warmup-runs must be >= 0 and runs must be > 0")
    if args.cpu_workers <= 0:
        raise ValueError("cpu-workers must be greater than 0")

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        frames_dir = root / "frames"
        outputs_dir = root / "outputs"
        _generate_frames(frames_dir, args.frames, args.width, args.height, args.seed)

        print(
            f"Benchmark input: {args.frames} frame(s), "
            f"{args.width}x{args.height}, seed={args.seed}"
        )
        _run_case(
            label="cpu_single_worker",
            input_dir=frames_dir,
            output_dir=outputs_dir,
            frame_count=args.frames,
            workers=1,
            use_cuda=False,
            warmup_runs=args.warmup_runs,
            measured_runs=args.runs,
        )
        _run_case(
            label=f"cpu_multi_worker_{args.cpu_workers}",
            input_dir=frames_dir,
            output_dir=outputs_dir,
            frame_count=args.frames,
            workers=args.cpu_workers,
            use_cuda=False,
            warmup_runs=args.warmup_runs,
            measured_runs=args.runs,
        )

        if not args.skip_cuda:
            try:
                _run_case(
                    label="cuda_single_worker_batched",
                    input_dir=frames_dir,
                    output_dir=outputs_dir,
                    frame_count=args.frames,
                    workers=1,
                    use_cuda=True,
                    warmup_runs=args.warmup_runs,
                    measured_runs=args.runs,
                )
            except RuntimeError as error:
                print(f"cuda_single_worker_batched: skipped ({error})")


if __name__ == "__main__":
    main()
