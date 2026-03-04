import io
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from img_timeline import core as core_module  # noqa: E402
from img_timeline.core import (  # noqa: E402
    DEFAULT_FILMIC_16_PALETTE,
    PALETTE_COLOR_COUNT,
    _parse_video_dimensions,
    _start_video_raw_extractor,
    build_timeline_from_frames,
    convert_to_strips,
    stack_tiff_images,
)


class TestImagePipeline(unittest.TestCase):
    class _FakeExtractorProcess:
        def __init__(self, returncode: int = 0, stderr_text: str = "") -> None:
            self.returncode = returncode
            self.stderr = io.StringIO(stderr_text)

        def poll(self) -> int:
            return self.returncode

    class _FakeRawExtractorProcess:
        def __init__(
            self,
            frame_bytes: bytes,
            returncode: int = 0,
            stderr_bytes: bytes = b"",
        ) -> None:
            self.returncode = returncode
            self.stdout = io.BytesIO(frame_bytes)
            self.stderr = io.BytesIO(stderr_bytes)

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    @staticmethod
    def _default_palette_hex() -> list[str]:
        return [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in DEFAULT_FILMIC_16_PALETTE]

    @staticmethod
    def _encode_solid_frames(
        colors: list[tuple[int, int, int]],
        width: int,
        height: int,
    ) -> bytes:
        frame_size = width * height
        payload = bytearray()
        for red, green, blue in colors:
            payload.extend(bytes([red, green, blue]) * frame_size)
        return bytes(payload)

    def test_pipeline_direct_output_sorted_order(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "stacked" / "timeline.tif"
            source_dir.mkdir(parents=True)

            frames = [
                ("frame_002.tif", (0, 255, 0)),
                ("frame_001.tif", (255, 0, 0)),
                ("frame_003.tif", (0, 0, 255)),
            ]

            for name, color in frames:
                img = Image.new("RGB", (4, 3), color)
                img.save(source_dir / name)

            count = build_timeline_from_frames(source_dir, output_file)
            self.assertEqual(count, 3)
            self.assertTrue(output_file.exists())

            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.size, (4, 3))
                pixels = timeline.load()
                self.assertEqual(pixels[0, 0], (255, 0, 0))
                self.assertEqual(pixels[0, 1], (0, 255, 0))
                self.assertEqual(pixels[0, 2], (0, 0, 255))

    def test_pipeline_with_intermediates(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            inter_dir = root / "intermediate"
            output_file = root / "stacked" / "timeline.tif"
            source_dir.mkdir(parents=True)

            Image.new("RGB", (5, 2), (10, 20, 30)).save(source_dir / "a.tif")
            Image.new("RGB", (3, 2), (40, 50, 60)).save(source_dir / "b.tiff")

            count = build_timeline_from_frames(source_dir, output_file, intermediate_dir=inter_dir)
            self.assertEqual(count, 2)

            strips = sorted(path.name for path in inter_dir.iterdir() if path.is_file())
            self.assertEqual(strips, ["a.tif", "b.tiff"])

            with Image.open(inter_dir / "a.tif") as strip_a:
                self.assertEqual(strip_a.size, (5, 1))
            with Image.open(inter_dir / "b.tiff") as strip_b:
                self.assertEqual(strip_b.size, (3, 1))

            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.size, (5, 2))
                pixels = timeline.load()
                self.assertEqual(pixels[0, 0], (10, 20, 30))
                self.assertEqual(pixels[0, 1], (40, 50, 60))

    def test_compat_convert_cli_behavior(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            out_dir = root / "converted"
            source_dir.mkdir(parents=True)

            Image.new("RGB", (4, 3), (1, 2, 3)).save(source_dir / "z.tif")
            Image.new("RGB", (2, 3), (9, 8, 7)).save(source_dir / "a.tif")

            count = convert_to_strips(source_dir, out_dir)
            self.assertEqual(count, 2)

            files = sorted(path.name for path in out_dir.glob("*.tif"))
            self.assertEqual(files, ["a.tif", "z.tif"])

            with Image.open(out_dir / "a.tif") as strip:
                self.assertEqual(strip.size, (2, 1))
                self.assertEqual(strip.load()[0, 0], (9, 8, 7))

    def test_compat_stack_cli_behavior(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            strips_dir = root / "strips"
            output_file = root / "out" / "stacked.tif"
            strips_dir.mkdir(parents=True)

            Image.new("RGB", (3, 1), (200, 0, 0)).save(strips_dir / "2.tif")
            Image.new("RGB", (3, 1), (0, 200, 0)).save(strips_dir / "1.tif")

            count = stack_tiff_images(strips_dir, output_file)
            self.assertEqual(count, 2)

            with Image.open(output_file) as stacked:
                self.assertEqual(stacked.size, (3, 2))
                px = stacked.load()
                self.assertEqual(px[0, 0], (0, 200, 0))
                self.assertEqual(px[0, 1], (200, 0, 0))

    def test_no_tiffs_error(self):
        with TemporaryDirectory() as temp_dir:
            empty_dir = Path(temp_dir) / "empty"
            empty_dir.mkdir(parents=True)

            with self.assertRaises(ValueError):
                build_timeline_from_frames(empty_dir, Path(temp_dir) / "out.tif")

    def test_accepts_common_image_formats(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "timeline.tif"
            source_dir.mkdir(parents=True)

            Image.new("RGB", (2, 2), (10, 20, 30)).save(source_dir / "001.png")
            Image.new("RGB", (2, 2), (40, 50, 60)).save(source_dir / "002.bmp")

            count = build_timeline_from_frames(source_dir, output_file)
            self.assertEqual(count, 2)

            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.size, (2, 2))
                px = timeline.load()
                self.assertEqual(px[0, 0], (10, 20, 30))
                self.assertEqual(px[0, 1], (40, 50, 60))

    def test_png_output_mode(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "timeline.png"
            inter_dir = root / "intermediate"
            source_dir.mkdir(parents=True)

            Image.new("RGB", (2, 2), (255, 0, 0)).save(source_dir / "a.tif")
            Image.new("RGB", (2, 2), (0, 255, 0)).save(source_dir / "b.tif")

            count = build_timeline_from_frames(
                source_dir,
                output_file,
                intermediate_dir=inter_dir,
                output_format="png",
            )
            self.assertEqual(count, 2)
            self.assertTrue(output_file.exists())

            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.format, "PNG")

            strips = sorted(path.name for path in inter_dir.iterdir() if path.is_file())
            self.assertEqual(strips, ["a.png", "b.png"])

    def test_invalid_output_extension_requires_explicit_format(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            source_dir.mkdir(parents=True)
            Image.new("RGB", (2, 2), (255, 0, 0)).save(source_dir / "a.tif")

            with self.assertRaises(ValueError):
                build_timeline_from_frames(source_dir, root / "timeline.jpg")

    def test_non_rgb_input_handling(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "timeline.tif"
            source_dir.mkdir(parents=True)

            Image.new("L", (2, 2), 128).save(source_dir / "a.tif")
            Image.new("RGBA", (2, 2), (10, 20, 30, 255)).save(source_dir / "b.tif")

            count = build_timeline_from_frames(source_dir, output_file)
            self.assertEqual(count, 2)

            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.size, (2, 2))
                px = timeline.load()
                self.assertEqual(px[0, 0], (128, 128, 128))
                self.assertEqual(px[0, 1], (10, 20, 30))

    def test_package_cli_build(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "out" / "timeline.tif"
            source_dir.mkdir(parents=True)

            Image.new("RGB", (2, 2), (255, 0, 0)).save(source_dir / "001.tif")
            Image.new("RGB", (2, 2), (0, 255, 0)).save(source_dir / "002.tif")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "img_timeline",
                    "build",
                    str(source_dir),
                    str(output_file),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertIn("Processed 2 frame(s)", proc.stdout)
            self.assertTrue(output_file.exists())

    def test_package_cli_convert_flow_mode(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_dir = root / "out"
            source_dir.mkdir(parents=True)

            frame = Image.new("RGB", (2, 3))
            px = frame.load()
            px[0, 0] = (255, 0, 0)
            px[0, 1] = (255, 0, 0)
            px[0, 2] = (0, 255, 0)
            px[1, 0] = (0, 0, 255)
            px[1, 1] = (40, 40, 40)
            px[1, 2] = (0, 0, 255)
            frame.save(source_dir / "001.png")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "img_timeline",
                    "convert",
                    str(source_dir),
                    str(output_dir),
                    "--output-format",
                    "png",
                    "--mode",
                    "flow",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertIn("Processed 1 image file(s)", proc.stdout)
            with Image.open(output_dir / "001.png") as strip:
                self.assertEqual(strip.load()[0, 0], (255, 0, 0))
                self.assertEqual(strip.load()[1, 0], (0, 0, 255))

    def test_package_cli_build_flow_workers(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "out" / "timeline.png"
            source_dir.mkdir(parents=True)

            for idx, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                Image.new("RGB", (4, 3), color).save(source_dir / f"{idx:03d}.png")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "img_timeline",
                    "build",
                    str(source_dir),
                    str(output_file),
                    "--mode",
                    "flow",
                    "--output-format",
                    "png",
                    "--workers",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertIn("Processed 3 frame(s)", proc.stdout)
            self.assertTrue(output_file.exists())

    def test_invalid_mode_rejected(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            out_dir = root / "out"
            source_dir.mkdir(parents=True)
            Image.new("RGB", (2, 2), (1, 2, 3)).save(source_dir / "a.tif")

            with self.assertRaises(ValueError):
                convert_to_strips(source_dir, out_dir, mode="not-a-mode")

            with self.assertRaises(ValueError):
                build_timeline_from_frames(source_dir, root / "timeline.tif", mode="not-a-mode")

    def test_invalid_workers_rejected(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            out_dir = root / "out"
            source_dir.mkdir(parents=True)
            Image.new("RGB", (2, 2), (1, 2, 3)).save(source_dir / "a.tif")

            with self.assertRaises(ValueError):
                convert_to_strips(source_dir, out_dir, workers=0)

            with self.assertRaises(ValueError):
                build_timeline_from_frames(source_dir, root / "timeline.tif", workers=0)

    def test_flow_mode_prefers_frequent_and_vibrant_column_colors(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "timeline.png"
            source_dir.mkdir(parents=True)

            frame = Image.new("RGB", (2, 4))
            px = frame.load()

            col0 = [(200, 10, 10), (200, 10, 10), (200, 10, 10), (10, 200, 10)]
            col1 = [(128, 128, 128), (10, 10, 200), (128, 128, 128), (10, 10, 200)]

            for y in range(4):
                px[0, y] = col0[y]
                px[1, y] = col1[y]

            frame.save(source_dir / "001.png")

            count = build_timeline_from_frames(
                source_dir,
                output_file,
                output_format="png",
                mode="flow",
            )
            self.assertEqual(count, 1)

            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.size, (2, 1))
                out_px = timeline.load()
                self.assertEqual(out_px[0, 0], (200, 10, 10))
                self.assertEqual(out_px[1, 0], (10, 10, 200))

    def test_flow_mode_prefers_brighter_candidate_with_equal_frequency(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "timeline.png"
            source_dir.mkdir(parents=True)

            frame = Image.new("RGB", (1, 2))
            px = frame.load()
            px[0, 0] = (80, 80, 80)
            px[0, 1] = (96, 96, 96)
            frame.save(source_dir / "001.png")

            count = build_timeline_from_frames(
                source_dir,
                output_file,
                output_format="png",
                mode="flow",
            )
            self.assertEqual(count, 1)

            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.load()[0, 0], (96, 96, 96))

    def test_flow_mode_uses_luminance_in_scoring(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "timeline.png"
            source_dir.mkdir(parents=True)

            frame = Image.new("RGB", (1, 4))
            px = frame.load()
            px[0, 0] = (32, 32, 32)
            px[0, 1] = (32, 32, 32)
            px[0, 2] = (96, 96, 96)
            px[0, 3] = (96, 96, 96)
            frame.save(source_dir / "001.png")

            count = build_timeline_from_frames(
                source_dir,
                output_file,
                output_format="png",
                mode="flow",
            )
            self.assertEqual(count, 1)

            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.load()[0, 0], (96, 96, 96))

    def test_flow_mode_deincentivizes_black_when_below_50_percent_frame_dominance(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "timeline.png"
            source_dir.mkdir(parents=True)

            frame = Image.new("RGB", (1, 10))
            px = frame.load()
            for y in range(4):
                px[0, y] = (0, 0, 0)
            for y in range(4, 10):
                px[0, y] = (120, 120, 120)
            frame.save(source_dir / "001.png")

            count = build_timeline_from_frames(
                source_dir,
                output_file,
                output_format="png",
                mode="flow",
            )
            self.assertEqual(count, 1)

            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.load()[0, 0], (120, 120, 120))

    def test_flow_mode_allows_black_when_column_dominates_at_or_above_50_percent(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "timeline.png"
            source_dir.mkdir(parents=True)

            frame = Image.new("RGB", (1, 10))
            px = frame.load()
            for y in range(6):
                px[0, y] = (0, 0, 0)
            for y in range(6, 10):
                px[0, y] = (120, 120, 120)
            frame.save(source_dir / "001.png")

            count = build_timeline_from_frames(
                source_dir,
                output_file,
                output_format="png",
                mode="flow",
            )
            self.assertEqual(count, 1)

            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.load()[0, 0], (0, 0, 0))

    def test_convert_flow_mode_emits_non_uniform_strip(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_dir = root / "strips"
            source_dir.mkdir(parents=True)

            frame = Image.new("RGB", (2, 3))
            px = frame.load()
            px[0, 0] = (255, 0, 0)
            px[0, 1] = (255, 0, 0)
            px[0, 2] = (0, 255, 0)
            px[1, 0] = (0, 0, 255)
            px[1, 1] = (40, 40, 40)
            px[1, 2] = (0, 0, 255)
            frame.save(source_dir / "001.png")

            count = convert_to_strips(source_dir, output_dir, output_format="png", mode="flow")
            self.assertEqual(count, 1)

            with Image.open(output_dir / "001.png") as strip:
                self.assertEqual(strip.size, (2, 1))
                strip_px = strip.load()
                self.assertEqual(strip_px[0, 0], (255, 0, 0))
                self.assertEqual(strip_px[1, 0], (0, 0, 255))

    def test_flow_parallel_matches_single_worker_output(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            single_out = root / "single.png"
            multi_out = root / "multi.png"
            source_dir.mkdir(parents=True)

            for index in range(12):
                frame = Image.new("RGB", (5, 4))
                px = frame.load()
                for x in range(5):
                    for y in range(4):
                        px[x, y] = ((index * 17 + x * 50) % 256, (y * 70) % 256, (x * y * 40) % 256)
                frame.save(source_dir / f"{index:03d}.png")

            build_timeline_from_frames(
                source_dir,
                single_out,
                mode="flow",
                output_format="png",
                workers=1,
            )
            build_timeline_from_frames(
                source_dir,
                multi_out,
                mode="flow",
                output_format="png",
                workers=4,
            )

            with Image.open(single_out) as single_img, Image.open(multi_out) as multi_img:
                self.assertEqual(single_img.size, multi_img.size)
                self.assertEqual(single_img.tobytes(), multi_img.tobytes())

    def test_flow_cuda_strip_matches_cpu_strip(self):
        if core_module.cp is None:
            self.skipTest("CuPy not installed")

        frame = Image.new("RGB", (4, 5))
        px = frame.load()
        for x in range(4):
            for y in range(5):
                if x == 0:
                    px[x, y] = (220, 20, 20) if y < 4 else (40, 40, 40)
                elif x == 1:
                    px[x, y] = (20, 220, 20) if y != 2 else (10, 10, 10)
                elif x == 2:
                    px[x, y] = (20, 20, 220) if y < 3 else (15, 15, 15)
                else:
                    px[x, y] = (220, 180, 30) if y < 4 else (0, 0, 0)

        try:
            cpu_strip = core_module._flow_strip_cpu(frame)
            cuda_strip = core_module._flow_strip_cuda(frame)
        except RuntimeError as error:
            self.skipTest(f"CUDA unavailable in environment: {error}")

        self.assertEqual(cpu_strip.size, cuda_strip.size)
        self.assertEqual(cpu_strip.tobytes(), cuda_strip.tobytes())

    def test_build_timeline_flow_cuda_matches_cpu_output(self):
        if core_module.cp is None:
            self.skipTest("CuPy not installed")

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            cpu_out = root / "cpu.png"
            cuda_out = root / "cuda.png"
            source_dir.mkdir(parents=True)

            for index in range(6):
                frame = Image.new("RGB", (6, 5))
                px = frame.load()
                for x in range(6):
                    for y in range(5):
                        px[x, y] = (
                            (index * 31 + x * 17 + y * 7) % 256,
                            (index * 29 + y * 43 + x * 5) % 256,
                            (index * 11 + x * y * 19) % 256,
                        )
                frame.save(source_dir / f"{index:03d}.png")

            build_timeline_from_frames(
                source_dir,
                cpu_out,
                output_format="png",
                mode="flow",
                workers=1,
                use_cuda=False,
            )
            try:
                build_timeline_from_frames(
                    source_dir,
                    cuda_out,
                    output_format="png",
                    mode="flow",
                    workers=1,
                    use_cuda=True,
                )
            except RuntimeError as error:
                self.skipTest(f"CUDA unavailable in environment: {error}")

            with Image.open(cpu_out) as cpu_img, Image.open(cuda_out) as cuda_img:
                self.assertEqual(cpu_img.size, cuda_img.size)
                self.assertEqual(cpu_img.tobytes(), cuda_img.tobytes())

    def test_parse_video_dimensions_from_default_ffprobe_output(self):
        width, height = _parse_video_dimensions("3840\n1600\n", Path("clip.mkv"))
        self.assertEqual((width, height), (3840, 1600))

    def test_parse_video_dimensions_from_csv_with_trailing_separator(self):
        width, height = _parse_video_dimensions("3840x1600x", Path("clip.mkv"))
        self.assertEqual((width, height), (3840, 1600))

    def test_parse_video_dimensions_rejects_invalid_output(self):
        with self.assertRaises(RuntimeError):
            _parse_video_dimensions("not-dimensions", Path("clip.mkv"))

    def test_build_flow_mode_with_video_input_smoke(self):
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            self.skipTest("ffmpeg not installed")

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_file = root / "clip.mp4"
            output_file = root / "timeline.png"

            subprocess.run(
                [
                    ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=4x2:rate=1:duration=3",
                    "-frames:v",
                    "3",
                    str(video_file),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            count = build_timeline_from_frames(
                video_file,
                output_file,
                output_format="png",
                mode="flow",
            )
            self.assertEqual(count, 3)
            self.assertTrue(output_file.exists())

            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.size, (4, 3))

    def test_video_in_memory_path_avoids_temp_frame_files(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_file = root / "clip.mkv"
            output_file = root / "nested" / "out" / "timeline.png"
            video_file.write_bytes(b"dummy")

            raw_bytes = self._encode_solid_frames([(10, 20, 30), (40, 50, 60)], width=2, height=2)
            with patch("img_timeline.core._probe_video_dimensions", return_value=(2, 2)):
                with patch(
                    "img_timeline.core._start_video_raw_extractor",
                    return_value=self._FakeRawExtractorProcess(raw_bytes),
                ):
                    count = build_timeline_from_frames(
                        video_file,
                        output_file,
                        output_format="png",
                        mode="flow",
                    )

            self.assertEqual(count, 2)
            self.assertTrue(output_file.exists())
            self.assertEqual(list(output_file.parent.glob("tmp*")), [])

    def test_video_in_memory_backpressure_limits_inflight_work(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_file = root / "clip.mkv"
            output_file = root / "out" / "timeline.png"
            video_file.write_bytes(b"dummy")

            frame_count = 12
            colors = [
                (idx * 10 % 255, idx * 20 % 255, idx * 30 % 255)
                for idx in range(frame_count)
            ]
            raw_bytes = self._encode_solid_frames(colors, width=2, height=2)
            max_outstanding = 0
            all_futures = []
            worker_count = 4
            expected_cap = worker_count * 2

            class _FakeFuture:
                def __init__(self, value):
                    self._value = value
                    self._done = False

                def result(self):
                    return self._value

                def done(self):
                    return self._done

            class _FakeExecutor:
                def __init__(self, *args, **kwargs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def submit(self, fn, *args, **kwargs):
                    nonlocal max_outstanding
                    value = fn(*args, **kwargs)
                    future = _FakeFuture(value)
                    all_futures.append(future)
                    outstanding = sum(1 for item in all_futures if not item.done())
                    max_outstanding = max(max_outstanding, outstanding)
                    return future

            def fake_wait(futures, timeout=None, return_when=None):
                pending = [future for future in futures if not future.done()]
                if not pending:
                    return set(), set()
                pending[0]._done = True
                done = {pending[0]}
                remaining = set(future for future in futures if not future.done())
                return done, remaining

            with patch("img_timeline.core._probe_video_dimensions", return_value=(2, 2)):
                with patch(
                    "img_timeline.core._start_video_raw_extractor",
                    return_value=self._FakeRawExtractorProcess(raw_bytes),
                ):
                    with patch(
                        "img_timeline.core.concurrent.futures.ProcessPoolExecutor",
                        _FakeExecutor,
                    ):
                        with patch(
                            "img_timeline.core.concurrent.futures.wait",
                            side_effect=fake_wait,
                        ):
                            count = build_timeline_from_frames(
                                video_file,
                                output_file,
                                output_format="png",
                                mode="flow",
                                workers=worker_count,
                            )

            self.assertEqual(count, frame_count)
            self.assertLessEqual(max_outstanding, expected_cap)
            self.assertEqual(max_outstanding, expected_cap)

    def test_video_flow_multi_worker_preserves_frame_order(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_file = root / "clip.mkv"
            output_file = root / "out" / "timeline.png"
            video_file.write_bytes(b"dummy")
            expected = [(200, 10, 10), (10, 200, 10), (10, 10, 200), (120, 120, 30)]

            raw_bytes = self._encode_solid_frames(expected, width=3, height=2)
            with patch("img_timeline.core._probe_video_dimensions", return_value=(3, 2)):
                with patch(
                    "img_timeline.core._start_video_raw_extractor",
                    return_value=self._FakeRawExtractorProcess(raw_bytes),
                ):
                    count = build_timeline_from_frames(
                        video_file,
                        output_file,
                        output_format="png",
                        mode="flow",
                        workers=3,
                    )

            self.assertEqual(count, len(expected))
            with Image.open(output_file) as timeline:
                pixels = timeline.load()
                self.assertEqual(timeline.size, (3, len(expected)))
                for row_index, color in enumerate(expected):
                    self.assertEqual(pixels[0, row_index], color)

    def test_video_flow_single_worker_preserves_frame_order(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_file = root / "clip.mkv"
            output_file = root / "out" / "timeline.png"
            video_file.write_bytes(b"dummy")
            expected = [(200, 10, 10), (10, 200, 10), (10, 10, 200), (120, 120, 30)]

            raw_bytes = self._encode_solid_frames(expected, width=3, height=2)
            with patch("img_timeline.core._probe_video_dimensions", return_value=(3, 2)):
                with patch(
                    "img_timeline.core._start_video_raw_extractor",
                    return_value=self._FakeRawExtractorProcess(raw_bytes),
                ):
                    count = build_timeline_from_frames(
                        video_file,
                        output_file,
                        output_format="png",
                        mode="flow",
                        workers=1,
                    )

            self.assertEqual(count, len(expected))
            with Image.open(output_file) as timeline:
                pixels = timeline.load()
                self.assertEqual(timeline.size, (3, len(expected)))
                for row_index, color in enumerate(expected):
                    self.assertEqual(pixels[0, row_index], color)

    def test_video_flow_cuda_single_worker_uses_batch_helper(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_file = root / "clip.mkv"
            output_file = root / "out" / "timeline.png"
            video_file.write_bytes(b"dummy")

            expected = [(200, 10, 10), (10, 200, 10), (10, 10, 200), (120, 120, 30), (80, 20, 220)]
            raw_bytes = self._encode_solid_frames(expected, width=2, height=2)
            batch_sizes: list[int] = []

            def fake_batch_helper(
                frame_batch: list[bytes],
                width: int,
                height: int,
                use_cuda: bool,
            ):
                batch_sizes.append(len(frame_batch))
                rows: list[tuple[int, bytes]] = []
                for frame_data in frame_batch:
                    red, green, blue = frame_data[0], frame_data[1], frame_data[2]
                    rows.append((width, bytes([red, green, blue]) * width))
                return rows

            with patch("img_timeline.core._probe_video_dimensions", return_value=(2, 2)):
                with patch(
                    "img_timeline.core._start_video_raw_extractor",
                    return_value=self._FakeRawExtractorProcess(raw_bytes),
                ):
                    with patch(
                        "img_timeline.core._build_flow_strips_from_frame_batch_bytes",
                        side_effect=fake_batch_helper,
                    ):
                        count = build_timeline_from_frames(
                            video_file,
                            output_file,
                            output_format="png",
                            mode="flow",
                            workers=1,
                            use_cuda=True,
                        )

            self.assertEqual(count, len(expected))
            self.assertTrue(any(size > 1 for size in batch_sizes))
            with Image.open(output_file) as timeline:
                pixels = timeline.load()
                self.assertEqual(timeline.size, (2, len(expected)))
                for row_index, color in enumerate(expected):
                    self.assertEqual(pixels[0, row_index], color)

    def test_video_streaming_cleans_temp_on_processing_failure(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_file = root / "clip.mkv"
            output_file = root / "out" / "timeline.png"
            video_file.write_bytes(b"dummy")

            raw_bytes = self._encode_solid_frames([(255, 0, 0)], width=2, height=2)

            def fake_build_strip(
                _frame_data: bytes,
                _width: int,
                _height: int,
                _mode: str,
                use_cuda: bool = False,
            ):
                raise RuntimeError("Synthetic frame processing failure")

            with patch("img_timeline.core._probe_video_dimensions", return_value=(2, 2)):
                with patch(
                    "img_timeline.core._start_video_raw_extractor",
                    return_value=self._FakeRawExtractorProcess(raw_bytes),
                ):
                    with patch(
                        "img_timeline.core._build_strip_from_frame_bytes",
                        side_effect=fake_build_strip,
                    ):
                        with self.assertRaises(RuntimeError):
                            build_timeline_from_frames(
                                video_file,
                                output_file,
                                output_format="png",
                                mode="flow",
                                workers=1,
                            )

            self.assertEqual(list(output_file.parent.glob("tmp*")), [])

    def test_video_streaming_cleans_temp_on_ffmpeg_failure(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_file = root / "clip.mkv"
            output_file = root / "out" / "timeline.png"
            video_file.write_bytes(b"dummy")

            failing_extractor = self._FakeRawExtractorProcess(
                frame_bytes=b"",
                returncode=1,
                stderr_bytes=b"ffmpeg failed",
            )

            with patch("img_timeline.core._probe_video_dimensions", return_value=(2, 2)):
                with patch(
                    "img_timeline.core._start_video_raw_extractor",
                    return_value=failing_extractor,
                ):
                    with self.assertRaises(RuntimeError):
                        build_timeline_from_frames(
                            video_file,
                            output_file,
                            output_format="png",
                            mode="flow",
                        )

            self.assertEqual(list(output_file.parent.glob("tmp*")), [])

    def test_video_disk_streaming_fallback_when_ffprobe_missing(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_file = root / "clip.mkv"
            output_file = root / "out" / "timeline.png"
            video_file.write_bytes(b"dummy")
            captured_frame_dir: Path | None = None

            def fake_start(_video_file: Path, frame_dir: Path, use_cuda: bool = False):
                nonlocal captured_frame_dir
                captured_frame_dir = frame_dir
                Image.new("RGB", (2, 2), (10, 20, 30)).save(frame_dir / "000000001.png")
                Image.new("RGB", (2, 2), (40, 50, 60)).save(frame_dir / "000000002.png")
                return self._FakeExtractorProcess()

            with patch(
                "img_timeline.core._probe_video_dimensions",
                side_effect=RuntimeError("ffprobe is required for in-memory video processing."),
            ):
                with patch(
                    "img_timeline.core._start_video_frame_extractor",
                    side_effect=fake_start,
                ):
                    count = build_timeline_from_frames(
                        video_file,
                        output_file,
                        output_format="png",
                        mode="flow",
                    )

            self.assertEqual(count, 2)
            self.assertIsNotNone(captured_frame_dir)
            assert captured_frame_dir is not None
            self.assertEqual(captured_frame_dir.parent, output_file.parent)
            self.assertFalse(captured_frame_dir.exists())

    def test_build_dither_default_palette_png(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "timeline.png"
            source_dir.mkdir(parents=True)

            colors = [
                (255, 30, 10),
                (10, 255, 80),
                (30, 60, 255),
                (220, 220, 40),
            ]
            for idx, color in enumerate(colors):
                Image.new("RGB", (4, 2), color).save(source_dir / f"{idx:03d}.png")

            count = build_timeline_from_frames(
                source_dir,
                output_file,
                output_format="png",
                dither="floyd-steinberg",
            )
            self.assertEqual(count, 4)

            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.mode, "P")
                color_count = len(timeline.getcolors(maxcolors=300000) or [])
                self.assertLessEqual(color_count, PALETTE_COLOR_COUNT)

    def test_convert_flow_mode_uses_cuda_path_when_requested(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_dir = root / "out"
            source_dir.mkdir(parents=True)
            Image.new("RGB", (2, 2), (9, 8, 7)).save(source_dir / "001.png")

            with patch("img_timeline.core._flow_strip_cpu", side_effect=AssertionError("cpu path")):
                with patch(
                    "img_timeline.core._flow_strip_cuda",
                    return_value=Image.new("RGB", (2, 1), (1, 2, 3)),
                ):
                    count = convert_to_strips(
                        source_dir,
                        output_dir,
                        output_format="png",
                        mode="flow",
                        use_cuda=True,
                    )

            self.assertEqual(count, 1)
            with Image.open(output_dir / "001.png") as strip:
                self.assertEqual(strip.size, (2, 1))
                self.assertEqual(strip.load()[0, 0], (1, 2, 3))

    def test_start_video_raw_extractor_adds_nvdec_flags_with_cuda(self):
        commands: list[list[str]] = []

        class _FakePopenProcess:
            stdout = None
            stderr = None

        def fake_popen(command, **kwargs):
            commands.append(command)
            return _FakePopenProcess()

        with patch("img_timeline.core.shutil.which", return_value="ffmpeg"):
            with patch("img_timeline.core.subprocess.Popen", side_effect=fake_popen):
                _start_video_raw_extractor(Path("clip.mp4"), use_cuda=True)

        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertIn("-hwaccel", command)
        self.assertIn("cuda", command)
        self.assertIn("-vf", command)
        self.assertIn("hwdownload,format=rgb24", command)

    def test_stack_dither_default_palette_png(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            strips_dir = root / "strips"
            output_file = root / "stacked.png"
            strips_dir.mkdir(parents=True)

            Image.new("RGB", (3, 1), (255, 0, 0)).save(strips_dir / "001.png")
            Image.new("RGB", (3, 1), (0, 255, 0)).save(strips_dir / "002.png")
            Image.new("RGB", (3, 1), (0, 0, 255)).save(strips_dir / "003.png")

            count = stack_tiff_images(
                strips_dir,
                output_file,
                output_format="png",
                dither="floyd-steinberg",
            )
            self.assertEqual(count, 3)

            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.mode, "P")
                self.assertEqual(timeline.size, (3, 3))

    def test_build_dither_tiff_keeps_paletted_output(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "timeline.tif"
            source_dir.mkdir(parents=True)

            Image.new("RGB", (3, 2), (200, 10, 10)).save(source_dir / "001.png")
            Image.new("RGB", (3, 2), (10, 200, 10)).save(source_dir / "002.png")

            count = build_timeline_from_frames(
                source_dir,
                output_file,
                output_format="tiff",
                dither="floyd-steinberg",
            )
            self.assertEqual(count, 2)

            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.mode, "P")

    def test_build_dither_custom_palette_success(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "timeline.png"
            source_dir.mkdir(parents=True)

            Image.new("RGB", (2, 2), (255, 0, 0)).save(source_dir / "001.png")
            Image.new("RGB", (2, 2), (0, 255, 0)).save(source_dir / "002.png")
            Image.new("RGB", (2, 2), (0, 0, 255)).save(source_dir / "003.png")

            custom_palette = self._default_palette_hex()
            count = build_timeline_from_frames(
                source_dir,
                output_file,
                output_format="png",
                dither="floyd-steinberg",
                palette_colors=custom_palette,
            )
            self.assertEqual(count, 3)

            allowed = set(DEFAULT_FILMIC_16_PALETTE)
            with Image.open(output_file) as timeline:
                colors = timeline.convert("RGB").getcolors(maxcolors=300000) or []
                used_colors = {color for _, color in colors}
                self.assertTrue(used_colors.issubset(allowed))

    def test_build_dither_palette_requires_exact_count(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            source_dir.mkdir(parents=True)
            Image.new("RGB", (2, 2), (255, 0, 0)).save(source_dir / "001.png")

            with self.assertRaises(ValueError):
                build_timeline_from_frames(
                    source_dir,
                    root / "timeline.png",
                    output_format="png",
                    dither="floyd-steinberg",
                    palette_colors=self._default_palette_hex()[:-1],
                )

    def test_build_dither_palette_rejects_duplicates(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            source_dir.mkdir(parents=True)
            Image.new("RGB", (2, 2), (255, 0, 0)).save(source_dir / "001.png")

            duplicate_palette = self._default_palette_hex()
            duplicate_palette[-1] = duplicate_palette[0]
            with self.assertRaises(ValueError):
                build_timeline_from_frames(
                    source_dir,
                    root / "timeline.png",
                    output_format="png",
                    dither="floyd-steinberg",
                    palette_colors=duplicate_palette,
                )

    def test_build_dither_palette_rejects_invalid_hex(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            source_dir.mkdir(parents=True)
            Image.new("RGB", (2, 2), (255, 0, 0)).save(source_dir / "001.png")

            bad_palette = self._default_palette_hex()
            bad_palette[3] = "#GG0000"
            with self.assertRaises(ValueError):
                build_timeline_from_frames(
                    source_dir,
                    root / "timeline.png",
                    output_format="png",
                    dither="floyd-steinberg",
                    palette_colors=bad_palette,
                )

    def test_build_dither_palette_rejects_too_similar_colors(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            source_dir.mkdir(parents=True)
            Image.new("RGB", (2, 2), (255, 0, 0)).save(source_dir / "001.png")

            similar_palette = self._default_palette_hex()
            similar_palette[0] = "#202020"
            similar_palette[1] = "#2A2A2A"
            with self.assertRaises(ValueError):
                build_timeline_from_frames(
                    source_dir,
                    root / "timeline.png",
                    output_format="png",
                    dither="floyd-steinberg",
                    palette_colors=similar_palette,
                )

    def test_palette_colors_without_dither_rejected(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            source_dir.mkdir(parents=True)
            Image.new("RGB", (2, 2), (255, 0, 0)).save(source_dir / "001.png")

            with self.assertRaises(ValueError):
                build_timeline_from_frames(
                    source_dir,
                    root / "timeline.png",
                    output_format="png",
                    dither="none",
                    palette_colors=self._default_palette_hex(),
                )

    def test_package_cli_build_with_dither(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "out" / "timeline.png"
            source_dir.mkdir(parents=True)
            Image.new("RGB", (2, 2), (255, 0, 0)).save(source_dir / "001.png")
            Image.new("RGB", (2, 2), (0, 255, 0)).save(source_dir / "002.png")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "img_timeline",
                    "build",
                    str(source_dir),
                    str(output_file),
                    "--output-format",
                    "png",
                    "--dither",
                    "floyd-steinberg",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertIn("Processed 2 frame(s)", proc.stdout)
            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.mode, "P")

    def test_package_cli_stack_with_custom_palette_dither(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            strips_dir = root / "strips"
            output_file = root / "out" / "timeline.png"
            strips_dir.mkdir(parents=True)
            Image.new("RGB", (3, 1), (255, 0, 0)).save(strips_dir / "001.png")
            Image.new("RGB", (3, 1), (0, 0, 255)).save(strips_dir / "002.png")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            command = [
                sys.executable,
                "-m",
                "img_timeline",
                "stack",
                str(strips_dir),
                str(output_file),
                "--output-format",
                "png",
                "--dither",
                "floyd-steinberg",
            ]
            for color in self._default_palette_hex():
                command.extend(["--palette-color", color])

            proc = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertIn("Stacked 2 image file(s)", proc.stdout)
            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.mode, "P")


if __name__ == "__main__":
    unittest.main()
