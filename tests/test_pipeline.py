import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from img_timeline.core import (  # noqa: E402
    DEFAULT_FILMIC_16_PALETTE,
    PALETTE_COLOR_COUNT,
    build_timeline_from_frames,
    convert_to_strips,
    stack_tiff_images,
)


class TestImagePipeline(unittest.TestCase):
    @staticmethod
    def _default_palette_hex() -> list[str]:
        return [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in DEFAULT_FILMIC_16_PALETTE]

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
