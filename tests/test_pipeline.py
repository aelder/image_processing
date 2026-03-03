import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from pipeline_core import build_timeline_from_frames, convert_to_strips, stack_tiff_images


class TestImagePipeline(unittest.TestCase):
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

            strips = sorted(path.name for path in inter_dir.iterdir() if path.suffix.lower() in {".tif", ".tiff"})
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

    def test_non_rgb_input_handling(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "input"
            output_file = root / "timeline.tif"
            source_dir.mkdir(parents=True)

            # grayscale and RGBA inputs should both be normalized to RGB safely
            Image.new("L", (2, 2), 128).save(source_dir / "a.tif")
            Image.new("RGBA", (2, 2), (10, 20, 30, 255)).save(source_dir / "b.tif")

            count = build_timeline_from_frames(source_dir, output_file)
            self.assertEqual(count, 2)

            with Image.open(output_file) as timeline:
                self.assertEqual(timeline.size, (2, 2))
                px = timeline.load()
                self.assertEqual(px[0, 0], (128, 128, 128))
                self.assertEqual(px[0, 1], (10, 20, 30))


if __name__ == "__main__":
    unittest.main()
