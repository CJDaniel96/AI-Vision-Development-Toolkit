import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from PIL import Image

from image_center_crop import center_crop, main


class CenterCropTests(unittest.TestCase):
    def test_crops_center_pixels_without_resizing(self):
        image = Image.new("RGB", (7, 5))
        image.putdata([(x * 30, y * 40, 0) for y in range(5) for x in range(7)])

        cropped = center_crop(image, 0.6)

        self.assertEqual(cropped.size, (4, 3))
        self.assertEqual(cropped.tobytes(), image.crop((1, 1, 5, 4)).tobytes())

    def test_full_ratio_preserves_pixels_and_tiny_ratio_keeps_one_pixel(self):
        image = Image.new("RGBA", (5, 3), (10, 20, 30, 40))

        self.assertEqual(center_crop(image, 1).tobytes(), image.tobytes())
        self.assertEqual(center_crop(image, 0.001).size, (1, 1))

    def test_rejects_invalid_ratios(self):
        image = Image.new("RGB", (10, 10))
        for ratio in (0, -0.5, 1.1, float("nan"), float("inf")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                center_crop(image, ratio)


class CliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "來源影像"
        self.source.mkdir()
        self.output = self.root / "裁切結果"

    def write_image(self, relative_path, *, mode="RGB", size=(20, 10)):
        path = self.source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(mode, size, 100).save(path)
        return path

    def run_cli(self, source=None, output=None, *options):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        with redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            return main([
                "-i", str(source or self.source),
                "-o", str(output or self.output), *options,
            ])

    def test_single_image_preserves_alpha_and_source(self):
        source = self.write_image("透明影像.PNG", mode="RGBA")
        original_bytes = source.read_bytes()

        self.assertEqual(self.run_cli(source, None, "--ratio", "0.8"), 0)

        with Image.open(self.output / source.name) as result:
            self.assertEqual(result.size, (16, 8))
            self.assertEqual(result.mode, "RGBA")
            self.assertEqual(result.getpixel((0, 0)), (100, 0, 0, 0))
        self.assertEqual(source.read_bytes(), original_bytes)

    def test_recursive_batch_preserves_structure_and_excludes_output(self):
        self.write_image("first.png")
        self.write_image("nested/second.png")
        (self.source / "notes.txt").write_text("skip", encoding="utf-8")
        output = self.source / "cropped"
        output.mkdir()
        Image.new("RGB", (4, 4)).save(output / "old.png")

        self.assertEqual(self.run_cli(None, output, "--recursive", "-r", "0.5"), 0)

        self.assertEqual(
            {str(path.relative_to(output)) for path in output.rglob("*.png")},
            {"first.png", "nested/second.png", "old.png"},
        )
        with Image.open(output / "nested/second.png") as result:
            self.assertEqual(result.size, (10, 5))

    def test_default_only_processes_top_level_and_supports_common_formats(self):
        filenames = ["a.JPG", "b.jpeg", "c.png", "d.bmp", "e.tif", "f.tiff", "g.webp"]
        for name in filenames:
            self.write_image(name)
        self.write_image("nested/ignored.png")

        self.assertEqual(self.run_cli(), 0)

        self.assertEqual({path.name for path in self.output.iterdir()}, set(filenames))
        for name in filenames:
            with self.subTest(name=name), Image.open(self.output / name) as result:
                self.assertEqual(result.size, (10, 5))

    def test_applies_exif_orientation_before_calculating_dimensions(self):
        path = self.source / "oriented.jpg"
        exif = Image.Exif()
        exif[274] = 6  # 旋轉 90 度後，顯示尺寸為 8×12。
        Image.new("RGB", (12, 8)).save(path, exif=exif)

        self.assertEqual(self.run_cli(path), 0)

        with Image.open(self.output / path.name) as result:
            self.assertEqual(result.size, (4, 6))
            self.assertIn(result.getexif().get(274), (None, 1))

    def test_existing_target_requires_overwrite_before_batch_starts(self):
        self.write_image("a.png")
        self.write_image("z.png")
        self.output.mkdir()
        existing = self.output / "z.png"
        existing.write_bytes(b"existing output")

        self.assertEqual(self.run_cli(), 1)
        self.assertIn("--overwrite", self.stderr.getvalue())
        self.assertEqual(existing.read_bytes(), b"existing output")
        self.assertFalse((self.output / "a.png").exists())
        self.assertEqual(self.run_cli(None, None, "--overwrite"), 0)
        with Image.open(existing) as result:
            self.assertEqual(result.size, (10, 5))

    def test_refuses_to_overwrite_source_even_with_overwrite_flag(self):
        source = self.write_image("original.png")
        original_bytes = source.read_bytes()

        for input_path in (source, self.source):
            with self.subTest(input_path=input_path):
                self.assertEqual(self.run_cli(input_path, self.source, "--overwrite"), 1)
                self.assertEqual(source.read_bytes(), original_bytes)

    def test_corrupt_image_reports_failure_and_continues_batch(self):
        (self.source / "broken.png").write_bytes(b"not an image")
        self.write_image("valid.png")

        self.assertEqual(self.run_cli(), 1)

        self.assertTrue((self.output / "valid.png").exists())
        self.assertIn("broken.png", self.stderr.getvalue())
        self.assertIn("成功 1 張，失敗 1 張", self.stdout.getvalue())

    def test_rejects_invalid_paths_and_empty_folders(self):
        unsupported = self.source / "notes.txt"
        unsupported.write_text("text", encoding="utf-8")
        for source in (self.root / "missing", unsupported, self.source):
            with self.subTest(source=source):
                self.assertEqual(self.run_cli(source), 1)
                self.assertFalse(self.output.exists())
        self.write_image("valid.png")
        self.output.write_text("not a folder", encoding="utf-8")
        self.assertEqual(self.run_cli(), 1)

    def test_cli_rejects_invalid_ratio_before_creating_output(self):
        self.write_image("valid.png")
        for ratio in ("0", "-0.1", "1.1", "nan", "inf", "bad"):
            with self.subTest(ratio=ratio), self.assertRaises(SystemExit) as error:
                self.run_cli(None, None, "--ratio", ratio)
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
