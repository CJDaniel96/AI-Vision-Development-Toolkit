import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from weighted_image_sampler import calculate_allocations, sample_and_copy


class CalculateAllocationsTests(unittest.TestCase):
    def test_uses_minimum_one_and_round_half_up(self):
        folders = {
            Path("class_a"): tuple(Path(f"a_{index}.jpg") for index in range(1)),
            Path("class_b"): tuple(Path(f"b_{index}.jpg") for index in range(3)),
            Path("class_c"): tuple(Path(f"c_{index}.jpg") for index in range(6)),
        }

        allocations = calculate_allocations(folders, requested_count=5)

        self.assertEqual([item.sample_count for item in allocations], [1, 2, 3])
        self.assertEqual(allocations[1].raw_count, Decimal("1.5"))

    def test_rejects_request_larger_than_available_images(self):
        folders = {Path("class_a"): (Path("a.jpg"),)}

        with self.assertRaisesRegex(ValueError, "超過可用影像總數"):
            calculate_allocations(folders, requested_count=2)


class SampleAndCopyTests(unittest.TestCase):
    def test_copies_images_and_preserves_subfolder_structure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            output = root / "output"
            (source / "class_a" / "nested").mkdir(parents=True)
            (source / "class_b").mkdir(parents=True)

            for relative_path in (
                "class_a/a.jpg",
                "class_a/nested/b.PNG",
                "class_b/c.jpeg",
                "class_b/not-an-image.txt",
            ):
                path = source / relative_path
                path.write_text(relative_path, encoding="utf-8")

            result = sample_and_copy(source, output, requested_count=2, seed=42)

            self.assertEqual(result.total_images, 3)
            self.assertEqual(result.copied_count, 2)
            self.assertEqual(len(list((output / "class_a").rglob("*.*"))), 1)
            self.assertEqual(len(list((output / "class_b").rglob("*.*"))), 1)
            self.assertFalse((output / "class_b" / "not-an-image.txt").exists())

    def test_ignores_output_images_when_output_is_inside_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source"
            output = source / "sampled"
            (source / "class_a").mkdir(parents=True)
            (output / "old_class").mkdir(parents=True)
            (source / "class_a" / "a.jpg").write_bytes(b"source")
            (output / "old_class" / "old.jpg").write_bytes(b"old output")

            result = sample_and_copy(source, output, requested_count=1, seed=1)

            self.assertEqual(result.total_images, 1)
            self.assertEqual(result.copied_count, 1)
            self.assertEqual((output / "class_a" / "a.jpg").read_bytes(), b"source")

    def test_requires_overwrite_for_existing_selected_target(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            output = root / "output"
            (source / "class_a").mkdir(parents=True)
            (source / "class_a" / "a.jpg").write_bytes(b"new")

            sample_and_copy(source, output, requested_count=1, seed=1)

            with self.assertRaises(FileExistsError):
                sample_and_copy(source, output, requested_count=1, seed=1)

            result = sample_and_copy(
                source,
                output,
                requested_count=1,
                seed=1,
                overwrite=True,
            )
            self.assertEqual(result.copied_count, 1)


if __name__ == "__main__":
    unittest.main()
