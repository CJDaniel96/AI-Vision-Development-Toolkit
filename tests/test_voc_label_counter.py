import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from voc_label_counter import collect_statistics, main


def write_voc_xml(path: Path, labels: list[str]) -> None:
    annotation = ET.Element("annotation")
    for label in labels:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = label
    ET.ElementTree(annotation).write(path, encoding="utf-8")


class CollectStatisticsTests(unittest.TestCase):
    def test_counts_instances_recursively_and_reports_dataset_summary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory)
            nested = dataset / "nested"
            nested.mkdir()

            (dataset / "first.JPG").write_bytes(b"image")
            write_voc_xml(dataset / "first.xml", ["cat", "cat", "dog"])
            (nested / "second.png").write_bytes(b"image")
            write_voc_xml(nested / "second.XML", ["dog"])
            (nested / "image_only.webp").write_bytes(b"image")
            write_voc_xml(nested / "xml_only.xml", [])

            statistics = collect_statistics(dataset)

            self.assertEqual(statistics.label_counts, {"cat": 2, "dog": 2})
            self.assertEqual(statistics.object_count, 4)
            self.assertEqual(statistics.image_count, 3)
            self.assertEqual(statistics.xml_count, 3)
            self.assertEqual(statistics.matched_pair_count, 2)
            self.assertEqual(statistics.images_without_xml_count, 1)
            self.assertEqual(statistics.xml_without_image_count, 1)
            self.assertEqual(statistics.unlabeled_xml_count, 1)

    def test_skips_malformed_xml_and_keeps_other_counts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory)
            write_voc_xml(dataset / "valid.xml", ["ok"])
            (dataset / "broken.xml").write_text("<annotation>", encoding="utf-8")

            statistics = collect_statistics(dataset)

            self.assertEqual(statistics.label_counts, {"ok": 1})
            self.assertEqual(statistics.parsed_xml_count, 1)
            self.assertEqual(
                tuple(path.name for path in statistics.malformed_xml_files),
                ("broken.xml",),
            )

    def test_non_recursive_mode_ignores_nested_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory)
            nested = dataset / "nested"
            nested.mkdir()
            write_voc_xml(dataset / "top.xml", ["top"])
            write_voc_xml(nested / "nested.xml", ["nested"])

            statistics = collect_statistics(dataset, recursive=False)

            self.assertEqual(statistics.label_counts, {"top": 1})
            self.assertEqual(statistics.xml_count, 1)

    def test_pair_summary_counts_duplicate_stems_as_separate_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory)
            first = dataset / "first"
            second = dataset / "second"
            first.mkdir()
            second.mkdir()
            (first / "sample.jpg").write_bytes(b"first")
            (second / "sample.png").write_bytes(b"second")
            write_voc_xml(first / "sample.xml", ["object"])

            statistics = collect_statistics(dataset)

            self.assertEqual(statistics.matched_pair_count, 1)
            self.assertEqual(statistics.images_without_xml_count, 1)
            self.assertEqual(statistics.xml_without_image_count, 0)

    def test_rejects_directory_without_xml(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "找不到 PASCAL VOC XML"):
                collect_statistics(temporary_directory)


class MainTests(unittest.TestCase):
    def test_returns_error_for_missing_input_directory(self):
        self.assertEqual(main(["--input", "/path/that/does/not/exist"]), 1)


if __name__ == "__main__":
    unittest.main()
