#!/usr/bin/env python3
"""統計資料夾中 PASCAL VOC XML 的物件標籤數量。"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
}


@dataclass(frozen=True)
class DatasetStatistics:
    """一次資料集掃描的統計結果。"""

    label_counts: Counter[str]
    image_count: int
    xml_count: int
    parsed_xml_count: int
    malformed_xml_files: tuple[Path, ...]
    unlabeled_xml_count: int
    matched_pair_count: int
    images_without_xml_count: int
    xml_without_image_count: int

    @property
    def object_count(self) -> int:
        return sum(self.label_counts.values())


def _find_files(input_dir: Path, recursive: bool) -> tuple[list[Path], list[Path]]:
    paths = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images: list[Path] = []
    xml_files: list[Path] = []

    for path in paths:
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            images.append(path)
        elif suffix == ".xml":
            xml_files.append(path)

    return sorted(images), sorted(xml_files)


def _labels_from_xml(xml_path: Path) -> list[str]:
    """讀取單一 PASCAL VOC XML 內所有非空白的 object/name。"""

    root = ET.parse(xml_path).getroot()
    labels: list[str] = []
    for obj in root.findall("object"):
        name = obj.find("name")
        if name is None or name.text is None:
            continue
        label = name.text.strip()
        if label:
            labels.append(label)
    return labels


def collect_statistics(
    input_dir: str | Path,
    *,
    recursive: bool = True,
) -> DatasetStatistics:
    """掃描資料夾並統計每個 label 的物件 instance 數。"""

    source = Path(input_dir).expanduser().resolve()
    if not source.exists():
        raise ValueError(f"輸入資料夾不存在: {source}")
    if not source.is_dir():
        raise ValueError(f"輸入路徑不是資料夾: {source}")

    images, xml_files = _find_files(source, recursive)
    if not xml_files:
        raise ValueError(f"在輸入資料夾中找不到 PASCAL VOC XML 檔案: {source}")

    label_counts: Counter[str] = Counter()
    malformed: list[Path] = []
    unlabeled_xml_count = 0

    for xml_path in xml_files:
        try:
            labels = _labels_from_xml(xml_path)
        except (ET.ParseError, OSError):
            malformed.append(xml_path)
            continue

        if labels:
            label_counts.update(labels)
        else:
            unlabeled_xml_count += 1

    # VOC 資料集常把影像和標註分放在不同資料夾，因此以檔名 stem 配對。
    image_stems = Counter(path.stem for path in images)
    xml_stems = Counter(path.stem for path in xml_files)
    matched_pair_count = sum((image_stems & xml_stems).values())

    return DatasetStatistics(
        label_counts=label_counts,
        image_count=len(images),
        xml_count=len(xml_files),
        parsed_xml_count=len(xml_files) - len(malformed),
        malformed_xml_files=tuple(malformed),
        unlabeled_xml_count=unlabeled_xml_count,
        matched_pair_count=matched_pair_count,
        images_without_xml_count=sum((image_stems - xml_stems).values()),
        xml_without_image_count=sum((xml_stems - image_stems).values()),
    )


def _print_statistics(statistics: DatasetStatistics) -> None:
    print("資料集摘要")
    print(f"  影像檔數量: {statistics.image_count}")
    print(f"  XML 檔數量: {statistics.xml_count}")
    print(f"  成功解析 XML: {statistics.parsed_xml_count}")
    print(f"  影像/XML 配對數: {statistics.matched_pair_count}")
    print(f"  無 XML 的影像數: {statistics.images_without_xml_count}")
    print(f"  無影像的 XML 數: {statistics.xml_without_image_count}")
    print(f"  無標籤 XML 數: {statistics.unlabeled_xml_count}")
    print(f"  物件總數: {statistics.object_count}")

    print("\nLabel 統計（依數量由多至少）")
    if not statistics.label_counts:
        print("  未找到任何 label")
    else:
        label_width = max(len("Label"), *(len(label) for label in statistics.label_counts))
        print(f"  {'Label':<{label_width}}  {'數量':>8}")
        print(f"  {'-' * label_width}  {'-' * 8}")
        for label, count in sorted(
            statistics.label_counts.items(), key=lambda item: (-item[1], item[0])
        ):
            print(f"  {label:<{label_width}}  {count:>8}")

    if statistics.malformed_xml_files:
        print("\n警告：下列 XML 無法解析，未納入 label 統計：", file=sys.stderr)
        for path in statistics.malformed_xml_files:
            print(f"  {path}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="統計資料夾內 PASCAL VOC XML 的各個 label instance 數量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python voc_label_counter.py -i ./voc_dataset
  python voc_label_counter.py -i ./voc_dataset --no-recursive
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="包含影像與 VOC XML 的資料夾")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="只掃描輸入資料夾第一層（預設會遞迴掃描子資料夾）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        statistics = collect_statistics(args.input, recursive=not args.no_recursive)
    except (ValueError, OSError) as error:
        print(f"錯誤: {error}", file=sys.stderr)
        return 1

    _print_statistics(statistics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
