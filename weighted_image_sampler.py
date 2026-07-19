#!/usr/bin/env python3
"""依第一層子資料夾的影像占比進行加權隨機抽樣。"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
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
class FolderAllocation:
    """單一子資料夾的統計與抽樣配置。"""

    folder: Path
    images: tuple[Path, ...]
    weight: Decimal
    raw_count: Decimal
    sample_count: int


@dataclass(frozen=True)
class SamplingResult:
    """抽樣完成後的摘要。"""

    total_images: int
    requested_count: int
    copied_count: int
    allocations: tuple[FolderAllocation, ...]
    copied_files: tuple[Path, ...]


def _is_inside(path: Path, directory: Path) -> bool:
    """判斷 path 是否位於 directory 內（包含 directory 本身）。"""

    return path == directory or path.is_relative_to(directory)


def find_images_by_subfolder(
    input_dir: Path,
    output_dir: Path,
    extensions: set[str] | None = None,
) -> dict[Path, tuple[Path, ...]]:
    """遞迴尋找每個第一層子資料夾中的影像。

    若輸出資料夾位於輸入資料夾內，會自動排除輸出資料夾中的檔案，避免
    重複執行時把先前輸出再次視為來源資料。
    """

    normalized_extensions = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in (extensions or SUPPORTED_IMAGE_EXTENSIONS)
    }
    resolved_output = output_dir.resolve()
    images_by_folder: dict[Path, tuple[Path, ...]] = {}

    for subfolder in sorted(input_dir.iterdir(), key=lambda path: path.name):
        if not subfolder.is_dir():
            continue

        resolved_subfolder = subfolder.resolve()
        if _is_inside(resolved_subfolder, resolved_output):
            continue

        images = tuple(
            path
            for path in sorted(subfolder.rglob("*"))
            if path.is_file()
            and path.suffix.lower() in normalized_extensions
            and not _is_inside(path.resolve(), resolved_output)
        )
        if images:
            images_by_folder[subfolder] = images

    return images_by_folder


def calculate_allocations(
    images_by_folder: dict[Path, tuple[Path, ...]],
    requested_count: int,
) -> tuple[FolderAllocation, ...]:
    """依影像占比計算各子資料夾的抽樣張數。

    原始計算值小於 1 時取 1；其餘採 ROUND_HALF_UP，亦即一般認知的
    四捨五入，而不是 Python round() 的銀行家捨入。
    """

    if requested_count <= 0:
        raise ValueError("抽樣數量必須是大於 0 的整數")

    total_images = sum(len(images) for images in images_by_folder.values())
    if total_images == 0:
        raise ValueError("輸入資料夾的子資料夾中沒有找到支援的影像")
    if requested_count > total_images:
        raise ValueError(
            f"抽樣數量 {requested_count} 超過可用影像總數 {total_images}；"
            "本工具不會重複抽取同一張影像"
        )

    total_decimal = Decimal(total_images)
    requested_decimal = Decimal(requested_count)
    allocations: list[FolderAllocation] = []

    for folder, images in images_by_folder.items():
        image_count = len(images)
        weight = Decimal(image_count) / total_decimal
        raw_count = weight * requested_decimal
        if raw_count < 1:
            sample_count = 1
        else:
            sample_count = int(raw_count.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

        allocations.append(
            FolderAllocation(
                folder=folder,
                images=images,
                weight=weight,
                raw_count=raw_count,
                sample_count=sample_count,
            )
        )

    return tuple(allocations)


def sample_and_copy(
    input_dir: str | Path,
    output_dir: str | Path,
    requested_count: int,
    *,
    seed: int | None = None,
    extensions: set[str] | None = None,
    overwrite: bool = False,
) -> SamplingResult:
    """計算各子資料夾配額、隨機抽樣，並將影像複製至輸出資料夾。"""

    source = Path(input_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()

    if not source.exists():
        raise ValueError(f"輸入資料夾不存在: {source}")
    if not source.is_dir():
        raise ValueError(f"輸入路徑不是資料夾: {source}")
    if source == destination:
        raise ValueError("輸入資料夾與輸出資料夾不可相同")
    if source.is_relative_to(destination):
        raise ValueError("輸出資料夾不可是輸入資料夾的上層資料夾")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"輸出路徑已存在且不是資料夾: {destination}")

    images_by_folder = find_images_by_subfolder(source, destination, extensions)
    allocations = calculate_allocations(images_by_folder, requested_count)
    rng = random.Random(seed)

    selected_pairs: list[tuple[Path, Path]] = []
    for allocation in allocations:
        selected_images = rng.sample(list(allocation.images), allocation.sample_count)
        for image_path in selected_images:
            relative_path = image_path.relative_to(allocation.folder)
            target_path = destination / allocation.folder.name / relative_path
            selected_pairs.append((image_path, target_path))

    existing_targets = [target for _, target in selected_pairs if target.exists()]
    if existing_targets and not overwrite:
        preview = ", ".join(str(path) for path in existing_targets[:3])
        more = "..." if len(existing_targets) > 3 else ""
        raise FileExistsError(
            f"有 {len(existing_targets)} 個輸出檔案已存在（例如: {preview}{more}）；"
            "如需覆寫請加上 --overwrite"
        )

    copied_files: list[Path] = []
    for source_path, target_path in selected_pairs:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied_files.append(target_path)

    return SamplingResult(
        total_images=sum(len(item.images) for item in allocations),
        requested_count=requested_count,
        copied_count=len(copied_files),
        allocations=allocations,
        copied_files=tuple(copied_files),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="依子資料夾影像占比，加權隨機抽樣並複製影像",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python weighted_image_sampler.py -i ./dataset -o ./sampled -n 100
  python weighted_image_sampler.py -i ./dataset -o ./sampled -n 100 --seed 42
  python weighted_image_sampler.py -i ./dataset -o ./sampled -n 100 --extensions jpg png
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="來源資料夾")
    parser.add_argument("--output", "-o", required=True, help="抽樣結果輸出資料夾")
    parser.add_argument("--count", "-n", required=True, type=int, help="希望抽樣的資料數量")
    parser.add_argument("--seed", type=int, help="隨機種子；指定後可重現抽樣結果")
    parser.add_argument(
        "--extensions",
        nargs="+",
        metavar="EXT",
        help="自訂影像副檔名，例如: jpg png webp",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允許覆寫輸出資料夾中的同名檔案",
    )
    return parser


def _print_result(result: SamplingResult, output_dir: Path) -> None:
    print("\n子資料夾抽樣配置:")
    print(f"{'子資料夾':<24} {'原始數量':>10} {'權重':>10} {'計算值':>10} {'抽樣數量':>10}")
    print("-" * 70)
    for item in result.allocations:
        print(
            f"{item.folder.name:<24} {len(item.images):>10} "
            f"{float(item.weight):>9.2%} {float(item.raw_count):>10.2f} "
            f"{item.sample_count:>10}"
        )

    print("\n抽樣完成")
    print(f"來源影像總數: {result.total_images}")
    print(f"要求抽樣數量: {result.requested_count}")
    print(f"實際複製數量: {result.copied_count}")
    if result.copied_count != result.requested_count:
        print("注意: 各子資料夾獨立套用最少 1 張與四捨五入規則，因此實際總數可能不同。")
    print(f"輸出資料夾: {output_dir.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    extensions = set(args.extensions) if args.extensions else None

    try:
        result = sample_and_copy(
            args.input,
            args.output,
            args.count,
            seed=args.seed,
            extensions=extensions,
            overwrite=args.overwrite,
        )
        _print_result(result, Path(args.output))
        return 0
    except (ValueError, FileExistsError, OSError) as error:
        print(f"錯誤: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
