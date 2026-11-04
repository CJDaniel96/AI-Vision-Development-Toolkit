#!/usr/bin/env python3
"""按原圖寬、高的比例，擷取影像中心區域並另存至指定資料夾。"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
}


def center_crop(image: Image.Image, ratio: float) -> Image.Image:
    """寬、高乘上 ratio 後無條件捨去，至少保留 1 像素，再從中心裁切。

    若無法對稱分配邊界，多出的 1 像素會從右側或下側裁掉。
    """
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("比例必須介於 0（不含）到 1（含）之間，例如 0.5")

    width, height = image.size
    crop_width = max(1, int(width * ratio))
    crop_height = max(1, int(height * ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def collect_images(source: Path, output: Path, recursive: bool) -> list[Path]:
    """尋找支援的影像；輸出位於來源資料夾內時，排除既有輸出。"""
    if not source.exists():
        raise ValueError(f"輸入路徑不存在: {source}")
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"不支援的影像副檔名: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise ValueError(f"輸入路徑不是影像檔或資料夾: {source}")
    if source == output:
        raise ValueError("輸入資料夾與輸出資料夾不可相同")
    if source.is_relative_to(output):
        raise ValueError("輸出資料夾不可是輸入資料夾的上層資料夾")

    paths = source.rglob("*") if recursive else source.iterdir()
    images = sorted(
        path for path in paths
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        and not path.resolve().is_relative_to(output)
    )
    if not images:
        raise ValueError(f"找不到支援的影像: {source}")
    return images


def crop_file(
    source: Path, target: Path, ratio: float, *, overwrite: bool = False,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """依 EXIF 校正方向後裁切第一幀，回傳原始尺寸與輸出尺寸。"""
    if source.resolve() == target.resolve():
        raise ValueError(f"輸出不可覆蓋來源影像: {source}")

    with Image.open(source) as original:
        with ImageOps.exif_transpose(original) as image:
            with center_crop(image, ratio) as cropped:
                target.parent.mkdir(parents=True, exist_ok=True)
                image_format = Image.registered_extensions()[target.suffix.lower()]
                options = {"quality": 95} if image_format == "JPEG" else {}
                # 排他建立可避免未指定 --overwrite 時覆寫既有檔案。
                with target.open("wb" if overwrite else "xb") as stream:
                    cropped.save(stream, format=image_format, **options)
                return image.size, cropped.size


def parse_ratio(value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("比例必須是數字，例如 0.5") from error
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise argparse.ArgumentTypeError("比例必須介於 0（不含）到 1（含）之間")
    return ratio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按原圖寬、高比例擷取影像中心區域（不縮放）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python image_center_crop.py -i ./photo.jpg -o ./cropped -r 0.5
  python image_center_crop.py -i ./images -o ./cropped --ratio 0.8
  python image_center_crop.py -i ./images -o ./cropped -r 0.6 --recursive

比例 0.5 表示寬、高各保留 50%，面積約為原圖的 25%。
支援 JPG、JPEG、PNG、BMP、TIF、TIFF、WebP；多幀影像僅處理第一幀。
        """,
    )
    parser.add_argument("-i", "--input", required=True, help="來源影像或資料夾路徑")
    parser.add_argument("-o", "--output", required=True, help="裁切影像的輸出資料夾")
    parser.add_argument(
        "-r", "--ratio", type=parse_ratio, default=0.5,
        help="寬、高保留比例，0 < ratio <= 1（預設: 0.5）",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="遞迴處理子資料夾並保留目錄結構（預設只處理第一層）",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="允許覆寫輸出資料夾中的同名檔案",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source = Path(args.input).expanduser().resolve()
        output = Path(args.output).expanduser().resolve()
        if output.exists() and not output.is_dir():
            raise ValueError(f"輸出路徑不是資料夾: {output}")
        images = collect_images(source, output, args.recursive)
        root = source if source.is_dir() else source.parent
        jobs = [(path, output / path.relative_to(root)) for path in images]

        # 先檢查所有目標，避免因為同名檔案而只完成部分批次。
        for image_path, target in jobs:
            if image_path.resolve() == target.resolve():
                raise ValueError(f"輸出不可覆蓋來源影像: {image_path}")
            if target.exists() and not args.overwrite:
                raise FileExistsError(f"輸出檔案已存在: {target}；如需覆寫請加上 --overwrite")
    except (OSError, ValueError) as error:
        print(f"錯誤: {error}", file=sys.stderr)
        return 1

    saved = failed = 0
    for image_path, target in jobs:
        try:
            before, after = crop_file(image_path, target, args.ratio, overwrite=args.overwrite)
            print(f"[儲存] {target} ({before[0]}×{before[1]} → {after[0]}×{after[1]})")
            saved += 1
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            print(f"[失敗] {image_path}: {error}", file=sys.stderr)
            failed += 1

    print(f"完成：成功 {saved} 張，失敗 {failed} 張。輸出資料夾: {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
