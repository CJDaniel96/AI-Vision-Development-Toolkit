#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
from __future__ import annotations
 
import argparse
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
 
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack YOLO txt annotations into CVAT YOLO 1.1 annotation-only zip."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="圖片資料夾，例如 ./images"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="YOLO txt 標註資料夾，例如 ./labels"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="類別名稱，順序需對應 YOLO class id，例如: --classes person car dog"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="輸出 zip 路徑，例如 ./cvat_yolo_annotations.zip"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        help="subset 名稱，預設 train，通常不用改"
    )
    parser.add_argument(
        "--empty-missing",
        action="store_true",
        help="若缺少對應 txt，則自動建立空白標註"
    )
    parser.add_argument(
        "--strict-label-check",
        action="store_true",
        help="檢查 txt 內容格式與 class id 是否合法"
    )
    return parser.parse_args()
 
 
def find_images(images_dir: Path) -> list[Path]:
    images = [
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(images)
 
 
def validate_label_file(label_path: Path, num_classes: int) -> None:
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return
 
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path} 第 {line_no} 行格式錯誤，YOLO detection 應為 5 欄: "
                f"class_id cx cy w h"
            )
 
        try:
            class_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(
                f"{label_path} 第 {line_no} 行 class_id 不是整數: {parts[0]}"
            ) from exc
 
        if not (0 <= class_id < num_classes):
            raise ValueError(
                f"{label_path} 第 {line_no} 行 class_id={class_id} 超出範圍 "
                f"[0, {num_classes - 1}]"
            )
 
        for idx, token in enumerate(parts[1:], start=2):
            try:
                value = float(token)
            except ValueError as exc:
                raise ValueError(
                    f"{label_path} 第 {line_no} 行第 {idx} 欄不是數值: {token}"
                ) from exc
 
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"{label_path} 第 {line_no} 行第 {idx} 欄數值不在 [0,1]: {value}"
                )
 
 
def collect_image_label_pairs(
    images_dir: Path,
    labels_dir: Path,
    empty_missing: bool,
    strict_label_check: bool,
    num_classes: int,
) -> list[tuple[Path, Path]]:
    images = find_images(images_dir)
    if not images:
        raise FileNotFoundError(f"找不到任何圖片: {images_dir}")
 
    pairs: list[tuple[Path, Path]] = []
    seen_names: set[str] = set()
 
    for img_path in images:
        img_name = img_path.name
        stem = img_path.stem
 
        if img_name in seen_names:
            raise ValueError(
                f"圖片檔名重複: {img_name}。建議所有圖片檔名唯一。"
            )
        seen_names.add(img_name)
 
        label_path = labels_dir / f"{stem}.txt"
 
        if not label_path.exists():
            if empty_missing:
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text("", encoding="utf-8")
            else:
                raise FileNotFoundError(
                    f"缺少對應標註檔: {label_path} (對應圖片 {img_name})"
                )
 
        if strict_label_check:
            validate_label_file(label_path, num_classes)
 
        pairs.append((img_path, label_path))
 
    return pairs
 
 
def build_obj_names(classes: list[str]) -> str:
    return "\n".join(classes) + "\n"
 
 
def build_obj_data(num_classes: int, subset: str) -> str:
    return (
        f"classes = {num_classes}\n"
        f"names = obj.names\n"
        f"train = {subset}.txt\n"
    )
 
 
def build_subset_txt(pairs: list[tuple[Path, Path]], subset: str) -> str:
    lines = [f"obj_{subset}_data/{img_path.name}" for img_path, _ in pairs]
    return "\n".join(lines) + "\n"
 
 
def write_zip(
    output_zip: Path,
    pairs: list[tuple[Path, Path]],
    classes: list[str],
    subset: str,
) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    ann_dir = f"obj_{subset}_data"
 
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("obj.names", build_obj_names(classes))
        zf.writestr("obj.data", build_obj_data(len(classes), subset))
        zf.writestr(f"{subset}.txt", build_subset_txt(pairs, subset))
 
        for img_path, label_path in pairs:
            arcname = f"{ann_dir}/{img_path.stem}.txt"
            zf.write(label_path, arcname=arcname)
 
 
def main() -> int:
    args = parse_args()
 
    images_dir: Path = args.images
    labels_dir: Path = args.labels
    classes: list[str] = args.classes
    output_zip: Path = args.output
    subset: str = args.subset
 
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"[ERROR] images 資料夾不存在: {images_dir}", file=sys.stderr)
        return 1
 
    if not labels_dir.exists() or not labels_dir.is_dir():
        print(f"[ERROR] labels 資料夾不存在: {labels_dir}", file=sys.stderr)
        return 1
 
    if not classes:
        print("[ERROR] classes 不可為空", file=sys.stderr)
        return 1
 
    try:
        pairs = collect_image_label_pairs(
            images_dir=images_dir,
            labels_dir=labels_dir,
            empty_missing=args.empty_missing,
            strict_label_check=args.strict_label_check,
            num_classes=len(classes),
        )
        write_zip(
            output_zip=output_zip,
            pairs=pairs,
            classes=classes,
            subset=subset,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
 
    print(f"[OK] 已建立: {output_zip}")
    print(f"[INFO] images = {len(pairs)}")
    print(f"[INFO] classes = {classes}")
    print(f"[INFO] subset = {subset}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())