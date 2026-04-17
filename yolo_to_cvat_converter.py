#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO → CVAT YOLO 格式封裝工具

將 YOLO 影像 + 標籤打包成可匯入 CVAT 的 ZIP 壓縮檔。

支援兩種模式：
  - 預設（含影像）：ZIP 內含影像與標籤，對應 CVAT「YOLO 1.1」格式匯入
  - 僅標注（--annotations-only）：ZIP 只含標籤檔，適合 annotations-only 匯入

使用範例：
  # 含影像（從類別名稱檔讀取）
  python yolo_to_cvat_converter.py \\
    --images-dir ./images --labels-dir ./labels \\
    --class-names-file classes.txt --output cvat_upload.zip

  # 含影像（直接指定類別名稱）
  python yolo_to_cvat_converter.py \\
    --images-dir ./images --labels-dir ./labels \\
    --classes person car dog --output cvat_upload.zip

  # 僅標注 + 嚴格格式驗證
  python yolo_to_cvat_converter.py \\
    --images-dir ./images --labels-dir ./labels \\
    --classes person car --output cvat_annotations.zip \\
    --annotations-only --strict-label-check
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from tqdm import tqdm

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


# ══════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════

def find_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def validate_label_file(label_path: Path, num_classes: int) -> None:
    """嚴格驗證 YOLO txt 格式（5 欄、class_id 範圍、浮點值在 [0,1]）"""
    text = label_path.read_text(encoding='utf-8').strip()
    if not text:
        return
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path} 第 {line_no} 行格式錯誤（應為 5 欄：class_id cx cy w h）"
            )
        try:
            class_id = int(parts[0])
        except ValueError:
            raise ValueError(f"{label_path} 第 {line_no} 行 class_id 不是整數：{parts[0]}")
        if not (0 <= class_id < num_classes):
            raise ValueError(
                f"{label_path} 第 {line_no} 行 class_id={class_id} 超出範圍 [0, {num_classes - 1}]"
            )
        for col, token in enumerate(parts[1:], start=2):
            try:
                val = float(token)
            except ValueError:
                raise ValueError(f"{label_path} 第 {line_no} 行第 {col} 欄不是數值：{token}")
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"{label_path} 第 {line_no} 行第 {col} 欄數值不在 [0,1]：{val}"
                )


def collect_pairs(
    images_dir: Path,
    labels_dir: Path,
    empty_missing: bool,
    strict_label_check: bool,
    num_classes: int,
) -> list[tuple[Path, Path]]:
    """
    配對影像與標籤，回傳 (image_path, label_path) 清單。
    若找不到對應標籤且 empty_missing=True，自動建立空白 txt。
    """
    images = find_images(images_dir)
    if not images:
        raise FileNotFoundError(f"找不到任何影像：{images_dir}")

    pairs: list[tuple[Path, Path]] = []
    seen: set[str] = set()

    for img_path in images:
        if img_path.name in seen:
            raise ValueError(f"影像檔名重複：{img_path.name}（所有影像檔名需唯一）")
        seen.add(img_path.name)

        label_path = labels_dir / f'{img_path.stem}.txt'
        if not label_path.exists():
            if empty_missing:
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text('', encoding='utf-8')
            else:
                raise FileNotFoundError(
                    f"找不到標籤檔：{label_path}（對應影像：{img_path.name}）"
                )

        if strict_label_check:
            validate_label_file(label_path, num_classes)

        pairs.append((img_path, label_path))

    return pairs


def load_class_names(args) -> list[str]:
    """從 --classes 或 --class-names-file 取得類別名稱。"""
    if args.classes:
        return args.classes
    if args.class_names_file:
        with open(args.class_names_file, 'r', encoding='utf-8') as f:
            names = [line.strip() for line in f if line.strip()]
        if not names:
            raise ValueError(f"類別名稱檔案是空的：{args.class_names_file}")
        return names
    raise ValueError("請提供 --classes 或 --class-names-file。")


# ══════════════════════════════════════════════════════
# ZIP 封裝（含影像）
# ══════════════════════════════════════════════════════

def pack_with_images(
    pairs: list[tuple[Path, Path]],
    class_names: list[str],
    output_zip: Path,
    subset: str,
) -> None:
    """
    打包影像 + 標籤至 ZIP（CVAT YOLO 1.1 格式）。
    ZIP 結構：
      obj_{subset}_data/{stem}.ext   ← 影像
      obj_{subset}_data/{stem}.txt   ← 標籤
      obj.names
      obj.data
      {subset}.txt
    """
    num_classes = len(class_names)
    data_folder_name = f'obj_{subset}_data'

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        data_folder = tmp / data_folder_name
        data_folder.mkdir()

        # obj.names
        (tmp / 'obj.names').write_text('\n'.join(class_names) + '\n', encoding='utf-8')

        # obj.data
        obj_data = (
            f'classes = {num_classes}\n'
            f'train = {subset}.txt\n'
            f'names = obj.names\n'
            f'backup = backup/\n'
        )
        (tmp / 'obj.data').write_text(obj_data, encoding='utf-8')

        # 複製影像與標籤
        train_lines = []
        for img_path, label_path in tqdm(pairs, desc="複製檔案"):
            shutil.copy2(img_path,   data_folder / img_path.name)
            shutil.copy2(label_path, data_folder / label_path.name)
            train_lines.append(f'{data_folder_name}/{img_path.name}')

        # {subset}.txt
        (tmp / f'{subset}.txt').write_text('\n'.join(train_lines), encoding='utf-8')

        # 壓縮
        archive_name = output_zip.with_suffix('')
        shutil.make_archive(str(archive_name), 'zip', root_dir=tmp)

    print(f"\n✅ 已建立（含影像）：{output_zip}")
    print(f"   影像數：{len(pairs)}　類別數：{num_classes}　subset：{subset}")


# ══════════════════════════════════════════════════════
# ZIP 封裝（僅標注）
# ══════════════════════════════════════════════════════

def pack_annotations_only(
    pairs: list[tuple[Path, Path]],
    class_names: list[str],
    output_zip: Path,
    subset: str,
) -> None:
    """
    僅打包標籤至 ZIP（CVAT YOLO 1.1 annotations-only 格式）。
    ZIP 結構：
      obj_{subset}_data/{stem}.txt   ← 標籤（依圖片列表）
      obj.names
      obj.data
      {subset}.txt
    """
    num_classes = len(class_names)
    ann_dir = f'obj_{subset}_data'

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zf:
        # obj.names
        zf.writestr('obj.names', '\n'.join(class_names) + '\n')
        # obj.data
        obj_data = (
            f'classes = {num_classes}\n'
            f'names = obj.names\n'
            f'train = {subset}.txt\n'
        )
        zf.writestr('obj.data', obj_data)
        # {subset}.txt
        subset_lines = [f'{ann_dir}/{img.name}' for img, _ in pairs]
        zf.writestr(f'{subset}.txt', '\n'.join(subset_lines) + '\n')
        # 標籤檔
        for img_path, label_path in pairs:
            zf.write(label_path, arcname=f'{ann_dir}/{img_path.stem}.txt')

    print(f"\n✅ 已建立（僅標注）：{output_zip}")
    print(f"   影像數：{len(pairs)}　類別數：{num_classes}　subset：{subset}")


# ══════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="將 YOLO 影像 + 標籤打包成可匯入 CVAT 的 ZIP 壓縮檔",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  # 含影像，類別名稱從檔案讀取
  python yolo_to_cvat_converter.py \\
    --images-dir ./images --labels-dir ./labels \\
    --class-names-file classes.txt --output cvat_upload.zip

  # 含影像，直接指定類別
  python yolo_to_cvat_converter.py \\
    --images-dir ./images --labels-dir ./labels \\
    --classes person car dog --output cvat_upload.zip

  # 僅標注 + 嚴格驗證 + 自動補空白標籤
  python yolo_to_cvat_converter.py \\
    --images-dir ./images --labels-dir ./labels \\
    --classes person --output cvat_ann.zip \\
    --annotations-only --strict-label-check --empty-missing
        """,
    )

    # ── 路徑
    parser.add_argument('--images-dir',  type=Path, required=True, help="影像資料夾路徑")
    parser.add_argument('--labels-dir',  type=Path, required=True, help="YOLO txt 標籤資料夾路徑")
    parser.add_argument('--output',      type=Path, required=True, help="輸出 ZIP 路徑")

    # ── 類別名稱（二擇一）
    cls_group = parser.add_mutually_exclusive_group(required=True)
    cls_group.add_argument('--classes', nargs='+',
                           help="直接指定類別名稱（順序對應 YOLO class id），例：--classes person car")
    cls_group.add_argument('--class-names-file', type=Path,
                           help="類別名稱文字檔路徑（每行一個類別名稱）")

    # ── 封裝選項
    parser.add_argument('--annotations-only', action='store_true',
                        help="僅打包標籤（不含影像），適合 CVAT annotations-only 匯入")
    parser.add_argument('--subset', type=str, default='train',
                        help="subset 名稱（預設：train）")
    parser.add_argument('--empty-missing', action='store_true',
                        help="若找不到對應標籤，自動建立空白 txt（預設：報錯）")
    parser.add_argument('--strict-label-check', action='store_true',
                        help="嚴格驗證標籤格式與 class_id 範圍")

    args = parser.parse_args()

    # ── 路徑驗證
    if not args.images_dir.is_dir():
        print(f"[ERROR] 影像資料夾不存在：{args.images_dir}", file=sys.stderr)
        return 1
    if not args.labels_dir.is_dir():
        print(f"[ERROR] 標籤資料夾不存在：{args.labels_dir}", file=sys.stderr)
        return 1

    try:
        class_names = load_class_names(args)
        print(f"類別（共 {len(class_names)} 個）：{class_names}")

        pairs = collect_pairs(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            empty_missing=args.empty_missing,
            strict_label_check=args.strict_label_check,
            num_classes=len(class_names),
        )
        print(f"找到 {len(pairs)} 組影像/標籤配對。")

        if args.annotations_only:
            pack_annotations_only(pairs, class_names, args.output, args.subset)
        else:
            pack_with_images(pairs, class_names, args.output, args.subset)

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
