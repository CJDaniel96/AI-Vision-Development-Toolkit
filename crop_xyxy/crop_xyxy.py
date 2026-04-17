#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依 xyxy 框裁切影像工具
支援單張影像模式（-i）和批次資料夾模式（--indir）

使用範例：
  # 單張影像
  python crop_xyxy.py -i image.jpg -b 10,20,200,180 300,50,500,220 -o crops/

  # 批次資料夾（每個框各自輸出到獨立子資料夾）
  python crop_xyxy.py --indir ./images -b 10,20,200,180 -f boxes.txt --outdir crops_batch/
"""

import argparse
import os
import sys
from typing import List, Tuple, Optional
import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ══════════════════════════════════════════════════════
# 共用工具函式
# ══════════════════════════════════════════════════════

def parse_xyxy(s: str) -> Tuple[int, int, int, int]:
    """將字串 "x1,y1,x2,y2" 轉成 int tuple，容忍空白/小數（四捨五入）。"""
    try:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError
        x1, y1, x2, y2 = [int(round(float(v))) for v in parts]
        return x1, y1, x2, y2
    except Exception:
        raise argparse.ArgumentTypeError(
            f"無法解析座標：{s}（格式需為 x1,y1,x2,y2）"
        )


def load_boxes_from_file(path: str) -> List[Tuple[int, int, int, int]]:
    """從文字檔讀取多個框（每行一個 x1,y1,x2,y2，可用 # 開頭作為註解）"""
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                boxes.append(parse_xyxy(line))
            except Exception as e:
                raise ValueError(f"讀取 {path} 第 {ln} 行失敗：{e}")
    if not boxes:
        raise ValueError(f"檔案 {path} 未讀到任何有效座標。")
    return boxes


def normalize_box(x1, y1, x2, y2):
    """修正反向座標（若 x2 < x1 或 y2 < y1 則自動交換）"""
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def apply_padding(x1, y1, x2, y2, pad: int):
    if pad <= 0:
        return x1, y1, x2, y2
    return x1 - pad, y1 - pad, x2 + pad, y2 + pad


def clip_to_bounds(x1, y1, x2, y2, W, H):
    return (
        max(0, min(x1, W - 1)),
        max(0, min(y1, H - 1)),
        max(0, min(x2, W - 1)),
        max(0, min(y2, H - 1)),
    )


def process_box(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    pad: int,
    no_clip: bool,
    keep_empty: bool,
) -> Tuple[Optional[np.ndarray], int, int, int, int]:
    """
    套用 normalize / padding / clipping，回傳 (crop, x1, y1, x2, y2)。
    若框無效且 keep_empty=False，crop 為 None。
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = normalize_box(x1, y1, x2, y2)
    x1, y1, x2, y2 = apply_padding(x1, y1, x2, y2, pad)
    if not no_clip:
        x1, y1, x2, y2 = clip_to_bounds(x1, y1, x2, y2, W, H)
    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    if x2 <= x1 or y2 <= y1:
        if keep_empty:
            x2, y2 = x1 + 1, y1 + 1
        else:
            return None, x1, y1, x2, y2

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        if keep_empty:
            crop = np.zeros((1, 1, 3), dtype=np.uint8)
        else:
            return None, x1, y1, x2, y2

    return crop, x1, y1, x2, y2


def collect_boxes(args) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    if args.boxes:
        boxes.extend(args.boxes)
    if args.boxes_file:
        boxes.extend(load_boxes_from_file(args.boxes_file))
    if not boxes:
        print("未提供任何框，請用 -b 或 -f 指定 xyxy。", file=sys.stderr)
        sys.exit(2)
    return boxes


def list_images(directory: str) -> List[str]:
    files = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name))
        and os.path.splitext(name)[1].lower() in IMG_EXTS
    ]
    return sorted(files)


# ══════════════════════════════════════════════════════
# 單張影像模式
# ══════════════════════════════════════════════════════

def run_single(args):
    """
    裁切單張影像，每個框輸出一個編號檔案。
    輸出：{outdir}/{prefix}001.ext, {prefix}002.ext, ...
    """
    img = cv2.imread(args.image)
    if img is None:
        print(f"無法讀取影像：{args.image}", file=sys.stderr)
        sys.exit(1)

    boxes = collect_boxes(args)
    os.makedirs(args.outdir, exist_ok=True)

    base_ext = os.path.splitext(args.image)[1].lower().lstrip(".")
    ext = args.ext or (base_ext if base_ext in {"jpg", "jpeg", "png", "bmp", "tiff"} else "png")

    saved = 0
    for idx, b in enumerate(boxes, start=1):
        crop, x1, y1, x2, y2 = process_box(img, *b, args.pad, args.no_clip, args.keep_empty)
        if crop is None:
            print(f"[跳過] 第 {idx} 個框無效或裁切結果為空：{x1},{y1},{x2},{y2}")
            continue
        out_path = os.path.join(args.outdir, f"{args.prefix}{idx:03d}.{ext}")
        cv2.imwrite(out_path, crop)
        print(f"[儲存] {out_path}")
        saved += 1

    print(f"\n完成，成功輸出 {saved}/{len(boxes)} 張。")


# ══════════════════════════════════════════════════════
# 批次資料夾模式
# ══════════════════════════════════════════════════════

def run_batch(args):
    """
    批次裁切資料夾中所有影像，每個框的結果各自放入獨立子資料夾。
    輸出結構：{outdir}/{prefix}001/{image_name}.ext
               {outdir}/{prefix}002/{image_name}.ext ...
    """
    if not os.path.isdir(args.indir):
        print(f"找不到資料夾：{args.indir}", file=sys.stderr)
        sys.exit(1)

    boxes = collect_boxes(args)
    images = list_images(args.indir)

    if not images:
        print(f"資料夾內沒有支援的影像檔：{args.indir}", file=sys.stderr)
        sys.exit(1)

    # 為每個框建立子資料夾
    os.makedirs(args.outdir, exist_ok=True)
    subdirs = []
    for idx in range(1, len(boxes) + 1):
        d = os.path.join(args.outdir, f"{args.prefix}{idx:03d}")
        os.makedirs(d, exist_ok=True)
        subdirs.append(d)

    total_saved = [0] * len(boxes)
    total_skipped = 0

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳過] 讀不到影像：{img_path}")
            total_skipped += 1
            continue

        stem, in_ext = os.path.splitext(os.path.basename(img_path))
        out_ext = (args.ext or in_ext.lstrip(".")).lower().lstrip(".")
        if out_ext not in {"jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"}:
            out_ext = "png"

        for k, b in enumerate(boxes, start=1):
            crop, x1, y1, x2, y2 = process_box(
                img, *b, args.pad, args.no_clip, args.keep_empty
            )
            if crop is None:
                print(f"[跳過] {os.path.basename(img_path)} 第 {k} 框無效：{x1},{y1},{x2},{y2}")
                continue

            out_dir = subdirs[k - 1]
            out_path = os.path.join(out_dir, f"{stem}.{out_ext}")

            # 避免覆蓋：自動加後綴
            if not args.overwrite and os.path.exists(out_path):
                suffix = 1
                while os.path.exists(os.path.join(out_dir, f"{stem}_{suffix}.{out_ext}")):
                    suffix += 1
                out_path = os.path.join(out_dir, f"{stem}_{suffix}.{out_ext}")

            if cv2.imwrite(out_path, crop):
                total_saved[k - 1] += 1
            else:
                print(f"[失敗] 無法寫入：{out_path}")

    print("\n=== 完成統計 ===")
    print(f"影像總數：{len(images)}，讀取失敗或跳過：{total_skipped}")
    for k, n in enumerate(total_saved, start=1):
        print(f"  {args.prefix}{k:03d}：成功 {n} 張")


# ══════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="依 xyxy 框裁切影像。支援單張（-i）和批次資料夾（--indir）兩種模式。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  # 單張影像，指定兩個框
  python crop_xyxy.py -i image.jpg -b 10,20,200,180 300,50,500,220 -o crops/

  # 批次資料夾，框座標從檔案讀取
  python crop_xyxy.py --indir ./images -f boxes.txt --outdir crops_batch/

  # 批次 + 加入邊距 + 覆蓋已存在的檔案
  python crop_xyxy.py --indir ./images -b 0,0,640,480 --pad 10 --overwrite --outdir out/
        """,
    )

    # ── 輸入來源（二擇一）
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("-i", "--image", help="單張影像路徑")
    src.add_argument("--indir", help="批次影像資料夾路徑")

    # ── 輸出
    parser.add_argument("-o", "--outdir", default="crops",
                        help="輸出資料夾（預設：crops）")

    # ── 裁切框（單張/批次共用）
    parser.add_argument("-b", "--boxes", nargs="*", type=parse_xyxy,
                        help="直接指定一個或多個框，例：-b 10,20,200,180 300,50,500,220")
    parser.add_argument("-f", "--boxes_file",
                        help="從文字檔讀取多個框（每行 x1,y1,x2,y2，# 開頭為註解）")

    # ── 裁切參數（共用）
    parser.add_argument("--pad", type=int, default=0,
                        help="四邊加入的像素邊距（預設：0）")
    parser.add_argument("--no_clip", action="store_true",
                        help="不自動裁齊到影像邊界（預設會裁齊）")
    parser.add_argument("--keep_empty", action="store_true",
                        help="裁切結果為空時仍強制輸出 1x1 像素的佔位圖")
    parser.add_argument("--ext", default=None,
                        help="強制指定輸出副檔名（預設沿用原始副檔名）")
    parser.add_argument("--prefix", default="crop_",
                        help="單張模式：檔名前綴；批次模式：子資料夾前綴（預設：crop_）")

    # ── 批次模式專屬
    parser.add_argument("--overwrite", action="store_true",
                        help="[批次模式] 若輸出檔案已存在則覆蓋（預設不覆蓋，自動加後綴）")

    args = parser.parse_args()

    if args.image:
        run_single(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
