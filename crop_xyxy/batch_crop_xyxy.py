#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import argparse
import os
import sys
from typing import List, Tuple
import cv2
import numpy as np
 
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
 
def parse_xyxy(s: str) -> Tuple[int, int, int, int]:
    """將字串 'x1,y1,x2,y2' 轉成 int tuple，容忍空白/小數（四捨五入）。"""
    try:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError
        x1, y1, x2, y2 = [int(round(float(v))) for v in parts]
        return x1, y1, x2, y2
    except Exception:
        raise argparse.ArgumentTypeError(f"無法解析座標：{s}（格式需為 x1,y1,x2,y2）")
 
def load_boxes_from_file(path: str) -> List[Tuple[int,int,int,int]]:
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            boxes.append(parse_xyxy(line))
    if not boxes:
        raise ValueError(f"檔案 {path} 未讀到任何有效座標。")
    return boxes
 
def normalize_box(x1, y1, x2, y2):
    """確保左上到右下，反向座標會自動交換。"""
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2
 
def apply_padding(x1, y1, x2, y2, pad):
    if pad <= 0:
        return x1, y1, x2, y2
    return (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
 
def clip_to_bounds(x1, y1, x2, y2, W, H):
    return (
        max(0, min(x1, W-1)),
        max(0, min(y1, H-1)),
        max(0, min(x2, W-1)),
        max(0, min(y2, H-1)),
    )
 
def list_images_in_dir(indir: str) -> List[str]:
    files = []
    for name in os.listdir(indir):
        p = os.path.join(indir, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)
 
def ensure_subfolders(out_root: str, n_boxes: int, prefix: str = "crop_") -> List[str]:
    subdirs = []
    for idx in range(1, n_boxes + 1):
        d = os.path.join(out_root, f"{prefix}{idx:03d}")
        os.makedirs(d, exist_ok=True)
        subdirs.append(d)
    return subdirs
 
def main():
    parser = argparse.ArgumentParser(
        description="批次裁切資料夾中的影像。依多個 xyxy 框，將各框結果分別存到 crop_001/、crop_002/... 子資料夾。"
    )
    parser.add_argument("--indir", required=True, help="輸入影像的資料夾路徑")
    parser.add_argument("--outdir", default="crops_batch", help="輸出根資料夾（預設：crops_batch）")
 
    # 取得裁切框：命令列直接給，或由檔案讀取
    parser.add_argument("-b", "--boxes", nargs="*", type=parse_xyxy,
                        help='直接在命令列提供一個或多個框，例如：-b 10,20,200,180 300,50,500,220')
    parser.add_argument("-f", "--boxes_file", type=str, help="從文字檔讀取多個框（每行一個 x1,y1,x2,y2，可用 # 註解）")
 
    # 裁切設定
    parser.add_argument("--pad", type=int, default=0, help="四邊加入像素邊距（可為 0 或正整數）")
    parser.add_argument("--no_clip", action="store_true", help="不要自動裁齊到影像邊界（預設會裁齊）")
    parser.add_argument("--keep_empty", action="store_true", help="若框越界導致空裁切仍強制輸出 1x1（通常不需要）")
 
    # 檔名副檔名設定
    parser.add_argument("--ext", default=None,
                        help="輸出副檔名（預設沿用各影像原始副檔名；可指定 jpg/png 等）")
    parser.add_argument("--prefix", default="crop_", help="子資料夾命名前綴（預設：crop_）")
    parser.add_argument("--overwrite", action="store_true", help="若輸出檔案已存在則覆蓋（預設不覆蓋，會跳過）")
 
    args = parser.parse_args()
 
    # 取得裁切框
    boxes: List[Tuple[int,int,int,int]] = []
    if args.boxes:
        boxes.extend(args.boxes)
    if args.boxes_file:
        boxes.extend(load_boxes_from_file(args.boxes_file))
    if not boxes:
        print("未提供任何框。請用 -b 或 -f 指定 xyxy。", file=sys.stderr)
        sys.exit(2)
 
    # 列出要處理的影像
    if not os.path.isdir(args.indir):
        print(f"找不到資料夾：{args.indir}", file=sys.stderr)
        sys.exit(1)
    images = list_images_in_dir(args.indir)
    if not images:
        print(f"資料夾內沒有支援的影像檔：{args.indir}", file=sys.stderr)
        sys.exit(1)
 
    # 準備輸出結構：每個框一個子資料夾
    os.makedirs(args.outdir, exist_ok=True)
    subdirs = ensure_subfolders(args.outdir, len(boxes), prefix=args.prefix)
 
    total_saved = [0] * len(boxes)
    total_skipped = 0
 
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳過] 讀不到影像：{img_path}")
            total_skipped += 1
            continue
        H, W = img.shape[:2]
 
        stem, in_ext = os.path.splitext(os.path.basename(img_path))
        # 決定輸出副檔名
        out_ext = (args.ext or in_ext.lstrip(".")).lower().lstrip(".")
        if out_ext not in {"jpg","jpeg","png","bmp","tif","tiff","webp"}:
            out_ext = "png"
 
        for k, b in enumerate(boxes, start=1):
            x1, y1, x2, y2 = normalize_box(*b)
            x1, y1, x2, y2 = apply_padding(x1, y1, x2, y2, args.pad)
 
            if not args.no_clip:
                x1, y1, x2, y2 = clip_to_bounds(x1, y1, x2, y2, W, H)
 
            x1, y1, x2, y2 = map(int, (round(x1), round(y1), round(x2), round(y2)))
            if x2 <= x1 or y2 <= y1:
                if args.keep_empty:
                    x2 = x1 + 1
                    y2 = y1 + 1
                else:
                    print(f"[跳過] {os.path.basename(img_path)} 第{k}框無效：{x1},{y1},{x2},{y2}")
                    continue
 
            crop = img[y1:y2, x1:x2]
            if crop.size == 0 and not args.keep_empty:
                print(f"[跳過] {os.path.basename(img_path)} 第{k}框裁切為空：{x1},{y1},{x2},{y2}")
                continue
 
            # 儲存到對應子資料夾，檔名沿用原檔名
            out_dir = subdirs[k-1]
            out_path = os.path.join(out_dir, f"{stem}.{out_ext}")
 
            if (not args.overwrite) and os.path.exists(out_path):
                # 若重名，避免覆蓋：在後面加序號
                suffix = 1
                candidate = os.path.join(out_dir, f"{stem}_{suffix}.{out_ext}")
                while os.path.exists(candidate):
                    suffix += 1
                    candidate = os.path.join(out_dir, f"{stem}_{suffix}.{out_ext}")
                out_path = candidate
 
            ok = cv2.imwrite(out_path, crop if crop.size > 0 else np.zeros((1,1,3), dtype=np.uint8))
            if ok:
                total_saved[k-1] += 1
                print(f"[儲存] {out_path}")
            else:
                print(f"[失敗] 無法寫入：{out_path}")
 
    # 總結
    print("\n=== 完成統計 ===")
    print(f"影像總數：{len(images)}，讀取失敗或跳過：{total_skipped}")
    for k, n in enumerate(total_saved, start=1):
        print(f"  {args.prefix}{k:03d}: 成功 {n} 張")
 
if __name__ == "__main__":
    main()