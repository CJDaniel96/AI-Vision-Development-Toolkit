#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import argparse
import os
import sys
from typing import List, Tuple
import cv2
import numpy as np
 
def parse_xyxy(s: str) -> Tuple[int, int, int, int]:
    """
    將字串 "x1,y1,x2,y2" 轉成 int tuple，並容忍空白/小數（四捨五入）。
    """
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
            try:
                boxes.append(parse_xyxy(line))
            except Exception as e:
                raise ValueError(f"讀取 {path} 第 {ln} 行失敗：{e}")
    return boxes
 
def normalize_box(x1, y1, x2, y2):
    # 修正反向座標（若使用者給 x2 < x1 或 y2 < y1）
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2
 
def apply_padding(x1, y1, x2, y2, pad, W, H):
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
 
def crop_and_save(img, box, idx, outdir, prefix, ext, keep_if_empty=False):
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]  # 注意：OpenCV 的切片是 [y1:y2, x1:x2)
    if crop.size == 0:
        if keep_if_empty:
            # 仍然輸出空裁切（不建議，一般跳過）
            out_path = os.path.join(outdir, f"{prefix}{idx:03d}_EMPTY.{ext}")
            cv2.imwrite(out_path, np.zeros((1,1,3), dtype=np.uint8))
            return out_path, False
        return None, False
    out_path = os.path.join(outdir, f"{prefix}{idx:03d}.{ext}")
    cv2.imwrite(out_path, crop)
    return out_path, True
 
def main():
    parser = argparse.ArgumentParser(
        description="依 xyxy 框裁切影像並另存。xyxy = x1,y1,x2,y2（像素座標）"
    )
    parser.add_argument("-i", "--image", required=True, help="輸入影像路徑")
    parser.add_argument("-b", "--boxes", nargs="*", type=parse_xyxy,
                        help='直接在命令列提供一個或多個框，例如：-b 10,20,200,180 300,50,500,220')
    parser.add_argument("-f", "--boxes_file", type=str,
                        help="從文字檔讀取多個框（每行一個 x1,y1,x2,y2，可用 # 註解）")
    parser.add_argument("-o", "--outdir", default="crops", help="輸出資料夾（預設：crops）")
    parser.add_argument("--prefix", default="crop_", help="輸出檔名前綴（預設：crop_）")
    parser.add_argument("--ext", default=None, help="輸出副檔名（預設沿用輸入影像副檔名，常見 jpg/png）")
    parser.add_argument("--pad", type=int, default=0, help="在四邊加入像素邊距（可為 0 或正整數）")
    parser.add_argument("--no_clip", action="store_true", help="不要自動裁齊到影像邊界（預設會裁齊）")
    parser.add_argument("--keep_empty", action="store_true", help="若框超出範圍導致空裁切，仍強制輸出（通常不需要）")
 
    args = parser.parse_args()
 
    # 讀影像
    img = cv2.imread(args.image)
    if img is None:
        print(f"無法讀取影像：{args.image}", file=sys.stderr)
        sys.exit(1)
    H, W = img.shape[:2]
 
    # 收集框
    all_boxes: List[Tuple[int,int,int,int]] = []
    if args.boxes:
        all_boxes.extend(args.boxes)
    if args.boxes_file:
        all_boxes.extend(load_boxes_from_file(args.boxes_file))
    if not all_boxes:
        print("未提供任何框。請用 -b 或 -f 指定 xyxy。", file=sys.stderr)
        sys.exit(2)
 
    # 準備輸出資料夾與副檔名
    os.makedirs(args.outdir, exist_ok=True)
    if args.ext is None:
        base_ext = os.path.splitext(args.image)[1].lower().lstrip(".")
        if base_ext in {"jpg", "jpeg", "png", "bmp", "tiff"}:
            args.ext = base_ext
        else:
            args.ext = "png"  # 不認得就用 png
 
    saved = 0
    for idx, b in enumerate(all_boxes, start=1):
        x1, y1, x2, y2 = normalize_box(*b)
        # 加 padding
        x1, y1, x2, y2 = apply_padding(x1, y1, x2, y2, args.pad, W, H)
        # (可選)裁齊到影像邊界
        if not args.no_clip:
            x1, y1, x2, y2 = clip_to_bounds(x1, y1, x2, y2, W, H)
        # 轉為整數且確保非負
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
 
        # 避免相同或倒置造成空區塊
        if x2 <= x1 or y2 <= y1:
            if args.keep_empty:
                x2 = x1 + 1
                y2 = y1 + 1
            else:
                print(f"[跳過] 第 {idx} 個框無效（x2<=x1 或 y2<=y1）：{x1},{y1},{x2},{y2}")
                continue
 
        out_path, ok = crop_and_save(img, (x1, y1, x2, y2), idx, args.outdir, args.prefix, args.ext, args.keep_empty)
        if ok:
            print(f"[儲存] {out_path}")
            saved += 1
        else:
            print(f"[跳過] 第 {idx} 個框裁切結果為空：{x1},{y1},{x2},{y2}")
 
    print(f"完成，成功輸出 {saved} 張。")
 
if __name__ == "__main__":
    main()