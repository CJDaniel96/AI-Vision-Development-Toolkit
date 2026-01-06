#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import cv2
import argparse
import os
import csv
 
PTS = []
IMG = None
SHOW = None
WIN = "Click to get (x, y)"
 
def draw_point(img, x, y, idx):
    cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
    cv2.putText(img, f"{idx}:{x},{y}", (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
 
def on_mouse(event, x, y, flags, param):
    global PTS, SHOW
    if event == cv2.EVENT_LBUTTONDOWN:
        PTS.append((x, y))
        print(f"({x}, {y})")
        draw_point(SHOW, x, y, len(PTS))
        cv2.imshow(WIN, SHOW)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右鍵撤銷最後一個點
        if PTS:
            PTS.pop()
            SHOW[:] = IMG
            for i, (px, py) in enumerate(PTS, start=1):
                draw_point(SHOW, px, py, i)
            cv2.imshow(WIN, SHOW)
 
def save_csv(path, points):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "x", "y"])
        for i, (x, y) in enumerate(points, start=1):
            w.writerow([i, x, y])
 
def main():
    parser = argparse.ArgumentParser(description="滑鼠點選影像回傳座標")
    parser.add_argument("-i", "--image", required=True, help="輸入影像路徑")
    parser.add_argument("-o", "--out", default="points.csv", help="輸出 CSV（按 S 儲存）")
    parser.add_argument("--fit", action="store_true",
                        help="視窗自適應螢幕顯示大小（僅縮放顯示，不改變回傳座標）")
    args = parser.parse_args()
 
    global IMG, SHOW
 
    IMG = cv2.imread(args.image)
    if IMG is None:
        raise SystemExit(f"讀不到影像：{args.image}")
 
    SHOW = IMG.copy()
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE if not args.fit else cv2.WINDOW_NORMAL)
    if args.fit:
        cv2.resizeWindow(WIN, 1280, 720)  # 僅調整視窗大小，實際座標仍以原圖像素為準
    cv2.setMouseCallback(WIN, on_mouse)
 
    print("操作說明：")
    print("  左鍵：加入座標   右鍵：撤銷上一個點")
    print("  S：存檔成 CSV   C：清除全部點   Q/ESC：離開")
    cv2.imshow(WIN, SHOW)
 
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q'), ord('Q')):  # ESC 或 Q
            break
        elif key in (ord('s'), ord('S')):
            save_csv(args.out, PTS)
            print(f"已儲存 {len(PTS)} 筆到 {args.out}")
        elif key in (ord('c'), ord('C')):
            PTS.clear()
            SHOW[:] = IMG
            cv2.imshow(WIN, SHOW)
 
    cv2.destroyAllWindows()
    # 離開前在終端機列出全部點
    if PTS:
        print("全部座標：")
        for i, (x, y) in enumerate(PTS, start=1):
            print(f"{i}: ({x}, {y})")
 
if __name__ == "__main__":
    main()