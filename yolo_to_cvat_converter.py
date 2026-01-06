#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO to CVAT for YOLO Converter

此腳本讀取一個影像資料夾和一個YOLO標記檔資料夾，
並將它們打包成一個與CVAT的 "YOLO" 格式匯入功能相容的ZIP檔案。
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from tqdm import tqdm

def create_cvat_yolo_zip(images_dir: Path, labels_dir: Path, class_names_file: Path, output_zip_path: Path):
    """
    建立用於CVAT YOLO匯入的ZIP壓縮檔。

    Args:
        images_dir (Path): 包含影像檔案的資料夾。
        labels_dir (Path): 包含YOLO .txt標記檔的資料夾。
        class_names_file (Path): 包含類別名稱的 .txt 檔案路徑 (每行一個)。
        output_zip_path (Path): 最終 .zip 檔案的儲存路徑。
    """
    # 使用暫存資料夾來準備檔案
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        print(f"[*] 正在暫存目錄中準備檔案: {tmp_path}")

        # 1. 建立用於存放影像和標籤的資料夾
        data_folder = tmp_path / 'obj_train_data'
        data_folder.mkdir()

        # 2. 讀取類別名稱並建立 obj.names
        try:
            with open(class_names_file, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f if line.strip()]
            
            if not class_names:
                raise ValueError("類別名稱檔案是空的。")

            # 將類別名稱檔案複製為 obj.names
            shutil.copy2(class_names_file, tmp_path / 'obj.names')
            print(f"[*] 找到 {len(class_names)} 個類別: {class_names}")

        except FileNotFoundError:
            print(f"❌ 錯誤: 在 '{class_names_file}' 找不到類別名稱檔案")
            return
        except Exception as e:
            print(f"❌ 讀取類別名稱檔案時發生錯誤: {e}")
            return

        # 3. 建立 obj.data 檔案
        num_classes = len(class_names)
        obj_data_content = (
            f"classes = {num_classes}\n"
            f"train = train.txt\n"
            f"names = obj.names\n"
            f"backup = backup/\n"
        )
        (tmp_path / 'obj.data').write_text(obj_data_content, encoding='utf-8')
        print("[*] 已建立 obj.data 檔案。")

        # 4. 遞迴掃描影像和標籤，並建立 train.txt
        print("[*] 正在遞迴掃描影像目錄...")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = sorted([p for p in images_dir.rglob('*') if p.suffix.lower() in image_extensions])
        
        if not image_files:
            print("❌ 錯誤: 在指定的影像目錄中找不到任何影像檔案。")
            return
        
        print(f"[*] 找到 {len(image_files)} 張影像。")

        print("[*] 正在遞迴掃描標籤目錄...")
        # 建立一個從標籤檔名到完整路徑的映射
        label_map = {p.name: p for p in labels_dir.rglob('*.txt')}
        print(f"[*] 找到 {len(label_map)} 個 .txt 標籤檔。")

        train_txt_content = []
        print(f"[*] 正在配對影像和標籤並複製檔案...")

        for img_path in tqdm(image_files, desc="複製檔案中"):
            label_filename = img_path.stem + '.txt'
            label_path = label_map.get(label_filename)

            if not label_path:
                relative_img_path = img_path.relative_to(images_dir)
                print(f"⚠️  警告: 找不到 '{relative_img_path}' 對應的標籤檔 '{label_filename}'，已跳過。")
                continue

            # 將影像和標籤複製到資料夾
            shutil.copy2(img_path, data_folder / img_path.name)
            shutil.copy2(label_path, data_folder / label_filename)

            # 在 train.txt 中新增一筆紀錄
            train_txt_content.append(f"obj_train_data/{img_path.name}")

        if not train_txt_content:
            print("❌ 錯誤: 找不到任何有效的影像-標籤配對來處理。")
            return
            
        (tmp_path / 'train.txt').write_text('\n'.join(train_txt_content), encoding='utf-8')
        print("[*] 已建立 train.txt 檔案。")

        # 5. 建立 ZIP 壓縮檔
        # make_archive 的第一個參數不應包含 .zip 副檔名
        archive_name = output_zip_path.with_suffix('')
        shutil.make_archive(str(archive_name), 'zip', root_dir=tmp_path)
        
        print("\n" + "="*50)
        print("✅ 成功！已建立CVAT YOLO資料包。")
        print(f"📦 輸出檔案: {output_zip_path}")
        print("="*50)


def main():
    """主函式，用於解析參數並執行腳本。"""
    parser = argparse.ArgumentParser(
        description="將影像和YOLO標記檔打包成用於CVAT匯入的ZIP檔案。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
使用範例:
  python yolo_to_cvat_converter.py \\
    --images-dir ./my_dataset/images \\
    --labels-dir ./my_dataset/labels \\
    --class-names ./my_dataset/classes.txt \\
    --output-zip ./cvat_upload.zip
        """
    )
    parser.add_argument('--images-dir', type=Path, required=True, help="包含影像檔案的來源資料夾路徑。")
    parser.add_argument('--labels-dir', type=Path, required=True, help="包含YOLO .txt標記檔的來源資料夾路徑。")
    parser.add_argument('--class-names', type=Path, required=True, help="包含類別名稱的 .txt 檔案路徑 (每行一個名稱)。")
    parser.add_argument('--output-zip', type=Path, required=True, help="最終輸出的 .zip 檔案路徑。")

    args = parser.parse_args()

    # --- 驗證路徑 ---
    if not args.images_dir.is_dir():
        print(f"❌ 錯誤: 影像目錄不存在或不是一個有效的資料夾: {args.images_dir}")
        return
    if not args.labels_dir.is_dir():
        print(f"❌ 錯誤: 標籤目錄不存在或不是一個有效的資料夾: {args.labels_dir}")
        return
    if not args.class_names.is_file():
        print(f"❌ 錯誤: 類別名稱檔案不存在: {args.class_names}")
        return
        
    # 確保輸出目錄存在
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)

    create_cvat_yolo_zip(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        class_names_file=args.class_names,
        output_zip_path=args.output_zip
    )

if __name__ == "__main__":
    main()