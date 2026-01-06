import os
import argparse
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import yaml
 
def parse_args():
    parser = argparse.ArgumentParser(description="Convert CVAT XML (Segmentation) to YOLOv8/Ultralytics format.")
    parser.add_argument('--xml', type=str, required=True, help='Path to the CVAT XML file (CVAT for Images 1.1 format)')
    parser.add_argument('--img-dir', type=str, required=True, help='Directory containing the original images')
    parser.add_argument('--out-dir', type=str, required=True, help='Output directory for the dataset')
    parser.add_argument('--split', type=float, nargs=3, default=[0.7, 0.2, 0.1], 
                        help='Split ratio for Train, Val, Test (e.g., 0.7 0.2 0.1). Must sum to 1.')
    return parser.parse_args()
 
def normalize_coords(points_str, width, height):
    """
    將 CVAT 的 points 字符串轉換為 YOLO 正規化格式 list
    支援格式: "x1,y1;x2,y2" (分號分隔) 或 "x1,y1 x2,y2" (空白分隔)
    """
    normalized_points = []
    # 1. 判斷分隔符號：如果有分號，先用分號切分；否則用空白切分
    if ';' in points_str:
        pairs = points_str.split(';')
    else:
        pairs = points_str.split()
    for pair in pairs:
        # 去除可能多餘的空白
        pair = pair.strip()
        if not pair:
            continue
        try:
            # 2. 解析 x,y (以逗號分隔)
            coords = pair.split(',')
            if len(coords) != 2:
                continue # 跳過格式錯誤的點
            x = float(coords[0])
            y = float(coords[1])
            # 3. 正規化 (0~1)
            nx = min(max(x / width, 0.0), 1.0)
            ny = min(max(y / height, 0.0), 1.0)
            normalized_points.extend([nx, ny])
        except ValueError:
            print(f"Warning: Could not parse coordinates: {pair}")
            continue
    return normalized_points
 
def main():
    args = parse_args()
    # 檢查比例總和
    if abs(sum(args.split) - 1.0) > 1e-5:
        print(f"Error: Split ratios must sum to 1. Current sum: {sum(args.split)}")
        return
 
    xml_path = Path(args.xml)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
 
    if not xml_path.exists():
        print(f"Error: XML file not found at {xml_path}")
        return
 
    # 1. 解析 XML
    print(f"Parsing XML: {xml_path}...")
    tree = ET.parse(xml_path)
    root = tree.getroot()
 
    # 2. 收集所有類別名稱以建立 ID 對應 (Class Mapping)
    labels = set()
    images_data = []
 
    print("Scanning annotations...")
    for image in root.findall('image'):
        filename = image.get('name')
        # 安全轉換寬高，避免部分 XML 屬性缺失
        try:
            width = float(image.get('width'))
            height = float(image.get('height'))
        except (TypeError, ValueError):
            print(f"Skipping image {filename}: Missing width/height attributes.")
            continue
        img_labels = []
        # 處理多邊形 (Polygons)
        for polygon in image.findall('polygon'):
            label = polygon.get('label')
            points_str = polygon.get('points')
            if not label or not points_str:
                continue
            labels.add(label)
            norm_points = normalize_coords(points_str, width, height)
            # 只有當點數足夠構成多邊形時才加入 (至少3個點 -> 6個數值)
            if len(norm_points) >= 6:
                img_labels.append({'class': label, 'points': norm_points})
        # 即使該圖片沒有標註，也建議加入 (Background image)，或視需求過濾
        # 這裡設定：如果有標註才加入資料集，或者您可以註解掉下面這行來包含空圖片
        if img_labels: 
            images_data.append({
                'filename': filename,
                'labels': img_labels
            })
 
    # 排序類別以確保 ID 一致
    class_names = sorted(list(labels))
    class_map = {name: i for i, name in enumerate(class_names)}
    print(f"Found classes: {class_map}")
    print(f"Total annotated images: {len(images_data)}")
 
    # 3. 隨機打亂並切分資料集
    random.seed(42)
    random.shuffle(images_data)
    total_count = len(images_data)
    train_end = int(total_count * args.split[0])
    val_end = train_end + int(total_count * args.split[1])
    datasets = {
        'train': images_data[:train_end],
        'val': images_data[train_end:val_end],
        'test': images_data[val_end:]
    }
 
    # 4. 建立目錄並寫入檔案
    for split_name, split_data in datasets.items():
        if not split_data:
            continue
        print(f"Processing {split_name} set ({len(split_data)} images)...")
        save_img_dir = out_dir / split_name / 'images'
        save_lbl_dir = out_dir / split_name / 'labels'
        save_img_dir.mkdir(parents=True, exist_ok=True)
        save_lbl_dir.mkdir(parents=True, exist_ok=True)
        for item in tqdm(split_data):
            # 處理圖片路徑：有些 CVAT 導出包含資料夾，我們只取檔名去原始目錄找
            img_filename = Path(item['filename']).name 
            src_img_path = img_dir / img_filename
            # 如果原始目錄找不到，嘗試直接用 XML 裡的完整路徑 (如果 XML 裡是相對路徑)
            if not src_img_path.exists():
                 src_img_path = img_dir / item['filename']
 
            if not src_img_path.exists():
                # 最後嘗試遞迴搜尋 (Optional: 視需求開啟，目前先印警告)
                print(f"Warning: Image not found {item['filename']}, skipping.")
                continue
            # 複製圖片
            dst_img_path = save_img_dir / img_filename
            shutil.copy2(src_img_path, dst_img_path)
            # 寫入 YOLO TXT
            txt_filename = Path(img_filename).stem + '.txt'
            txt_path = save_lbl_dir / txt_filename
            with open(txt_path, 'w') as f:
                for label_data in item['labels']:
                    class_id = class_map[label_data['class']]
                    points = label_data['points']
                    points_str = " ".join([f"{p:.6f}" for p in points])
                    f.write(f"{class_id} {points_str}\n")
 
    # 5. 生成 data.yaml
    yaml_content = {
        'path': str(out_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    yaml_path = out_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"\nSuccess! Dataset generated at: {out_dir}")
    print(f"Data YAML created at: {yaml_path}")
 
if __name__ == "__main__":
    main()