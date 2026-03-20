import argparse
import os
import shutil
import xml.etree.ElementTree as ET
import random
from pathlib import Path
from tqdm import tqdm
import yaml
 
# 支援的圖片格式
IMG_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
 
def parse_args():
    parser = argparse.ArgumentParser(description='VOC XML to YOLO Converter & Splitter')
    parser.add_argument('--input', type=str, required=True, help='包含圖片和XML的來源資料夾路徑')
    parser.add_argument('--output', type=str, required=True, help='輸出資料夾路徑')
    parser.add_argument('--split', type=float, nargs='+', default=[0.7, 0.2, 0.1], 
                        help='資料切分比例: Train Val Test (例如: 0.7 0.2 0.1)')
    parser.add_argument('--classes', type=str, default=None, 
                        help='(選用) 類別名稱列表 txt 檔路徑，將依照行號順序決定 ID')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子碼')
    return parser.parse_args()
 
def load_classes_from_txt(path):
    """讀取自訂類別檔案，回傳列表"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到類別檔案: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        # 去除空白行與前後空白
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes
 
def convert_bbox(size, box):
    """將 VOC bbox 轉為 YOLO (x, y, w, h)"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)
 
def analyze_and_group_files(input_dir, predefined_classes=None):
    """
    掃描資料夾，將檔案依照「類別組合」分組 (Stratified Split by Label Combination)。
    如果提供了 predefined_classes，只會統計在清單內的類別。
    不在清單內的標籤會被忽略，若整張圖都沒有有效標籤，視為 background。
    """
    files_by_signature = {}
    detected_classes = set()
    no_label_files = []
    
    valid_classes_set = set(predefined_classes) if predefined_classes else None
    
    input_path = Path(input_dir)
    image_files = [p for p in input_path.iterdir() if p.suffix.lower() in IMG_FORMATS]
    print(f"🔍 正在分析 {len(image_files)} 張影像...")

    for img_path in tqdm(image_files, desc="Analyzing"):
        xml_path = img_path.with_suffix('.xml')
        # 如果沒有對應 XML，視為無標籤 (或背景)
        if not xml_path.exists():
            no_label_files.append(img_path)
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = root.findall('object')
            
            # 收集該張圖片內所有出現的有效類別
            current_img_classes = set()
            if objects:
                for obj in objects:
                    name = obj.find('name').text
                    if valid_classes_set is not None:
                        if name in valid_classes_set:
                            current_img_classes.add(name)
                    else:
                        current_img_classes.add(name)
            
            # 判斷結果
            if not current_img_classes:
                # 存在的 XML 但沒有有效物件 (或物件都被過濾了)
                no_label_files.append(img_path)
            else:
                detected_classes.update(current_img_classes)
                # 建立簽名 (排序後的 tuple，例如 ('dog', 'person'))
                signature = tuple(sorted(list(current_img_classes)))
                
                if signature not in files_by_signature:
                    files_by_signature[signature] = []
                files_by_signature[signature].append(img_path)

        except Exception as e:
            print(f"⚠️ XML 解析錯誤: {xml_path.name} -> {e}")
            # 解析失敗則跳過，不加入任何列表

    # 處理背景圖
    if no_label_files:
        files_by_signature[('__background__',)] = no_label_files

    # 決定最終回傳的類別列表
    if predefined_classes:
        final_classes = predefined_classes 
    else:
        final_classes = sorted(list(detected_classes))

    return files_by_signature, final_classes
 
def process_dataset(files, split_name, class_mapping, output_dir):
    """處理單一分割的檔案複製與轉換"""
    img_save_dir = output_dir / 'images' / split_name
    lbl_save_dir = output_dir / 'labels' / split_name
    img_save_dir.mkdir(parents=True, exist_ok=True)
    lbl_save_dir.mkdir(parents=True, exist_ok=True)
    for img_path in files:
        xml_path = img_path.with_suffix('.xml')
        txt_path = lbl_save_dir / img_path.with_suffix('.txt').name
        shutil.copy2(img_path, img_save_dir / img_path.name)
        label_data = []
        if xml_path.exists():
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find('size')
                w = int(size.find('width').text)
                h = int(size.find('height').text)
                if w == 0 or h == 0: continue
 
                for obj in root.iter('object'):
                    cls_name = obj.find('name').text
                    # 只轉換在 mapping 內的類別
                    if cls_name in class_mapping:
                        cls_id = class_mapping[cls_name]
                        xmlbox = obj.find('bndbox')
                        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                        bb = convert_bbox((w, h), b)
                        label_data.append(f"{cls_id} {' '.join(f'{a:.6f}' for a in bb)}")
            except Exception:
                pass
        with open(txt_path, 'w') as f:
            f.write('\n'.join(label_data))
 
def main():
    args = parse_args()
    random.seed(args.seed)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    # 1. 處理自訂類別
    target_classes = None
    if args.classes:
        print(f"📜 讀取類別檔: {args.classes}")
        target_classes = load_classes_from_txt(args.classes)
        print(f"   -> 指定順序: {target_classes}")
 
    # 2. 分析資料
    if sum(args.split) != 1.0:
        total = sum(args.split)
        args.split = [x/total for x in args.split]
    files_by_signature, classes = analyze_and_group_files(input_dir, target_classes)
    # 建立 ID 對應表 (依照 classes 列表的順序)
    class_mapping = {name: i for i, name in enumerate(classes)}

    print("\n📊 資料統計 (基於影像內類別組合分組):")
    total_images = 0
    # 統計各單一類別出現的影像數
    class_counts = {c: 0 for c in classes}
    bg_count = 0
    
    # files_by_signature key 為 tuple, value 為 list
    for signature, files in files_by_signature.items():
        count = len(files)
        total_images += count
        if signature == ('__background__',):
            bg_count = count
            continue
        
        # 統計這組 signature 中的類別
        for c in signature:
            if c in class_counts:
                class_counts[c] += count

    for cls in classes:
        print(f"  - [ID: {class_mapping[cls]}] {cls}: {class_counts[cls]} 張 (含此類別)")
    
    if bg_count > 0:
        print(f"  - [Background]: {bg_count} 張 (無標記或不在清單內)")
        
    print(f"  - 總計處理: {total_images} 張")
    print("-" * 30)
 
    # 3. 分層抽樣
    split_groups = {'train': [], 'val': [], 'test': []}
    # 這裡的 key 改為 signature (tuple)
    all_signatures = list(files_by_signature.keys())
    
    for sig in all_signatures:
        files = files_by_signature[sig]
        if not files: continue
        random.shuffle(files)
        n_total = len(files)
        n_train = int(n_total * args.split[0])
        n_val = int(n_total * args.split[1])
        # 剩餘的給 test，確保總數正確
        split_groups['train'].extend(files[:n_train])
        split_groups['val'].extend(files[n_train : n_train + n_val])
        split_groups['test'].extend(files[n_train + n_val:])
 
    # 4. 轉換與輸出
    print(f"\n🚀 開始轉換並輸出至 {output_dir} ...")
    for split_name, files in split_groups.items():
        if not files: continue
        print(f"  正在處理 {split_name} 集 ({len(files)} 張)...")
        process_dataset(files, split_name, class_mapping, output_dir)
 
    # 5. 建立 data.yaml
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes  # 這裡會嚴格依照 txt 的順序
    }
    if not split_groups['test']:
        del yaml_content['test']
 
    with open(output_dir / 'data.yaml', 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
 
    print("\n✅ 完成！ data.yaml 已依照指定順序生成。")
 
if __name__ == '__main__':
    main()