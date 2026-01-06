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
    掃描資料夾，將檔案依照類別分組。
    如果提供了 predefined_classes，只會統計在清單內的類別，
    不在清單內的標籤會導致該圖片被視為 background (無標籤)。
    """
    files_by_class = {}
    detected_classes = set()
    no_label_files = []
    # 如果有預定義類別，先初始化字典，確保順序正確
    if predefined_classes:
        for cls in predefined_classes:
            files_by_class[cls] = []
    input_path = Path(input_dir)
    image_files = [p for p in input_path.iterdir() if p.suffix.lower() in IMG_FORMATS]
    print(f"🔍 正在分析 {len(image_files)} 張影像...")
 
    for img_path in tqdm(image_files, desc="Analyzing"):
        xml_path = img_path.with_suffix('.xml')
        if not xml_path.exists():
            no_label_files.append(img_path)
            continue
 
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = root.findall('object')
            if not objects:
                no_label_files.append(img_path)
                continue
            # 取得第一個物件的名稱
            cls_name = objects[0].find('name').text
            # 邏輯判斷
            if predefined_classes is not None:
                # 【模式 A：自訂類別】
                if cls_name in files_by_class:
                    files_by_class[cls_name].append(img_path)
                    detected_classes.add(cls_name)
                else:
                    # 標籤存在但不在我們的清單內 -> 視為背景圖
                    no_label_files.append(img_path)
            else:
                # 【模式 B：自動偵測】
                if cls_name not in files_by_class:
                    files_by_class[cls_name] = []
                files_by_class[cls_name].append(img_path)
                detected_classes.add(cls_name)
        except Exception as e:
            print(f"⚠️ XML 解析錯誤: {xml_path.name} -> {e}")
 
    # 將無標籤 (或標籤被過濾掉) 的檔案加入 background
    if no_label_files:
        files_by_class['__background__'] = no_label_files
 
    # 決定最終回傳的類別列表
    if predefined_classes:
        final_classes = predefined_classes # 保持使用者定義的順序
    else:
        final_classes = sorted(list(detected_classes)) # 自動排序
 
    return files_by_class, final_classes
 
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
    files_by_class, classes = analyze_and_group_files(input_dir, target_classes)
    # 建立 ID 對應表 (依照 classes 列表的順序)
    class_mapping = {name: i for i, name in enumerate(classes)}
    print("\n📊 資料統計:")
    total_images = 0
    for cls in classes:
        # 注意：如果是自訂類別，有些類別可能沒有圖片，要防止 Key Error
        count = len(files_by_class.get(cls, []))
        total_images += count
        print(f"  - [ID: {class_mapping[cls]}] {cls}: {count} 張")
    bg_count = len(files_by_class.get('__background__', []))
    if bg_count > 0:
        print(f"  - [Background]: {bg_count} 張 (無標記或不在清單內)")
        total_images += bg_count
    print(f"  - 總計處理: {total_images} 張")
    print("-" * 30)
 
    # 3. 分層抽樣
    split_groups = {'train': [], 'val': [], 'test': []}
    # 包含背景圖的所有類別 (包括 __background__)
    all_keys = list(files_by_class.keys())
    for cls in all_keys:
        files = files_by_class[cls]
        if not files: continue
        random.shuffle(files)
        n_total = len(files)
        n_train = int(n_total * args.split[0])
        n_val = int(n_total * args.split[1])
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