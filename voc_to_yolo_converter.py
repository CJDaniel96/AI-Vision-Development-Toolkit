#!/usr/bin/env python3
"""
PASCAL VOC to YOLO format converter
將PASCAL VOC XML標記檔轉換為YOLO TXT格式
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

class VOCToYOLOConverter:
    def __init__(self, class_mapping: Optional[Dict[str, int]] = None):
        """
        初始化轉換器
        
        Args:
            class_mapping: 類別名稱到ID的映射，如果為None則會自動生成
        """
        self.class_mapping = class_mapping or {}
        self.class_counter = 0
        
    def parse_xml(self, xml_path: str) -> Tuple[str, int, int, List[Dict]]:
        """
        解析PASCAL VOC XML檔案
        
        Args:
            xml_path: XML檔案路徑
            
        Returns:
            檔名, 圖片寬度, 圖片高度, 物件列表
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 獲取圖片資訊
            filename = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # 解析物件
            objects = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                
                # 處理類別映射
                if name not in self.class_mapping:
                    self.class_mapping[name] = self.class_counter
                    self.class_counter += 1
                
                # 獲取邊界框
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                objects.append({
                    'name': name,
                    'class_id': self.class_mapping[name],
                    'bbox': (xmin, ymin, xmax, ymax)
                })
                
            return filename, width, height, objects
            
        except Exception as e:
            print(f"錯誤：無法解析XML檔案 {xml_path}: {e}")
            return None, None, None, None
    
    def convert_bbox_to_yolo(self, bbox: Tuple[float, float, float, float], 
                           img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        將PASCAL VOC邊界框轉換為YOLO格式
        
        Args:
            bbox: (xmin, ymin, xmax, ymax)
            img_width: 圖片寬度
            img_height: 圖片高度
            
        Returns:
            (x_center, y_center, width, height) 正規化後的座標
        """
        xmin, ymin, xmax, ymax = bbox
        
        # 計算中心點和寬高
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        
        # 正規化到0-1範圍
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return x_center, y_center, width, height
    
    def convert_file(self, xml_path: str, output_dir: str) -> bool:
        """
        轉換單個XML檔案
        
        Args:
            xml_path: XML檔案路徑
            output_dir: 輸出目錄
            
        Returns:
            轉換是否成功
        """
        filename, width, height, objects = self.parse_xml(xml_path)
        
        if filename is None:
            return False
            
        # 生成輸出檔案路徑
        base_name = Path(xml_path).stem
        output_path = os.path.join(output_dir, f"{base_name}.txt")
        
        try:
            with open(output_path, 'w') as f:
                for obj in objects:
                    class_id = obj['class_id']
                    x_center, y_center, w, h = self.convert_bbox_to_yolo(
                        obj['bbox'], width, height
                    )
                    
                    # YOLO格式：class_id x_center y_center width height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                    
            return True
            
        except Exception as e:
            print(f"錯誤：無法寫入檔案 {output_path}: {e}")
            return False
    
    def save_class_mapping(self, output_dir: str):
        """
        儲存類別映射到檔案
        
        Args:
            output_dir: 輸出目錄
        """
        # 儲存為classes.txt (YOLO常用格式)
        classes_path = os.path.join(output_dir, "classes.txt")
        with open(classes_path, 'w', encoding='utf-8') as f:
            # 按照ID順序排列
            sorted_classes = sorted(self.class_mapping.items(), key=lambda x: x[1])
            for class_name, _ in sorted_classes:
                f.write(f"{class_name}\n")
        
        # 儲存為JSON格式 (便於程式讀取)
        mapping_path = os.path.join(output_dir, "class_mapping.json")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.class_mapping, f, ensure_ascii=False, indent=2)
        
        print(f"類別映射已儲存到: {classes_path} 和 {mapping_path}")

def find_image_files(image_dir: str) -> Dict[str, str]:
    """
    尋找圖片檔案並建立檔名映射
    
    Args:
        image_dir: 圖片目錄
        
    Returns:
        檔名(不含副檔名)到完整路徑的映射
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = {}
    
    for ext in image_extensions:
        for img_path in Path(image_dir).glob(f"*{ext}"):
            stem = img_path.stem
            image_files[stem] = str(img_path)
        for img_path in Path(image_dir).glob(f"*{ext.upper()}"):
            stem = img_path.stem
            image_files[stem] = str(img_path)
    
    return image_files

def load_class_mapping(mapping_file: str) -> Dict[str, int]:
    """
    從檔案載入類別映射
    
    Args:
        mapping_file: 映射檔案路徑
        
    Returns:
        類別映射字典
    """
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            if mapping_file.endswith('.json'):
                return json.load(f)
            else:
                # 假設是classes.txt格式
                mapping = {}
                for i, line in enumerate(f):
                    class_name = line.strip()
                    if class_name:
                        mapping[class_name] = i
                return mapping
    except Exception as e:
        print(f"錯誤：無法載入類別映射檔案 {mapping_file}: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(
        description="將PASCAL VOC XML標記檔轉換為YOLO TXT格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  %(prog)s --xml-dir /path/to/xml --image-dir /path/to/images --output-dir /path/to/output
  %(prog)s -x ./annotations -i ./images -o ./yolo_labels
  %(prog)s -x ./xml -i ./img -o ./out --class-mapping classes.json
        """
    )
    
    parser.add_argument('-x', '--xml-dir', required=True,
                       help='XML標記檔案目錄')
    parser.add_argument('-i', '--image-dir', required=True,
                       help='圖片檔案目錄')
    parser.add_argument('-o', '--output-dir', required=True,
                       help='輸出目錄')
    parser.add_argument('-c', '--class-mapping',
                       help='類別映射檔案 (JSON或classes.txt格式)')
    parser.add_argument('--check-images', action='store_true',
                       help='檢查XML檔案是否有對應的圖片檔案')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='顯示詳細信息')
    
    args = parser.parse_args()
    
    # 檢查輸入目錄
    if not os.path.isdir(args.xml_dir):
        print(f"錯誤：XML目錄不存在: {args.xml_dir}")
        sys.exit(1)
        
    if not os.path.isdir(args.image_dir):
        print(f"錯誤：圖片目錄不存在: {args.image_dir}")
        sys.exit(1)
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 載入類別映射 (如果提供)
    class_mapping = None
    if args.class_mapping:
        class_mapping = load_class_mapping(args.class_mapping)
        if args.verbose:
            print(f"載入類別映射: {class_mapping}")
    
    # 初始化轉換器
    converter = VOCToYOLOConverter(class_mapping)
    
    # 尋找圖片檔案 (如果需要檢查)
    image_files = {}
    if args.check_images:
        image_files = find_image_files(args.image_dir)
        if args.verbose:
            print(f"找到 {len(image_files)} 個圖片檔案")
    
    # 尋找XML檔案
    xml_files = list(Path(args.xml_dir).glob("*.xml"))
    if not xml_files:
        print(f"錯誤：在 {args.xml_dir} 中未找到XML檔案")
        sys.exit(1)
    
    print(f"找到 {len(xml_files)} 個XML檔案")
    
    # 轉換檔案
    success_count = 0
    failed_files = []
    
    for xml_path in xml_files:
        # 檢查對應的圖片是否存在
        if args.check_images:
            stem = xml_path.stem
            if stem not in image_files:
                print(f"警告：找不到對應的圖片檔案: {stem}")
                continue
        
        if args.verbose:
            print(f"轉換: {xml_path.name}")
            
        if converter.convert_file(str(xml_path), args.output_dir):
            success_count += 1
        else:
            failed_files.append(xml_path.name)
    
    # 儲存類別映射
    if converter.class_mapping:
        converter.save_class_mapping(args.output_dir)
    
    # 顯示結果
    print(f"\n轉換完成！")
    print(f"成功轉換: {success_count} 個檔案")
    print(f"轉換失敗: {len(failed_files)} 個檔案")
    
    if failed_files:
        print(f"失敗的檔案: {', '.join(failed_files)}")
    
    if converter.class_mapping:
        print(f"找到 {len(converter.class_mapping)} 個類別: {list(converter.class_mapping.keys())}")
    
    print(f"輸出目錄: {args.output_dir}")

if __name__ == "__main__":
    main()