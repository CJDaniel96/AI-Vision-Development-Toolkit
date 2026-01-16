import os
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="合併並重新命名 CVAT PASCAL VOC 資料集 (包含 XML 與 影像)。"
    )
    parser.add_argument(
        '--inputs', 
        nargs='+', 
        required=True, 
        help='輸入資料夾路徑列表 (可以輸入多個資料夾，例如: video1 video2 video3)'
    )
    parser.add_argument(
        '--output', 
        required=True, 
        help='輸出資料夾路徑 (所有檔案將被複製到此處)'
    )
    return parser.parse_args()

def find_image_file(xml_path: Path, image_filename: str) -> Path:
    """
    嘗試尋找對應的影像檔案。
    CVAT PASCAL VOC 結構通常是:
    Root/
      Annotations/xxx.xml
      JPEGImages/xxx.jpg
    
    但也可能是扁平結構。此函式會嘗試在不同位置尋找影像。
    """
    # 1. 嘗試在 XML 同級目錄尋找 (扁平結構)
    if (xml_path.parent / image_filename).exists():
        return xml_path.parent / image_filename
    
    # 2. 嘗試在 XML 上一層的 JPEGImages 資料夾尋找 (標準 VOC 結構)
    # 假設 xml 在 Annotations 資料夾內
    voc_img_path = xml_path.parent.parent / 'JPEGImages' / image_filename
    if voc_img_path.exists():
        return voc_img_path
        
    return None

def process_datasets(input_dirs, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"準備處理 {len(input_dirs)} 個輸入資料夾...")
    print(f"輸出位置: {output_path.resolve()}")

    total_processed = 0
    
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"警告: 找不到輸入資料夾 {input_dir}，已跳過。")
            continue

        # 使用資料夾名稱作為前綴 (Prefix)
        folder_name = input_path.name
        print(f"\n正在處理資料夾: {folder_name}")

        # 遞迴搜尋所有 .xml 檔案
        xml_files = list(input_path.rglob("*.xml"))
        
        if not xml_files:
            print(f"  - 在 {folder_name} 中未發現 XML 檔案。")
            continue

        for xml_file in tqdm(xml_files, desc=f"  Processing {folder_name}"):
            try:
                # 解析 XML
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # 獲取原始檔名 (例如 frame_000000.jpg)
                filename_node = root.find('filename')
                if filename_node is None:
                    continue
                    
                original_img_name = filename_node.text
                
                # 尋找實際的影像檔案
                src_img_path = find_image_file(xml_file, original_img_name)
                
                if src_img_path is None:
                    # 嘗試處理副檔名大小寫不一致的問題 (例如 xml寫 .jpg 但檔案是 .JPG)
                    # 這裡做一個簡單的 fallback
                    found = False
                    for ext in ['.jpg', '.JPG', '.png', '.PNG', '.jpeg']:
                        stem = Path(original_img_name).stem
                        temp_name = stem + ext
                        src_img_path = find_image_file(xml_file, temp_name)
                        if src_img_path:
                            original_img_name = temp_name # 更新為實際檔名
                            found = True
                            break
                    if not found:
                        # print(f"    警告: 找不到影像 {original_img_name} (對應 XML: {xml_file.name})")
                        continue

                # --- 定義新檔名 ---
                # 格式: {資料夾名}_{原始檔名}
                # 例如: video1_frame_000000.jpg
                new_img_name = f"{folder_name}_{original_img_name}"
                new_xml_name = f"{folder_name}_{xml_file.name}"
                
                dst_img_path = output_path / new_img_name
                dst_xml_path = output_path / new_xml_name

                # --- 修改 XML 內容 ---
                # 1. 修改 <filename>
                filename_node.text = new_img_name
                
                # 2. 修改 <path> (通常包含絕對路徑，建議更新或移除，這裡我們更新為新路徑)
                path_node = root.find('path')
                if path_node is not None:
                    path_node.text = str(dst_img_path.absolute())
                else:
                    # 如果沒有 path node，有些工具需要它，可以選擇新增 (視需求而定)
                    pass

                # --- 寫入檔案 ---
                # 1. 儲存修改後的 XML
                tree.write(dst_xml_path, encoding='utf-8')
                
                # 2. 複製影像檔案並重新命名
                shutil.copy2(src_img_path, dst_img_path)
                
                total_processed += 1

            except Exception as e:
                print(f"    處理 {xml_file.name} 時發生錯誤: {e}")

    print(f"\n完成！共處理並合併了 {total_processed} 組影像與 XML 到 {output_dir}。")

if __name__ == "__main__":
    args = parse_args()
    process_datasets(args.inputs, args.output)
