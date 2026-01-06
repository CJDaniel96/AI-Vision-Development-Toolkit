#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5物件偵測與PASCAL VOC XML轉換程式 (CLI版本)
功能：使用YOLOv5對影像進行物件偵測，並將結果轉換為PASCAL VOC XML格式
作者：高階程式設計師助手
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import cv2
import argparse
import sys
from typing import List, Dict, Tuple
import torch
from ultralytics import YOLO

def load_yolo_model(model_path: Path, device='cpu'):
    """
    載入YOLOv5模型
    
    Args:
        model_path (str): 模型路徑
        device (str): 計算設備 ('cpu' 或 'cuda')
        
    Returns:
        model: YOLOv5模型物件
    """
    try:
        print(f"正在使用 Ultralytics 引擎載入 {model_path.stem} 模型到 {device}...")
        model = YOLO(model_path)
        model.to(device)
        print("模型載入成功！")
        return model
    except Exception as e:
        print(f"模型載入失敗: {e}")
        return None

def create_pascal_voc_xml(image_path: str, detections: List[Dict], output_path: str, width: int, height: int, depth: int = 3) -> bool:
    """
    創建PASCAL VOC格式的XML檔案
    
    Args:
        image_path (str): 影像檔案路徑
        detections (List[Dict]): YOLOv5偵測結果
        output_path (str): XML輸出路徑
        width (int): 影像寬度
        height (int): 影像高度
        depth (int): 影像深度
        
    Returns:
        bool: 成功返回True，失敗返回False
    """
    # 影像資訊已由外部傳入
    # 創建XML根元素
    annotation = ET.Element("annotation")
    
    # 添加folder元素
    folder = ET.SubElement(annotation, "folder")
    folder.text = os.path.dirname(image_path).split(os.sep)[-1] if os.path.dirname(image_path) else "images"
    
    # 添加filename元素
    filename = ET.SubElement(annotation, "filename")
    filename.text = os.path.basename(image_path)
    
    # 添加path元素
    path = ET.SubElement(annotation, "path")
    path.text = os.path.abspath(image_path)
    
    # 添加source元素
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    
    # 添加size元素
    size = ET.SubElement(annotation, "size")
    width_elem = ET.SubElement(size, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, "height")
    height_elem.text = str(height)
    depth_elem = ET.SubElement(size, "depth")
    depth_elem.text = str(depth)
    
    # 添加segmented元素
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"
    
    # 處理每個偵測到的物件
    for detection in detections:
        obj = ET.SubElement(annotation, "object")
        
        # 物件名稱
        name = ET.SubElement(obj, "name")
        name.text = str(detection['name'])
        
        # pose
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        
        # truncated
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        
        # difficult
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        
        # 邊界框
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(detection['xmin']))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(detection['ymin']))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(detection['xmax']))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(detection['ymax']))
    
    # 格式化XML並儲存
    try:
        tree = ET.ElementTree(annotation)
        ET.indent(tree, space="  ", level=0)  # 美化XML格式
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        return True
    except Exception as e:
        print(f"XML儲存失敗 {output_path}: {e}")
        return False

def process_yolo_results(results, confidence_threshold: float = 0.25) -> List[Dict]:
    """
    處理YOLOv5偵測結果
    
    Args:
        results: YOLOv5偵測結果
        confidence_threshold (float): 信心度閾值
        
    Returns:
        List[Dict]: 處理後的偵測結果列表
    """
    detections = []
    # Ultralytics results object is a list, process the first one
    result = results[0]
    boxes = result.boxes  # Boxes object for bounding box outputs

    for i in range(len(boxes)):
        box = boxes[i]
        if box.conf[0] >= confidence_threshold:
            xyxy = box.xyxy[0]
            detection = {
                'name': result.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'xmin': max(0, float(xyxy[0])),
                'ymin': max(0, float(xyxy[1])),
                'xmax': float(xyxy[2]),
                'ymax': float(xyxy[3])
            }
            detections.append(detection)

    return detections

def get_image_files(input_folder: str) -> List[str]:
    """
    取得資料夾中的所有影像檔案
    
    Args:
        input_folder (str): 輸入資料夾路徑
        
    Returns:
        List[str]: 影像檔案路徑列表
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for file_path in Path(input_folder).rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))
    
    return image_files

def get_video_files(input_folder: str) -> List[str]:
    """
    取得資料夾中的所有影片檔案
    
    Args:
        input_folder (str): 輸入資料夾路徑
        
    Returns:
        List[str]: 影片檔案路徑列表
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for file_path in Path(input_folder).rglob('*'):
        if file_path.suffix.lower() in video_extensions:
            video_files.append(str(file_path))
    
    return video_files

def process_video(video_path: str, model, output_folder: str, confidence_threshold: float, frame_interval: int, save_frames: bool, verbose: bool) -> Tuple[int, int]:
    """
    處理單一影片檔案
    
    Args:
        video_path (str): 影片檔案路徑
        model: YOLOv5模型
        output_folder (str): 輸出資料夾
        confidence_threshold (float): 信心度閾值
        frame_interval (int): 幀間隔
        save_frames (bool): 是否儲存偵測到的幀
        verbose (bool): 是否顯示詳細資訊
        
    Returns:
        (成功計數, 失敗計數)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片檔案 {video_path}")
        return 0, 1

    frame_idx = 0
    success_count = 0
    error_count = 0
    video_base_name = Path(video_path).stem

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"影片總幀數: {total_frames}，將以每 {frame_interval} 幀的間隔處理。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if (frame_idx - 1) % frame_interval != 0:
            continue

        if verbose:
            print(f"處理中 (幀 {frame_idx}/{total_frames}):")

        try:
            # YOLOv5需要RGB格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 執行物件偵測
            results = model.predict(frame_rgb, verbose=False)
            
            # 處理偵測結果
            detections = process_yolo_results(results, confidence_threshold)
            
            # 取得幀的尺寸
            height, width, depth = frame.shape
            
            # 生成輸出檔案名稱
            frame_filename = f"{video_base_name}_frame_{frame_idx:06d}.jpg"
            xml_filename = f"{video_base_name}_frame_{frame_idx:06d}.xml"
            frame_output_path = os.path.join(output_folder, frame_filename)
            xml_output_path = os.path.join(output_folder, xml_filename)

            # 儲存偵測到的幀
            if save_frames:
                cv2.imwrite(frame_output_path, frame)
            
            # 創建PASCAL VOC XML
            if create_pascal_voc_xml(frame_output_path, detections, xml_output_path, width, height, depth):
                if verbose:
                    print(f"  ✓ 偵測到 {len(detections)} 個物件，XML已儲存。")
                success_count += 1
            else:
                if verbose:
                    print("  ✗ XML創建失敗")
                error_count += 1
                
        except Exception as e:
            if verbose:
                print(f"  ✗ 處理幀 {frame_idx} 失敗: {e}")
            error_count += 1
    
    cap.release()
    return success_count, error_count

def validate_paths(input_folder: str, output_folder: str) -> bool:
    """
    驗證輸入和輸出路徑
    
    Args:
        input_folder (str): 輸入資料夾路徑
        output_folder (str): 輸出資料夾路徑
        
    Returns:
        bool: 驗證通過返回True
    """
    # 檢查輸入資料夾
    if not os.path.exists(input_folder):
        print(f"錯誤：輸入路徑 '{input_folder}' 不存在！")
        return False

    # 輸入可以是檔案（影片）或資料夾（影像）
    if not os.path.isdir(input_folder) and not os.path.isfile(input_folder):
        print(f"錯誤：輸入路徑 '{input_folder}' 必須是資料夾或檔案！")
        return False
    
    # 創建輸出資料夾
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"輸出資料夾已準備: {output_folder}")
    except Exception as e:
        print(f"無法創建輸出資料夾: {e}")
        return False
    
    return True

def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="YOLOv5物件偵測與PASCAL VOC XML轉換程式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例 (影像資料夾):
  python %(prog)s -i ./images_folder -o ./output
  python %(prog)s -i ./images_folder -o ./output -c 0.5 -m yolov5m.pt --gpu

使用範例 (單一影片):
  python %(prog)s -i ./my_video.mp4 -o ./output --frame-interval 30

使用範例 (影片資料夾):
  python %(prog)s -i ./videos_folder -o ./output --frame-interval 30
        """
    )
    
    # 必要參數
    parser.add_argument('-i', '--input', 
                       required=True,
                       help='輸入影像資料夾路徑 或 影片檔案路徑')
    
    parser.add_argument('-o', '--output',
                       required=True, 
                       help='輸出資料夾路徑')
    
    # 可選參數
    parser.add_argument('-c', '--confidence',
                       type=float,
                       default=0.25,
                       help='信心度閾值 (預設: 0.25)')
    
    parser.add_argument('-m', '--model',
                       default='yolov5s.pt',
                       help='YOLOv5模型版本 (預設: yolov5s)')
    
    parser.add_argument('--gpu',
                       action='store_true',
                       help='使用GPU加速 (需要CUDA支援)')

    parser.add_argument('--frame-interval',
                       type=int,
                       default=1,
                       help='處理影片時，每隔多少幀進行一次偵測 (預設: 1，即每幀都處理)')
    
    parser.add_argument('--copy-images',
                       action='store_true',
                       default=True,
                       help='複製原始影像或儲存影片幀到輸出資料夾 (預設: True)')
    
    parser.add_argument('--no-copy-images',
                       action='store_false',
                       dest='copy_images',
                       help='不複製原始影像到輸出資料夾')
    
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='顯示詳細資訊')
    
    parser.add_argument('--version',
                       action='version',
                       version='YOLOv5 PASCAL VOC Converter v1.0')
    
    return parser.parse_args()

def main():
    """主程式"""
    # 解析命令列參數
    args = parse_arguments()
    
    if args.verbose:
        print("=" * 60)
        print("YOLOv5物件偵測與PASCAL VOC XML轉換程式")
        print("=" * 60)
        print(f"輸入資料夾: {args.input}")
        print(f"輸出資料夾: {args.output}")
        print(f"信心度閾值: {args.confidence}")
        print(f"模型版本: {args.model}")
        print(f"使用設備: {'GPU' if args.gpu else 'CPU'}")
        print(f"影片幀間隔: {args.frame_interval}")
        print(f"複製影像/幀: {'是' if args.copy_images else '否'}")
        print("-" * 60)
    
    # 驗證信心度範圍
    if not 0 <= args.confidence <= 1:
        print("錯誤：信心度閾值必須介於0-1之間！")
        sys.exit(1)
    
    # 驗證路徑
    if not validate_paths(args.input, args.output):
        sys.exit(1)

    # 設定計算設備
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    if args.gpu and not torch.cuda.is_available():
        print("警告：要求使用GPU但CUDA不可用，改用CPU")
        device = 'cpu'
    
    # 載入YOLOv5模型
    model = load_yolo_model(Path(args.model), device)
    if model is None:
        print("錯誤：模型載入失敗！")
        sys.exit(1)
    
    # 處理每張影像
    success_count = 0
    error_count = 0
    
    if os.path.isdir(args.input):
        # --- 處理資料夾 (可能包含影像和影片) ---
        image_files = get_image_files(args.input)
        video_files = get_video_files(args.input)

        if not image_files and not video_files:
            print(f"錯誤：在 '{args.input}' 中找不到任何影像或影片檔案！")
            sys.exit(1)
        
        # --- 處理影像檔案 ---
        if image_files:
            print(f"找到 {len(image_files)} 個影像檔案")
            
            if args.verbose:
                print("\n開始處理影像...")
                print("-" * 60)
            
            for i, image_path in enumerate(image_files, 1):
                if args.verbose:
                    print(f"處理中 ({i}/{len(image_files)}): {os.path.basename(image_path)}")
                
                try:
                    # 讀取影像以獲取尺寸
                    with Image.open(image_path) as img:
                        width, height = img.size
                        depth = len(img.getbands()) if hasattr(img, 'getbands') else 3

                    # 執行物件偵測
                    results = model.predict(image_path, verbose=False)
                    
                    # 處理偵測結果
                    detections = process_yolo_results(results, args.confidence)
                    
                    # 生成輸出檔案名稱
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    xml_filename = f"{base_name}.xml"
                    xml_output_path = os.path.join(args.output, xml_filename)
                    
                    # 創建PASCAL VOC XML
                    if create_pascal_voc_xml(image_path, detections, xml_output_path, width, height, depth):
                        # 複製原始影像到輸出資料夾（如果需要）
                        if args.copy_images:
                            image_output_path = os.path.join(args.output, os.path.basename(image_path))
                            shutil.copy2(image_path, image_output_path)
                        
                        if args.verbose:
                            print(f"  ✓ 偵測到 {len(detections)} 個物件")
                        success_count += 1
                    else:
                        if args.verbose:
                            print("  ✗ XML創建失敗")
                        error_count += 1
                        
                except Exception as e:
                    if args.verbose:
                        print(f"  ✗ 處理失敗: {e}")
                    error_count += 1
        
        # --- 處理影片檔案 ---
        if video_files:
            print(f"找到 {len(video_files)} 個影片檔案")

            if args.verbose:
                print("\n開始處理影片...")
                print("-" * 60)
            
            for i, video_path in enumerate(video_files, 1):
                if args.verbose:
                    print(f"處理中 ({i}/{len(video_files)}): {os.path.basename(video_path)}")
                
                s_count, e_count = process_video(video_path, model, args.output, args.confidence, args.frame_interval, args.copy_images, args.verbose)
                success_count += s_count
                error_count += e_count

    elif os.path.isfile(args.input):
        # --- 處理單一影片檔案 ---
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        if Path(args.input).suffix.lower() not in video_extensions:
            print(f"錯誤：輸入檔案 '{args.input}' 不是支援的影片格式。")
            print(f"支援的格式: {', '.join(video_extensions)}")
            sys.exit(1)

        if args.verbose:
            print("\n開始處理影片...")
            print("-" * 60)
        
        s_count, e_count = process_video(args.input, model, args.output, args.confidence, args.frame_interval, args.copy_images, args.verbose)
        success_count += s_count
        error_count += e_count
    
    # 顯示總結
    print("-" * 60)
    print("處理完成！")
    print(f"成功處理: {success_count} 個檔案")
    print(f"處理失敗: {error_count} 個檔案")
    print(f"輸出資料夾: {args.output}")
    
    if success_count > 0:
        print(f"所有XML檔案{'和影像/幀' if args.copy_images else ''}已儲存到: {args.output}")
    
    # 設定退出碼
    sys.exit(0 if error_count == 0 else 1)

if __name__ == "__main__":
    main()