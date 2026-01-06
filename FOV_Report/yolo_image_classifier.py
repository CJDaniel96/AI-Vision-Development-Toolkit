import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import argparse

def classify_images(input_folder, output_folder, model_path='yolov8n-cls.pt', confidence_threshold=0.5):
    """
    使用 YOLO 模型對資料夾中的所有影像進行分類，並將結果複製到對應的子資料夾中
    
    Args:
        input_folder (str): 輸入影像資料夾路徑
        output_folder (str): 輸出結果資料夾路徑
        model_path (str): YOLO 模型路徑，預設使用 YOLOv8n 分類模型
        confidence_threshold (float): 信心度閾值，預設 0.5
    """
    
    # 載入 YOLO 分類模型
    print(f"載入模型: {model_path}")
    model = YOLO(model_path)
    
    # 建立輸出資料夾
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支援的影像格式
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # 取得所有影像檔案
    input_path = Path(input_folder)
    image_files = []
    
    for ext in supported_extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
    
    if not image_files:
        print(f"在 {input_folder} 中沒有找到支援的影像檔案")
        return
    
    print(f"找到 {len(image_files)} 個影像檔案")
    
    # 統計分類結果
    classification_stats = {}
    processed_count = 0
    
    # 處理每個影像檔案
    for image_file in image_files:
        try:
            print(f"處理中: {image_file.name}")
            
            # 執行推論
            results = model(str(image_file))
            
            # 取得分類結果
            result = results[0]
            
            # 取得最高信心度的分類
            top_class_idx = result.probs.top1  # 最高信心度的類別索引
            top_confidence = result.probs.top1conf.item()  # 最高信心度
            class_name = result.names[top_class_idx]  # 類別名稱
            
            print(f"  分類結果: {class_name} (信心度: {top_confidence:.3f})")
            
            # 檢查是否達到信心度閾值
            if top_confidence >= confidence_threshold:
                # 建立對應的分類子資料夾
                class_folder = output_path / class_name
                class_folder.mkdir(exist_ok=True)
                
                # 複製影像到對應的分類資料夾
                destination = class_folder / image_file.name
                shutil.copy2(image_file, destination)
                
                # 統計分類結果
                if class_name not in classification_stats:
                    classification_stats[class_name] = 0
                classification_stats[class_name] += 1
                
                print(f"  已複製到: {destination}")
            else:
                # 信心度不足的影像放到 "low_confidence" 資料夾
                low_conf_folder = output_path / "low_confidence"
                low_conf_folder.mkdir(exist_ok=True)
                
                destination = low_conf_folder / image_file.name
                shutil.copy2(image_file, destination)
                
                if "low_confidence" not in classification_stats:
                    classification_stats["low_confidence"] = 0
                classification_stats["low_confidence"] += 1
                
                print(f"  信心度不足，已複製到: {destination}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"處理 {image_file.name} 時發生錯誤: {e}")
            continue
    
    # 顯示統計結果
    print("\n" + "="*50)
    print("分類統計結果:")
    print("="*50)
    
    for class_name, count in classification_stats.items():
        print(f"{class_name}: {count} 張影像")
    
    print(f"\n總共處理: {processed_count} 張影像")
    print(f"結果已保存到: {output_folder}")

def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='使用 YOLO 對影像進行分類')
    parser.add_argument('input_folder', help='輸入影像資料夾路徑')
    parser.add_argument('output_folder', help='輸出結果資料夾路徑')
    parser.add_argument('--model', default='yolov8n-cls.pt', 
                       help='YOLO 模型路徑 (預設: yolov8n-cls.pt)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='信心度閾值 (預設: 0.5)')
    
    args = parser.parse_args()
    
    # 檢查輸入資料夾是否存在
    if not os.path.exists(args.input_folder):
        print(f"錯誤: 輸入資料夾 '{args.input_folder}' 不存在")
        return
    
    # 執行分類
    classify_images(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model_path=args.model,
        confidence_threshold=args.confidence
    )

if __name__ == "__main__":
    main()

# 使用範例:
# python yolo_classifier.py input_images output_results
# python yolo_classifier.py input_images output_results --model yolov8s-cls.pt --confidence 0.7