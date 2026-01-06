import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from paddleocr import PaddleOCR
import argparse
from typing import List, Tuple, Optional
import logging

class AOIImageClassifier:
    def __init__(self, target_texts: List[str], output_dir: str = "classified_images", confidence_threshold: float = 0.6):
        """
        初始化AOI影像分類器
        
        Args:
            target_texts: 目標文字列表，符合任一文字即為正常
            output_dir: 輸出目錄路徑
            confidence_threshold: OCR識別信心度閾值
        """
        self.target_texts = [text.upper() for text in target_texts]  # 轉為大寫比較
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        
        # 初始化PaddleOCR
        self.ocr = PaddleOCR(lang='en')
        
        # 創建輸出資料夾
        self.create_output_folders()
        
        # 設置日誌
        self.setup_logging()
    
    def setup_logging(self):
        """設置日誌系統"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_output_folders(self):
        """創建輸出資料夾結構"""
        folders = ['0_degree', '90_degree', '180_degree', '270_degree', 'NG']
        
        # 創建主目錄
        self.output_dir.mkdir(exist_ok=True)
        
        # 創建子資料夾
        for folder in folders:
            (self.output_dir / folder).mkdir(exist_ok=True)
        
        print(f"已創建輸出資料夾結構於: {self.output_dir}")
    
    def rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """
        旋轉影像
        
        Args:
            image: 輸入影像
            angle: 旋轉角度 (90, 180, 270)
            
        Returns:
            旋轉後的影像
        """
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image
    
    def extract_text_from_image(self, image: np.ndarray) -> Tuple[List[str], float]:
        """
        從影像中提取文字
        
        Args:
            image: 輸入影像
            
        Returns:
            (提取的文字列表, 平均信心度)
        """
        try:
            results = self.ocr.ocr(image)
            
            if not results or not results[0]:
                return [], 0.0
            
            if 'rec_texts' in results[0] and 'rec_scores' in results[0]:
                texts = results[0]['rec_texts']
                confidences = results[0]['rec_scores']
            else:
                texts = []
                confidences = []
            
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return texts, avg_confidence
            
        except Exception as e:
            self.logger.error(f"OCR處理錯誤: {e}")
            return [], 0.0
    
    def check_target_text(self, texts: List[str]) -> bool:
        """
        檢查是否包含目標文字
        
        Args:
            texts: 檢測到的文字列表
            
        Returns:
            是否包含目標文字
        """
        for text in texts:
            for target in self.target_texts:
                if target in text or text in target:
                    return True
        return False
    
    def process_single_image(self, image_path: str) -> dict:
        """
        處理單張影像
        
        Args:
            image_path: 影像路徑
            
        Returns:
            處理結果字典
        """
        try:
            # 讀取影像
            image = cv2.imread(image_path)
            if image is None:
                return {"status": "error", "message": "無法讀取影像"}
            
            # 測試四個角度
            angles = [0, 90, 180, 270]
            best_result = {
                "angle": None,
                "texts": [],
                "confidence": 0.0,
                "has_target": False
            }
            
            for angle in angles:
                # 旋轉影像
                rotated_image = self.rotate_image(image, angle)
                
                # 提取文字
                texts, confidence = self.extract_text_from_image(rotated_image)
                
                # 檢查是否包含目標文字
                has_target = self.check_target_text(texts)
                
                self.logger.info(f"角度 {angle}度: 文字={texts}, 信心度={confidence:.2f}, 包含目標={has_target}")
                
                # 如果找到目標文字且信心度更高，更新最佳結果
                if has_target and confidence > best_result["confidence"]:
                    best_result.update({
                        "angle": angle,
                        "texts": texts,
                        "confidence": confidence,
                        "has_target": True
                    })
            
            return best_result
            
        except Exception as e:
            self.logger.error(f"處理影像 {image_path} 時發生錯誤: {e}")
            return {"status": "error", "message": str(e)}
    
    def save_rotated_images(self, original_path: str, correct_angle: int):
        """
        保存所有角度的旋轉影像
        
        Args:
            original_path: 原始影像路徑
            correct_angle: 正確角度
        """
        try:
            image = cv2.imread(original_path)
            filename = Path(original_path).name
            
            # 保存各個角度的影像
            angles = [0, 90, 180, 270]
            
            for angle in angles:
                # 計算相對於正確角度的旋轉
                relative_angle = (angle - correct_angle) % 360
                rotated_image = self.rotate_image(image, relative_angle)
                
                # 確定保存資料夾
                folder_name = f"{angle}_degree"
                save_path = self.output_dir / folder_name / filename
                
                # 保存影像
                cv2.imwrite(str(save_path), rotated_image)
                
            self.logger.info(f"已保存影像 {filename} 的所有角度版本")
            
        except Exception as e:
            self.logger.error(f"保存旋轉影像時發生錯誤: {e}")
    
    def save_ng_image(self, original_path: str):
        """
        保存NG影像
        
        Args:
            original_path: 原始影像路徑
        """
        try:
            filename = Path(original_path).name
            ng_path = self.output_dir / "NG" / filename
            shutil.copy2(original_path, ng_path)
            self.logger.info(f"已保存NG影像: {filename}")
        except Exception as e:
            self.logger.error(f"保存NG影像時發生錯誤: {e}")
    
    def process_directory(self, input_dir: str):
        """
        處理整個目錄的影像
        
        Args:
            input_dir: 輸入目錄路徑
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"輸入目錄不存在: {input_dir}")
        
        # 支援的影像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 獲取所有影像檔案
        image_files = [
            f for f in input_path.rglob('*') 
            if f.suffix.lower() in image_extensions and f.is_file()
        ]
        
        if not image_files:
            print(f"在 {input_dir} 中未找到影像檔案")
            return
        
        print(f"找到 {len(image_files)} 個影像檔案，開始處理...")
        
        # 統計資訊
        stats = {
            "total": len(image_files),
            "processed": 0,
            "ng": 0,
            "normal": 0
        }
        
        # 處理每個影像
        for i, image_file in enumerate(image_files, 1):
            print(f"\n處理進度: {i}/{len(image_files)} - {image_file.name}")
            
            result = self.process_single_image(str(image_file))
            
            if result.get("status") == "error":
                self.logger.error(f"跳過檔案 {image_file.name}: {result.get('message')}")
                stats["ng"] += 1
                self.save_ng_image(str(image_file))
                continue
            
            # 根據結果分類
            if result["has_target"]:
                # 正常影像，保存所有角度
                self.save_rotated_images(str(image_file), result["angle"])
                stats["normal"] += 1
                self.logger.info(f"✓ {image_file.name} - 正確角度: {result['angle']}度")
            else:
                # NG影像
                self.save_ng_image(str(image_file))
                stats["ng"] += 1
                self.logger.warning(f"✗ {image_file.name} - 未找到目標文字")
            
            stats["processed"] += 1
        
        # 顯示統計結果
        print(f"\n處理完成！")
        print(f"總計檔案: {stats['total']}")
        print(f"成功處理: {stats['processed']}")
        print(f"正常影像: {stats['normal']}")
        print(f"NG影像: {stats['ng']}")
        print(f"輸出目錄: {self.output_dir}")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='AOI電子元件影像文字識別分類程式')
    parser.add_argument('--input', '-i', required=True, help='輸入影像目錄路徑')
    parser.add_argument('--output', '-o', default='classified_images', help='輸出目錄路徑')
    parser.add_argument('--target', '-t', nargs='+', required=True, help='目標文字列表')
    parser.add_argument('--confidence', '-c', type=float, default=0.25, help='OCR信心度閾值')
    
    args = parser.parse_args()
    
    # 創建分類器
    classifier = AOIImageClassifier(
        target_texts=args.target,
        output_dir=args.output,
        confidence_threshold=args.confidence
    )
    
    # 處理影像
    try:
        classifier.process_directory(args.input)
    except Exception as e:
        print(f"處理過程中發生錯誤: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()