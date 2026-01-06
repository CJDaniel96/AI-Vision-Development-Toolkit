#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
影像資料集分割程式
將影像資料集及其對應的標記檔分割為訓練集和測試集

作者: AI助理
版本: 2.0
"""

import os
import shutil
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from sklearn.model_selection import train_test_split
import argparse
import logging
import sys
from dataclasses import dataclass


@dataclass
class SplitConfig:
    """資料集分割配置"""
    source_dir: Union[str, Path]
    output_dir: Union[str, Path]
    train_size: float = 0.8
    annotation_ext: Optional[str] = None
    random_state: int = 42
    create_val_set: bool = False
    val_size: float = 0.1
    verbose: bool = True
    coco_format: bool = False  # 是否使用 COCO 格式 (images/labels 分離)


@dataclass
class SplitResult:
    """分割結果"""
    total_images: int
    train_count: int
    test_count: int
    val_count: int = 0
    images_with_annotations: int = 0
    success: bool = True
    error_message: Optional[str] = None


class ImageDatasetSplitter:
    """影像資料集分割器"""
    
    # 支援的影像格式
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
    
    # 支援的標記檔格式
    SUPPORTED_ANNOTATION_EXTENSIONS = {'.txt', '.xml', '.json', '.yaml', '.yml'}
    
    def __init__(self, config: SplitConfig):
        """
        初始化資料集分割器
        
        Args:
            config: 分割配置
        """
        self.config = config
        self.source_dir = Path(config.source_dir)
        self.output_dir = Path(config.output_dir)
        
        # 設定日誌
        log_level = logging.INFO if config.verbose else logging.WARNING
        logging.basicConfig(level=log_level, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # 驗證輸入參數
        self._validate_config()
        
    def _validate_config(self):
        """驗證配置參數"""
        if not self.source_dir.exists():
            raise ValueError(f"來源資料夾不存在: {self.source_dir}")
            
        if not self.source_dir.is_dir():
            raise ValueError(f"來源路徑不是資料夾: {self.source_dir}")
            
        if not 0 < self.config.train_size < 1:
            raise ValueError(f"訓練集比例必須在 0 和 1 之間: {self.config.train_size}")
            
        if self.config.create_val_set:
            if not 0 < self.config.val_size < 1:
                raise ValueError(f"驗證集比例必須在 0 和 1 之間: {self.config.val_size}")
            if self.config.train_size + self.config.val_size >= 1:
                raise ValueError("訓練集 + 驗證集比例必須小於 1")
                
        if (self.config.annotation_ext and 
            self.config.annotation_ext not in self.SUPPORTED_ANNOTATION_EXTENSIONS):
            self.logger.warning(f"不常見的標記檔格式: {self.config.annotation_ext}")
            
    def find_image_files(self) -> List[Path]:
        """
        尋找所有影像檔案
        
        Returns:
            影像檔案路徑列表
        """
        image_files = []
        
        for ext in self.SUPPORTED_IMAGE_EXTENSIONS:
            # 使用 glob 尋找所有符合的檔案（不區分大小寫）
            pattern = str(self.source_dir / f"*{ext}")
            image_files.extend(glob.glob(pattern, recursive=False))
            
            # 也搜尋大寫版本
            pattern = str(self.source_dir / f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern, recursive=False))
            
        # 轉換為 Path 物件並去重
        image_files = list(set([Path(f) for f in image_files]))
        image_files.sort()  # 排序以確保一致性
        
        self.logger.info(f"找到 {len(image_files)} 個影像檔案")
        return image_files
        
    def find_annotation_file(self, image_path: Path) -> Optional[Path]:
        """
        為指定影像尋找對應的標記檔
        
        Args:
            image_path: 影像檔案路徑
            
        Returns:
            標記檔路徑，如果不存在則返回 None
        """
        if not self.config.annotation_ext:
            return None
            
        # 建構標記檔路徑
        annotation_path = image_path.with_suffix(self.config.annotation_ext)
        
        return annotation_path if annotation_path.exists() else None
        
    def create_file_pairs(self, image_files: List[Path]) -> List[Dict]:
        """
        建立影像與標記檔的配對
        
        Args:
            image_files: 影像檔案列表
            
        Returns:
            檔案配對字典列表
        """
        file_pairs = []
        images_with_annotations = 0
        
        for image_path in image_files:
            annotation_path = self.find_annotation_file(image_path)
            
            file_pair = {
                'image': image_path,
                'annotation': annotation_path
            }
            
            file_pairs.append(file_pair)
            
            if annotation_path:
                images_with_annotations += 1
                
        self.logger.info(f"影像總數: {len(file_pairs)}")
        self.logger.info(f"有標記檔的影像: {images_with_annotations}")
        self.logger.info(f"無標記檔的影像: {len(file_pairs) - images_with_annotations}")
        
        return file_pairs, images_with_annotations
        
    def split_dataset(self, file_pairs: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        分割資料集
        
        Args:
            file_pairs: 檔案配對列表
            
        Returns:
            (訓練集, 測試集, 驗證集) 元組
        """
        if len(file_pairs) == 0:
            raise ValueError("沒有找到任何影像檔案")
        
        val_pairs = []
        
        if self.config.create_val_set:
            # 三分割: train/val/test
            # 先分出訓練集
            train_pairs, temp_pairs = train_test_split(
                file_pairs,
                train_size=self.config.train_size,
                random_state=self.config.random_state,
                shuffle=True
            )
            
            # 從剩餘的資料中分出驗證集和測試集
            remaining_size = 1 - self.config.train_size
            val_ratio = self.config.val_size / remaining_size
            
            val_pairs, test_pairs = train_test_split(
                temp_pairs,
                train_size=val_ratio,
                random_state=self.config.random_state,
                shuffle=True
            )
            
            self.logger.info(f"訓練集大小: {len(train_pairs)} ({len(train_pairs)/len(file_pairs)*100:.1f}%)")
            self.logger.info(f"驗證集大小: {len(val_pairs)} ({len(val_pairs)/len(file_pairs)*100:.1f}%)")
            self.logger.info(f"測試集大小: {len(test_pairs)} ({len(test_pairs)/len(file_pairs)*100:.1f}%)")
            
        else:
            # 二分割: train/test
            train_pairs, test_pairs = train_test_split(
                file_pairs,
                train_size=self.config.train_size,
                random_state=self.config.random_state,
                shuffle=True
            )
            
            self.logger.info(f"訓練集大小: {len(train_pairs)} ({len(train_pairs)/len(file_pairs)*100:.1f}%)")
            self.logger.info(f"測試集大小: {len(test_pairs)} ({len(test_pairs)/len(file_pairs)*100:.1f}%)")
        
        return train_pairs, test_pairs, val_pairs
        
    def create_output_directories(self) -> Dict[str, Dict[str, Path]]:
        """建立輸出資料夾結構"""
        if self.config.coco_format:
            # COCO 格式: images/ 和 labels/ 分離
            directories = {
                'train': {
                    'images': self.output_dir / 'images' / 'train',
                    'labels': self.output_dir / 'labels' / 'train' if self.config.annotation_ext else None
                },
                'test': {
                    'images': self.output_dir / 'images' / 'test',
                    'labels': self.output_dir / 'labels' / 'test' if self.config.annotation_ext else None
                }
            }
            
            if self.config.create_val_set:
                directories['val'] = {
                    'images': self.output_dir / 'images' / 'val',
                    'labels': self.output_dir / 'labels' / 'val' if self.config.annotation_ext else None
                }
        else:
            # 傳統格式: train/, test/, val/ 各自包含所有檔案
            directories = {
                'train': {
                    'images': self.output_dir / 'train',
                    'labels': self.output_dir / 'train'
                },
                'test': {
                    'images': self.output_dir / 'test',
                    'labels': self.output_dir / 'test'
                }
            }
            
            if self.config.create_val_set:
                directories['val'] = {
                    'images': self.output_dir / 'val',
                    'labels': self.output_dir / 'val'
                }
        
        # 建立所有資料夾
        for split_name, paths in directories.items():
            paths['images'].mkdir(parents=True, exist_ok=True)
            if paths['labels'] and paths['labels'] != paths['images']:
                paths['labels'].mkdir(parents=True, exist_ok=True)
        
        if self.config.coco_format:
            self.logger.info(f"建立 COCO 格式輸出資料夾: {self.output_dir}")
        else:
            self.logger.info(f"建立傳統格式輸出資料夾: {self.output_dir}")
        
        return directories
        
    def copy_files(self, file_pairs: List[Dict], directories: Dict[str, Path], split_name: str) -> Tuple[int, int]:
        """
        複製檔案到目標資料夾
        
        Args:
            file_pairs: 檔案配對列表
            directories: 目標資料夾字典 {'images': Path, 'labels': Path}
            split_name: 分割名稱 (用於日誌)
            
        Returns:
            (複製的影像數, 複製的標記檔數)
        """
        copied_images = 0
        copied_annotations = 0
        
        images_dir = directories['images']
        labels_dir = directories['labels']
        
        for pair in file_pairs:
            try:
                # 複製影像檔案
                image_src = pair['image']
                image_dst = images_dir / image_src.name
                shutil.copy2(image_src, image_dst)
                copied_images += 1
                
                # 複製標記檔（如果存在）
                if pair['annotation'] and labels_dir:
                    annotation_src = pair['annotation']
                    annotation_dst = labels_dir / annotation_src.name
                    shutil.copy2(annotation_src, annotation_dst)
                    copied_annotations += 1
                    
            except Exception as e:
                self.logger.error(f"複製檔案失敗 {image_src}: {e}")
                
        self.logger.info(f"{split_name}集複製完成: {copied_images} 個影像, {copied_annotations} 個標記檔")
        return copied_images, copied_annotations
        
    def run(self) -> SplitResult:
        """
        執行資料集分割
        
        Returns:
            分割結果
        """
        self.logger.info("開始分割資料集...")
        self.logger.info(f"來源資料夾: {self.source_dir}")
        self.logger.info(f"輸出資料夾: {self.output_dir}")
        self.logger.info(f"訓練集比例: {self.config.train_size:.2f}")
        if self.config.create_val_set:
            self.logger.info(f"驗證集比例: {self.config.val_size:.2f}")
        self.logger.info(f"標記檔格式: {self.config.annotation_ext or '無'}")
        
        try:
            # 1. 尋找影像檔案
            image_files = self.find_image_files()
            
            if not image_files:
                raise ValueError("在來源資料夾中沒有找到任何影像檔案")
                
            # 2. 建立檔案配對
            file_pairs, images_with_annotations = self.create_file_pairs(image_files)
            
            # 3. 分割資料集
            train_pairs, test_pairs, val_pairs = self.split_dataset(file_pairs)
            
            # 4. 建立輸出資料夾
            directories = self.create_output_directories()
            
            # 5. 複製檔案
            self.copy_files(train_pairs, directories['train'], "訓練")
            self.copy_files(test_pairs, directories['test'], "測試")
            val_count = 0
            if val_pairs and 'val' in directories:
                self.copy_files(val_pairs, directories['val'], "驗證")
                val_count = len(val_pairs)
            
            self.logger.info("資料集分割完成！")
            
            return SplitResult(
                total_images=len(file_pairs),
                train_count=len(train_pairs),
                test_count=len(test_pairs),
                val_count=val_count,
                images_with_annotations=images_with_annotations,
                success=True
            )
            
        except Exception as e:
            error_msg = f"資料集分割失敗: {e}"
            self.logger.error(error_msg)
            return SplitResult(
                total_images=0,
                train_count=0,
                test_count=0,
                success=False,
                error_message=str(e)
            )


# 便利函數
def split_image_dataset(source_dir: Union[str, Path],
                       output_dir: Union[str, Path],
                       train_size: float = 0.8,
                       annotation_ext: Optional[str] = None,
                       random_state: int = 42,
                       create_val_set: bool = False,
                       val_size: float = 0.1,
                       verbose: bool = True,
                       coco_format: bool = False) -> SplitResult:
    """
    便利函數：分割影像資料集
    
    Args:
        source_dir: 來源資料夾路徑
        output_dir: 輸出資料夾路徑
        train_size: 訓練集比例 (0-1)
        annotation_ext: 標記檔副檔名 (例如: '.txt', '.xml')
        random_state: 隨機種子
        create_val_set: 是否建立驗證集
        val_size: 驗證集比例 (0-1)
        verbose: 是否顯示詳細日誌
        coco_format: 是否使用 COCO 格式 (images/labels 分離)
        
    Returns:
        分割結果
        
    Example:
        >>> # 傳統格式
        >>> result = split_image_dataset(
        ...     source_dir="./images",
        ...     output_dir="./dataset",
        ...     train_size=0.7,
        ...     annotation_ext=".txt",
        ...     create_val_set=True,
        ...     val_size=0.15
        ... )
        
        >>> # COCO 格式
        >>> result = split_image_dataset(
        ...     source_dir="./images",
        ...     output_dir="./dataset",
        ...     train_size=0.8,
        ...     annotation_ext=".txt",
        ...     create_val_set=True,
        ...     val_size=0.1,
        ...     coco_format=True
        ... )
        >>> print(f"成功分割 {result.total_images} 張影像")
    """
    config = SplitConfig(
        source_dir=source_dir,
        output_dir=output_dir,
        train_size=train_size,
        annotation_ext=annotation_ext,
        random_state=random_state,
        create_val_set=create_val_set,
        val_size=val_size,
        verbose=verbose,
        coco_format=coco_format
    )
    
    splitter = ImageDatasetSplitter(config)
    return splitter.run()


def get_user_input():
    """獲取使用者輸入（互動模式）"""
    print("=== 影像資料集分割程式 ===\n")
    
    # 來源資料夾
    while True:
        source_dir = input("請輸入來源資料夾路徑: ").strip()
        if os.path.exists(source_dir) and os.path.isdir(source_dir):
            break
        print("❌ 資料夾不存在或不是有效路徑，請重新輸入")
    
    # 輸出資料夾
    output_dir = input("請輸入輸出資料夾路徑: ").strip()
    
    # 訓練集比例
    while True:
        try:
            train_size = float(input("請輸入訓練集比例 (0-1，預設 0.8): ").strip() or "0.8")
            if 0 < train_size < 1:
                break
            print("❌ 比例必須在 0 和 1 之間")
        except ValueError:
            print("❌ 請輸入有效的數字")
    
    # 是否建立驗證集
    create_val = input("是否建立驗證集？ (y/N): ").strip().lower() in ['y', 'yes']
    val_size = 0.1
    if create_val:
        while True:
            try:
                val_size = float(input("請輸入驗證集比例 (0-1，預設 0.1): ").strip() or "0.1")
                if 0 < val_size < 1 and train_size + val_size < 1:
                    break
                print("❌ 驗證集比例無效，請確保訓練集+驗證集比例小於1")
            except ValueError:
                print("❌ 請輸入有效的數字")
    
    # 標記檔副檔名
    annotation_ext = input("請輸入標記檔副檔名 (例如: .txt, .xml，留空表示無): ").strip()
    if annotation_ext and not annotation_ext.startswith('.'):
        annotation_ext = '.' + annotation_ext
    
    # 隨機種子
    while True:
        try:
            random_state = int(input("請輸入隨機種子 (預設 42): ").strip() or "42")
            break
        except ValueError:
            print("❌ 請輸入有效的整數")
    
    # 是否使用 COCO 格式
    coco_format = input("是否使用 COCO 格式？ (images/labels 分離) (y/N): ").strip().lower() in ['y', 'yes']
    
    return source_dir, output_dir, train_size, annotation_ext or None, random_state, create_val, val_size, coco_format


def main():
    """主程式"""
    parser = argparse.ArgumentParser(
        description='影像資料集分割程式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本用法
  python %(prog)s -s ./images -o ./dataset
  
  # 帶驗證集
  python %(prog)s -s ./images -o ./dataset --val --val-size 0.15
  
  # 指定標記檔格式
  python %(prog)s -s ./images -o ./dataset -a .txt
  
  # 互動模式
  python %(prog)s --interactive
        """
    )
    
    parser.add_argument('--source', '-s', help='來源資料夾路徑')
    parser.add_argument('--output', '-o', help='輸出資料夾路徑')
    parser.add_argument('--train-size', '-t', type=float, default=0.8, 
                       help='訓練集比例 (預設: 0.8)')
    parser.add_argument('--annotation-ext', '-a', 
                       help='標記檔副檔名 (例如: .txt, .xml)')
    parser.add_argument('--random-state', '-r', type=int, default=42, 
                       help='隨機種子 (預設: 42)')
    parser.add_argument('--val', '--validation', action='store_true',
                       help='建立驗證集')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='驗證集比例 (預設: 0.1)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='互動模式')
    parser.add_argument('--coco', '--coco-format', action='store_true',
                       help='使用 COCO 格式 (images/labels 分離)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='安靜模式（減少輸出）')
    
    args = parser.parse_args()
    
    try:
        # 互動模式
        if args.interactive or (not args.source or not args.output):
            source_dir, output_dir, train_size, annotation_ext, random_state, create_val, val_size, coco_format = get_user_input()
        else:
            source_dir = args.source
            output_dir = args.output
            train_size = args.train_size
            annotation_ext = args.annotation_ext
            random_state = args.random_state
            create_val = args.val
            val_size = args.val_size
            coco_format = args.coco
        
        # 執行分割
        result = split_image_dataset(
            source_dir=source_dir,
            output_dir=output_dir,
            train_size=train_size,
            annotation_ext=annotation_ext,
            random_state=random_state,
            create_val_set=create_val,
            val_size=val_size,
            verbose=not args.quiet,
            coco_format=coco_format
        )
        
        # 顯示結果
        if result.success:
            print(f"\n✅ 資料集分割完成！")
            print(f"📊 總影像數: {result.total_images}")
            print(f"📚 訓練集: {result.train_count}")
            if result.val_count > 0:
                print(f"🔍 驗證集: {result.val_count}")
            print(f"🧪 測試集: {result.test_count}")
            print(f"🏷️  有標記檔的影像: {result.images_with_annotations}")
        else:
            print(f"❌ 分割失敗: {result.error_message}")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n程式已被使用者中斷")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 程式執行失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
