#!/usr/bin/env python3
"""
影像檔案整理工具
遞迴搜尋資料夾中的影像檔案，解析檔名並輸出為 JSON 格式
"""

import os
import json
import re
import argparse
import sys
import shutil
from pathlib import Path
from typing import Dict, Any

class ImageOrganizer:
    def __init__(self):
        # 支援的影像檔案副檔名
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
        
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        解析檔名，提取 component name 和光源類型
        
        檔名格式範例: 20250612120523_DF1756L06G_T@Q812_2_top.jpg
        - Component name: Q812 (從 T@Q812 部分提取)
        - 光源類型: top (檔名最後的部分，去掉副檔名)
        
        Args:
            filename: 檔案名稱
            
        Returns:
            Dict 包含 'component' 和 'light_source'
        """
        # 移除副檔名
        name_without_ext = os.path.splitext(filename)[0]
        
        # 解析 component name (T@後面的部分，直到遇到 _ 或結尾)
        component_match = re.search(r'T@([^_]+)', name_without_ext)
        component_name = component_match.group(1) if component_match else 'unknown'
        
        # 解析光源類型 (檔名最後一個 _ 後面的部分)
        parts = name_without_ext.split('_')
        light_source = parts[-1] if parts else 'unknown'
        
        return {
            'component': component_name,
            'light_source': light_source
        }
    
    def parse_folder_name(self, folder_name: str) -> Dict[str, str]:
        """
        解析資料夾名稱，提取產品名和元件名

        資料夾格式: {產品}_{元件名}_{光源} (例如: 1005-102206-02B_Q1_top)

        Args:
            folder_name: 資料夾名稱

        Returns:
            Dict 包含 'product' 和 'component'
        """
        parts = folder_name.split('_')
        if len(parts) >= 2:
            product = parts[0]
            component = parts[1]
            return {'product': product, 'component': component}
        return {'product': 'Default', 'component': 'Default'}

    def scan_directory(self, root_path: str) -> Dict[str, Any]:
        """
        遞迴搜尋目錄中的影像檔案

        Args:
            root_path: 根目錄路徑

        Returns:
            層次化的字典結構: {資料夾名}/{產品名}/{元件名}/{光源類型}
        """
        result = {}
        product_component_count = {}  # 記錄每個 {產品}_{元件名} 的影像數量
        product_component_files = {}  # 記錄每個 {產品}_{元件名} 的所有檔案
        root_path = Path(root_path)

        if not root_path.exists():
            raise ValueError(f"路徑不存在: {root_path}")

        try:
            # 遞迴搜尋所有檔案
            for root, dirs, files in os.walk(root_path):
                for file in files:
                    # 檢查是否為影像檔案
                    file_path = Path(root) / file
                    if file_path.suffix.lower() in self.image_extensions:
                        # 取得相對路徑
                        relative_path = file_path.relative_to(root_path)
                        path_parts = relative_path.parts[:-1]  # 除了檔名的路徑部分
                        filename = relative_path.name

                        # 解析檔名
                        try:
                            parsed = self.parse_filename(filename)
                            light_source = parsed['light_source']

                            # 判斷是否在子資料夾中
                            if len(path_parts) > 0:
                                # 取得最後一層資料夾名稱（假設為 {產品}_{元件名}_{光源} 格式）
                                subfolder = path_parts[-1]
                                folder_info = self.parse_folder_name(subfolder)
                                product = folder_info['product']
                                component = folder_info['component']

                                # 取得父資料夾名稱（作為第一層）
                                if len(path_parts) > 1:
                                    parent_folder = path_parts[0]
                                else:
                                    parent_folder = root_path.name  # 使用根目錄名稱

                                # 建構層次化結構: {資料夾名}/{產品名}/{元件名}/{光源類型}
                                if parent_folder not in result:
                                    result[parent_folder] = {}
                                if product not in result[parent_folder]:
                                    result[parent_folder][product] = {}
                                if component not in result[parent_folder][product]:
                                    result[parent_folder][product][component] = {}

                                # 加入檔案資訊
                                if light_source:
                                    result[parent_folder][product][component][light_source] = filename

                                # 統計 {產品}_{元件名} 的影像數量
                                key = f"{product}_{component}"
                                if key not in product_component_count:
                                    product_component_count[key] = 0
                                    product_component_files[key] = {'product': product, 'component': component, 'files': {}}
                                product_component_count[key] += 1
                                product_component_files[key]['files'][light_source] = filename

                        except Exception as e:
                            print(f"警告: 解析檔案 {filename} 時發生錯誤: {e}", file=sys.stderr)
                            continue
        except Exception as e:
            print(f"警告: 掃描目錄時發生錯誤: {e}", file=sys.stderr)

        # 找出影像數量最多的 {產品}_{元件名}
        if product_component_count:
            max_key = max(product_component_count, key=product_component_count.get)
            max_data = product_component_files[max_key]

            # 在每個父資料夾下添加 Default
            for parent_folder in result:
                if 'Default' not in result[parent_folder]:
                    result[parent_folder]['Default'] = {}
                result[parent_folder]['Default']['Default'] = max_data['files'].copy()

        return result
    
    def get_score_filename(self, original_filename: str, suffix: str = '_scores') -> str:
        """
        根據原檔案名生成分數檔案名
        
        Args:
            original_filename: 原始檔案名
            suffix: 後綴名
            
        Returns:
            分數檔案名
        """
        if not original_filename:
            return None
            
        path = Path(original_filename)
        stem = path.stem  # 檔名不含副檔名
        extension = path.suffix  # 副檔名
        parent = path.parent
        
        score_filename = f"{stem}{suffix}{extension}"
        return str(parent / score_filename)
    
    def save_json(self, data: Dict[str, Any], output_path: str, pretty: bool = True):
        """
        將資料存檔為 JSON 格式
        
        Args:
            data: 要存檔的資料
            output_path: 輸出檔案路徑
            pretty: 是否格式化 JSON
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, ensure_ascii=False, indent=4)
            else:
                json.dump(data, f, ensure_ascii=False)
    
    def count_files(self, data: Dict[str, Any]) -> int:
        """
        遞迴計算影像檔案總數
        
        Args:
            data: 巢狀字典資料
            
        Returns:
            檔案總數
        """
        count = 0
        for key, value in data.items():
            if isinstance(value, dict):
                count += self.count_files(value)
            else:
                # 如果 value 是字串，表示是檔案名稱
                count += 1
        
        return count
    def convert_to_scores(self, data: Dict[str, Any], score_value: float) -> Dict[str, Any]:
        """
        將檔名替換為固定分數值

        Args:
            data: 原始巢狀字典資料
            score_value: 要替換的分數值

        Returns:
            替換後的字典
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                # 如果是字典，遞迴處理
                result[key] = self.convert_to_scores(value, score_value)
            else:
                # 如果是檔名字串，替換為分數值
                result[key] = score_value
        return result

    def organize_files(self, data: Dict[str, Any], source_root: str, output_root: str, product: str = "", model: str = ""):
        """
        根據 JSON 結構創建資料夾並複製影像檔案
        結構: {產品型號}/{元件名}/影像檔案

        Args:
            data: 巢狀字典資料
            source_root: 原始影像檔案的根目錄
            output_root: 輸出資料夾的根目錄
            product: 產品名稱（第一層）
            model: 產品型號（第二層）
        """
        source_root = Path(source_root)
        output_root = Path(output_root)

        for key, value in data.items():
            if isinstance(value, dict):
                # 判斷當前層級
                if not product:
                    # 第一層：產品名稱
                    self.organize_files(value, source_root, output_root, product=key, model="")
                elif not model:
                    # 第二層：產品型號
                    self.organize_files(value, source_root, output_root, product=product, model=key)
                else:
                    # 第三層：元件名，遞迴處理光源類型
                    component = key
                    for light_source, filename in value.items():
                        if isinstance(filename, str):
                            # 創建資料夾: {產品型號}/{元件名}
                            target_dir = output_root / model / component
                            target_dir.mkdir(parents=True, exist_ok=True)

                            # 在原始目錄中搜尋該檔案
                            source_file = None
                            for root, dirs, files in os.walk(source_root):
                                if filename in files:
                                    source_file = Path(root) / filename
                                    break

                            if source_file and source_file.exists():
                                target_file = target_dir / filename
                                try:
                                    shutil.copy2(source_file, target_file)
                                except Exception as e:
                                    print(f"警告: 複製檔案 {filename} 時發生錯誤: {e}", file=sys.stderr)
                            else:
                                print(f"警告: 找不到檔案 {filename}", file=sys.stderr)

def main():
    """主程式 - CLI 介面"""
    parser = argparse.ArgumentParser(
        description="影像檔案整理工具 - 遞迴搜尋影像檔案並解析檔名",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python image_organizer.py /path/to/images
  python image_organizer.py /path/to/images -o output.json
  python image_organizer.py /path/to/images --output result.json --verbose
  python image_organizer.py /path/to/images -o files.json -s 0.7
  python image_organizer.py /path/to/images -o data.json -s 0.8 --score-suffix _ratings
  python image_organizer.py /path/to/images -o output.json --organize ./organized

支援的影像格式: jpg, jpeg, png, bmp, tiff, tif, gif

檔名解析規則:
  - Component name: 從 T@component_name 部分提取
  - 光源類型: 檔名最後 _ 後的部分 (如 top, side)

分數 JSON:
  使用 -s 參數可以額外生成一個分數 JSON 檔案，結構相同但檔名都替換為指定數值
  例如：-s 0.7 會將所有檔名替換為 0.7

檔案整理:
  使用 --organize 參數可以根據 JSON 階層結構創建資料夾並複製影像檔案
  例如：--organize ./organized 會創建資料夾結構並複製檔案到 ./organized 目錄
        """
    )
    
    # 必需參數
    parser.add_argument(
        'input_path',
        help='要掃描的輸入資料夾路徑'
    )
    
    # 選用參數
    parser.add_argument(
        '-o', '--output',
        help='輸出 JSON 檔案路徑 (如不指定則只顯示在終端)',
        metavar='FILE'
    )
    
    parser.add_argument(
        '-s', '--score',
        type=float,
        help='生成分數 JSON，將所有檔名替換為指定的數值 (例如: -s 0.7)',
        metavar='VALUE'
    )
    
    parser.add_argument(
        '--score-suffix',
        default='_scores',
        help='分數 JSON 檔案的後綴名 (預設: _scores)',
        metavar='SUFFIX'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='顯示詳細處理資訊'
    )
    
    parser.add_argument(
        '--pretty',
        action='store_true',
        default=True,
        help='格式化 JSON 輸出 (預設啟用)'
    )
    
    parser.add_argument(
        '--no-pretty',
        action='store_false',
        dest='pretty',
        help='不格式化 JSON 輸出'
    )

    parser.add_argument(
        '--organize',
        help='根據 JSON 結構創建資料夾並複製影像檔案到指定的輸出目錄',
        metavar='DIR'
    )
    
    # 解析參數
    args = parser.parse_args()
    
    # 檢查輸入路徑是否存在
    if not os.path.exists(args.input_path):
        print(f"錯誤: 輸入路徑不存在: {args.input_path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isdir(args.input_path):
        print(f"錯誤: 輸入路徑不是資料夾: {args.input_path}", file=sys.stderr)
        sys.exit(1)
    
    # 建立處理器
    organizer = ImageOrganizer()
    
    if args.verbose:
        print(f"正在掃描目錄: {args.input_path}")
        print(f"支援的影像格式: {', '.join(organizer.image_extensions)}")
        if args.output:
            print(f"檔名 JSON 輸出: {args.output}")
        if args.score is not None:
            score_output = organizer.get_score_filename(args.output, args.score_suffix) if args.output else None
            print(f"分數值: {args.score}")
            if score_output:
                print(f"分數 JSON 輸出: {score_output}")
        print("-" * 50)
    
    try:
        # 掃描目錄
        result = organizer.scan_directory(args.input_path)
        
        if not result:
            print("未找到任何影像檔案")
            sys.exit(0)
        
        # 計算統計資訊
        if result:
            total_files = organizer.count_files(result)
            if args.verbose:
                print(f"找到 {total_files} 個影像檔案")
        else:
            total_files = 0
            if args.verbose:
                print("未找到任何影像檔案")
        
        # 輸出結果到終端 (除非指定了輸出檔案且不是 verbose 模式)
        if not args.output or args.verbose:
            print("\n檔名掃描結果:")
            if args.pretty:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(json.dumps(result, ensure_ascii=False))
        
        # 存儲檔名 JSON (如果指定輸出路徑)
        if args.output:
            organizer.save_json(result, args.output, pretty=args.pretty)
            if args.verbose:
                print(f"\n檔名 JSON 已儲存至: {args.output}")
            else:
                print(f"檔名 JSON 已儲存至: {args.output}")
        
        # 生成並存儲分數 JSON (如果指定分數值)
        if args.score is not None:
            score_result = organizer.convert_to_scores(result, args.score)

            if args.verbose:
                print(f"\n分數掃描結果 (分數值: {args.score}):")
                if args.pretty:
                    print(json.dumps(score_result, ensure_ascii=False, indent=2))
                else:
                    print(json.dumps(score_result, ensure_ascii=False))

            # 存儲分數 JSON
            if args.output:
                score_filename = organizer.get_score_filename(args.output, args.score_suffix)
                organizer.save_json(score_result, score_filename, pretty=args.pretty)
                if args.verbose:
                    print(f"\n分數 JSON 已儲存至: {score_filename}")
                else:
                    print(f"分數 JSON 已儲存至: {score_filename}")
            elif not args.output and not args.verbose:
                # 如果沒有指定輸出檔案，但有分數，顯示分數結果
                print(f"\n分數結果 (分數值: {args.score}):")
                if args.pretty:
                    print(json.dumps(score_result, ensure_ascii=False, indent=2))
                else:
                    print(json.dumps(score_result, ensure_ascii=False))

        # 整理檔案到資料夾結構 (如果指定整理目錄)
        if args.organize:
            if args.verbose:
                print(f"\n正在整理檔案到: {args.organize}")

            try:
                organizer.organize_files(result, args.input_path, args.organize)
                if args.verbose:
                    print(f"檔案整理完成: {args.organize}")
                else:
                    print(f"檔案已整理至: {args.organize}")
            except Exception as e:
                print(f"錯誤: 整理檔案時發生錯誤: {e}", file=sys.stderr)
        
    except KeyboardInterrupt:
        print("\n操作已取消", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"錯誤: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
