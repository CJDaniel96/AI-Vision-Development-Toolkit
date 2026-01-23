import os
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

def filter_voc_dataset(input_dir: str, output_dir: str, target_labels: list[str] | None, exclude_labels: list[str] | None, filter_no_labels: bool = False, move_files: bool = False):
    """
    解析資料夾中的 PASCAL VOC XML 檔案，並將包含指定標記的
    影像和 XML 檔案複製到新的資料夾。此版本會遞迴搜尋子資料夾。

    Args:
        input_dir (str): 包含原始影像和 XML 檔案的資料夾路徑。
        output_dir (str): 用於存放篩選後檔案的新資料夾路徑。
        target_labels (list[str] | None): 要篩選的物件標記名稱列表。
        exclude_labels (list[str] | None): 要排除的物件標記名稱列表。
        filter_no_labels (bool): 是否篩選出沒有標記的資料。
        move_files (bool): 是否移動檔案而非複製。
    """
    # 步驟 1: 建立輸出資料夾 (如果不存在)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"輸出資料夾 '{output_path}' 已建立或已存在。")

    copied_files_count = 0
    
    # 定義操作模式
    file_op = shutil.move if move_files else shutil.copy
    op_name = "搬移" if move_files else "複製"

    # 使用 rglob 遞迴搜尋所有 .xml 檔案
    input_path = Path(input_dir)
    xml_files = list(input_path.rglob('*.xml'))

    if not xml_files:
        print(f"警告：在 '{input_dir}' 及其子資料夾中沒有找到任何 XML 檔案。")
        return

    print_msg = f"在 '{input_dir}' 及其子資料夾中找到 {len(xml_files)} 個 XML 檔案，開始篩選...\n"
    if target_labels:
        print_msg += f"  - 目標標記: {', '.join(target_labels)}"
    
    if filter_no_labels:
        prefix = " + " if target_labels else "  - 目標標記: "
        print_msg += f"{prefix}[無標記資料 (No Label)]"

    if not target_labels and not filter_no_labels:
        print_msg += "  - 目標標記: [無] (篩選所有未被排除的資料)"
    if exclude_labels:
        print_msg += f"\n  - 排除標記: {', '.join(exclude_labels)}"
    print(print_msg)

    # 步驟 2: 遍歷所有找到的 XML 檔案
    for xml_path in xml_files:
        try:
            # 步驟 3: 解析 XML 檔案
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 步驟 4: 尋找檔案中的所有標記
            labels_in_file = {obj.find('name').text for obj in root.findall('object') if obj.find('name') is not None and obj.find('name').text is not None}

            # 步驟 5: 排除條件檢查
            if exclude_labels and not set(exclude_labels).isdisjoint(labels_in_file):
                excluded_found = set(exclude_labels).intersection(labels_in_file)
                print(f"  [排除] -> '{xml_path.name}' 包含排除標記: {', '.join(excluded_found)}，已跳過。")
                continue

            # 步驟 6: 包含條件檢查
            keep_file = False
            reasons = []

            if target_labels:
                found_targets = set(target_labels).intersection(labels_in_file)
                if found_targets:
                    keep_file = True
                    reasons.append(f"找到目標標記 '{list(found_targets)[0]}'")
            
            if filter_no_labels and not labels_in_file:
                keep_file = True
                reasons.append("未包含任何標記")

            if not target_labels and not filter_no_labels:
                keep_file = True
                reasons.append("未包含任何排除標記")

            if not keep_file:
                continue

            # 檔案符合篩選條件，執行操作
            reason = " / ".join(reasons)

            # 處理 XML 檔案
            file_op(xml_path, output_path)

            # 尋找並複製對應的影像檔
            image_filename_base = xml_path.stem
            image_found = False
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            potential_image_paths = [xml_path.with_suffix(ext) for ext in image_extensions]
            if xml_path.parent.name.lower() in ['annotations', 'xml']:
                for folder in ['JPEGImages', 'images']:
                    image_dir = xml_path.parent.parent / folder
                    potential_image_paths.extend([image_dir / f"{image_filename_base}{ext}" for ext in image_extensions])
            for image_path in potential_image_paths:
                if image_path.exists():
                    file_op(image_path, output_path)
                    print(f"  [符合] -> {reason}。正在{op_name} '{xml_path.name}' 和 '{image_path.name}'...")
                    copied_files_count += 1
                    image_found = True
                    break
            
            if not image_found:
                print(f"  [警告] -> XML 檔案 '{xml_path.name}' 符合條件，但找不到對應的影像檔。")

        except ET.ParseError:
            print(f"  [錯誤] -> 無法解析 XML 檔案 '{xml_path.name}'，已跳過。")
        except Exception as e:
            print(f"  [錯誤] -> 處理 '{xml_path.name}' 時發生未知錯誤: {e}")

    print(f"\n處理完成！總共{op_name}了 {copied_files_count} 組影像與 XML 檔案到 '{output_dir}'。")


def main():
    """
    主函式，用於解析命令列參數。
    """
    parser = argparse.ArgumentParser(
        description="根據指定的標記名稱篩選 PASCAL VOC 資料集。\n"
                    "程式會將包含該標記的影像及其對應的 XML 檔案複製(或搬移)到新目錄。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="包含原始影像和 PASCAL VOC XML 檔案的資料夾路徑。"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="用於存放篩選後檔案的新資料夾路徑。"
    )
    parser.add_argument(
        "-l", "--label",
        nargs='+',
        type=str,
        default=None,
        help="要篩選的一個或多個標記 (class) 名稱，以空格分隔。\n例如: --label dog person"
    )
    parser.add_argument(
        "-e", "--exclude",
        nargs='+',
        type=str,
        default=None,
        help="要排除的一個或多個標記 (class) 名稱，以空格分隔。\n"
             "如果影像包含這些標記中的任何一個，將不會被複製。\n例如: --exclude person"
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="篩選出沒有任何標記的資料 (Empty Labels)。"
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="將檔案搬移到輸出目錄，而不是複製 (預設為複製)。"
    )

    args = parser.parse_args()

    if not args.label and not args.exclude and not args.no_label:
        parser.error("錯誤：您必須至少提供 --label, --exclude 或 --no-label 其中一個參數。")

    # 檢查輸入路徑是否存在
    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        print(f"錯誤：輸入路徑 '{input_path}' 不存在或不是一個有效的資料夾。")
        return

    filter_voc_dataset(args.input_dir, args.output_dir, args.label, args.exclude, args.no_label, args.move)


if __name__ == "__main__":
    main()
