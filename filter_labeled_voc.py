import argparse
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm

def parse_arguments() -> argparse.Namespace:
    """
    解析命令列參數。
    """
    parser = argparse.ArgumentParser(
        description="分析 PASCAL VOC 1.0 資料集，並將有標記的 XML 和影像檔複製到新的資料夾。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
範例用法:
  # 基本使用，指定輸入和輸出目錄
  python filter_labeled_voc.py --input_dir ./voc_dataset --output_dir ./labeled_data

  # 指定不同的影像和標註資料夾名稱
  python filter_labeled_voc.py -i ./my_dataset -o ./filtered --image_folder Images --annotation_folder Annotations_XML
        """
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=Path,
        required=True,
        help="包含 PASCAL VOC 資料集的來源資料夾路徑。"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        required=True,
        help="用於存放已標記檔案的目標資料夾路徑。"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="JPEGImages",
        help="存放影像的資料夾名稱 (預設: JPEGImages)。"
    )
    parser.add_argument(
        "--annotation_folder",
        type=str,
        default="Annotations",
        help="存放 XML 標註檔的資料夾名稱 (預設: Annotations)。"
    )
    return parser.parse_args()

def filter_labeled_files(
    input_dir: Path,
    output_dir: Path,
    image_folder: str,
    annotation_folder: str
) -> None:
    """
    過濾出有標記的檔案並複製到目標目錄。

    Args:
        input_dir (Path): 來源 VOC 資料集目錄。
        output_dir (Path): 目標目錄。
        image_folder (str): 影像資料夾名稱。
        annotation_folder (str): 標註資料夾名稱。
    """
    # 1. 驗證來源路徑
    source_annotations_dir = input_dir / annotation_folder
    source_images_dir = input_dir / image_folder

    if not source_annotations_dir.is_dir():
        print(f"❌ 錯誤: 標註資料夾不存在: {source_annotations_dir}")
        return

    if not source_images_dir.is_dir():
        print(f"❌ 錯誤: 影像資料夾不存在: {source_images_dir}")
        return

    # 2. 建立目標路徑
    dest_annotations_dir = output_dir / annotation_folder
    dest_images_dir = output_dir / image_folder
    dest_annotations_dir.mkdir(parents=True, exist_ok=True)
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] 結果將儲存至: {output_dir.resolve()}")

    # 3. 迭代處理所有 XML 檔案
    xml_files = sorted(list(source_annotations_dir.glob("*.xml")))
    if not xml_files:
        print("⚠️  警告: 在來源標註資料夾中找不到任何 XML 檔案。")
        return

    print(f"\n[*] 正在掃描 {len(xml_files)} 個 XML 檔案...")

    labeled_count = 0
    unlabeled_count = 0
    image_not_found_count = 0

    for xml_path in tqdm(xml_files, desc="處理進度"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 檢查是否存在 <object> 標籤
            objects = root.findall("object")
            if len(objects) > 0:
                # 檔案有標記，執行複製操作
                labeled_count += 1

                # 複製 XML 檔案
                shutil.copy2(xml_path, dest_annotations_dir / xml_path.name)

                # 從 XML 中讀取對應的影像檔名
                image_filename_tag = root.find("filename")
                if image_filename_tag is None or not image_filename_tag.text:
                    print(f"⚠️  警告: {xml_path.name} 中缺少 <filename> 標籤，無法複製對應影像。")
                    image_not_found_count += 1
                    continue

                image_filename = image_filename_tag.text
                source_image_path = source_images_dir / image_filename

                if source_image_path.exists():
                    # 複製影像檔案
                    shutil.copy2(source_image_path, dest_images_dir / image_filename)
                else:
                    print(f"⚠️  警告: 找不到 {xml_path.name} 對應的影像檔案: {source_image_path}")
                    image_not_found_count += 1
            else:
                # 檔案沒有標記
                unlabeled_count += 1

        except ET.ParseError:
            print(f"❌ 錯誤: 無法解析 XML 檔案: {xml_path.name}")
        except Exception as e:
            print(f"❌ 處理檔案 {xml_path.name} 時發生未知錯誤: {e}")

    # 4. 輸出總結報告
    print("\n" + "="*50)
    print("✅ 處理完成！")
    print("\n📊 總結報告:")
    print(f"  - 總共掃描的 XML 檔案數: {len(xml_files)}")
    print(f"  - 有標記的檔案數 (已複製): {labeled_count}")
    print(f"  - 無標記的檔案數 (已跳過): {unlabeled_count}")
    if image_not_found_count > 0:
        print(f"  - 找不到對應影像的檔案數: {image_not_found_count}")
    print("="*50)


def main():
    """主執行函式"""
    args = parse_arguments()
    filter_labeled_files(
        args.input_dir,
        args.output_dir,
        args.image_folder,
        args.annotation_folder
    )

if __name__ == "__main__":
    main()
