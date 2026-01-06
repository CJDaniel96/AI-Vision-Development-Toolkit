import argparse
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm
from xml.dom import minidom

def parse_arguments() -> argparse.Namespace:
    """
    解析命令列參數。
    """
    parser = argparse.ArgumentParser(
        description="分析 CVAT for Image 1.0 格式的資料集，並將有標記的影像和標註資訊複製到新的資料夾。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
範例用法:
  # 基本使用，指定輸入和輸出目錄
  python filter_labeled_cvat.py --cvat_xml annotations.xml --image_dir ./images --output_dir ./labeled_data
        """
    )
    parser.add_argument(
        "--cvat_xml", "-x",
        type=Path,
        required=True,
        help="CVAT for Image 1.0 格式的 XML 標註檔案路徑。"
    )
    parser.add_argument(
        "--image_dir", "-i",
        type=Path,
        required=True,
        help="包含所有原始影像的來源資料夾路徑。"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        required=True,
        help="用於存放已標記影像和新 XML 檔案的目標資料夾路徑。"
    )
    return parser.parse_args()

def filter_labeled_files(
    cvat_xml_path: Path,
    image_dir: Path,
    output_dir: Path,
) -> None:
    """
    過濾出有標記的影像和標註，並複製到目標目錄。

    Args:
        cvat_xml_path (Path): CVAT XML 檔案路徑。
        image_dir (Path): 來源影像目錄。
        output_dir (Path): 目標目錄。
    """
    # 1. 驗證來源路徑
    if not cvat_xml_path.is_file():
        print(f"❌ 錯誤: CVAT XML 檔案不存在: {cvat_xml_path}")
        return

    if not image_dir.is_dir():
        print(f"❌ 錯誤: 影像資料夾不存在: {image_dir}")
        return

    # 2. 建立目標路徑
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] 結果將儲存至: {output_dir.resolve()}")

    # 3. 解析 CVAT XML 檔案
    try:
        tree = ET.parse(cvat_xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"❌ 錯誤: 無法解析 XML 檔案: {cvat_xml_path} - {e}")
        return

    # 建立新的 XML 結構，用於存放過濾後的結果
    new_root = ET.Element("annotations")
    # 複製 meta 資訊
    meta = root.find('meta')
    if meta is not None:
        new_root.append(meta)

    image_tags = root.findall("image")
    print(f"\n[*] 正在掃描 {len(image_tags)} 個影像標註...")

    labeled_count = 0
    unlabeled_count = 0
    image_not_found_count = 0

    # 定義可能的標註標籤
    annotation_tags = ['box', 'polygon', 'polyline', 'points', 'cuboid', 'skeleton', 'tag']

    for image_tag in tqdm(image_tags, desc="處理進度"):
        try:
            # 檢查 <image> 標籤下是否有任何標註
            has_annotation = any(image_tag.find(tag) is not None for tag in annotation_tags)

            if has_annotation:
                labeled_count += 1
                image_name = image_tag.get('name')

                if not image_name:
                    print(f"⚠️  警告: ID 為 {image_tag.get('id')} 的影像缺少 'name' 屬性，跳過。")
                    image_not_found_count += 1
                    continue

                source_image_path = image_dir / image_name

                if source_image_path.exists():
                    # 複製影像檔案
                    shutil.copy2(source_image_path, output_dir / Path(image_name).name)
                    # 將這個 <image> 標籤加入到新的 XML 樹中
                    new_root.append(image_tag)
                else:
                    print(f"⚠️  警告: 找不到影像檔案: {source_image_path}")
                    image_not_found_count += 1
            else:
                unlabeled_count += 1

        except Exception as e:
            print(f"❌ 處理 ID 為 {image_tag.get('id')} 的影像時發生未知錯誤: {e}")

    # 4. 儲存新的 XML 檔案
    if labeled_count > 0:
        new_xml_path = output_dir / cvat_xml_path.name
        xml_string = ET.tostring(new_root, 'utf-8')
        reparsed = minidom.parseString(xml_string)
        
        with open(new_xml_path, 'w', encoding='utf-8') as f:
            f.write(reparsed.toprettyxml(indent="  "))
        print(f"\n[*] 新的標註檔案已儲存至: {new_xml_path.resolve()}")

    # 5. 輸出總結報告
    print("\n" + "="*50)
    print("✅ 處理完成！")
    print("\n📊 總結報告:")
    print(f"  - 總共掃描的影像數: {len(image_tags)}")
    print(f"  - 有標記的檔案數 (已複製): {labeled_count}")
    print(f"  - 無標記的檔案數 (已跳過): {unlabeled_count}")
    if image_not_found_count > 0:
        print(f"  - 找不到對應影像的檔案數: {image_not_found_count}")
    print("="*50)


def main():
    """主執行函式"""
    args = parse_arguments()
    filter_labeled_files(
        args.cvat_xml,
        args.image_dir,
        args.output_dir,
    )

if __name__ == "__main__":
    main()
