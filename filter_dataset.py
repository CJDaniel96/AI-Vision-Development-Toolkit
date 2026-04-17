#!/usr/bin/env python3
"""
VOC / CVAT 資料集過濾工具

支援三種模式：
  1. VOC 平坦模式（預設）：遞迴搜尋 XML，依標籤條件過濾，輸出到平坦資料夾
  2. VOC 結構化模式（--voc-structure）：讀取 Annotations/ + JPEGImages/ 結構，保留輸出目錄結構
  3. CVAT 模式（--format cvat）：讀取單一 CVAT XML，過濾有標注的影像並輸出過濾後的 XML

使用範例：
  # VOC 平坦：只保留含 cat 或 dog 的檔案
  python filter_dataset.py -i ./dataset -o ./output -l cat dog

  # VOC 平坦：只保留有任何標注的檔案
  python filter_dataset.py -i ./dataset -o ./output --labeled-only

  # VOC 結構化：保留有標注的檔案，維持 Annotations/JPEGImages 目錄結構
  python filter_dataset.py -i ./voc_root -o ./output --voc-structure --labeled-only

  # CVAT：過濾有標注的影像，並輸出過濾後的 XML
  python filter_dataset.py --format cvat --cvat-xml ./annotations.xml -i ./images -o ./output
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom
from tqdm import tqdm


IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
ANNOTATION_TAGS  = ['box', 'polygon', 'polyline', 'points', 'cuboid', 'skeleton', 'tag']


# ══════════════════════════════════════════════════════
# 模式 1 & 2：VOC 格式（平坦 / 結構化）
# ══════════════════════════════════════════════════════

def _get_labels_from_voc_xml(xml_path: Path) -> set:
    """解析 VOC XML，回傳所有 <name> 集合。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return {
        obj.find('name').text
        for obj in root.findall('object')
        if obj.find('name') is not None and obj.find('name').text
    }


def _should_keep(labels: set, target_labels, exclude_labels, labeled_only: bool, filter_no_labels: bool) -> tuple:
    """回傳 (keep: bool, reason: str)"""
    # 排除條件
    if exclude_labels and not set(exclude_labels).isdisjoint(labels):
        found = set(exclude_labels) & labels
        return False, f"包含排除標記 {found}"

    # 包含條件
    if labeled_only:
        if labels:
            return True, "有標注"
        return False, "無標注"

    if target_labels:
        found = set(target_labels) & labels
        if found:
            return True, f"找到目標標記 {found}"

    if filter_no_labels and not labels:
        return True, "未包含任何標記"

    if not target_labels and not filter_no_labels:
        return True, "未包含任何排除標記"

    return False, "不符合條件"


def _find_image(xml_path: Path) -> Path | None:
    """在 XML 旁邊或 VOC 結構的 JPEGImages/ 中尋找對應影像。"""
    for ext in IMAGE_EXTENSIONS:
        # 同目錄
        candidate = xml_path.with_suffix(ext)
        if candidate.exists():
            return candidate
        # VOC 結構：../JPEGImages/
        if xml_path.parent.name.lower() in ('annotations', 'xml'):
            for folder in ('JPEGImages', 'images'):
                candidate = xml_path.parent.parent / folder / f"{xml_path.stem}{ext}"
                if candidate.exists():
                    return candidate
    return None


def run_voc_flat(input_dir: str, output_dir: str,
                 target_labels, exclude_labels,
                 labeled_only: bool, filter_no_labels: bool,
                 move_files: bool):
    """
    VOC 平坦模式：遞迴搜尋 XML，過濾後輸出到平坦資料夾。
    （整合自原 filter_dataset.py）
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_op = shutil.move if move_files else shutil.copy
    op_name = "搬移" if move_files else "複製"

    xml_files = list(input_path.rglob('*.xml'))
    if not xml_files:
        print(f"警告：在 '{input_dir}' 中找不到任何 XML 檔案。")
        return

    print(f"找到 {len(xml_files)} 個 XML 檔案，開始過濾...")
    copied = 0

    for xml_path in xml_files:
        try:
            labels = _get_labels_from_voc_xml(xml_path)
        except ET.ParseError:
            print(f"  [錯誤] 無法解析 '{xml_path.name}'，跳過。")
            continue
        except Exception as e:
            print(f"  [錯誤] 處理 '{xml_path.name}' 時發生錯誤：{e}")
            continue

        keep, reason = _should_keep(labels, target_labels, exclude_labels,
                                    labeled_only, filter_no_labels)
        if not keep:
            if exclude_labels and "排除" in reason:
                print(f"  [排除] {xml_path.name}：{reason}")
            continue

        file_op(xml_path, output_path)
        image_path = _find_image(xml_path)
        if image_path:
            file_op(image_path, output_path)
            print(f"  [符合] {xml_path.name}：{reason}")
            copied += 1
        else:
            print(f"  [警告] {xml_path.name} 符合條件，但找不到對應影像。")

    print(f"\n完成！{op_name}了 {copied} 組影像/XML 到 '{output_dir}'。")


def run_voc_structured(input_dir: str, output_dir: str,
                       image_folder: str, annotation_folder: str,
                       target_labels, exclude_labels,
                       labeled_only: bool, filter_no_labels: bool,
                       move_files: bool):
    """
    VOC 結構化模式：讀取 Annotations/ + JPEGImages/ 結構，輸出維持相同結構。
    （整合自原 filter_labeled_voc.py）
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    src_ann = input_path / annotation_folder
    src_img = input_path / image_folder

    if not src_ann.is_dir():
        print(f"錯誤：標注資料夾不存在：{src_ann}")
        return
    if not src_img.is_dir():
        print(f"錯誤：影像資料夾不存在：{src_img}")
        return

    dst_ann = output_path / annotation_folder
    dst_img = output_path / image_folder
    dst_ann.mkdir(parents=True, exist_ok=True)
    dst_img.mkdir(parents=True, exist_ok=True)

    file_op = shutil.move if move_files else shutil.copy2
    op_name = "搬移" if move_files else "複製"

    xml_files = sorted(src_ann.glob('*.xml'))
    if not xml_files:
        print(f"警告：在 {src_ann} 中找不到 XML 檔案。")
        return

    print(f"找到 {len(xml_files)} 個 XML 檔案，開始過濾...")
    labeled = unlabeled = img_not_found = 0

    for xml_path in tqdm(xml_files, desc="處理進度"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            labels = {
                obj.find('name').text
                for obj in root.findall('object')
                if obj.find('name') is not None and obj.find('name').text
            }
        except ET.ParseError:
            print(f"  [錯誤] 無法解析 '{xml_path.name}'，跳過。")
            continue

        keep, reason = _should_keep(labels, target_labels, exclude_labels,
                                    labeled_only, filter_no_labels)
        if not keep:
            unlabeled += 1
            continue

        labeled += 1
        file_op(xml_path, dst_ann / xml_path.name)

        # 從 XML 讀取 <filename> 尋找影像
        img_filename_tag = root.find('filename')
        if img_filename_tag is None or not img_filename_tag.text:
            print(f"  [警告] {xml_path.name} 缺少 <filename>，無法複製影像。")
            img_not_found += 1
            continue

        src_image = src_img / img_filename_tag.text
        if src_image.exists():
            file_op(src_image, dst_img / src_image.name)
        else:
            print(f"  [警告] 找不到影像：{src_image}")
            img_not_found += 1

    print(f"\n{'=' * 50}")
    print("處理完成！")
    print(f"  掃描 XML 總數：{len(xml_files)}")
    print(f"  符合條件（已{op_name}）：{labeled}")
    print(f"  不符合條件（已跳過）：{unlabeled}")
    if img_not_found:
        print(f"  找不到對應影像：{img_not_found}")
    print(f"{'=' * 50}")


# ══════════════════════════════════════════════════════
# 模式 3：CVAT 格式
# ══════════════════════════════════════════════════════

def _get_labels_from_cvat_image(image_tag) -> set:
    """從 CVAT <image> 標籤取得所有 label 名稱。"""
    labels = set()
    for ann_tag in ANNOTATION_TAGS:
        for elem in image_tag.findall(ann_tag):
            label = elem.get('label')
            if label:
                labels.add(label)
    return labels


def run_cvat(cvat_xml: str, image_dir: str, output_dir: str,
             target_labels, exclude_labels, labeled_only: bool):
    """
    CVAT 模式：讀取單一 CVAT XML，過濾後複製影像並輸出過濾後的 XML。
    （整合自原 filter_labeled_cvat.py）
    """
    cvat_xml_path = Path(cvat_xml)
    image_dir_path = Path(image_dir)
    output_path   = Path(output_dir)

    if not cvat_xml_path.is_file():
        print(f"錯誤：找不到 CVAT XML：{cvat_xml_path}")
        return
    if not image_dir_path.is_dir():
        print(f"錯誤：影像資料夾不存在：{image_dir_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    try:
        tree = ET.parse(cvat_xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"錯誤：無法解析 CVAT XML：{e}")
        return

    new_root = ET.Element('annotations')
    meta = root.find('meta')
    if meta is not None:
        new_root.append(meta)

    image_tags = root.findall('image')
    print(f"找到 {len(image_tags)} 個影像標注，開始過濾...")

    kept = skipped = img_not_found = 0

    for image_tag in tqdm(image_tags, desc="處理進度"):
        try:
            labels = _get_labels_from_cvat_image(image_tag)

            # 若 --labeled-only 且沒有指定其他條件，使用 has_annotation 判斷
            if labeled_only and not target_labels and not exclude_labels:
                has_ann = any(image_tag.find(t) is not None for t in ANNOTATION_TAGS)
                if not has_ann:
                    skipped += 1
                    continue
            else:
                keep, _ = _should_keep(labels, target_labels, exclude_labels,
                                       labeled_only, False)
                if not keep:
                    skipped += 1
                    continue

            kept += 1
            image_name = image_tag.get('name')
            if not image_name:
                print(f"  [警告] id={image_tag.get('id')} 缺少 name 屬性，跳過。")
                img_not_found += 1
                continue

            src_image = image_dir_path / image_name
            if src_image.exists():
                shutil.copy2(src_image, output_path / Path(image_name).name)
                new_root.append(image_tag)
            else:
                print(f"  [警告] 找不到影像：{src_image}")
                img_not_found += 1

        except Exception as e:
            print(f"  [錯誤] 處理 id={image_tag.get('id')} 時發生錯誤：{e}")

    # 儲存過濾後的 XML
    if kept > 0:
        new_xml_path = output_path / cvat_xml_path.name
        xml_str = ET.tostring(new_root, 'utf-8')
        reparsed = minidom.parseString(xml_str)
        with open(new_xml_path, 'w', encoding='utf-8') as f:
            f.write(reparsed.toprettyxml(indent='  '))
        print(f"\n過濾後的 XML 已儲存至：{new_xml_path}")

    print(f"\n{'=' * 50}")
    print("處理完成！")
    print(f"  掃描影像總數：{len(image_tags)}")
    print(f"  符合條件（已複製）：{kept}")
    print(f"  不符合條件（已跳過）：{skipped}")
    if img_not_found:
        print(f"  找不到對應影像：{img_not_found}")
    print(f"{'=' * 50}")


# ══════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VOC / CVAT 資料集過濾工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  # VOC 平坦：篩選含 cat 或 dog 的標注
  python filter_dataset.py -i ./dataset -o ./output -l cat dog

  # VOC 平坦：只保留有任何標注的檔案（排除空標注）
  python filter_dataset.py -i ./dataset -o ./output --labeled-only

  # VOC 平坦：排除含 person 的標注
  python filter_dataset.py -i ./dataset -o ./output -e person

  # VOC 結構化：維持 Annotations/JPEGImages 目錄結構，只保留有標注的
  python filter_dataset.py -i ./voc_root -o ./output --voc-structure --labeled-only

  # CVAT：過濾有標注的影像並輸出過濾後的 XML
  python filter_dataset.py --format cvat --cvat-xml ./annotations.xml -i ./images -o ./output

  # CVAT：只保留含特定標籤的影像
  python filter_dataset.py --format cvat --cvat-xml ./annotations.xml -i ./images -o ./output -l defect
        """,
    )

    # ── 格式選擇
    parser.add_argument('--format', choices=['voc', 'cvat'], default='voc',
                        help='標注格式（預設：voc）')

    # ── 路徑
    parser.add_argument('-i', '--input_dir', required=True,
                        help='輸入資料夾（VOC 模式：含 XML 的目錄；CVAT 模式：影像資料夾）')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='輸出資料夾')
    parser.add_argument('--cvat-xml', type=str, default=None,
                        help='[CVAT 模式] CVAT XML 標注檔路徑（--format cvat 時必填）')

    # ── VOC 結構化選項
    parser.add_argument('--voc-structure', action='store_true',
                        help='[VOC] 以 Annotations/JPEGImages 結構讀取並輸出')
    parser.add_argument('--image-folder', type=str, default='JPEGImages',
                        help='[VOC 結構化] 影像子資料夾名稱（預設：JPEGImages）')
    parser.add_argument('--annotation-folder', type=str, default='Annotations',
                        help='[VOC 結構化] 標注子資料夾名稱（預設：Annotations）')

    # ── 過濾條件
    parser.add_argument('-l', '--label', nargs='+', default=None,
                        help='保留含有這些標籤的檔案（空格分隔）')
    parser.add_argument('-e', '--exclude', nargs='+', default=None,
                        help='排除含有這些標籤的檔案（空格分隔）')
    parser.add_argument('--no-label', action='store_true',
                        help='保留沒有任何標注的檔案（空標注）')
    parser.add_argument('--labeled-only', action='store_true',
                        help='只保留有任何標注的檔案（忽略空標注）')

    # ── 其他
    parser.add_argument('--move', action='store_true',
                        help='搬移檔案而非複製（預設：複製）')

    args = parser.parse_args()

    # ── 驗證
    if args.format == 'cvat' and not args.cvat_xml:
        parser.error("使用 --format cvat 時必須提供 --cvat-xml 路徑。")

    if args.format == 'voc':
        has_condition = args.label or args.exclude or args.no_label or args.labeled_only
        if not has_condition:
            parser.error("請至少提供 --label、--exclude、--no-label 或 --labeled-only 其中一個過濾條件。")

    if not Path(args.input_dir).is_dir():
        print(f"錯誤：輸入路徑 '{args.input_dir}' 不存在或不是資料夾。")
        return

    # ── 執行對應模式
    if args.format == 'cvat':
        run_cvat(
            cvat_xml=args.cvat_xml,
            image_dir=args.input_dir,
            output_dir=args.output_dir,
            target_labels=args.label,
            exclude_labels=args.exclude,
            labeled_only=args.labeled_only or (not args.label and not args.exclude),
        )
    elif args.voc_structure:
        run_voc_structured(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            image_folder=args.image_folder,
            annotation_folder=args.annotation_folder,
            target_labels=args.label,
            exclude_labels=args.exclude,
            labeled_only=args.labeled_only,
            filter_no_labels=args.no_label,
            move_files=args.move,
        )
    else:
        run_voc_flat(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_labels=args.label,
            exclude_labels=args.exclude,
            labeled_only=args.labeled_only,
            filter_no_labels=args.no_label,
            move_files=args.move,
        )


if __name__ == "__main__":
    main()
