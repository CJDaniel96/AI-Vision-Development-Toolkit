#!/usr/bin/env python3
"""
影像擴增腳本 (Image Augmentation Script)
功能：
  - 支援 PASCAL VOC (.xml)、YOLO (.txt)、CVAT (.xml) 格式的標註同步擴增
  - 支援純影像擴增（augment image-only）
  - 多種擴增管道：horizontal_flip, brightness_contrast, hue_saturation,
                   mixed, smart_camera, hz_pde, connector, rotation, image_only

使用範例：
  # VOC 格式（影像與 XML 同一資料夾）
  python main.py augment annotated -i ./dataset -o ./augmented -a mixed -n 3

  # YOLO/多格式（影像與標籤分開）
  python main.py augment annotated -i ./images -l ./labels -o ./augmented -a mixed -n 5

  # Connector 標註同步擴增（保留 pin row 幾何，不旋轉、不翻轉、不透視）
  python main.py augment annotated -i ./images -l ./labels -o ./augmented -a connector -n 5

  # 旋轉擴增（隨機角度）
  python main.py augment annotated -i ./images -l ./labels -o ./augmented -a rotation --rotate_limit 40 -n 5

  # 旋轉擴增（固定步長，每隔 45° 產生一個版本）
  python main.py augment annotated -i ./images -l ./labels -o ./augmented -a rotation --rotate_step 45

  # 純影像擴增（無標註，預設不旋轉）
  python main.py augment image-only -i ./images -o ./augmented -n 3

  # 純影像擴增並加入保守的隨機旋轉（預設 ±5°，此處自訂為 ±10°）
  python main.py augment image-only -i ./images -o ./augmented --image_only_rotate 10 -n 3

  # 整批資料總共產生 100 張擴增影像（不是每張原圖各 100 張）
  python main.py augment image-only -i ./images -o ./augmented --total_count 100

  # 類別平衡：讓標註裡每個 label 在輸出中都累積到 100 個 bbox instance（輸出不含原圖）
  python main.py augment annotated -i ./dataset -o ./balanced -a mixed --augment_per_class 100

  # 純影像擴增：每個第一層子資料夾產生 200 張擴增影像（輸出不含原圖）
  python main.py augment image-only -i ./class_images -o ./balanced --augment_per_folder 200

  # Connector 純影像擴增：保留 pin row 幾何，不旋轉、不翻轉、不透視
  python main.py augment image-only -i ./connector_images -o ./connector_aug --image_only_profile connector --augment_per_folder 200
"""

import argparse
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

from .augmentation_profiles import IMAGE_EXTENSIONS as VALID_IMAGE_EXTENSIONS
from .augmentation_profiles import build_transforms, load_profiles


# ══════════════════════════════════════════════════════════════
# 標註格式 讀取 / 儲存
# ══════════════════════════════════════════════════════════════

def load_yolo_annotations(label_path: Path) -> Tuple[list, list, str]:
    bboxes, class_labels = [], []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line_number, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    raise ValueError(
                        f'{label_path.name} 第 {line_number} 行：YOLO 水平框需要 5 欄；'
                        '四角點 OBB 請使用 augment obb'
                    )
                class_labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])
    return bboxes, class_labels, 'yolo'


def save_yolo_annotations(label_path: Path, bboxes: list, class_labels: list):
    with open(label_path, 'w') as f:
        for bbox, label in zip(bboxes, class_labels):
            f.write(f"{int(label)} {' '.join(f'{v:.6f}' for v in bbox)}\n")


def load_pascal_voc_annotations(xml_path: Path) -> Tuple[list, list, str]:
    bboxes, class_labels = [], []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        bboxes.append([xmin, ymin, xmax, ymax])
        class_labels.append(name)
    return bboxes, class_labels, 'pascal_voc'


def save_pascal_voc_annotations(label_path: Path, bboxes: list, class_labels: list,
                                 filename: str, width: int, height: int):
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'augmented'
    ET.SubElement(annotation, 'filename').text = filename
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'
    for bbox, label in zip(bboxes, class_labels):
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = str(label)
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(bbox[0]))
        ET.SubElement(bndbox, 'ymin').text = str(int(bbox[1]))
        ET.SubElement(bndbox, 'xmax').text = str(int(bbox[2]))
        ET.SubElement(bndbox, 'ymax').text = str(int(bbox[3]))
    tree = ET.ElementTree(annotation)
    ET.indent(tree, space='  ')
    tree.write(str(label_path), encoding='utf-8', xml_declaration=True)


def load_cvat_annotations(xml_path: Path) -> Tuple[list, list, str]:
    """載入 CVAT for Images XML 格式（bbox 座標與 pascal_voc 相同）"""
    bboxes, class_labels = [], []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for image_tag in root.findall('image'):
        for box in image_tag.findall('box'):
            label = box.get('label')
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            bboxes.append([xtl, ytl, xbr, ybr])
            class_labels.append(label)
    return bboxes, class_labels, 'pascal_voc'


def save_cvat_annotations(label_path: Path, bboxes: list, class_labels: list,
                           filename: str, width: int, height: int):
    annotations = ET.Element('annotations')
    ET.SubElement(annotations, 'version').text = '1.1'
    image = ET.SubElement(annotations, 'image',
                           id='0', name=filename, width=str(width), height=str(height))
    for bbox, label in zip(bboxes, class_labels):
        ET.SubElement(image, 'box', label=str(label), occluded='0', source='manual',
                      xtl=str(bbox[0]), ytl=str(bbox[1]),
                      xbr=str(bbox[2]), ybr=str(bbox[3]), z_order='0')
    tree = ET.ElementTree(annotations)
    ET.indent(tree, space='  ')
    tree.write(str(label_path), encoding='utf-8', xml_declaration=True)


def detect_and_load_annotations(
    image_path: Path,
    labels_dir: Path
) -> Tuple[Optional[list], Optional[list], Optional[str], Optional[Callable], Optional[str]]:
    """
    自動偵測並載入對應的標籤檔。
    回傳 (bboxes, class_labels, ann_format, save_func, ann_ext)，
    偵測不到時回傳 (None, None, None, None, None)。
    """
    base_name = image_path.stem

    xml_path = labels_dir / f'{base_name}.xml'
    if xml_path.exists():
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            if root.tag == 'annotations':
                bboxes, labels, fmt = load_cvat_annotations(xml_path)
                return bboxes, labels, fmt, save_cvat_annotations, '.xml'
            else:
                bboxes, labels, fmt = load_pascal_voc_annotations(xml_path)
                return bboxes, labels, fmt, save_pascal_voc_annotations, '.xml'
        except ET.ParseError:
            print(f'警告：XML 檔案 {xml_path} 解析失敗，跳過。')
            return None, None, None, None, None

    txt_path = labels_dir / f'{base_name}.txt'
    if txt_path.exists():
        try:
            bboxes, labels, fmt = load_yolo_annotations(txt_path)
        except (OSError, ValueError) as error:
            tqdm.write(f'警告：無法讀取 {txt_path}：{error}')
            return None, None, None, None, None
        return bboxes, labels, fmt, save_yolo_annotations, '.txt'

    return None, None, None, None, None


# ══════════════════════════════════════════════════════════════
# 擴增管道定義
# ══════════════════════════════════════════════════════════════

def build_image_only_pipeline(
    rotate_limit: Optional[int] = None,
    profile: str = 'default',
    *,
    profiles: dict | None = None,
) -> A.Compose:
    """建立純影像擴增管道，旋轉可由參數額外開啟。"""
    if profile == 'connector' and rotate_limit is not None:
        raise ValueError('connector profile 不允許旋轉，請移除 --image_only_rotate')
    if profiles is None:
        profiles = load_profiles()['image_only']
    if profile not in profiles:
        raise ValueError(f'不支援的 image_only profile: {profile}')

    transforms = build_transforms(profiles[profile])
    if rotate_limit is not None:
        transforms.insert(0, A.SafeRotate(
            limit=rotate_limit, p=1.0, border_mode=cv2.BORDER_REFLECT_101,
        ))
    return A.Compose(transforms)


def build_pipeline(transforms_list: list, ann_format: str) -> A.Compose:
    """根據標註格式動態建立帶 bbox_params 的 A.Compose"""
    return A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(
            format=ann_format,
            label_fields=['class_labels'],
            min_visibility=0.2,
            min_area=16,
            clip=True,
        )
    )


def build_rotation_transforms(rotate_limit: Optional[int],
                               rotate_step: Optional[int],
                               profiles: dict) -> Tuple[list, bool]:
    """
    建立旋轉管道的 transforms list。
    回傳 (transforms_list, is_fixed_step)。
    is_fixed_step=True 表示需要在主迴圈中使用固定角度模式。
    """
    base_transforms = build_transforms(profiles['rotation'])
    if rotate_step is not None:
        print(f'啟用固定角度步長旋轉，步長: {rotate_step}°')
        return base_transforms, True
    elif rotate_limit is not None:
        print(f'啟用隨機角度旋轉，範圍: ±{rotate_limit}°')
        rotation = [
            A.Rotate(limit=rotate_limit, p=0.7,
                     border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.Affine(scale=(0.9, 1.1), translate_percent=0.1, p=0.5,
                     border_mode=cv2.BORDER_CONSTANT, fill=0),
        ]
        return rotation + base_transforms, False
    else:
        print('啟用 90/180/270° 固定角度旋轉')
        return [A.RandomRotate90(p=0.75)] + base_transforms, False


def make_transforms_resolver(
    rotation_transforms: Optional[list],
    is_fixed_step: bool,
    rotate_step: Optional[int],
    annotated_transforms: dict[str, list],
) -> Callable[[str], Optional[list]]:
    """
    建立「擴增類型 → transforms list」的解析函式。
    rotation 在固定步長模式下每被取用一次就前進一個角度；
    無法解析的類型回傳 None，由呼叫端決定跳過。
    """
    rotation_index = 0

    def resolve(aug_type: str) -> Optional[list]:
        nonlocal rotation_index
        if aug_type != 'rotation':
            return annotated_transforms.get(aug_type)
        if rotation_transforms is None or not is_fixed_step:
            return rotation_transforms
        angle = (rotation_index * rotate_step) % 360
        rotation_index += 1
        return [
            A.Rotate(limit=(angle, angle), p=1.0,
                     border_mode=cv2.BORDER_CONSTANT, fill=0)
        ] + rotation_transforms

    return resolve


# ══════════════════════════════════════════════════════════════
# 純影像擴增模式
# ══════════════════════════════════════════════════════════════

def list_image_files(directory: Path) -> List[Path]:
    """列出資料夾中的影像檔，固定排序讓輸出可重現。"""
    return sorted(
        [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ],
        key=lambda p: p.name.lower(),
    )


def save_image_only_augmentation(
    img_path: Path,
    pipeline: A.Compose,
    output_path: Path,
    attempt_label: str,
) -> bool:
    """對單張影像執行純影像擴增並儲存。"""
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        tqdm.write(f'[警告] 無法讀取影像: {img_path.name}')
        return False

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    try:
        transformed = pipeline(image=image_rgb)
        aug_rgb = transformed['image']
    except Exception as e:
        tqdm.write(f'[錯誤] 處理 {img_path.name} {attempt_label} 時發生例外：{e}')
        return False

    cv2.imwrite(str(output_path), cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR))
    return True


def run_image_only(
    images_dir: Path,
    output_dir: Path,
    num_augmentations: int,
    rotate_limit: Optional[int] = None,
    image_only_profile: str = 'default',
    *,
    profiles: dict | None = None,
):
    """產生每張輸入影像所需的擴增版本。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    img_files = list_image_files(images_dir)

    if not img_files:
        print(f'[提示] 在 {images_dir} 中找不到影像檔案。')
        return

    pipeline = build_image_only_pipeline(
        rotate_limit=rotate_limit,
        profile=image_only_profile,
        profiles=profiles,
    )
    print(f'找到 {len(img_files)} 張影像，開始純影像擴增...')
    for img_path in tqdm(img_files):
        for i in range(num_augmentations):
            prefix = f'aug_{i + 1}_' if num_augmentations > 1 else 'aug_'
            out_name = f'{prefix}{img_path.name}'
            save_image_only_augmentation(
                img_path,
                pipeline,
                output_dir / out_name,
                f'第 {i + 1} 次',
            )

    print(f'純影像擴增完成！輸出目錄：{output_dir}')


def run_image_only_total_count(
    images_dir: Path,
    output_dir: Path,
    total_count: int,
    rotate_limit: Optional[int] = None,
    image_only_profile: str = 'default',
    *,
    profiles: dict | None = None,
) -> int:
    """輪流取用來源影像，整批產生指定數量的純影像擴增結果。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    img_files = list_image_files(images_dir)

    if not img_files:
        print(f'[提示] 在 {images_dir} 中找不到影像檔案。')
        return 0

    pipeline = build_image_only_pipeline(
        rotate_limit=rotate_limit,
        profile=image_only_profile,
        profiles=profiles,
    )
    print(
        f'找到 {len(img_files)} 張影像，開始產生指定總數 '
        f'{total_count} 張擴增影像...'
    )

    success_count = 0
    attempt_count = 0
    # 允許略過損壞影像或偶發的擴增錯誤，同時避免所有來源都無法處理時無限迴圈。
    max_attempts = max(total_count * max(2, len(img_files)), len(img_files))
    with tqdm(total=total_count, desc='整批擴增') as progress:
        while success_count < total_count and attempt_count < max_attempts:
            img_path = img_files[attempt_count % len(img_files)]
            output_index = success_count + 1
            out_name = f'aug_{output_index:05d}_{img_path.name}'
            attempt_count += 1

            ok = save_image_only_augmentation(
                img_path,
                pipeline,
                output_dir / out_name,
                f'總數模式第 {output_index} 張',
            )
            if ok:
                success_count += 1
                progress.update(1)

    if success_count < total_count:
        print(
            f'[警告] 指定產生 {total_count} 張，但只有 {success_count} 張成功；'
            '請檢查無法讀取的來源影像或上方錯誤訊息。'
        )
    else:
        print(f'純影像擴增完成！已產生指定的 {success_count} 張影像。')
    print(f'輸出目錄：{output_dir}')
    return success_count


def run_image_only_augment_per_folder(
    images_dir: Path,
    output_dir: Path,
    augment_per_folder: int,
    rotate_limit: Optional[int] = None,
    image_only_profile: str = 'default',
    *,
    profiles: dict | None = None,
):
    """每個第一層子資料夾產生固定數量的擴增影像，輸出不複製原圖。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_image_only_pipeline(
        rotate_limit=rotate_limit,
        profile=image_only_profile,
        profiles=profiles,
    )

    output_root = output_dir.resolve()
    subdirs = sorted(
        [
            d for d in images_dir.iterdir()
            if d.is_dir() and d.resolve() != output_root
        ],
        key=lambda p: p.name.lower(),
    )

    if not subdirs:
        print(f'[提示] 在 {images_dir} 中找不到子資料夾。')
        return

    total_augmented = 0
    print(
        f'找到 {len(subdirs)} 個子資料夾，開始對每資料夾產生 '
        f'{augment_per_folder} 張擴增影像（輸出不含原圖）...'
    )

    for subdir in subdirs:
        img_files = list_image_files(subdir)
        if not img_files:
            tqdm.write(f'[提示] {subdir.name} 中找不到影像檔案，跳過。')
            continue

        folder_output_dir = output_dir / subdir.name
        folder_output_dir.mkdir(parents=True, exist_ok=True)

        existing_outputs = list_image_files(folder_output_dir)
        if existing_outputs:
            tqdm.write(
                f'[提醒] 輸出子資料夾 {folder_output_dir} 已有 '
                f'{len(existing_outputs)} 張影像；腳本不會自動刪除舊檔。'
            )

        source_count = len(img_files)
        folder_augmented = 0
        for i in tqdm(
            range(augment_per_folder),
            desc=f'{subdir.name} 擴增',
            leave=False,
        ):
            src_img = img_files[i % source_count]
            out_name = f'aug_{i + 1:05d}_{src_img.name}'
            ok = save_image_only_augmentation(
                src_img,
                pipeline,
                folder_output_dir / out_name,
                f'擴增第 {i + 1} 張',
            )
            if ok:
                total_augmented += 1
                folder_augmented += 1

        output_count = len(list_image_files(folder_output_dir))
        tqdm.write(
            f'{subdir.name}: 來源原圖 {source_count} 張，新增擴增 '
            f'{folder_augmented} 張，輸出目前 {output_count} 張。'
        )

    print(f'子資料夾擴增完成！新增擴增 {total_augmented} 張，未複製原圖。')
    print(f'輸出目錄：{output_dir}')


# ══════════════════════════════════════════════════════════════
# 標註同步擴增模式
# ══════════════════════════════════════════════════════════════

def augment_single(image_path: Path, bboxes: list, class_labels: list,
                   ann_format: str, save_func: Callable, ann_ext: str,
                   pipeline_name: str, transforms_list: list,
                   output_images_dir: Path, output_labels_dir: Path,
                   suffix: str) -> Optional[list]:
    """
    對單張影像執行一次擴增並儲存結果。
    成功時回傳實際寫出的 class_labels（擴增後可能少於原始標註），失敗回傳 None。
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = build_pipeline(transforms_list, ann_format)
    try:
        augmented = transform(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
    except Exception as e:
        print(f'  錯誤：{image_path.name} ({pipeline_name}) 擴增失敗：{e}')
        return None

    aug_bboxes = augmented['bboxes']
    aug_labels = augmented['class_labels']
    if not aug_bboxes:
        return None  # 物件全部超出邊界，跳過

    aug_image = augmented['image']
    aug_h, aug_w = aug_image.shape[:2]
    new_stem = f'{image_path.stem}_{suffix}'
    new_img_name = f'{new_stem}{image_path.suffix}'

    cv2.imwrite(str(output_images_dir / new_img_name),
                cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
    label_path = output_labels_dir / f'{new_stem}{ann_ext}'
    if ann_format == 'yolo':
        save_func(label_path, aug_bboxes, aug_labels)
    else:
        save_func(label_path, aug_bboxes, aug_labels, filename=new_img_name, width=aug_w, height=aug_h)
    return list(aug_labels)


def load_annotated_sources(image_files: List[Path], labels_dir: Path) -> list:
    """載入所有可用的 (影像, 標註) 組合，缺標籤檔的影像會被跳過。"""
    sources = []
    for image_path in image_files:
        annotation = detect_and_load_annotations(image_path, labels_dir)
        if annotation[0] is None:
            tqdm.write(f'警告：{image_path.name} 沒有可用的標註，跳過。')
            continue
        sources.append((image_path, *annotation))
    return sources


# ══════════════════════════════════════════════════════════════
# 類別平衡擴增（每個 label 的 instance 數擴增到指定值）
# ══════════════════════════════════════════════════════════════

# 同一張來源影像連續失敗達此次數後不再取用（多半是檔案損毀或標註全部超界）
MAX_SOURCE_FAILURES = 3


def score_source_against_deficits(
    label_counts: Counter,
    deficits: Dict[str, int],
) -> Tuple[int, int]:
    """
    評估單一來源影像對目前缺口的貢獻。
    回傳 (gain, waste)：gain 是能補進缺口的 instance 數，
    waste 是已達標類別上多產生的 instance 數。
    """
    gain = 0
    waste = 0
    for label, count in label_counts.items():
        need = max(deficits.get(label, 0), 0)
        gain += min(count, need)
        waste += max(count - need, 0)
    return gain, waste


def select_balance_source(
    source_label_counts: List[Counter],
    deficits: Dict[str, int],
    usage: List[int],
    blocked: Optional[set] = None,
) -> Optional[int]:
    """
    挑出最適合下一次擴增的來源索引。

    一張影像常同時含多個類別，補足某個類別必然會讓其他類別超額，
    因此以 gain - waste 為主要準則，優先選「補得最多、超額最少」的來源；
    同分時選使用次數最少的，讓同一類別的擴增結果分散到不同原圖上。
    沒有任何來源能再補足缺口時回傳 None。
    """
    blocked = blocked or set()
    best_index: Optional[int] = None
    best_key: Optional[tuple] = None

    for index, label_counts in enumerate(source_label_counts):
        if index in blocked:
            continue
        gain, waste = score_source_against_deficits(label_counts, deficits)
        if gain <= 0:
            continue
        key = (gain - waste, -usage[index], -index)
        if best_key is None or key > best_key:
            best_key = key
            best_index = index

    return best_index


def print_balance_report(
    original_totals: Counter,
    produced_totals: Counter,
    per_class_count: int,
):
    """列出每個類別的原始 / 輸出 instance 數對照表。"""
    label_width = max([len(str(label)) for label in original_totals] + [8])
    print(f'{"類別".ljust(label_width)}{"原始":>10}{"輸出":>10}{"目標":>10}')
    for label in sorted(original_totals, key=str):
        print(
            f'{str(label).ljust(label_width)}'
            f'{original_totals[label]:>10}'
            f'{produced_totals.get(label, 0):>10}'
            f'{per_class_count:>10}'
        )

    shortfall = [
        label for label in original_totals
        if produced_totals.get(label, 0) < per_class_count
    ]
    if shortfall:
        names = ', '.join(str(label) for label in sorted(shortfall, key=str))
        print(
            f'[警告] 下列類別未達目標：{names}；'
            '可能是來源影像過少、影像無法讀取，或擴增後物件全部超出邊界。'
        )


def run_per_class_balance(
    annotated_sources: list,
    augmentation_types: List[str],
    resolve_transforms: Callable[[str], Optional[list]],
    per_class_count: int,
    output_images_dir: Path,
    output_labels_dir: Path,
) -> int:
    """
    將每個類別的 bbox instance 數擴增到 per_class_count，並列出結果對照表。
    輸出只含擴增結果，不複製原圖。回傳成功產生的影像/標籤組數。
    """
    source_label_counts = [
        Counter(str(label) for label in source[2]) for source in annotated_sources
    ]
    original_totals: Counter = Counter()
    for label_counts in source_label_counts:
        original_totals.update(label_counts)

    if not original_totals:
        print('[提示] 所有標籤檔都沒有標註物件，未產生擴增結果。')
        return 0

    deficits: Dict[str, int] = {label: per_class_count for label in original_totals}
    produced_totals: Counter = Counter()
    usage = [0] * len(annotated_sources)
    failures = [0] * len(annotated_sources)
    blocked: set = set()

    print(
        f'找到 {len(annotated_sources)} 組可用影像/標籤、{len(original_totals)} 個類別，'
        f'開始將每個類別擴增到 {per_class_count} 個 instance...'
    )

    total_needed = per_class_count * len(original_totals)
    max_attempts = max(total_needed * 10, len(annotated_sources) * 10)
    attempt_count = 0
    success_count = 0
    aug_cycle = 0

    with tqdm(total=total_needed, desc='類別平衡擴增') as progress:
        while attempt_count < max_attempts and any(v > 0 for v in deficits.values()):
            index = select_balance_source(source_label_counts, deficits, usage, blocked)
            if index is None:
                break

            attempt_count += 1
            aug_type = augmentation_types[aug_cycle % len(augmentation_types)]
            aug_cycle += 1
            transforms_list = resolve_transforms(aug_type)
            if transforms_list is None:
                tqdm.write(f'警告：不支援的擴增類型 "{aug_type}"，跳過。')
                continue

            image_path, bboxes, class_labels, ann_format, save_func, ann_ext = \
                annotated_sources[index]
            suffix = f'{aug_type}_bal_{success_count + 1:05d}'
            saved_labels = augment_single(
                image_path, bboxes, class_labels, ann_format,
                save_func, ann_ext, aug_type, transforms_list,
                output_images_dir, output_labels_dir, suffix,
            )

            if saved_labels is None:
                failures[index] += 1
                if failures[index] >= MAX_SOURCE_FAILURES:
                    blocked.add(index)
                    tqdm.write(
                        f'[警告] {image_path.name} 連續 {failures[index]} 次擴增失敗，'
                        '後續不再取用。'
                    )
                continue

            failures[index] = 0
            usage[index] += 1
            success_count += 1

            produced = Counter(str(label) for label in saved_labels)
            produced_totals.update(produced)
            filled = 0
            for label, count in produced.items():
                filled += min(count, max(deficits.get(label, 0), 0))
                deficits[label] = deficits.get(label, 0) - count
            if filled:
                progress.update(filled)

    print(f'\n類別平衡擴增完成！成功產生 {success_count} 組影像/標籤（未複製原圖）')
    print_balance_report(original_totals, produced_totals, per_class_count)
    return success_count


def run_annotated_augmentation(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    augmentation_types: List[str],
    num_augmentations: int,
    rotate_limit: Optional[int],
    rotate_step: Optional[int],
    total_count: Optional[int] = None,
    augment_per_class: Optional[int] = None,
    *,
    profiles: dict | None = None,
):
    """標註同步擴增，支援 VOC / YOLO / CVAT 自動偵測"""
    if profiles is None:
        profiles = load_profiles()['annotated']
    annotated_transforms = {
        name: build_transforms(profiles[name])
        for name in dict.fromkeys(augmentation_types) if name != 'rotation' and name in profiles
    }
    output_images_dir = output_dir / 'images'
    output_labels_dir = output_dir / 'labels'
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = list_image_files(images_dir)

    if not image_files:
        print(f'[提示] 在 {images_dir} 中找不到影像檔案。')
        return

    print(f'找到 {len(image_files)} 張影像，開始標註同步擴增...')

    # 預先建立旋轉 transforms（若有指定 rotation 管道）
    rotation_transforms = None
    is_fixed_step = False
    if 'rotation' in augmentation_types:
        rotation_transforms, is_fixed_step = build_rotation_transforms(rotate_limit, rotate_step, profiles)
    resolve_transforms = make_transforms_resolver(
        rotation_transforms, is_fixed_step, rotate_step, annotated_transforms,
    )

    if augment_per_class is not None:
        annotated_sources = load_annotated_sources(image_files, labels_dir)
        if not annotated_sources:
            print('[提示] 找不到任何可用的影像/標籤組合，未產生擴增結果。')
            return

        run_per_class_balance(
            annotated_sources=annotated_sources,
            augmentation_types=augmentation_types,
            resolve_transforms=resolve_transforms,
            per_class_count=augment_per_class,
            output_images_dir=output_images_dir,
            output_labels_dir=output_labels_dir,
        )
        print(f'  影像輸出：{output_images_dir}')
        print(f'  標籤輸出：{output_labels_dir}')
        return

    if total_count is not None:
        annotated_sources = load_annotated_sources(image_files, labels_dir)
        if not annotated_sources:
            print('[提示] 找不到任何可用的影像/標籤組合，未產生擴增結果。')
            return

        success_count = 0
        attempt_count = 0
        candidate_count = len(annotated_sources) * len(augmentation_types)
        max_attempts = max(total_count * 10, candidate_count)
        print(
            f'找到 {len(annotated_sources)} 組可用影像/標籤，'
            f'將整批產生指定總數 {total_count} 組擴增結果...'
        )

        with tqdm(total=total_count, desc='整批標註擴增') as progress:
            while success_count < total_count and attempt_count < max_attempts:
                source_index = (attempt_count // len(augmentation_types)) % len(annotated_sources)
                aug_type = augmentation_types[attempt_count % len(augmentation_types)]
                image_path, bboxes, class_labels, ann_format, save_func, ann_ext = \
                    annotated_sources[source_index]
                attempt_count += 1

                transforms_list = resolve_transforms(aug_type)
                if transforms_list is None:
                    continue

                suffix = f'{aug_type}_total_{success_count + 1:05d}'
                saved_labels = augment_single(
                    image_path, bboxes, class_labels, ann_format,
                    save_func, ann_ext, aug_type, transforms_list,
                    output_images_dir, output_labels_dir, suffix,
                )
                if saved_labels is not None:
                    success_count += 1
                    progress.update(1)

        if success_count < total_count:
            print(
                f'\n[警告] 指定產生 {total_count} 組，但只有 {success_count} 組成功；'
                '部分影像可能缺少有效標註，或擴增後所有物件皆超出範圍。'
            )
        else:
            print(f'\n擴增完成！已產生指定的 {success_count} 組影像/標籤')
        print(f'  影像輸出：{output_images_dir}')
        print(f'  標籤輸出：{output_labels_dir}')
        return

    success_count = 0
    for image_path in tqdm(image_files):
        bboxes, class_labels, ann_format, save_func, ann_ext = \
            detect_and_load_annotations(image_path, labels_dir)

        if bboxes is None:
            tqdm.write(f'警告：{image_path.name} 沒有可用的標註，跳過。')
            continue

        for aug_type in augmentation_types:
            # ── rotation 管道 ──
            if aug_type == 'rotation':
                if rotation_transforms is None:
                    continue
                if is_fixed_step:
                    # 固定步長：每個步長角度產生一個版本
                    num_steps = 360 // rotate_step
                    for step_idx in range(num_steps):
                        angle = step_idx * rotate_step
                        step_transforms = [
                            A.Rotate(limit=(angle, angle), p=1.0,
                                     border_mode=cv2.BORDER_CONSTANT, fill=0)
                        ] + rotation_transforms
                        suffix = f'rot{angle}'
                        saved_labels = augment_single(
                            image_path, bboxes, class_labels, ann_format,
                            save_func, ann_ext, aug_type, step_transforms,
                            output_images_dir, output_labels_dir, suffix,
                        )
                        if saved_labels is not None:
                            success_count += 1
                else:
                    for i in range(num_augmentations):
                        suffix = f'rotation_{i + 1}' if num_augmentations > 1 else 'rotation'
                        saved_labels = augment_single(
                            image_path, bboxes, class_labels, ann_format,
                            save_func, ann_ext, aug_type, rotation_transforms,
                            output_images_dir, output_labels_dir, suffix,
                        )
                        if saved_labels is not None:
                            success_count += 1

            # ── 一般管道 ──
            elif aug_type in annotated_transforms:
                transforms_list = annotated_transforms[aug_type]
                for i in range(num_augmentations):
                    suffix = f'{aug_type}_{i + 1}' if num_augmentations > 1 else aug_type
                    saved_labels = augment_single(
                        image_path, bboxes, class_labels, ann_format,
                        save_func, ann_ext, aug_type, transforms_list,
                        output_images_dir, output_labels_dir, suffix,
                    )
                    if saved_labels is not None:
                        success_count += 1
            else:
                tqdm.write(f'警告：不支援的擴增類型 "{aug_type}"，跳過。')

    print(f'\n擴增完成！成功產生 {success_count} 組影像/標籤')
    print(f'  影像輸出：{output_images_dir}')
    print(f'  標籤輸出：{output_labels_dir}')


# ══════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None, *, image_only: bool = False):
    mode = 'image-only' if image_only else 'annotated'
    description = '純影像擴增' if image_only else 'VOC、CVAT 與 YOLO 水平框同步擴增'
    parser = argparse.ArgumentParser(prog=f'main.py augment {mode}', description=description)
    parser.add_argument('-i', '--images_dir', type=Path, required=True, help='輸入影像資料夾')
    parser.add_argument('-o', '--output_dir', type=Path, required=True, help='輸出資料夾')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子（預設 42）')
    parser.add_argument('--config', type=Path, help='新增或覆寫擴增配方的 JSON 檔')
    counts = parser.add_mutually_exclusive_group()
    counts.add_argument('-n', '--num_augmentations', type=int, default=1,
                        help='每張原圖、每種配方的版本數（預設 1）')
    counts.add_argument('--total_count', '--total_augmentations', type=int,
                        metavar='COUNT', help='整批擴增的輸出總數')
    if image_only:
        parser.add_argument('--image_only_rotate', type=int, nargs='?', const=5,
                            metavar='MAX_ANGLE', help='啟用隨機旋轉（預設 ±5°）')
        parser.add_argument('--image_only_profile', metavar='PROFILE', default='default',
                            help='擴增配方（預設 default，connector 僅改變拍攝條件）')
        counts.add_argument('--augment_per_folder', type=int, metavar='COUNT',
                            help='輸入的每個第一層子資料夾各產生 COUNT 張影像')
    else:
        parser.add_argument('-l', '--labels_dir', type=Path, help='標籤資料夾（預設同影像目錄）')
        parser.add_argument('-a', '--augmentations', nargs='+', metavar='PROFILE',
                            default=['mixed'], help='擴增配方，可指定多種（預設 mixed）')
        parser.add_argument('--rotate_limit', type=int,
                            help='rotation 配方的隨機旋轉最大角度')
        parser.add_argument('--rotate_step', type=int,
                            help='rotation 配方的固定旋轉步長（度）')
        counts.add_argument('--augment_per_class', type=int, metavar='COUNT',
                            help='每個標註類別擴增至至少 COUNT 個物件框')
    args = parser.parse_args(argv)
    try:
        profiles = load_profiles(args.config)
    except (OSError, ValueError) as error:
        parser.error(f'擴增設定檔錯誤：{error}')

    images_dir, output_dir = args.images_dir, args.output_dir
    if not images_dir.exists():
        print(f'錯誤：輸入影像資料夾 "{images_dir}" 不存在')
        return

    num_augmentations = args.num_augmentations
    for option in ['num_augmentations', 'total_count',
                   'augment_per_folder' if image_only else 'augment_per_class']:
        value = getattr(args, option)
        if value is not None and value <= 0:
            parser.error(f'--{option} 的數量必須大於 0')

    if image_only:
        if args.image_only_profile not in profiles['image_only']:
            options = ', '.join(profiles['image_only'])
            parser.error(f'不支援的純影像配方 {args.image_only_profile!r}（可用：{options}）')
        if args.image_only_rotate is not None:
            if args.image_only_rotate <= 0:
                parser.error('--image_only_rotate 的角度必須大於 0')
            if args.image_only_profile == 'connector':
                parser.error('connector profile 不允許旋轉，請移除 --image_only_rotate')
    else:
        labels_dir = args.labels_dir or images_dir
        for profile in args.augmentations:
            if profile not in profiles['annotated']:
                options = ', '.join(profiles['annotated'])
                parser.error(f'不支援的標註擴增配方 {profile!r}（可用：{options}）')
        if args.rotate_step is not None and not 1 <= args.rotate_step < 360:
            parser.error('--rotate_step 必須為 1～359 的整數')

    random.seed(args.seed)
    np.random.seed(args.seed)

    print('=' * 60)
    print('影像擴增腳本')
    print('=' * 60)
    print(f'影像來源：{images_dir}')
    if not image_only:
        print(f'標籤來源：{labels_dir}')
    print(f'輸出目錄：{output_dir}')
    print(f'模式    ：{"純影像擴增" if image_only else "標註同步擴增"}')
    if not image_only:
        print(f'擴增類型：{", ".join(args.augmentations)}')
    else:
        rotation_status = (
            f'開啟（±{args.image_only_rotate}°）'
            if args.image_only_rotate is not None else '關閉'
        )
        print(f'旋轉    ：{rotation_status}')
        print(f'純影像 profile：{args.image_only_profile}')
        if args.augment_per_folder is not None:
            print(f'每資料夾擴增：{args.augment_per_folder} 張（輸出不含原圖）')
        elif args.total_count is not None:
            print(f'整批擴增總數：{args.total_count} 張（輸出不含原圖）')
        else:
            print(f'擴增倍數：{num_augmentations}')
    if not image_only:
        if args.augment_per_class is not None:
            print(f'每類別擴增：{args.augment_per_class} 個 instance（輸出不含原圖）')
        elif args.total_count is not None:
            print(f'整批擴增總數：{args.total_count} 組')
        else:
            print(f'擴增倍數：{num_augmentations}')
    print(f'隨機種子：{args.seed}')
    print('=' * 60)

    if image_only:
        if args.augment_per_folder is not None:
            run_image_only_augment_per_folder(
                images_dir,
                output_dir,
                args.augment_per_folder,
                rotate_limit=args.image_only_rotate,
                image_only_profile=args.image_only_profile,
                profiles=profiles['image_only'],
            )
        elif args.total_count is not None:
            run_image_only_total_count(
                images_dir,
                output_dir,
                args.total_count,
                rotate_limit=args.image_only_rotate,
                image_only_profile=args.image_only_profile,
                profiles=profiles['image_only'],
            )
        else:
            run_image_only(
                images_dir,
                output_dir,
                num_augmentations,
                rotate_limit=args.image_only_rotate,
                image_only_profile=args.image_only_profile,
                profiles=profiles['image_only'],
            )
    else:
        if not labels_dir.exists():
            print(f'錯誤：標籤資料夾 "{labels_dir}" 不存在')
            return
        run_annotated_augmentation(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=output_dir,
            augmentation_types=args.augmentations,
            num_augmentations=num_augmentations,
            rotate_limit=args.rotate_limit,
            rotate_step=args.rotate_step,
            total_count=args.total_count,
            augment_per_class=args.augment_per_class,
            profiles=profiles['annotated'],
        )

    print('=' * 60)
    print('全部完成！')
