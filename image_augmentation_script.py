#!/usr/bin/env python3
"""
影像擴增腳本 (Image Augmentation Script)
功能：
  - 支援 PASCAL VOC (.xml)、YOLO (.txt)、CVAT (.xml) 格式的標註同步擴增
  - 支援純影像擴增（無標註，--image_only）
  - 多種擴增管道：horizontal_flip, brightness_contrast, hue_saturation,
                   mixed, smart_camera, hz_pde, rotation, image_only

使用範例：
  # VOC 格式（影像與 XML 同一資料夾）
  python image_augmentation_script.py -i ./dataset -o ./augmented -a mixed -n 3

  # YOLO/多格式（影像與標籤分開）
  python image_augmentation_script.py -i ./images -l ./labels -o ./augmented -a mixed -n 5

  # 旋轉擴增（隨機角度）
  python image_augmentation_script.py -i ./images -l ./labels -o ./augmented -a rotation --rotate_limit 40 -n 5

  # 旋轉擴增（固定步長，每隔 45° 產生一個版本）
  python image_augmentation_script.py -i ./images -l ./labels -o ./augmented -a rotation --rotate_step 45

  # 純影像擴增（無標註，預設不旋轉）
  python image_augmentation_script.py -i ./images -o ./augmented --image_only -n 3

  # 純影像擴增並加入保守的隨機旋轉（預設 ±5°，此處自訂為 ±10°）
  python image_augmentation_script.py -i ./images -o ./augmented --image_only --image_only_rotate 10 -n 3

  # 純影像擴增：將每個第一層子資料夾補到 200 張（包含原圖）
  python image_augmentation_script.py -i ./class_images -o ./balanced --image_only --target_per_folder 200
"""

import os
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


# ══════════════════════════════════════════════════════════════
# 標註格式 讀取 / 儲存
# ══════════════════════════════════════════════════════════════

def load_yolo_annotations(label_path: Path) -> Tuple[list, list, str]:
    bboxes, class_labels = [], []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    bboxes.append(bbox)
                    class_labels.append(class_id)
    return bboxes, class_labels, 'yolo'


def save_yolo_annotations(label_path: Path, bboxes: list, class_labels: list, **kwargs):
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
                                 filename: str, width: int, height: int, depth: int = 3, **kwargs):
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'augmented'
    ET.SubElement(annotation, 'filename').text = filename
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)
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
                           filename: str, width: int, height: int, **kwargs):
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
        bboxes, labels, fmt = load_yolo_annotations(txt_path)
        return bboxes, labels, fmt, save_yolo_annotations, '.txt'

    return None, None, None, None, None


# ══════════════════════════════════════════════════════════════
# 擴增管道定義
# ══════════════════════════════════════════════════════════════

# 各有標註管道的 transforms list（不含 bbox_params，執行時動態組合）
ANNOTATED_PIPELINE_TRANSFORMS: Dict[str, list] = {
    'horizontal_flip': [
        A.HorizontalFlip(p=1.0),
    ],
    'brightness_contrast': [
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    ],
    'hue_saturation': [
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
    ],
    'mixed': [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5),
    ],
    'smart_camera': [
        A.LongestMaxSize(max_size=960, p=1.0),
        A.PadIfNeeded(min_height=960, min_width=960,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
        A.Affine(scale=(0.95, 1.05),
                 translate_percent={'x': (-0.03, 0.03), 'y': (-0.03, 0.03)},
                 rotate=(-5, 5), shear=(-3, 3),
                 border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.6),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.RandomGamma(gamma_limit=(85, 115), p=1.0),
        ], p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.ImageCompression(quality_range=(70, 95), p=1.0),
        ], p=0.25),
        A.CoarseDropout(num_holes_range=(1, 3),
                        hole_height_range=(0.03, 0.08),
                        hole_width_range=(0.03, 0.08),
                        fill=0, p=0.15),
    ],
    'hz_pde': [
        A.OneOf([
            A.Affine(scale=(0.95, 1.05),
                     translate_percent={'x': (-0.03, 0.03), 'y': (-0.03, 0.03)},
                     rotate=(-5, 5), shear=(-3, 3),
                     interpolation=cv2.INTER_LINEAR,
                     border_mode=cv2.BORDER_CONSTANT, p=1.0),
        ], p=0.6),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.03, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.ImageCompression(quality_range=(70, 95), p=1.0),
            A.GaussNoise(std_range=(0.01, 0.03), p=1.0),
        ], p=0.35),
        A.CoarseDropout(num_holes_range=(1, 3),
                        hole_height_range=(0.03, 0.08),
                        hole_width_range=(0.03, 0.08),
                        fill=0, p=0.15),
    ],
}

# 旋轉管道的基礎 transforms（rotation 管道專用）
_ROTATION_BASE_TRANSFORMS = [
    A.HorizontalFlip(p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
]

def build_image_only_pipeline(rotate_limit: Optional[int] = None) -> A.Compose:
    """建立保守的純影像擴增管道。

    預設只做輕微的光照、色彩、模糊、雜訊與壓縮變化；旋轉只會在
    rotate_limit 有值時加入，避免未明確指定就改變影像幾何。
    """
    transforms = []
    if rotate_limit is not None:
        transforms.append(
            A.SafeRotate(
                limit=rotate_limit,
                p=1.0,
                border_mode=cv2.BORDER_REFLECT_101,
            )
        )

    transforms.extend([
        A.RandomBrightnessContrast(
            brightness_limit=0.08,
            contrast_limit=0.08,
            p=0.6,
        ),
        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=8,
            val_shift_limit=8,
            p=0.3,
        ),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            A.ImageCompression(quality_range=(90, 100), p=1.0),
        ], p=0.2),
        A.OneOf([
            A.GaussNoise(
                std_range=(0.01, 0.03),
                mean_range=(0.0, 0.0),
                per_channel=True,
                noise_scale_factor=1.0,
                p=1.0,
            ),
            A.ISONoise(
                color_shift=(0.01, 0.02),
                intensity=(0.05, 0.15),
                p=1.0,
            ),
        ], p=0.1),
    ])
    return A.Compose(transforms)


def build_pipeline(transforms_list: list, ann_format: str,
                   min_visibility: float = 0.2) -> A.Compose:
    """根據標註格式動態建立帶 bbox_params 的 A.Compose"""
    return A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(
            format=ann_format,
            label_fields=['class_labels'],
            min_visibility=min_visibility,
            min_area=16,
            clip=True,
        )
    )


def build_rotation_transforms(rotate_limit: Optional[int],
                               rotate_step: Optional[int]) -> Tuple[list, bool]:
    """
    建立旋轉管道的 transforms list。
    回傳 (transforms_list, is_fixed_step)。
    is_fixed_step=True 表示需要在主迴圈中使用固定角度模式。
    """
    if rotate_step is not None:
        print(f'啟用固定角度步長旋轉，步長: {rotate_step}°')
        return _ROTATION_BASE_TRANSFORMS, True
    elif rotate_limit is not None:
        print(f'啟用隨機角度旋轉，範圍: ±{rotate_limit}°')
        rotation = [
            A.Rotate(limit=rotate_limit, p=0.7,
                     border_mode=cv2.BORDER_CONSTANT, fill_value=0),
            A.Affine(scale=(0.9, 1.1), translate_percent=0.1, p=0.5,
                     border_mode=cv2.BORDER_CONSTANT, fill_value=0),
        ]
        return rotation + _ROTATION_BASE_TRANSFORMS, False
    else:
        print('啟用 90/180/270° 固定角度旋轉')
        return [A.RandomRotate90(p=0.75)] + _ROTATION_BASE_TRANSFORMS, False


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


def run_image_only(images_dir: Path, output_dir: Path, num_augmentations: int,
                   rotate_limit: Optional[int] = None):
    """純影像擴增（無標註），對應原 augment_image_only.py 的功能"""
    output_dir.mkdir(parents=True, exist_ok=True)
    img_files = list_image_files(images_dir)

    if not img_files:
        print(f'[提示] 在 {images_dir} 中找不到影像檔案。')
        return

    pipeline = build_image_only_pipeline(rotate_limit)
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


def run_image_only_target_per_folder(
    images_dir: Path,
    output_dir: Path,
    target_per_folder: int,
    rotate_limit: Optional[int] = None,
):
    """將每個第一層子資料夾補到固定影像總數（包含原圖）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_image_only_pipeline(rotate_limit)

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

    total_copied = 0
    total_augmented = 0
    print(
        f'找到 {len(subdirs)} 個子資料夾，開始補齊到每資料夾 '
        f'{target_per_folder} 張影像（包含原圖）...'
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

        originals_to_copy = img_files[:target_per_folder]
        for img_path in originals_to_copy:
            target_path = folder_output_dir / img_path.name
            if img_path.resolve() != target_path.resolve():
                shutil.copy2(img_path, target_path)
            total_copied += 1

        source_count = len(img_files)
        if source_count >= target_per_folder:
            if source_count > target_per_folder:
                tqdm.write(
                    f'[提示] {subdir.name} 原始影像有 {source_count} 張，'
                    f'超過目標 {target_per_folder} 張；輸出僅保留排序後前 '
                    f'{target_per_folder} 張。'
                )
            output_count = len(list_image_files(folder_output_dir))
            tqdm.write(f'{subdir.name}: 已達目標，輸出目前 {output_count} 張。')
            continue

        augment_needed = target_per_folder - source_count
        folder_augmented = 0
        for i in tqdm(
            range(augment_needed),
            desc=f'{subdir.name} 補齊',
            leave=False,
        ):
            src_img = img_files[i % source_count]
            out_name = f'aug_{i + 1:05d}_{src_img.name}'
            ok = save_image_only_augmentation(
                src_img,
                pipeline,
                folder_output_dir / out_name,
                f'補齊第 {i + 1} 張',
            )
            if ok:
                total_augmented += 1
                folder_augmented += 1

        output_count = len(list_image_files(folder_output_dir))
        tqdm.write(
            f'{subdir.name}: 原圖 {source_count} 張，新增擴增 '
            f'{folder_augmented} 張，輸出目前 {output_count} 張。'
        )

    print(f'子資料夾補齊完成！複製原圖 {total_copied} 張，新增擴增 {total_augmented} 張。')
    print(f'輸出目錄：{output_dir}')


# ══════════════════════════════════════════════════════════════
# 標註同步擴增模式
# ══════════════════════════════════════════════════════════════

def augment_single(image_path: Path, bboxes: list, class_labels: list,
                   ann_format: str, save_func: Callable, ann_ext: str,
                   pipeline_name: str, transforms_list: list,
                   output_images_dir: Path, output_labels_dir: Path,
                   suffix: str) -> bool:
    """
    對單張影像執行一次擴增並儲存結果。
    回傳是否成功。
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return False
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = build_pipeline(transforms_list, ann_format)
    try:
        augmented = transform(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
    except Exception as e:
        print(f'  錯誤：{image_path.name} ({pipeline_name}) 擴增失敗：{e}')
        return False

    aug_bboxes = augmented['bboxes']
    aug_labels = augmented['class_labels']
    if not aug_bboxes:
        return False  # 物件全部超出邊界，跳過

    aug_image = augmented['image']
    aug_h, aug_w = aug_image.shape[:2]
    new_stem = f'{image_path.stem}_{suffix}'
    new_img_name = f'{new_stem}{image_path.suffix}'

    cv2.imwrite(str(output_images_dir / new_img_name),
                cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
    save_func(
        output_labels_dir / f'{new_stem}{ann_ext}',
        aug_bboxes, aug_labels,
        filename=new_img_name, width=aug_w, height=aug_h, depth=3,
    )
    return True


def run_annotated_augmentation(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    augmentation_types: List[str],
    num_augmentations: int,
    rotate_limit: Optional[int],
    rotate_step: Optional[int],
):
    """標註同步擴增，支援 VOC / YOLO / CVAT 自動偵測"""
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
        rotation_transforms, is_fixed_step = build_rotation_transforms(rotate_limit, rotate_step)

    success_count = 0
    for image_path in tqdm(image_files):
        bboxes, class_labels, ann_format, save_func, ann_ext = \
            detect_and_load_annotations(image_path, labels_dir)

        if bboxes is None:
            tqdm.write(f'警告：找不到 {image_path.name} 的標籤檔，跳過。')
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
                                     border_mode=cv2.BORDER_CONSTANT, fill_value=0)
                        ] + rotation_transforms
                        suffix = f'rot{angle}'
                        augment_single(
                            image_path, bboxes, class_labels, ann_format,
                            save_func, ann_ext, aug_type, step_transforms,
                            output_images_dir, output_labels_dir, suffix,
                        )
                else:
                    for i in range(num_augmentations):
                        suffix = f'rotation_{i + 1}' if num_augmentations > 1 else 'rotation'
                        ok = augment_single(
                            image_path, bboxes, class_labels, ann_format,
                            save_func, ann_ext, aug_type, rotation_transforms,
                            output_images_dir, output_labels_dir, suffix,
                        )
                        if ok:
                            success_count += 1

            # ── 一般管道 ──
            elif aug_type in ANNOTATED_PIPELINE_TRANSFORMS:
                transforms_list = ANNOTATED_PIPELINE_TRANSFORMS[aug_type]
                for i in range(num_augmentations):
                    suffix = f'{aug_type}_{i + 1}' if num_augmentations > 1 else aug_type
                    ok = augment_single(
                        image_path, bboxes, class_labels, ann_format,
                        save_func, ann_ext, aug_type, transforms_list,
                        output_images_dir, output_labels_dir, suffix,
                    )
                    if ok:
                        success_count += 1
            else:
                tqdm.write(f'警告：不支援的擴增類型 "{aug_type}"，跳過。')

    print(f'\n擴增完成！成功產生 {success_count} 組影像/標籤')
    print(f'  影像輸出：{output_images_dir}')
    print(f'  標籤輸出：{output_labels_dir}')


# ══════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════

def main():
    ALL_ANNOTATED_TYPES = list(ANNOTATED_PIPELINE_TRANSFORMS.keys()) + ['rotation']

    parser = argparse.ArgumentParser(
        description='影像擴增腳本 — 支援 VOC/YOLO/CVAT 標註同步擴增及純影像擴增',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用的擴增類型（-a）：
  horizontal_flip     水平翻轉
  brightness_contrast 亮度對比度調整
  hue_saturation      色調飽和度調整
  mixed               混合擴增
  smart_camera        Smart Camera 擴增（含 resize/pad/光照/模糊）
  hz_pde              HZ PDE 擴增（保守幾何 + 光照 + 模糊）
  rotation            旋轉擴增（搭配 --rotate_limit 或 --rotate_step）
  image_only          純影像擴增管道（需加 --image_only 旗標）

使用範例：
  # VOC（images 與 XML 同資料夾）
  python image_augmentation_script.py -i ./dataset -o ./out -a mixed -n 3

  # YOLO / 多格式（images 與 labels 分開）
  python image_augmentation_script.py -i ./images -l ./labels -o ./out -a hz_pde -n 5

  # 旋轉（固定步長）
  python image_augmentation_script.py -i ./images -l ./labels -o ./out -a rotation --rotate_step 45

  # 純影像擴增
  python image_augmentation_script.py -i ./images -o ./out --image_only -n 3

  # 純影像擴增 + 隨機旋轉（不給角度時預設 ±5°）
  python image_augmentation_script.py -i ./images -o ./out --image_only --image_only_rotate 10 -n 3

  # 純影像擴增：將每個第一層子資料夾補到 200 張（包含原圖）
  python image_augmentation_script.py -i ./class_images -o ./balanced --image_only --target_per_folder 200
        """,
    )

    parser.add_argument('-i', '--images_dir', type=str, required=True,
                        help='輸入影像資料夾路徑')
    parser.add_argument('-l', '--labels_dir', type=str, default=None,
                        help='標籤資料夾路徑（預設與 images_dir 相同，適用 VOC 同資料夾格式）')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='輸出資料夾路徑（標註模式會建立 images/ 和 labels/ 子目錄）')
    parser.add_argument('-a', '--augmentations', nargs='+',
                        choices=ALL_ANNOTATED_TYPES,
                        default=['mixed'],
                        help='要執行的擴增類型（可指定多個）')
    parser.add_argument('-n', '--num_augmentations', type=int, default=1,
                        help='每種擴增類型要產生的版本數量（預設：1）')
    parser.add_argument('--rotate_limit', type=int, default=None,
                        help='[rotation] 隨機旋轉最大角度，例如 40 表示 ±40°')
    parser.add_argument('--rotate_step', type=int, default=None,
                        help='[rotation] 固定旋轉步長（度），例如 45 會產生 0°/45°/90°/... 版本')
    parser.add_argument('--image_only', action='store_true',
                        help='純影像擴增模式（不處理標籤，使用專用管道）')
    parser.add_argument('--image_only_rotate', type=int, nargs='?', const=5, default=None,
                        metavar='MAX_ANGLE',
                        help='[image_only] 啟用隨機旋轉；預設 ±5°，也可指定最大角度')
    parser.add_argument('--target_per_folder', type=int, default=None,
                        metavar='COUNT',
                        help='[image_only] 將輸入資料夾第一層每個子資料夾補到 COUNT 張影像（包含原圖）')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子（預設：42）')

    args = parser.parse_args()

    # 設定隨機種子
    random.seed(args.seed)
    np.random.seed(args.seed)

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir) if args.labels_dir else images_dir
    output_dir = Path(args.output_dir)

    if not images_dir.exists():
        print(f'錯誤：輸入影像資料夾 "{images_dir}" 不存在')
        return
    if args.image_only_rotate is not None and args.image_only_rotate <= 0:
        parser.error('--image_only_rotate 的角度必須大於 0')
    if args.image_only_rotate is not None and not args.image_only:
        parser.error('--image_only_rotate 只能與 --image_only 一起使用')
    if args.target_per_folder is not None and args.target_per_folder <= 0:
        parser.error('--target_per_folder 的數量必須大於 0')
    if args.target_per_folder is not None and not args.image_only:
        parser.error('--target_per_folder 只能與 --image_only 一起使用')

    print('=' * 60)
    print('影像擴增腳本')
    print('=' * 60)
    print(f'影像來源：{images_dir}')
    if not args.image_only:
        print(f'標籤來源：{labels_dir}')
    print(f'輸出目錄：{output_dir}')
    print(f'模式    ：{"純影像擴增" if args.image_only else "標註同步擴增"}')
    if not args.image_only:
        print(f'擴增類型：{", ".join(args.augmentations)}')
    else:
        rotation_status = (
            f'開啟（±{args.image_only_rotate}°）'
            if args.image_only_rotate is not None else '關閉'
        )
        print(f'旋轉    ：{rotation_status}')
        if args.target_per_folder is not None:
            print(f'每資料夾目標：{args.target_per_folder} 張（包含原圖）')
        else:
            print(f'擴增倍數：{args.num_augmentations}')
    if not args.image_only:
        print(f'擴增倍數：{args.num_augmentations}')
    print(f'隨機種子：{args.seed}')
    print('=' * 60)

    if args.image_only:
        if args.target_per_folder is not None:
            run_image_only_target_per_folder(
                images_dir,
                output_dir,
                args.target_per_folder,
                rotate_limit=args.image_only_rotate,
            )
        else:
            run_image_only(
                images_dir,
                output_dir,
                args.num_augmentations,
                rotate_limit=args.image_only_rotate,
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
            num_augmentations=args.num_augmentations,
            rotate_limit=args.rotate_limit,
            rotate_step=args.rotate_step,
        )

    print('=' * 60)
    print('全部完成！')


if __name__ == '__main__':
    main()
