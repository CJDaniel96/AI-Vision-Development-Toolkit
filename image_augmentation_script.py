#!/usr/bin/env python3
"""
影像擴增腳本 (Image Augmentation Script)
功能：
  - 支援 PASCAL VOC (.xml)、YOLO (.txt)、CVAT (.xml) 格式的標註同步擴增
  - 支援純影像擴增（無標註，--image_only）
  - 多種擴增管道：horizontal_flip, brightness_contrast, hue_saturation,
                   mixed, smart_camera, hz_pde, connector, rotation, image_only

使用範例：
  # VOC 格式（影像與 XML 同一資料夾）
  python image_augmentation_script.py -i ./dataset -o ./augmented -a mixed -n 3

  # YOLO/多格式（影像與標籤分開）
  python image_augmentation_script.py -i ./images -l ./labels -o ./augmented -a mixed -n 5

  # Connector 標註同步擴增（保留 pin row 幾何，不旋轉、不翻轉、不透視）
  python image_augmentation_script.py -i ./images -l ./labels -o ./augmented -a connector -n 5

  # 旋轉擴增（隨機角度）
  python image_augmentation_script.py -i ./images -l ./labels -o ./augmented -a rotation --rotate_limit 40 -n 5

  # 旋轉擴增（固定步長，每隔 45° 產生一個版本）
  python image_augmentation_script.py -i ./images -l ./labels -o ./augmented -a rotation --rotate_step 45

  # 純影像擴增（無標註，預設不旋轉）
  python image_augmentation_script.py -i ./images -o ./augmented --image_only -n 3

  # 純影像擴增並加入保守的隨機旋轉（預設 ±5°，此處自訂為 ±10°）
  python image_augmentation_script.py -i ./images -o ./augmented --image_only --image_only_rotate 10 -n 3

  # 整批資料總共產生 100 張擴增影像（不是每張原圖各 100 張）
  python image_augmentation_script.py -i ./images -o ./augmented --image_only --total_count 100

  # 類別平衡：讓標註裡每個 label 在輸出中都累積到 100 個 bbox instance（輸出不含原圖）
  python image_augmentation_script.py -i ./dataset -o ./balanced -a mixed --augment_per_class 100

  # 純影像擴增：每個第一層子資料夾產生 200 張擴增影像（輸出不含原圖）
  python image_augmentation_script.py -i ./class_images -o ./balanced --image_only --augment_per_folder 200

  # Connector 純影像擴增：保留 pin row 幾何，不旋轉、不翻轉、不透視
  python image_augmentation_script.py -i ./connector_images -o ./connector_aug --image_only --image_only_profile connector --augment_per_folder 200
"""

import os
import argparse
import random
import xml.etree.ElementTree as ET
from collections import Counter
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

def build_connector_transforms() -> List[Any]:
    """Connector 專用：只模擬拍攝條件變化，不改變 pin row 幾何。"""
    return [
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.08,
                contrast_limit=0.10,
                p=1.0,
            ),
            A.RandomGamma(
                gamma_limit=(90, 110),
                p=1.0,
            ),
            A.CLAHE(
                clip_limit=(1.0, 2.0),
                tile_grid_size=(8, 8),
                p=1.0,
            ),
        ], p=0.65),
        A.HueSaturationValue(
            hue_shift_limit=3,
            sat_shift_limit=6,
            val_shift_limit=6,
            p=0.25,
        ),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            A.ImageCompression(quality_range=(92, 100), p=1.0),
        ], p=0.20),
        A.OneOf([
            A.GaussNoise(
                std_range=(0.005, 0.02),
                mean_range=(0.0, 0.0),
                per_channel=True,
                noise_scale_factor=1.0,
                p=1.0,
            ),
            A.ISONoise(
                color_shift=(0.005, 0.015),
                intensity=(0.03, 0.10),
                p=1.0,
            ),
        ], p=0.12),
    ]


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
    'hz_pde_traditional': [
        # 先增加 5% 邊界，避免底座本來就太靠近影像邊緣
        # 不要填黑色，否則可能和黑色墊片特徵混淆
        A.CropAndPad(
            percent=0.05,
            keep_size=False,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(114, 114, 114),
            p=1.0,
        ),
 
        # 輕微旋轉與縮放；fit_output=True 會擴張畫布保留完整影像
        A.Affine(
            scale=(0.95, 1.05),
            rotate=(-2, 2),
            shear=(-1, 1),
            fit_output=True,
            keep_ratio=True,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(114, 114, 114),
            interpolation=cv2.INTER_LINEAR,
            p=0.4,
        ),
 
        # 若設備左右方向在實際環境中都可能出現才開啟
        A.HorizontalFlip(p=0.0),
 
        # 光照變化：黑色墊片仍需保持可辨識
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4,
        ),
 
        A.HueSaturationValue(
            hue_shift_limit=3,
            sat_shift_limit=10,
            val_shift_limit=10,
            p=0.25,
        ),
 
        # 少量相機畫質變化，不要模糊過強
        A.OneOf(
            [
                A.GaussianBlur(
                    blur_limit=(3, 3),
                    p=1.0,
                ),
                A.GaussNoise(
                    std_range=(0.01, 0.03),
                    p=1.0,
                ),
                A.ImageCompression(
                    quality_range=(85, 100),
                    p=1.0,
                ),
            ],
            p=0.15,
        ),
    ],
    'connector': build_connector_transforms(),
}

# 旋轉管道的基礎 transforms（rotation 管道專用）
_ROTATION_BASE_TRANSFORMS = [
    A.HorizontalFlip(p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
]

def build_image_only_pipeline(
    rotate_limit: Optional[int] = None,
    profile: str = 'default',
) -> A.Compose:
    """建立保守的純影像擴增管道。

    預設只做輕微的光照、色彩、模糊、雜訊與壓縮變化；旋轉只會在
    rotate_limit 有值時加入，避免未明確指定就改變影像幾何。
    """
    if profile == 'connector':
        if rotate_limit is not None:
            raise ValueError('connector profile 不允許旋轉，請移除 --image_only_rotate')
        return A.Compose(build_connector_transforms())
    if profile != 'default':
        raise ValueError(f'不支援的 image_only profile: {profile}')

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
                     border_mode=cv2.BORDER_CONSTANT, fill=0),
            A.Affine(scale=(0.9, 1.1), translate_percent=0.1, p=0.5,
                     border_mode=cv2.BORDER_CONSTANT, fill=0),
        ]
        return rotation + _ROTATION_BASE_TRANSFORMS, False
    else:
        print('啟用 90/180/270° 固定角度旋轉')
        return [A.RandomRotate90(p=0.75)] + _ROTATION_BASE_TRANSFORMS, False


def make_transforms_resolver(
    rotation_transforms: Optional[list],
    is_fixed_step: bool,
    rotate_step: Optional[int],
) -> Callable[[str], Optional[list]]:
    """
    建立「擴增類型 → transforms list」的解析函式。
    rotation 在固定步長模式下每被取用一次就前進一個角度；
    無法解析的類型回傳 None，由呼叫端決定跳過。
    """
    state = {'rotation_index': 0}

    def resolve(aug_type: str) -> Optional[list]:
        if aug_type != 'rotation':
            return ANNOTATED_PIPELINE_TRANSFORMS.get(aug_type)
        if rotation_transforms is None:
            return None
        if is_fixed_step:
            angle = (state['rotation_index'] * rotate_step) % 360
            resolved = [
                A.Rotate(limit=(angle, angle), p=1.0,
                         border_mode=cv2.BORDER_CONSTANT, fill=0)
            ] + rotation_transforms
        else:
            resolved = rotation_transforms
        state['rotation_index'] += 1
        return resolved

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
):
    """純影像擴增（無標註），對應原 augment_image_only.py 的功能"""
    output_dir.mkdir(parents=True, exist_ok=True)
    img_files = list_image_files(images_dir)

    if not img_files:
        print(f'[提示] 在 {images_dir} 中找不到影像檔案。')
        return

    pipeline = build_image_only_pipeline(
        rotate_limit=rotate_limit,
        profile=image_only_profile,
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
):
    """每個第一層子資料夾產生固定數量的擴增影像，輸出不複製原圖。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_image_only_pipeline(
        rotate_limit=rotate_limit,
        profile=image_only_profile,
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
    save_func(
        output_labels_dir / f'{new_stem}{ann_ext}',
        aug_bboxes, aug_labels,
        filename=new_img_name, width=aug_w, height=aug_h, depth=3,
    )
    return list(aug_labels)


def load_annotated_sources(image_files: List[Path], labels_dir: Path) -> list:
    """載入所有可用的 (影像, 標註) 組合，缺標籤檔的影像會被跳過。"""
    sources = []
    for image_path in image_files:
        annotation = detect_and_load_annotations(image_path, labels_dir)
        if annotation[0] is None:
            tqdm.write(f'警告：找不到 {image_path.name} 的標籤檔，跳過。')
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
    resolve_transforms = make_transforms_resolver(
        rotation_transforms, is_fixed_step, rotate_step,
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
            elif aug_type in ANNOTATED_PIPELINE_TRANSFORMS:
                transforms_list = ANNOTATED_PIPELINE_TRANSFORMS[aug_type]
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
  connector           Connector 擴增（保留 pin row 幾何，無旋轉/翻轉/透視）
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

  # 指定整批總擴增數量（總共 100 張，不是每張原圖各 100 張）
  python image_augmentation_script.py -i ./images -o ./out --image_only --total_count 100

  # 類別平衡：每個 label 在輸出中累積到 100 個 bbox instance（輸出不含原圖）
  python image_augmentation_script.py -i ./dataset -o ./balanced -a mixed --augment_per_class 100

  # 純影像擴增：每個第一層子資料夾產生 200 張擴增影像（輸出不含原圖）
  python image_augmentation_script.py -i ./class_images -o ./balanced --image_only --augment_per_folder 200

  # Connector 純影像擴增（無旋轉、無翻轉、無透視）
  python image_augmentation_script.py -i ./connector_images -o ./connector_aug --image_only --image_only_profile connector --augment_per_folder 200
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
    count_group = parser.add_mutually_exclusive_group()
    count_group.add_argument('-n', '--num_augmentations', type=int, default=None,
                             help='每張原圖、每種擴增類型要產生的版本數量（預設：1）')
    count_group.add_argument('--total_count', '--total_augmentations',
                             dest='total_count', type=int, default=None,
                             metavar='COUNT',
                             help='整批資料要產生的擴增結果總數；會輪流取用來源影像與擴增類型')
    parser.add_argument('--rotate_limit', type=int, default=None,
                        help='[rotation] 隨機旋轉最大角度，例如 40 表示 ±40°')
    parser.add_argument('--rotate_step', type=int, default=None,
                        help='[rotation] 固定旋轉步長（度），例如 45 會產生 0°/45°/90°/... 版本')
    parser.add_argument('--image_only', action='store_true',
                        help='純影像擴增模式（不處理標籤，使用專用管道）')
    parser.add_argument('--image_only_rotate', type=int, nargs='?', const=5, default=None,
                        metavar='MAX_ANGLE',
                        help='[image_only] 啟用隨機旋轉；預設 ±5°，也可指定最大角度')
    parser.add_argument('--image_only_profile', choices=['default', 'connector'],
                        default='default',
                        help='[image_only] 純影像擴增 profile；connector 只做拍攝條件變化，不做旋轉/翻轉/透視')
    parser.add_argument('--augment_per_folder', '--target_per_folder',
                        dest='augment_per_folder', type=int, default=None,
                        metavar='COUNT',
                        help='[image_only] 對輸入資料夾第一層每個子資料夾產生 COUNT 張擴增影像（輸出不含原圖；--target_per_folder 為舊別名）')
    parser.add_argument('--augment_per_class',
                        type=int, default=None, metavar='COUNT',
                        help='[標註模式] 依標註內的類別平衡擴增，讓每個類別在輸出中累積到 COUNT 個 bbox instance（輸出不含原圖）')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子（預設：42）')

    args = parser.parse_args()
    num_augmentations = (
        args.num_augmentations if args.num_augmentations is not None else 1
    )

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
    if args.image_only_profile != 'default' and not args.image_only:
        parser.error('--image_only_profile 只能與 --image_only 一起使用')
    if args.image_only_profile == 'connector' and args.image_only_rotate is not None:
        parser.error('connector profile 不允許旋轉，請移除 --image_only_rotate')
    if num_augmentations <= 0:
        parser.error('--num_augmentations 的數量必須大於 0')
    if args.total_count is not None and args.total_count <= 0:
        parser.error('--total_count 的數量必須大於 0')
    if args.augment_per_folder is not None and args.augment_per_folder <= 0:
        parser.error('--augment_per_folder 的數量必須大於 0')
    if args.augment_per_folder is not None and not args.image_only:
        parser.error('--augment_per_folder 只能與 --image_only 一起使用')
    if args.augment_per_folder is not None and args.total_count is not None:
        parser.error('--augment_per_folder 不能與 --total_count 同時使用')
    if args.augment_per_folder is not None and args.num_augmentations is not None:
        parser.error('--augment_per_folder 不能與 --num_augmentations 同時使用')
    if args.augment_per_class is not None and args.augment_per_class <= 0:
        parser.error('--augment_per_class 的數量必須大於 0')
    if args.augment_per_class is not None and args.image_only:
        parser.error('--augment_per_class 需要標註資料，不能與 --image_only 一起使用')
    if args.augment_per_class is not None and args.total_count is not None:
        parser.error('--augment_per_class 不能與 --total_count 同時使用')
    if args.augment_per_class is not None and args.num_augmentations is not None:
        parser.error('--augment_per_class 不能與 --num_augmentations 同時使用')

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
        print(f'純影像 profile：{args.image_only_profile}')
        if args.augment_per_folder is not None:
            print(f'每資料夾擴增：{args.augment_per_folder} 張（輸出不含原圖）')
        elif args.total_count is not None:
            print(f'整批擴增總數：{args.total_count} 張（輸出不含原圖）')
        else:
            print(f'擴增倍數：{num_augmentations}')
    if not args.image_only:
        if args.augment_per_class is not None:
            print(f'每類別擴增：{args.augment_per_class} 個 instance（輸出不含原圖）')
        elif args.total_count is not None:
            print(f'整批擴增總數：{args.total_count} 組')
        else:
            print(f'擴增倍數：{num_augmentations}')
    print(f'隨機種子：{args.seed}')
    print('=' * 60)

    if args.image_only:
        if args.augment_per_folder is not None:
            run_image_only_augment_per_folder(
                images_dir,
                output_dir,
                args.augment_per_folder,
                rotate_limit=args.image_only_rotate,
                image_only_profile=args.image_only_profile,
            )
        elif args.total_count is not None:
            run_image_only_total_count(
                images_dir,
                output_dir,
                args.total_count,
                rotate_limit=args.image_only_rotate,
                image_only_profile=args.image_only_profile,
            )
        else:
            run_image_only(
                images_dir,
                output_dir,
                num_augmentations,
                rotate_limit=args.image_only_rotate,
                image_only_profile=args.image_only_profile,
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
        )

    print('=' * 60)
    print('全部完成！')


if __name__ == '__main__':
    main()
