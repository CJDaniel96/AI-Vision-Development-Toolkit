#!/usr/bin/env python3
"""同步擴增影像與 YOLO OBB 四角點標註（需要 albumentations >= 2.0.8）。

標註每行：class_id x1 y1 x2 y2 x3 y3 x4 y4，座標正規化至 [0, 1]，
四個頂點須沿框的邊依序排列。輸出只含擴增結果，分放 images/ 與 labels/。

使用範例：
  python obb_image_augmentation_script.py -i ./images -l ./labels -o ./aug -a mixed -n 5
  python obb_image_augmentation_script.py -i ./dataset -o ./aug -a rotation --rotate_step 45
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
AUGMENTATION_TYPES = (
    'horizontal_flip', 'vertical_flip', 'brightness_contrast',
    'hue_saturation', 'mixed', 'rotation',
)


def load_yolo_obb_annotations(label_path: Path) -> tuple[list, list[int]]:
    """嚴格讀取四角點標註；空檔代表背景影像，缺檔或格式錯誤則拋出例外。"""
    polygons, class_labels = [], []
    for line_number, line in enumerate(
        label_path.read_text(encoding='utf-8-sig').splitlines(), start=1
    ):
        parts = line.split()
        if not parts:
            continue
        prefix = f'{label_path}:{line_number}'
        if len(parts) != 9:
            raise ValueError(f'{prefix}：YOLO OBB 每行必須恰好有 9 個欄位')
        try:
            class_id = int(parts[0])
            points = np.array([float(value) for value in parts[1:]], dtype=np.float64)
        except ValueError as error:
            raise ValueError(f'{prefix}：類別須為整數，座標須為數值') from error
        if class_id < 0:
            raise ValueError(f'{prefix}：class_id 不可小於 0')
        if not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
            raise ValueError(f'{prefix}：座標必須為介於 0 與 1 的有限數值')
        points = points.reshape(4, 2)
        contour = points.astype(np.float32)
        if not cv2.isContourConvex(contour) or cv2.contourArea(contour) <= 0:
            raise ValueError(f'{prefix}：四角點須依序形成非退化的凸四邊形')
        polygons.append(points.tolist())
        class_labels.append(class_id)
    return polygons, class_labels


def save_yolo_obb_annotations(
    label_path: Path, polygons: list, class_labels: list[int]
) -> None:
    """寫出 YOLO OBB，不覆寫既有標註檔。"""
    lines = [
        f"{class_id} {' '.join(f'{value:.8f}' for value in np.asarray(points).ravel())}\n"
        for points, class_id in zip(polygons, class_labels, strict=True)
    ]
    with label_path.open('x', encoding='utf-8') as label_file:
        label_file.writelines(lines)


def build_pipeline(
    augmentation_type: str,
    rotate_limit: float = 30,
    angle: float | None = None,
    seed: int = 42,
) -> A.Compose:
    """只採用保持四角點幾何的變換，旋轉與等比縮放會擴張畫布。"""
    def affine(rotation, scale=(1.0, 1.0), probability=1.0):
        return A.Sequential([
            # fit_output 的範圍計算使用像素中心；先補 1px，保護座標為 0/1
            # 的半像素外緣，避免任意角度旋轉時貼邊框仍略微超出畫布。
            A.CropAndPad(
                px=1, keep_size=False, border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114), p=1.0,
            ),
            A.Affine(
                rotate=rotation,
                scale=scale,
                keep_ratio=True,
                fit_output=True,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
                p=1.0,
            ),
        ], p=probability)

    if augmentation_type == 'horizontal_flip':
        transforms = [A.HorizontalFlip(p=1.0)]
    elif augmentation_type == 'vertical_flip':
        transforms = [A.VerticalFlip(p=1.0)]
    elif augmentation_type == 'brightness_contrast':
        transforms = [A.NoOp(p=1.0), A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=1.0,
        )]
    elif augmentation_type == 'hue_saturation':
        transforms = [A.NoOp(p=1.0), A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=1.0,
        )]
    elif augmentation_type == 'mixed':
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            affine((-rotate_limit, rotate_limit), (0.9, 1.1), 0.7),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.6,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4,
            ),
            A.GaussianBlur(blur_limit=(3, 3), p=0.15),
        ]
    elif augmentation_type == 'rotation':
        # 整數直角用轉置/翻轉，避免插值與 Affine 浮點取整造成額外邊框。
        if angle is not None and angle % 90 == 0:
            quarter_turns = int(angle % 360 // 90)
            transforms = []
            if quarter_turns in (1, 3):
                transforms.append(A.Transpose(p=1.0))
            if quarter_turns in (2, 3):
                transforms.append(A.HorizontalFlip(p=1.0))
            if quarter_turns in (1, 2):
                transforms.append(A.VerticalFlip(p=1.0))
            if not transforms:
                transforms.append(A.NoOp(p=1.0))
        else:
            rotation = (-rotate_limit, rotate_limit) if angle is None else (angle, angle)
            transforms = [affine(rotation)]
    else:
        raise ValueError(f'不支援的擴增類型：{augmentation_type}')

    return A.Compose(
        transforms,
        # 不可單獨刪除不可見角點，否則四個一組的物件與類別會錯位。
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        seed=seed,
        strict=True,
    )


def augment_obb(
    image_rgb: np.ndarray,
    polygons: list,
    class_labels: list[int],
    pipeline: A.Compose,
) -> tuple[np.ndarray, list, list[int]]:
    """以 keypoints 同步轉換各 OBB，並依輸出畫布尺寸重新正規化。"""
    if len(polygons) != len(class_labels):
        raise ValueError('OBB 數量與類別數量不一致')
    height, width = image_rgb.shape[:2]
    points = np.asarray(polygons, dtype=np.float32).reshape(-1, 2)
    # YOLO 描述像素邊界 [0, width]；Albumentations keypoints 描述像素中心
    # [0, width-1]。以半像素轉換，讓翻轉後 x 正確等於 1-x，且邊界 0/1 合法。
    keypoints = points * [width, height] - 0.5
    augmented = pipeline(image=image_rgb, keypoints=keypoints)
    output_image = augmented['image']
    output_height, output_width = output_image.shape[:2]
    output_points = np.asarray(augmented['keypoints'], dtype=np.float64).reshape(-1, 4, 2)
    if len(output_points) != len(class_labels):
        raise ValueError('擴增後角點數量改變，無法配對 OBB 類別')
    output_points = (output_points + 0.5) / [output_width, output_height]
    # 僅修正浮點誤差；實際超界時整組失敗，不把框裁成錯誤的四邊形。
    if not np.isfinite(output_points).all() or np.any(
        (output_points < -1e-6) | (output_points > 1 + 1e-6)
    ):
        raise ValueError('擴增後 OBB 超出畫布，請使用可保留完整影像的變換')
    return output_image, np.clip(output_points, 0, 1).tolist(), list(class_labels)


@dataclass
class AugmentationSummary:
    generated: int = 0
    skipped_sources: int = 0
    failed_outputs: int = 0


def run_augmentation(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    augmentation_types: list[str],
    num_augmentations: int = 1,
    rotate_limit: float = 30,
    rotate_step: int | None = None,
    seed: int = 42,
    recursive: bool = False,
) -> AugmentationSummary:
    """掃描配對標註並輸出擴增；遞迴模式保留相對路徑，缺失來源會列警告。"""
    images_dir, labels_dir, output_dir = (
        Path(directory).resolve() for directory in (images_dir, labels_dir, output_dir)
    )
    for source_dir in (images_dir, labels_dir):
        if not source_dir.is_dir():
            raise ValueError(f'輸入資料夾不存在或不是資料夾：{source_dir}')
        if source_dir.is_relative_to(output_dir):
            raise ValueError('輸出目錄不可等於輸入目錄或包含輸入目錄')
    image_files = sorted(
        path for path in (images_dir.rglob('*') if recursive else images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        and not path.resolve().is_relative_to(output_dir)
    )
    if not image_files:
        raise ValueError(f'在 {images_dir} 找不到影像檔案')
    # 不同副檔名的同名影像會共用 TXT，須在寫檔前拒絕，避免互相覆蓋。
    seen = set()
    for image_path in image_files:
        key = image_path.relative_to(images_dir).with_suffix('').as_posix().casefold()
        if key in seen:
            raise ValueError(f'同一相對路徑有重複影像檔名（不含副檔名）：{image_path}')
        seen.add(key)

    # 每個 pipeline 只建一次，讓 Albumentations 的獨立亂數狀態隨呼叫前進。
    pipelines = []
    for augmentation_type in dict.fromkeys(augmentation_types):
        if augmentation_type == 'rotation' and rotate_step is not None:
            for angle in range(0, 360, rotate_step):
                pipelines.append((
                    f'rot{angle}',
                    build_pipeline(augmentation_type, angle=angle, seed=seed),
                    1,
                ))
        else:
            pipelines.append((
                augmentation_type,
                build_pipeline(augmentation_type, rotate_limit=rotate_limit, seed=seed),
                num_augmentations,
            ))

    summary = AugmentationSummary()
    print(f'找到 {len(image_files)} 張影像，開始 OBB 標註同步擴增...')
    for image_path in tqdm(image_files, desc='OBB 擴增'):
        relative_path = image_path.relative_to(images_dir)
        label_path = labels_dir / relative_path.with_suffix('.txt')
        try:
            polygons, class_labels = load_yolo_obb_annotations(label_path)
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise ValueError(f'無法讀取影像：{image_path}')
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except (OSError, ValueError, cv2.error) as error:
            summary.skipped_sources += 1
            tqdm.write(f'[警告] 跳過 {relative_path}：{error}')
            continue

        output_images = output_dir / 'images' / relative_path.parent
        output_labels = output_dir / 'labels' / relative_path.parent
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        for pipeline_name, pipeline, count in pipelines:
            for index in range(1, count + 1):
                stem = f'{image_path.stem}_{pipeline_name}_{index}'
                image_output = output_images / f'{stem}{image_path.suffix}'
                label_output = output_labels / f'{stem}.txt'
                image_written = False
                try:
                    if image_output.exists() or label_output.exists():
                        raise FileExistsError(f'輸出已存在，請改用其他輸出目錄：{stem}')
                    aug_image, aug_polygons, aug_labels = augment_obb(
                        image_rgb, polygons, class_labels, pipeline,
                    )
                    if not cv2.imwrite(
                        str(image_output), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    ):
                        raise OSError(f'影像寫入失敗：{image_output}')
                    image_written = True
                    save_yolo_obb_annotations(label_output, aug_polygons, aug_labels)
                    summary.generated += 1
                except (OSError, ValueError, cv2.error) as error:
                    if image_written:
                        image_output.unlink(missing_ok=True)
                    summary.failed_outputs += 1
                    tqdm.write(f'[錯誤] {relative_path} ({pipeline_name})：{error}')

    print(
        f'擴增完成：產生 {summary.generated} 組影像/標註，'
        f'跳過 {summary.skipped_sources} 張來源，失敗 {summary.failed_outputs} 組。'
    )
    print(f'影像輸出：{output_dir / "images"}')
    print(f'標籤輸出：{output_dir / "labels"}')
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='YOLO OBB 影像與四角點標註同步擴增（Albumentations 2.x）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''標註每行：class_id x1 y1 x2 y2 x3 y3 x4 y4（座標介於 0～1）。
旋轉會擴張畫布；輸出至 images/ 與 labels/，只含擴增結果。
--rotate_step 45 會為每張影像產生 0°、45°、…、315° 共 8 個版本。
空標註檔視為背景影像；缺失或無效的標註會列出警告並跳過來源。''',
    )
    parser.add_argument('-i', '--images_dir', type=Path, required=True, help='輸入影像目錄')
    parser.add_argument('-l', '--labels_dir', type=Path, help='TXT 標註目錄（預設同影像目錄）')
    parser.add_argument('-o', '--output_dir', type=Path, required=True, help='輸出目錄')
    parser.add_argument('-a', '--augmentations', nargs='+', choices=AUGMENTATION_TYPES,
                        default=['mixed'], help='擴增類型，可指定多種（預設 mixed）')
    parser.add_argument('-n', '--num_augmentations', type=int,
                        help='每張影像、每種擴增的版本數（預設 1）')
    rotation = parser.add_mutually_exclusive_group()
    rotation.add_argument('--rotate_limit', type=float,
                          help='mixed/rotation 隨機旋轉範圍 ±ANGLE（預設 30°，0 可關閉旋轉）')
    rotation.add_argument('--rotate_step', type=int,
                          help='rotation 固定步長；1～359 的整數，從 0° 產生至小於 360°')
    parser.add_argument('--recursive', action='store_true', help='遞迴掃描並保留相對子目錄')
    parser.add_argument('--seed', type=int, default=42, help='Albumentations 隨機種子（預設 42）')
    args = parser.parse_args(argv)

    if args.num_augmentations is not None and args.num_augmentations <= 0:
        parser.error('-n 必須大於 0')
    if args.rotate_limit is not None and not 0 <= args.rotate_limit <= 180:
        parser.error('--rotate_limit 必須介於 0 與 180 度')
    if args.rotate_limit is not None and not {'mixed', 'rotation'} & set(args.augmentations):
        parser.error('--rotate_limit 需要選擇 mixed 或 rotation 擴增')
    if args.rotate_step is not None:
        if not 1 <= args.rotate_step < 360:
            parser.error('--rotate_step 必須為 1～359 的整數')
        if 'rotation' not in args.augmentations:
            parser.error('--rotate_step 需要選擇 -a rotation')
        if args.num_augmentations is not None:
            parser.error('--rotate_step 不可與 -n 同時使用；每個角度固定產生一個版本')
    try:
        summary = run_augmentation(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir or args.images_dir,
            output_dir=args.output_dir,
            augmentation_types=args.augmentations,
            num_augmentations=args.num_augmentations or 1,
            rotate_limit=30 if args.rotate_limit is None else args.rotate_limit,
            rotate_step=args.rotate_step,
            seed=args.seed,
            recursive=args.recursive,
        )
    except (ValueError, OSError) as error:
        parser.exit(1, f'錯誤：{error}\n')
    return 0 if summary.generated and not summary.failed_outputs else 1


if __name__ == '__main__':
    raise SystemExit(main())
