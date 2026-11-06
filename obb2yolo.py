#!/usr/bin/env python3
"""將 YOLO OBB 四角點資料整理成訓練資料集，並以補邊修正 CVAT 越界標註。

輸入 TXT 每行：class_id x1 y1 x2 y2 x3 y3 x4 y4，座標以原圖寬高正規化。
支援影像/TXT 同目錄，或 images/、labels/ 分離；遞迴掃描並保留相對路徑。
輸出 images/{train,val,test}、labels/{train,val,test} 與 data.yaml。

使用範例：
  python obb2yolo.py --input ./obb_data --output ./yolo_obb --classes ./classes.txt
  python obb2yolo.py -i ./images -l ./labels -o ./yolo_obb --split 0.8 0.2
"""

import argparse
import math
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from yolo_obb_utils import (
    load_yolo_obb_annotations,
    pad_image_to_fit_obb,
    save_yolo_obb_annotations,
)


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
SPLIT_NAMES = ('train', 'val', 'test')


@dataclass
class OBBSample:
    image_path: Path
    relative_path: Path
    polygons: list
    class_labels: list[int]


@dataclass
class DatasetSummary:
    split_counts: dict[str, int]
    padded_images: int
    skipped_sources: int
    names: list[str]


def load_class_names(path: Path) -> list[str]:
    """TXT 依行號對應 ID；YAML 讀取 names，保留原始 ID 順序。"""
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() in {'.yaml', '.yml'}:
        data = yaml.safe_load(text)
        names = data.get('names') if isinstance(data, dict) else None
        if isinstance(names, dict):
            try:
                if any(isinstance(key, (bool, float)) for key in names):
                    raise ValueError
                indexed = {int(key): value for key, value in names.items()}
                if len(indexed) != len(names) or set(indexed) != set(range(len(names))):
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError(f'{path}：names 的類別 ID 必須從 0 開始且連續') from error
            names = [indexed[index] for index in range(len(indexed))]
    else:
        names = [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(names, list) or not names or any(
        not isinstance(name, str) or not name.strip() for name in names
    ):
        raise ValueError(f'{path}：須提供非空的類別名稱列表（YAML 使用 names 欄位）')
    names = [name.strip() for name in names]
    if len(set(names)) != len(names):
        raise ValueError(f'{path}：類別名稱不可重複')
    return names


def normalize_split(ratios: list[float] | tuple[float, ...]) -> list[float]:
    """接受 train/val 或 train/val/test 比例，並正規化總和。"""
    if len(ratios) not in (2, 3) or any(not math.isfinite(value) or value < 0 for value in ratios):
        raise ValueError('--split 須為 2 或 3 個非負有限數值：train val [test]')
    if ratios[0] <= 0 or ratios[1] <= 0:
        raise ValueError('--split 的 train 與 val 比例必須大於 0')
    total = sum(ratios)
    if not math.isfinite(total):
        raise ValueError('--split 比例總和必須為有限數值')
    values = list(ratios) + ([0.0] if len(ratios) == 2 else [])
    return [value / total for value in values]


def split_samples(
    samples: list[OBBSample], ratios: list[float] | tuple[float, ...], seed: int = 42,
) -> dict[str, list[OBBSample]]:
    """依類別組合近似分層，同時維持整體切分數量，避免小組全落入 test。"""
    weights = normalize_split(ratios)
    if len(samples) < 2:
        raise ValueError('至少需要 2 張有效影像，才能建立非空的 train 與 val')
    quotas = [len(samples) * weight for weight in weights]
    targets = [math.floor(quota) for quota in quotas]
    remainder_order = sorted(range(3), key=lambda index: quotas[index] - targets[index], reverse=True)
    for index in remainder_order[:len(samples) - sum(targets)]:
        targets[index] += 1
    # 小資料集優先保留 train、val；有足夠影像時才保留 test。
    for index, weight in enumerate(weights):
        if weight and not targets[index]:
            donors = [
                other for other in range(3)
                if targets[other] > 1 or (other > index and targets[other] > 0)
            ]
            if donors:
                donor = max(donors, key=lambda other: targets[other] - quotas[other])
                targets[donor] -= 1
                targets[index] += 1

    groups = {}
    for sample in sorted(samples, key=lambda item: item.relative_path.as_posix()):
        signature = tuple(sorted(set(sample.class_labels)))
        groups.setdefault(signature, []).append(sample)
    rng = random.Random(seed)
    remaining = targets.copy()
    result = {name: [] for name in SPLIT_NAMES}
    training_has_object = False
    for signature in sorted(groups, key=lambda key: (not bool(key), len(groups[key]), key)):
        group = groups[signature].copy()
        rng.shuffle(group)
        assigned = [0, 0, 0]
        for sample in group:
            candidates = [index for index in range(3) if remaining[index]]
            # 有標註資料時，先保留一張給 train，避免小資料集只用背景訓練。
            if signature and not training_has_object and remaining[0]:
                index = 0
            else:
                index = max(candidates, key=lambda item: (
                    len(group) * targets[item] / len(samples) - assigned[item], remaining[item],
                ))
            result[SPLIT_NAMES[index]].append(sample)
            assigned[index] += 1
            remaining[index] -= 1
            if index == 0 and signature:
                training_has_object = True
    return result


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    labels_dir: Path | None = None,
    classes_path: Path | None = None,
    split: list[float] | tuple[float, ...] = (0.7, 0.2, 0.1),
    seed: int = 42,
    missing_as_background: bool = False,
) -> DatasetSummary:
    """驗證、切分並整理 OBB；補邊時同步修正全部角點，來源檔案不變。"""
    normalize_split(split)
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not input_dir.is_dir():
        raise ValueError(f'輸入資料夾不存在：{input_dir}')
    images_dir = input_dir / 'images' if (input_dir / 'images').is_dir() else input_dir
    default_labels = (
        images_dir.parent / 'labels' if images_dir.name == 'images' else images_dir
    )
    labels_dir = Path(labels_dir).resolve() if labels_dir is not None else (
        default_labels if default_labels.is_dir() else images_dir
    )
    for source_dir in (images_dir, labels_dir):
        if not source_dir.is_dir():
            raise ValueError(f'來源資料夾不存在：{source_dir}')
        if source_dir.is_relative_to(output_dir):
            raise ValueError('輸出目錄不可等於或包含輸入影像／標註目錄')
    if output_dir.exists():
        raise ValueError(f'輸出目錄已存在，請使用新的目錄：{output_dir}')

    if classes_path is None:
        classes_path = next((
            root / filename
            for root in dict.fromkeys((input_dir, images_dir.parent))
            for filename in ('data.yaml', 'classes.txt')
            if (root / filename).is_file()
        ), None)
    names = load_class_names(Path(classes_path)) if classes_path is not None else None
    image_files = sorted(
        path for path in images_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.resolve().is_relative_to(output_dir)
    )
    if not image_files:
        raise ValueError(f'在 {images_dir} 找不到影像')
    seen = set()
    for image_path in image_files:
        key = image_path.relative_to(images_dir).with_suffix('').as_posix().casefold()
        if key in seen:
            raise ValueError(f'同一相對路徑有重複影像檔名（不含副檔名）：{image_path}')
        seen.add(key)

    samples, skipped = [], 0
    for image_path in tqdm(image_files, desc='驗證 OBB'):
        relative_path = image_path.relative_to(images_dir)
        label_path = labels_dir / relative_path.with_suffix('.txt')
        try:
            if missing_as_background and not label_path.exists():
                polygons, class_labels = [], []
            else:
                polygons, class_labels = load_yolo_obb_annotations(label_path)
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError('無法讀取影像')
        except (OSError, ValueError, cv2.error) as error:
            tqdm.write(f'[警告] 跳過 {relative_path}：{error}')
            skipped += 1
            continue
        if names is not None and any(class_id >= len(names) for class_id in class_labels):
            raise ValueError(f'{label_path}：class_id 超出類別名稱列表範圍 0～{len(names) - 1}')
        samples.append(OBBSample(image_path, relative_path, polygons, class_labels))

    if names is None:
        detected_ids = {class_id for sample in samples for class_id in sample.class_labels}
        if not detected_ids:
            raise ValueError('無法推斷類別，請用 --classes 指定 classes.txt 或 data.yaml')
        names = [f'class_{index}' for index in range(max(detected_ids) + 1)]
        print('未提供類別名稱，使用 class_0、class_1…，保留來源 class_id。')
    groups = split_samples(samples, split, seed)
    counts = {name: len(group) for name, group in groups.items()}
    padded = 0
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # 全部寫入成功才移到正式目錄，寫檔失敗時不留下可被誤用的半成品資料集。
    with tempfile.TemporaryDirectory(prefix=f'.{output_dir.name}-', dir=output_dir.parent) as temporary:
        staging = Path(temporary) / 'dataset'
        for split_name, group in groups.items():
            if not group:
                continue
            for sample in tqdm(group, desc=f'輸出 {split_name}'):
                image_output = staging / 'images' / split_name / sample.relative_path
                label_output = staging / 'labels' / split_name / sample.relative_path.with_suffix('.txt')
                image_output.parent.mkdir(parents=True, exist_ok=True)
                label_output.parent.mkdir(parents=True, exist_ok=True)
                image = cv2.imread(str(sample.image_path))
                if image is None:
                    raise ValueError(f'無法讀取影像：{sample.image_path}')
                corrected_image, polygons = pad_image_to_fit_obb(image, sample.polygons)
                if corrected_image.shape != image.shape:
                    if not cv2.imwrite(str(image_output), corrected_image):
                        raise OSError(f'補邊影像寫入失敗：{image_output}')
                    padded += 1
                else:
                    shutil.copy2(sample.image_path, image_output)
                save_yolo_obb_annotations(label_output, polygons, sample.class_labels)
        config = {
            'path': str(output_dir), 'train': 'images/train', 'val': 'images/val',
            'nc': len(names), 'names': names,
        }
        if counts['test']:
            config['test'] = 'images/test'
        with (staging / 'data.yaml').open('x', encoding='utf-8') as stream:
            yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
        staging.rename(output_dir)

    print(
        f'完成：train {counts["train"]}、val {counts["val"]}、test {counts["test"]} 張；'
        f'補邊 {padded} 張，跳過 {skipped} 張來源。'
    )
    print(f'訓練設定：{output_dir / "data.yaml"}')
    return DatasetSummary(counts, padded, skipped, names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='整理 YOLO OBB 訓練資料集，並以影像補邊修正越界座標',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''輸入為九欄 TXT：class_id x1 y1 x2 y2 x3 y3 x4 y4（以原圖寬高正規化）。
自動辨識 images/、labels/ 結構，或在影像同目錄配對 TXT；一律遞迴掃描。
未指定 --classes 時尋找 data.yaml 或 classes.txt，否則以 class_ID 命名。
保留原 class_id；越界時補灰邊，不裁切或重新擬合 OBB。
空 TXT 是背景；缺失/無效標註或損壞影像預設列出警告並跳過。
輸出目錄必須不存在；至少需要 2 張有效影像建立 train、val。''',
    )
    parser.add_argument('-i', '--input', type=Path, required=True, help='影像目錄或含 images/、labels/ 的資料集根目錄')
    parser.add_argument('-l', '--labels', type=Path, help='指定 TXT 標註目錄，須與影像有相同相對路徑')
    parser.add_argument('-o', '--output', type=Path, required=True, help='新的輸出資料夾')
    parser.add_argument('--split', type=float, nargs='+', default=[0.7, 0.2, 0.1], help='train val [test] 比例，預設 0.7 0.2 0.1')
    parser.add_argument('--classes', type=Path, help='類別名稱 TXT，或含 names 的 YAML；不重新編號')
    parser.add_argument('--seed', type=int, default=42, help='資料分割隨機種子，預設 42')
    parser.add_argument('--missing-as-background', action='store_true', help='將缺少 TXT 的影像明確視為背景')
    args = parser.parse_args(argv)
    try:
        normalize_split(args.split)
    except ValueError as error:
        parser.error(str(error))
    try:
        prepare_dataset(
            args.input, args.output, labels_dir=args.labels, classes_path=args.classes,
            split=args.split, seed=args.seed, missing_as_background=args.missing_as_background,
        )
    except (OSError, ValueError, cv2.error, yaml.YAMLError) as error:
        parser.exit(1, f'錯誤：{error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
