#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
voc_to_yolo_augment.py

Offline CLI tool:
1. Read images + PASCAL VOC XML annotations
2. Split dataset into train/val/test
3. Apply Albumentations augmentation by split (train/val/test configurable)
4. Export Ultralytics YOLO detection folder structure
5. Generate data.yaml

Example:
python voc_to_yolo_augment.py \
  --input-dir /data/voc_raw \
  --output-dir /data/yolo_dataset \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --train-aug-per-image 3 \
  --val-aug-per-image 1 \
  --test-aug-per-image 1 \
  --aug-only \
  --image-size 960 \
  --seed 42

Optional class override:
python voc_to_yolo_augment.py \
  --input-dir /data/voc_raw \
  --output-dir /data/yolo_dataset \
  --classes left_hand_manual right_hand_manual box_region

Expected input layout (flat or nested both OK):
input-dir/
  images...
  annotations...
or
input-dir/
  JPEGImages/
  Annotations/

Each image should have a same-stem XML annotation.
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class VocObject:
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    difficult: int = 0
    truncated: int = 0
    pose: str = "Unspecified"


@dataclass
class VocSample:
    image_path: Path
    xml_path: Path
    width: int
    height: int
    objects: List[VocObject]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PASCAL VOC XML dataset to Ultralytics YOLO format with offline augmentation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Root directory containing images and VOC XML files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output YOLO dataset directory.")
    parser.add_argument(
        "--classes",
        nargs="+",
        required=False,
        default=None,
        help="Optional ordered class list. If omitted, classes are auto-detected from XML and sorted alphabetically.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio. Default: 0.8")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio. Default: 0.1")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio. Default: 0.1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument(
        "--train-aug-per-image",
        type=int,
        default=3,
        help="How many augmented copies to generate per train image. Default: 3",
    )
    parser.add_argument(
        "--val-aug-per-image",
        type=int,
        default=0,
        help="How many augmented copies to generate per val image. Default: 0",
    )
    parser.add_argument(
        "--test-aug-per-image",
        type=int,
        default=0,
        help="How many augmented copies to generate per test image. Default: 0",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=960,
        help="Output square image size. Resize longest edge and pad to this size. Default: 960",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.3,
        help="Albumentations bbox min_visibility for train augmentation. Default: 0.3",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=16.0,
        help="Albumentations bbox min_area for train augmentation. Default: 16",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep transformed samples even if all boxes are dropped. Default: False",
    )
    parser.add_argument(
        "--copy-original-train",
        action="store_true",
        help="Copy original train images in addition to augmented train images. Recommended: True if dataset is small.",
    )
    parser.add_argument(
        "--copy-original-val",
        action="store_true",
        help="Copy original val images in addition to augmented val images.",
    )
    parser.add_argument(
        "--copy-original-test",
        action="store_true",
        help="Copy original test images in addition to augmented test images.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable all split augmentations and only convert format.",
    )
    parser.add_argument(
        "--aug-only",
        action="store_true",
        help="Output augmented samples only. Do not keep any original images in any split.",
    )
    parser.add_argument(
        "--include-difficult",
        action="store_true",
        help="Include VOC objects with difficult=1. Default: skip difficult objects.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise error on class mismatch or malformed annotation instead of skipping sample.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=None,
        help="Optional image extensions filter, e.g. --extensions .jpg .png",
    )
    parser.add_argument(
        "--yaml-name",
        type=str,
        default="data.yaml",
        help="Output YAML file name. Default: data.yaml",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="custom_voc_yolo",
        help="Dataset name written into YAML metadata.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratio sum must be 1.0, got {total}")


def validate_augmentation_args(args: argparse.Namespace) -> None:
    if args.aug_only:
        if args.copy_original_train or args.copy_original_val or args.copy_original_test:
            raise ValueError("--aug-only cannot be used together with any --copy-original-* option.")
        if args.no_augment:
            raise ValueError("--aug-only cannot be used together with --no-augment.")
        if args.train_aug_per_image <= 0 or args.val_aug_per_image <= 0 or args.test_aug_per_image <= 0:
            raise ValueError(
                "--aug-only requires --train-aug-per-image, --val-aug-per-image, and --test-aug-per-image to be > 0."
            )


def find_xml_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.rglob("*.xml"))


def find_image_for_stem(stem: str, search_root: Path, exts: Sequence[str] | None = None) -> Optional[Path]:
    allowed = {e.lower() for e in exts} if exts else IMG_EXTS
    matches = []
    for p in search_root.rglob(f"{stem}.*"):
        if p.suffix.lower() in allowed:
            matches.append(p)
    if not matches:
        return None
    # Prefer nearest/common image dirs if multiple found
    matches.sort(key=lambda x: (len(x.parts), str(x)))
    return matches[0]


def parse_voc_xml(xml_path: Path, include_difficult: bool = False) -> Tuple[int, int, List[VocObject], Optional[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_node = root.find("filename")
    filename = filename_node.text.strip() if filename_node is not None and filename_node.text else None

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in XML: {xml_path}")

    width_node = size.find("width")
    height_node = size.find("height")
    if width_node is None or height_node is None:
        raise ValueError(f"Missing width/height in XML: {xml_path}")

    width = int(float(width_node.text))
    height = int(float(height_node.text))

    objects: List[VocObject] = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        bnd = obj.find("bndbox")
        if name_node is None or bnd is None:
            continue

        class_name = name_node.text.strip()
        difficult = int(obj.findtext("difficult", default="0"))
        truncated = int(obj.findtext("truncated", default="0"))
        pose = obj.findtext("pose", default="Unspecified")

        if difficult and not include_difficult:
            continue

        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))

        if xmax <= xmin or ymax <= ymin:
            logging.warning("Invalid bbox in %s: (%s, %s, %s, %s)", xml_path, xmin, ymin, xmax, ymax)
            continue

        objects.append(
            VocObject(
                class_name=class_name,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                difficult=difficult,
                truncated=truncated,
                pose=pose,
            )
        )

    return width, height, objects, filename


def detect_classes_from_xml(input_dir: Path, include_difficult: bool = False, strict: bool = False) -> List[str]:
    xml_files = find_xml_files(input_dir)
    if not xml_files:
        raise FileNotFoundError(f"No XML files found under {input_dir}")

    class_names = set()
    failed = 0

    for xml_path in xml_files:
        try:
            _, _, objects, _ = parse_voc_xml(xml_path, include_difficult=include_difficult)
            for obj in objects:
                if obj.class_name:
                    class_names.add(obj.class_name)
        except Exception as e:
            if strict:
                raise
            logging.warning("Failed to scan classes from %s: %s", xml_path, e)
            failed += 1

    if not class_names:
        raise RuntimeError("No classes detected from XML files.")

    classes = sorted(class_names)
    logging.info("Auto-detected %d classes from XML%s.", len(classes), f" (scan skipped {failed} files)" if failed else "")
    logging.info("Detected classes from XML:")
    for idx, name in enumerate(classes):
        logging.info("  %d -> %s", idx, name)
    return classes


def collect_samples(
    input_dir: Path,
    class_to_id: Dict[str, int],
    include_difficult: bool = False,
    strict: bool = False,
    extensions: Sequence[str] | None = None,
) -> List[VocSample]:
    xml_files = find_xml_files(input_dir)
    if not xml_files:
        raise FileNotFoundError(f"No XML files found under {input_dir}")

    samples: List[VocSample] = []
    skipped = 0
    allowed_exts = [e.lower() for e in extensions] if extensions else None

    for xml_path in xml_files:
        try:
            width, height, objects, filename = parse_voc_xml(xml_path, include_difficult=include_difficult)

            if not objects:
                logging.warning("Skip XML without valid objects: %s", xml_path)
                skipped += 1
                continue

            unknown = sorted({o.class_name for o in objects if o.class_name not in class_to_id})
            if unknown:
                msg = f"Unknown classes {unknown} in {xml_path}"
                if strict:
                    raise ValueError(msg)
                logging.warning("%s. Skip sample.", msg)
                skipped += 1
                continue

            image_path = None
            if filename:
                candidate = xml_path.parent / filename
                if candidate.exists() and candidate.suffix.lower() in (set(allowed_exts) if allowed_exts else IMG_EXTS):
                    image_path = candidate
                else:
                    image_path = find_image_for_stem(Path(filename).stem, input_dir, exts=allowed_exts)
            else:
                image_path = find_image_for_stem(xml_path.stem, input_dir, exts=allowed_exts)

            if image_path is None or not image_path.exists():
                msg = f"No matching image found for XML: {xml_path}"
                if strict:
                    raise FileNotFoundError(msg)
                logging.warning("%s. Skip sample.", msg)
                skipped += 1
                continue

            samples.append(
                VocSample(
                    image_path=image_path,
                    xml_path=xml_path,
                    width=width,
                    height=height,
                    objects=objects,
                )
            )

        except Exception as e:
            if strict:
                raise
            logging.warning("Failed to parse %s: %s", xml_path, e)
            skipped += 1

    if not samples:
        raise RuntimeError("No valid samples collected.")

    logging.info("Collected %d valid samples, skipped %d.", len(samples), skipped)
    return samples


def train_val_test_split(
    samples: List[VocSample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[VocSample], List[VocSample], List[VocSample]]:
    validate_ratios(train_ratio, val_ratio, test_ratio)
    rnd = random.Random(seed)
    shuffled = samples[:]
    rnd.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val

    train_samples = shuffled[:n_train]
    val_samples = shuffled[n_train:n_train + n_val]
    test_samples = shuffled[n_train + n_val:]

    logging.info(
        "Split sizes -> train: %d, val: %d, test: %d",
        len(train_samples),
        len(val_samples),
        len(test_samples),
    )
    return train_samples, val_samples, test_samples


def make_dirs(output_dir: Path) -> Dict[str, Dict[str, Path]]:
    dirs = {
        "train": {
            "images": output_dir / "images" / "train",
            "labels": output_dir / "labels" / "train",
        },
        "val": {
            "images": output_dir / "images" / "val",
            "labels": output_dir / "labels" / "val",
        },
        "test": {
            "images": output_dir / "images" / "test",
            "labels": output_dir / "labels" / "test",
        },
    }
    for split in dirs.values():
        split["images"].mkdir(parents=True, exist_ok=True)
        split["labels"].mkdir(parents=True, exist_ok=True)
    return dirs


def voc_boxes_to_pascal_voc(objects: List[VocObject], img_w: int, img_h: int, class_to_id: Dict[str, int]):
    bboxes = []
    class_labels = []
    for o in objects:
        xmin = max(0.0, min(float(img_w - 1), o.xmin))
        ymin = max(0.0, min(float(img_h - 1), o.ymin))
        xmax = max(0.0, min(float(img_w - 1), o.xmax))
        ymax = max(0.0, min(float(img_h - 1), o.ymax))
        if xmax <= xmin or ymax <= ymin:
            continue
        bboxes.append([xmin, ymin, xmax, ymax])
        class_labels.append(class_to_id[o.class_name])
    return bboxes, class_labels


def pascal_voc_to_yolo_line(box: Sequence[float], class_id: int, img_w: int, img_h: int) -> str:
    xmin, ymin, xmax, ymax = map(float, box)
    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def save_yolo_sample(
    image,
    bboxes: Sequence[Sequence[float]],
    class_labels: Sequence[int],
    out_image_path: Path,
    out_label_path: Path,
) -> None:
    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    out_label_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(out_image_path), image)
    if not success:
        raise IOError(f"Failed to write image: {out_image_path}")

    h, w = image.shape[:2]
    lines = [pascal_voc_to_yolo_line(box, cls_id, w, h) for box, cls_id in zip(bboxes, class_labels)]

    with open(out_label_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
        else:
            f.write("")


def build_train_transform(image_size: int, min_visibility: float, min_area: float) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, p=1.0),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=1.0,
            ),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                rotate=(-5, 5),
                shear=(-3, 3),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.6,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15,
                        contrast_limit=0.15,
                        p=1.0,
                    ),
                    A.RandomGamma(
                        gamma_limit=(85, 115),
                        p=1.0,
                    ),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.ImageCompression(quality_range=(70, 95), p=1.0),
                ],
                p=0.25,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.03, 0.08),
                hole_width_range=(0.03, 0.08),
                fill=0,
                p=0.15,
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=min_visibility,
            min_area=min_area,
            clip=True,
        ),
    )


def build_eval_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, p=1.0),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            clip=True,
        ),
    )


def build_split_transform(
    split_name: str,
    image_size: int,
    min_visibility: float,
    min_area: float,
    no_augment: bool,
) -> A.Compose:
    if no_augment:
        return build_eval_transform(image_size)

    if split_name == "train":
        return build_train_transform(
            image_size=image_size,
            min_visibility=min_visibility,
            min_area=min_area,
        )

    # For val/test:
    # keep resize+pad always, but apply lighter random perturbation so the offline-expanded
    # dataset can preserve the requested split proportions while remaining closer to evaluation data.
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, p=1.0),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=1.0,
            ),
            A.Affine(
                scale=(0.98, 1.02),
                translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                rotate=(-2, 2),
                shear=(-1, 1),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.35,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.08,
                        contrast_limit=0.08,
                        p=1.0,
                    ),
                    A.RandomGamma(
                        gamma_limit=(95, 105),
                        p=1.0,
                    ),
                ],
                p=0.20,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 3), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                    A.ImageCompression(quality_range=(85, 95), p=1.0),
                ],
                p=0.10,
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=min_visibility,
            min_area=min_area,
            clip=True,
        ),
    )


def read_image_bgr(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise IOError(f"Failed to read image: {image_path}")
    return image


def process_split(
    split_name: str,
    samples: List[VocSample],
    split_dirs: Dict[str, Path],
    class_to_id: Dict[str, int],
    image_size: int,
    aug_per_image: int,
    copy_original: bool,
    no_augment: bool,
    aug_only: bool,
    keep_empty: bool,
    min_visibility: float,
    min_area: float,
) -> int:
    total_written = 0

    deterministic_transform = build_eval_transform(image_size)
    random_transform = build_split_transform(
        split_name=split_name,
        image_size=image_size,
        min_visibility=min_visibility,
        min_area=min_area,
        no_augment=no_augment,
    )

    for idx, sample in enumerate(samples):
        image = read_image_bgr(sample.image_path)
        bboxes, class_labels = voc_boxes_to_pascal_voc(sample.objects, sample.width, sample.height, class_to_id)

        if not bboxes:
            logging.warning("Skip sample with zero valid boxes: %s", sample.image_path)
            continue

        base_name = sample.image_path.stem
        ext = ".jpg"

        num_versions = aug_per_image
        if not aug_only:
            if copy_original or num_versions == 0:
                num_versions += 1

        for version_idx in range(num_versions):
            use_original = (not aug_only) and copy_original and version_idx == 0
            if (not aug_only) and num_versions == 1 and aug_per_image == 0:
                use_original = True

            if use_original:
                transformed = deterministic_transform(image=image, bboxes=bboxes, class_labels=class_labels)
                out_stem = f"{base_name}_orig"
            else:
                transformed = random_transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_idx = version_idx if not copy_original else version_idx - 1
                out_stem = f"{base_name}_aug{aug_idx:02d}"

            out_img = transformed["image"]
            out_boxes = transformed["bboxes"]
            out_labels = transformed["class_labels"]

            if (not out_boxes) and (not keep_empty):
                logging.debug("Drop empty transformed sample: %s [%s]", sample.image_path, out_stem)
                continue

            img_path = split_dirs["images"] / f"{out_stem}{ext}"
            lbl_path = split_dirs["labels"] / f"{out_stem}.txt"

            collision_counter = 1
            while img_path.exists() or lbl_path.exists():
                img_path = split_dirs["images"] / f"{out_stem}_{collision_counter}{ext}"
                lbl_path = split_dirs["labels"] / f"{out_stem}_{collision_counter}.txt"
                collision_counter += 1

            save_yolo_sample(out_img, out_boxes, out_labels, img_path, lbl_path)
            total_written += 1

        if (idx + 1) % 50 == 0 or idx == len(samples) - 1:
            logging.info("[%s] processed %d / %d samples", split_name, idx + 1, len(samples))

    return total_written

def write_yaml(output_dir: Path, yaml_name: str, dataset_name: str, classes: List[str]) -> Path:
    yaml_path = output_dir / yaml_name
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)},
        "nc": len(classes),
        "dataset_name": dataset_name,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return yaml_path


def copy_script_snapshot(output_dir: Path) -> None:
    try:
        this_script = Path(__file__).resolve()
        shutil.copy2(this_script, output_dir / this_script.name)
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    if args.classes is None:
        args.classes = detect_classes_from_xml(
            input_dir=args.input_dir,
            include_difficult=args.include_difficult,
            strict=args.strict,
        )
    else:
        logging.info("Using user-provided classes:")
        for idx, name in enumerate(args.classes):
            logging.info("  %d -> %s", idx, name)

    class_to_id = {name: idx for idx, name in enumerate(args.classes)}
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    validate_augmentation_args(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dirs = make_dirs(args.output_dir)

    samples = collect_samples(
        input_dir=args.input_dir,
        class_to_id=class_to_id,
        include_difficult=args.include_difficult,
        strict=args.strict,
        extensions=args.extensions,
    )

    train_samples, val_samples, test_samples = train_val_test_split(
        samples=samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    counts = {}
    counts["train"] = process_split(
        split_name="train",
        samples=train_samples,
        split_dirs=dirs["train"],
        class_to_id=class_to_id,
        image_size=args.image_size,
        aug_per_image=args.train_aug_per_image,
        copy_original=args.copy_original_train,
        no_augment=args.no_augment,
        aug_only=args.aug_only,
        keep_empty=args.keep_empty,
        min_visibility=args.min_visibility,
        min_area=args.min_area,
    )
    counts["val"] = process_split(
        split_name="val",
        samples=val_samples,
        split_dirs=dirs["val"],
        class_to_id=class_to_id,
        image_size=args.image_size,
        aug_per_image=args.val_aug_per_image,
        copy_original=args.copy_original_val,
        no_augment=args.no_augment,
        aug_only=args.aug_only,
        keep_empty=args.keep_empty,
        min_visibility=args.min_visibility,
        min_area=args.min_area,
    )
    counts["test"] = process_split(
        split_name="test",
        samples=test_samples,
        split_dirs=dirs["test"],
        class_to_id=class_to_id,
        image_size=args.image_size,
        aug_per_image=args.test_aug_per_image,
        copy_original=args.copy_original_test,
        no_augment=args.no_augment,
        aug_only=args.aug_only,
        keep_empty=args.keep_empty,
        min_visibility=args.min_visibility,
        min_area=args.min_area,
    )

    yaml_path = write_yaml(
        output_dir=args.output_dir,
        yaml_name=args.yaml_name,
        dataset_name=args.dataset_name,
        classes=args.classes,
    )
    copy_script_snapshot(args.output_dir)

    logging.info("Done.")
    logging.info("Output dir: %s", args.output_dir.resolve())
    logging.info("YAML: %s", yaml_path.resolve())
    logging.info("Written images -> train: %d, val: %d, test: %d", counts["train"], counts["val"], counts["test"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
