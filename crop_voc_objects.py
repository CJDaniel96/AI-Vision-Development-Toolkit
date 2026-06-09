#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
 
 
def parse_voc_xml(xml_path: Path):
    """
    解析 PASCAL VOC XML，回傳:
    {
        "filename": str | None,
        "objects": [
            {
                "name": str,
                "difficult": int,
                "bbox": (xmin, ymin, xmax, ymax)
            },
            ...
        ]
    }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
 
    filename = root.findtext("filename")
    objects = []
 
    for obj in root.findall("object"):
        name = obj.findtext("name")
        difficult_text = obj.findtext("difficult", default="0")
        difficult = int(difficult_text) if difficult_text.isdigit() else 0
 
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
 
        try:
            xmin = int(float(bndbox.findtext("xmin")))
            ymin = int(float(bndbox.findtext("ymin")))
            xmax = int(float(bndbox.findtext("xmax")))
            ymax = int(float(bndbox.findtext("ymax")))
        except (TypeError, ValueError):
            continue
 
        objects.append({
            "name": name,
            "difficult": difficult,
            "bbox": (xmin, ymin, xmax, ymax),
        })
 
    return {
        "filename": filename,
        "objects": objects,
    }
 
 
def find_image_for_xml(xml_path: Path, images_dir: Path):
    """
    根據 xml 檔名尋找對應影像。
    先嘗試 xml 內的 filename，再嘗試同 stem 的常見副檔名。
    """
    data = parse_voc_xml(xml_path)
 
    # 1) 優先使用 XML 內的 filename
    filename = data.get("filename")
    if filename:
        candidate = images_dir / filename
        if candidate.exists():
            return candidate
 
    # 2) 使用同 stem 搭配常見副檔名尋找
    stem = xml_path.stem
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    for ext in exts:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
 
    return None
 
 
def sanitize_name(name: str) -> str:
    """
    避免類別名稱含有不適合路徑的字元
    """
    invalid = '<>:"/\\|?*'
    for ch in invalid:
        name = name.replace(ch, "_")
    return name.strip() or "unknown"
 
 
def clamp_bbox(xmin, ymin, xmax, ymax, width, height):
    """
    將 bbox 限制在影像範圍內
    """
    xmin = max(0, min(xmin, width))
    ymin = max(0, min(ymin, height))
    xmax = max(0, min(xmax, width))
    ymax = max(0, min(ymax, height))
    return xmin, ymin, xmax, ymax
 
 
def resize_crop(crop: Image.Image, target_size: tuple, mode: str) -> Image.Image:
    """
    將裁切影像調整至目標尺寸。
    mode='stretch'   : 直接拉伸，不保持比例
    mode='letterbox' : 保持比例縮放，其餘位置填黑
    """
    tw, th = target_size
    if mode == "stretch":
        return crop.resize((tw, th), Image.LANCZOS)

    # letterbox: 等比例縮放後置中，其餘填黑
    cw, ch = crop.size
    scale = min(tw / cw, th / ch)
    new_w = int(cw * scale)
    new_h = int(ch * scale)
    resized = crop.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    paste_x = (tw - new_w) // 2
    paste_y = (th - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def crop_objects_from_annotation(
    xml_path: Path,
    images_dir: Path,
    output_dir: Path,
    include_difficult: bool = False,
    only_classes=None,
    resize: tuple = None,
    resize_mode: str = "letterbox",
):
    """
    從單一 VOC XML 裁切物件
    回傳裁切成功數量
    """
    data = parse_voc_xml(xml_path)
    image_path = find_image_for_xml(xml_path, images_dir)

    if image_path is None:
        print(f"[WARN] 找不到對應影像: {xml_path.name}", file=sys.stderr)
        return 0

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] 無法開啟影像 {image_path.name}: {e}", file=sys.stderr)
        return 0

    width, height = image.size
    saved_count = 0

    for idx, obj in enumerate(data["objects"], start=1):
        cls_name = obj["name"] or "unknown"

        if only_classes and cls_name not in only_classes:
            continue

        if not include_difficult and obj["difficult"] == 1:
            continue

        xmin, ymin, xmax, ymax = obj["bbox"]
        xmin, ymin, xmax, ymax = clamp_bbox(xmin, ymin, xmax, ymax, width, height)

        # 避免無效框
        if xmax <= xmin or ymax <= ymin:
            print(
                f"[WARN] 無效 bbox，已略過: {xml_path.name} object#{idx} {cls_name}",
                file=sys.stderr
            )
            continue

        crop = image.crop((xmin, ymin, xmax, ymax))

        if resize is not None:
            crop = resize_crop(crop, resize, resize_mode)

        class_dir = output_dir / sanitize_name(cls_name)
        class_dir.mkdir(parents=True, exist_ok=True)

        out_name = (
            f"{image_path.stem}"
            f"__obj{idx:03d}"
            f"__{sanitize_name(cls_name)}"
            f"__x{xmin}_y{ymin}_x{xmax}_y{ymax}.jpg"
        )
        out_path = class_dir / out_name

        try:
            crop.save(out_path, quality=95)
            saved_count += 1
        except Exception as e:
            print(f"[WARN] 儲存失敗 {out_path}: {e}", file=sys.stderr)

    return saved_count
 
 
def main():
    parser = argparse.ArgumentParser(
        description="讀取影像與 PASCAL VOC XML 標記，將標記物件裁切後另存。"
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="影像資料夾路徑"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="PASCAL VOC XML 標記資料夾路徑"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="裁切結果輸出資料夾"
    )
    parser.add_argument(
        "--include-difficult",
        action="store_true",
        help="若指定，則連 difficult=1 的物件也一起裁切"
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="只裁切指定類別，例如: --classes cat dog person"
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="將裁切影像調整至指定尺寸，例如: --resize 224 224"
    )
    parser.add_argument(
        "--resize-mode",
        choices=["stretch", "letterbox"],
        default="letterbox",
        help=(
            "resize 模式 (預設: letterbox)。"
            "stretch: 直接拉伸至目標尺寸；"
            "letterbox: 保持比例縮放，其餘位置填黑"
        )
    )

    args = parser.parse_args()

    images_dir = args.images
    annotations_dir = args.annotations
    output_dir = args.output
    include_difficult = args.include_difficult
    only_classes = set(args.classes) if args.classes else None
    resize = tuple(args.resize) if args.resize else None
    resize_mode = args.resize_mode
 
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"[ERROR] 影像資料夾不存在或不是資料夾: {images_dir}", file=sys.stderr)
        sys.exit(1)
 
    if not annotations_dir.exists() or not annotations_dir.is_dir():
        print(f"[ERROR] 標記資料夾不存在或不是資料夾: {annotations_dir}", file=sys.stderr)
        sys.exit(1)
 
    output_dir.mkdir(parents=True, exist_ok=True)
 
    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        print(f"[ERROR] 在 {annotations_dir} 找不到任何 XML 檔案", file=sys.stderr)
        sys.exit(1)
 
    total_saved = 0
    total_xml = 0
 
    for xml_path in xml_files:
        saved = crop_objects_from_annotation(
            xml_path=xml_path,
            images_dir=images_dir,
            output_dir=output_dir,
            include_difficult=include_difficult,
            only_classes=only_classes,
            resize=resize,
            resize_mode=resize_mode,
        )
        total_saved += saved
        total_xml += 1
        print(f"[INFO] {xml_path.name}: 已裁切 {saved} 個物件")
 
    print("-" * 60)
    print(f"[DONE] 共處理 XML: {total_xml}")
    print(f"[DONE] 共輸出裁切物件: {total_saved}")
    print(f"[DONE] 輸出資料夾: {output_dir}")
 
 
if __name__ == "__main__":
    main()