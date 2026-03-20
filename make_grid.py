#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import argparse
import math
import os
from pathlib import Path
from typing import List, Tuple
 
from PIL import Image, ImageOps
 
 
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine multiple images into a grid image."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input image paths. Example: --input img1.jpg img2.jpg img3.jpg img4.jpg",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output grid image path. Example: grid.jpg",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Number of grid rows. If not set, it will be inferred automatically.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Number of grid columns. If not set, it will be inferred automatically.",
    )
    parser.add_argument(
        "--cell-width",
        type=int,
        default=512,
        help="Width of each cell in the grid. Default: 512",
    )
    parser.add_argument(
        "--cell-height",
        type=int,
        default=512,
        help="Height of each cell in the grid. Default: 512",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding between images in pixels. Default: 10",
    )
    parser.add_argument(
        "--bg-color",
        type=str,
        default="white",
        help="Background color. Example: white, black, #cccccc. Default: white",
    )
    parser.add_argument(
        "--keep-aspect",
        action="store_true",
        help="Keep aspect ratio and pad the remaining area instead of stretching.",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort input image paths before processing.",
    )
    return parser.parse_args()
 
 
def validate_images(image_paths: List[str]) -> List[Path]:
    valid_paths = []
    for path_str in image_paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported image format: {path}")
        valid_paths.append(path)
    return valid_paths
 
 
def infer_grid_shape(n: int, rows: int = None, cols: int = None) -> Tuple[int, int]:
    if rows is not None and cols is not None:
        if rows * cols < n:
            raise ValueError(
                f"Grid size {rows}x{cols} is too small for {n} images."
            )
        return rows, cols
 
    if rows is not None:
        cols = math.ceil(n / rows)
        return rows, cols
 
    if cols is not None:
        rows = math.ceil(n / cols)
        return rows, cols
 
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols
 
 
def load_and_prepare_image(
    image_path: Path,
    cell_width: int,
    cell_height: int,
    keep_aspect: bool,
    bg_color: str,
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
 
    if keep_aspect:
        # 保持比例，剩餘區域補背景色
        return ImageOps.pad(
            img,
            (cell_width, cell_height),
            color=bg_color,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
    else:
        # 直接拉伸到固定大小
        return img.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
 
 
def make_grid(
    image_paths: List[Path],
    output_path: str,
    rows: int,
    cols: int,
    cell_width: int,
    cell_height: int,
    padding: int,
    bg_color: str,
    keep_aspect: bool,
) -> None:
    grid_width = cols * cell_width + (cols + 1) * padding
    grid_height = rows * cell_height + (rows + 1) * padding
 
    canvas = Image.new("RGB", (grid_width, grid_height), color=bg_color)
 
    for idx, image_path in enumerate(image_paths):
        row = idx // cols
        col = idx % cols
 
        x = padding + col * (cell_width + padding)
        y = padding + row * (cell_height + padding)
 
        img = load_and_prepare_image(
            image_path=image_path,
            cell_width=cell_width,
            cell_height=cell_height,
            keep_aspect=keep_aspect,
            bg_color=bg_color,
        )
        canvas.paste(img, (x, y))
 
    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
 
    canvas.save(output_path)
    print(f"Saved grid image to: {output_path}")
    print(f"Grid shape: {rows} x {cols}")
    print(f"Canvas size: {canvas.size[0]} x {canvas.size[1]}")
 
 
def main() -> None:
    args = parse_args()
 
    image_paths = validate_images(args.input)
 
    if args.sort:
        image_paths = sorted(image_paths)
 
    rows, cols = infer_grid_shape(
        n=len(image_paths),
        rows=args.rows,
        cols=args.cols,
    )
 
    make_grid(
        image_paths=image_paths,
        output_path=args.output,
        rows=rows,
        cols=cols,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        padding=args.padding,
        bg_color=args.bg_color,
        keep_aspect=args.keep_aspect,
    )
 
 
if __name__ == "__main__":
    main()