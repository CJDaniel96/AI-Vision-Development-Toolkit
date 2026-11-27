"""YOLO OBB 四角點標註的讀寫與越界補邊，供整理及擴增工具共用。"""

from pathlib import Path

import cv2
import numpy as np


PADDING_FILL = (114, 114, 114)


def load_yolo_obb_annotations(label_path: Path) -> tuple[list, list[int]]:
    """讀取四角點標註，允許 CVAT 越界座標；空檔代表背景影像。"""
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
        if not np.isfinite(points).all():
            raise ValueError(f'{prefix}：座標必須為有限數值')
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


def pad_image_to_fit_obb(image: np.ndarray, polygons: list) -> tuple[np.ndarray, list]:
    """向外取整補邊並同步平移全部角點；回傳影像與重新正規化的四角點。"""
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError('影像寬高必須大於 0')
    points = np.asarray(polygons, dtype=np.float64).reshape(-1, 4, 2)
    if not np.isfinite(points).all():
        raise ValueError('OBB 座標必須為有限數值')
    if not points.size:
        return image, []

    pixel_points = points * [width, height]
    min_x, min_y = pixel_points.min(axis=(0, 1))
    max_x, max_y = pixel_points.max(axis=(0, 1))
    # 不可逐點 clip：裁切角點會改變旋轉框的形狀。
    left = int(np.ceil(max(0, -min_x)))
    top = int(np.ceil(max(0, -min_y)))
    right = int(np.ceil(max(0, max_x - width)))
    bottom = int(np.ceil(max(0, max_y - height)))
    if any((left, top, right, bottom)):
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=PADDING_FILL,
        )
        pixel_points += [left, top]
        points = pixel_points / [width + left + right, height + top + bottom]
    return image, points.tolist()
