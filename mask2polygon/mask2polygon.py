import os
import cv2
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_cvat_xml(image_annotations, output_file_path, label_name="bubble"):
    """
    根據提供的標註資料產生 CVAT XML 格式的檔案。

    :param image_annotations: 一個字典，key 為影像檔案名稱，value 為該影像的多邊形列表。
    :param output_file_path: 輸出的 XML 檔案路徑。
    :param label_name: 瑕疵的標籤名稱。
    """
    annotations_node = ET.Element("annotations")
    ET.SubElement(annotations_node, "version").text = "1.1"

    # Meta 資訊
    meta_node = ET.SubElement(annotations_node, "meta")
    task_node = ET.SubElement(meta_node, "task")
    ET.SubElement(task_node, "name").text = "xray_bubble_detection"
    ET.SubElement(task_node, "size").text = str(len(image_annotations))
    ET.SubElement(task_node, "mode").text = "annotation"
    
    labels_node = ET.SubElement(task_node, "labels")
    label_node = ET.SubElement(labels_node, "label")
    ET.SubElement(label_node, "name").text = label_name
    ET.SubElement(label_node, "color").text = "#ff0000" # 紅色
    ET.SubElement(label_node, "attributes")

    # 影像與多邊形標註
    for image_id, (image_name, data) in enumerate(image_annotations.items()):
        image_node = ET.SubElement(annotations_node, "image", {
            "id": str(image_id),
            "name": image_name,
            "width": str(data["width"]),
            "height": str(data["height"])
        })

        for polygon_points in data["polygons"]:
            ET.SubElement(image_node, "polygon", {
                "label": label_name,
                "occluded": "0",
                "source": "manual",
                "points": polygon_points
            })

    # 將 XML 結構轉換為格式化的字串
    rough_string = ET.tostring(annotations_node, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml_as_string = reparsed.toprettyxml(indent="  ")

    # 寫入檔案
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml_as_string)

def masks_to_cvat_polygons(original_images_dir, mask_images_dir, output_xml_path, label_name, min_area=2000, epsilon_factor=0.005):
    """
    讀取 mask 影像，轉換為多邊形，並產生 CVAT XML 檔案。
    :param min_area: 瑕疵的最小面積，小於此值的瑕疵將被忽略。
    :param epsilon_factor: 輪廓簡化的 epsilon 因子，值越小點越多。
    """
    image_annotations = {}
    
    # 確保原始影像目錄存在
    if not os.path.isdir(original_images_dir):
        print(f"錯誤: 找不到原始影像目錄 '{original_images_dir}'")
        return

    print(f"正在從 '{mask_images_dir}' 讀取 mask 影像...")

    for mask_filename in sorted(os.listdir(mask_images_dir)):
        mask_path = os.path.join(mask_images_dir, mask_filename)
        if not os.path.isfile(mask_path):
            continue

        # 讀取 mask 影像
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            print(f"警告: 無法讀取 mask 影像 '{mask_filename}'，已跳過。")
            continue

        height, width = mask_image.shape
        
        # 尋找對應的原始影像檔案名稱
        base_name, _ = os.path.splitext(mask_filename)
        original_image_name = None
        # 支援多種常見影像格式
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            potential_original_path = os.path.join(original_images_dir, base_name + ext)
            if os.path.exists(potential_original_path):
                original_image_name = base_name + ext
                break
        
        if not original_image_name:
            print(f"警告: 找不到與 '{mask_filename}' 對應的原始影像，已跳過。")
            continue

        # 尋找輪廓
        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for contour in contours:
            # 計算瑕疵的面積
            area = cv2.contourArea(contour)

            # 只有當面積大於指定的最小值時，才進行處理
            if area > min_area:
                # 簡化輪廓點以減少點的數量，可以調整 epsilon 值
                arc_length = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon_factor * arc_length, True)

                if len(approx) >= 3: # 一個多邊形至少需要3個點
                    points_str = ";".join([f"{p[0][0]},{p[0][1]}" for p in approx])
                    polygons.append(points_str)
        
        if polygons:
            image_annotations[original_image_name] = {
                "width": width,
                "height": height,
                "polygons": polygons
            }
            print(f"處理完成: '{original_image_name}'，找到 {len(polygons)} 個瑕疵區域。")

    if not image_annotations:
        print("錯誤: 沒有找到任何有效的 mask 影像進行處理。")
        return

    # 產生 XML 檔案
    create_cvat_xml(image_annotations, output_xml_path, label_name=label_name)
    print(f"\n成功！CVAT 標記檔案已儲存至 '{output_xml_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="將 Mask 影像轉換為 CVAT 多邊形標註格式的 XML 檔案，並可依面積篩選。")
    parser.add_argument('--original_dir', type=str, default='original_images',
                        help='包含原始影像的資料夾路徑。')
    parser.add_argument('--mask_dir', type=str, default='mask_images',
                        help='包含 mask 影像的資料夾路徑。')
    parser.add_argument('--output_xml', type=str, default='annotations.xml',
                        help='輸出的 CVAT XML 檔案路徑。')
    parser.add_argument('--label', type=str, default='bubble',
                        help='瑕疵的標籤名稱。')
    parser.add_argument('--min_area', type=int, default=2000,
                        help='瑕疵的最小面積，小於此值的瑕疵將被忽略。')
    parser.add_argument('--epsilon', type=float, default=0.005,
                        help='輪廓簡化的 epsilon 因子。值越小，產生的標記點越多，輪廓越精細。預設為 0.005。')

    args = parser.parse_args()

    masks_to_cvat_polygons(
        original_images_dir=args.original_dir,
        mask_images_dir=args.mask_dir,
        output_xml_path=args.output_xml,
        label_name=args.label,
        min_area=args.min_area,
        epsilon_factor=args.epsilon
    )
