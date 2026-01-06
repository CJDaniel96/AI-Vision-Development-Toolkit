import argparse
import os
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

# CVAT 中 "Defect" 標籤對應的 RGB 顏色
# The RGB color corresponding to the "Defect" label in CVAT
DEFECT_COLOR = [32, 175, 98]

def process_and_colorize_mask(input_path: str, output_path: str):
    """
    Opens a grayscale mask, thresholds it, and saves a new RGB mask
    with a specific color for the segmented area.

    Args:
        input_path (str): Path to the source grayscale mask image.
        output_path (str): Path to save the new colorized RGB mask image.
    """
    try:
        # 使用 PIL 開啟原始灰階遮罩影像
        # Open the original grayscale mask image using PIL
        with Image.open(input_path) as gray_mask_img:
            # 將影像轉換為 NumPy 陣列以進行高效能的像素操作
            # Convert the image to a NumPy array for efficient pixel manipulation
            gray_mask_array = np.array(gray_mask_img)

            # 確保影像是 2D 的灰階影像
            # Ensure the image is a 2D grayscale image
            if gray_mask_array.ndim != 2:
                print(f"\nWarning: Skipping non-grayscale image: {input_path}")
                return

            # 建立一個與原始影像相同尺寸的全新 RGB 影像陣列，並初始化為全黑
            # Create a new RGB image array of the same size as the original, initialized to all black
            height, width = gray_mask_array.shape
            color_mask_array = np.zeros((height, width, 3), dtype=np.uint8)

            # 找出所有非黑色像素的位置 (像素值 > 0)
            # Find all non-black pixel locations (pixel value > 0)
            defect_pixels = gray_mask_array > 0

            # 將這些位置的像素值設定為指定的 "Defect" 顏色
            # Set the pixel values at these locations to the specified "Defect" color
            color_mask_array[defect_pixels] = DEFECT_COLOR

            # 將 NumPy 陣列轉換回 PIL 影像物件
            # Convert the NumPy array back to a PIL Image object
            color_mask_img = Image.fromarray(color_mask_array, 'RGB')

            # 儲存處理完成的彩色遮罩影像
            # Save the processed color mask image
            color_mask_img.save(output_path)

    except Exception as e:
        print(f"\nError processing {input_path}: {e}")

def main():
    """
    Main function to parse arguments and process all mask images in a directory.
    """
    parser = argparse.ArgumentParser(
        description="Convert grayscale masks to colorized masks for CVAT segmentation tasks."
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="Directory containing the original grayscale mask images."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Directory to save the processed colorized mask images."
    )
    args = parser.parse_args()

    # 建立輸出資料夾 (如果不存在)
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir}")

    # 尋找所有支援的影像格式
    # Find all supported image formats
    supported_formats = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    image_paths = []
    for fmt in supported_formats:
        image_paths.extend(glob(os.path.join(args.input_dir, fmt)))

    if not image_paths:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(image_paths)} images to process.")

    # 使用 tqdm 顯示進度條並處理每一張影像
    # Use tqdm to display a progress bar and process each image
    for img_path in tqdm(image_paths, desc="Processing masks"):
        filename = os.path.basename(img_path)
        output_path = os.path.join(args.output_dir, filename)
        process_and_colorize_mask(img_path, output_path)

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()
