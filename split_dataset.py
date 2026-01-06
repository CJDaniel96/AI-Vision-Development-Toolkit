import os
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def copy_files(image_paths, dest_image_dir, labels_dir=None, dest_label_dir=None):
    """
    複製影像及其對應的標籤檔到目標資料夾。
    """
    os.makedirs(dest_image_dir, exist_ok=True)
    if dest_label_dir:
        os.makedirs(dest_label_dir, exist_ok=True)
    
    for img_path in tqdm(image_paths, desc=f"複製到 {Path(dest_image_dir).parent.name}"):
        # 複製影像檔
        shutil.copy(img_path, dest_image_dir)
        
        # 如果提供了標籤資料夾，則尋找並複製對應的標籤檔
        if labels_dir and dest_label_dir:
            base_name = img_path.stem
            label_path_txt = Path(labels_dir) / f"{base_name}.txt"
            label_path_xml = Path(labels_dir) / f"{base_name}.xml"
            
            if label_path_txt.exists():
                shutil.copy(label_path_txt, dest_label_dir)
            elif label_path_xml.exists():
                shutil.copy(label_path_xml, dest_label_dir)

def main(args):
    """主程式"""
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    output_dir = Path(args.output_dir)

    # 獲取所有影像檔案的路徑
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions]
    
    if not image_paths:
        print(f"錯誤：在資料夾 '{images_dir}' 中找不到任何影像檔案。")
        return

    print(f"找到 {len(image_paths)} 張影像。")

    # 使用 train_test_split 切分資料
    train_paths, val_paths = train_test_split(
        image_paths,
        test_size=args.test_size,
        random_state=args.random_state
    )

    print(f"訓練集大小: {len(train_paths)} 張影像")
    print(f"驗證集大小: {len(val_paths)} 張影像")

    # 定義輸出的資料夾路徑
    train_images_dest = output_dir / 'train' / 'images'
    val_images_dest = output_dir / 'val' / 'images'
    train_labels_dest = output_dir / 'train' / 'labels' if labels_dir else None
    val_labels_dest = output_dir / 'val' / 'labels' if labels_dir else None

    # 複製檔案到對應的資料夾
    copy_files(train_paths, train_images_dest, labels_dir, train_labels_dest)
    copy_files(val_paths, val_images_dest, labels_dir, val_labels_dest)

    print("\n資料集切分完成！")
    print(f"結果已儲存至: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="將物件偵測資料集切分為訓練集與驗證集")
    parser.add_argument("--images_dir", type=str, required=True, help="包含所有影像的來源資料夾路徑")
    parser.add_argument("--labels_dir", type=str, default=None, help="(選用) 包含所有標籤的來源資料夾路徑")
    parser.add_argument("--output_dir", type=str, required=True, help="儲存切分後資料集的目標資料夾路徑")
    parser.add_argument("--test_size", type=float, default=0.2, help="驗證集所佔的比例 (例如: 0.2 表示 20%%)")
    parser.add_argument("--random_state", type=int, default=42, help="隨機種子，確保每次切分結果一致")
    
    args = parser.parse_args()
    main(args)
