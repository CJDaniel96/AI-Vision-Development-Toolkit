# AI 視覺開發工具箱 (AI Vision Development Toolkit)

一套專為加速電腦視覺與機器學習專案開發流程而設計的命令列實用工具集合。

## 總覽

在開發AI模型的過程中，我們經常需要花費大量時間在資料前處理、格式轉換、資料擴增和品質檢驗上。本工具箱旨在將這些重複性高、耗時的工作自動化，讓開發者能更專注於模型本身的建構與優化。從標註格式的轉換、資料集的智慧分割，到專為特定場景（如AOI）設計的影像分析與分類工具，這裡的每一個腳本都是為了解決一個真實世界的問題而生。

## 核心功能

- **標註格式轉換**: 輕鬆在 YOLO, PASCAL VOC, CVAT 等主流格式之間進行轉換。
- **資料集整備**: 提供智慧分割、分層抽樣、資料篩選與過濾等功能。
- **影像擴增**: 同步更新影像與標註檔，支援多種擴增策略。
- **自動化分析**: 自動偵測影像品質、讀取元件文字(OCR)，並進行分類。
- **專案打包**: 一鍵將 YOLO 資料集打包成 CVAT 相容的格式，簡化團隊協作流程。

---

## 工具詳解

為了方便使用，相似功能的工具被歸納在一起。

### 標註格式轉換 (Annotation Conversion)

`voc2yolo.py` (推薦)
- **功能**: 更進階的 VOC 轉 YOLO 工具。支援**分層抽樣**來確保各類別在訓練/驗證/測試集中的分佈均衡，並會自動產生 YOLO 訓練所需的 `data.yaml` 檔案。
- **使用範例**:
  ```bash
  python voc2yolo.py --input ./voc_data --output ./yolo_dataset --split 0.7 0.2 0.1
  ```

`yolo_pascal_voc_converter.py`
- **功能**: 使用 YOLO 模型對影像或影片進行物件偵測，並將偵測結果儲存為 PASCAL VOC (.xml) 格式。
- **使用範例**:
  ```bash
  python yolo_pascal_voc_converter.py -i ./images_folder -o ./output -m yolov5s.pt
  ```

`cvat2yolo_seg.py`
- **功能**: 將 CVAT 導出的多邊形分割 (Segmentation) 標註檔 (.xml) 轉換為 YOLOv8 分割格式，並自動切分資料集與產生 `data.yaml`。
- **使用範例**:
  ```bash
  python cvat2yolo_seg.py --xml annotations.xml --img-dir ./images --out-dir ./yolo_seg_dataset
  ```

`yolo_to_cvat_converter.py`
- **功能**: 將 YOLO 格式的資料集（包含影像、標籤檔、類別檔）打包成一個 .zip 壓縮檔，以便直接匯入 CVAT 進行標註或審閱。
- **使用範例**:
  ```bash
  python yolo_to_cvat_converter.py --images-dir ./img --labels-dir ./lab --class-names ./classes.txt --output-zip ./cvat_upload.zip
  ```

`voc_to_yolo_converter.py`
- **功能**: 基礎的 PASCAL VOC (.xml) 轉 YOLO (.txt) 工具。
- **使用範例**:
  ```bash
  python voc_to_yolo_converter.py -x ./annotations -i ./images -o ./yolo_labels
  ```

### 資料集整備與分割 (Dataset Preparation & Splitting)

`image_dataset_splitter.py` (推薦)
- **功能**: 功能強大的資料集分割工具，可將影像和對應的標註檔（支援 .xml, .txt 等）按比例分割為訓練、驗證、測試集。支援傳統模式與 COCO-like (images/labels 分離) 模式。
- **使用範例**:
  ```bash
  python image_dataset_splitter.py -s ./source_data -o ./split_dataset -a .xml --val --coco
  ```

`filter_dataset.py`
- **功能**: 根據指定的類別名稱，遞迴地篩選 PASCAL VOC 資料集。您可以選擇只保留包含特定標籤的影像，或排除包含某些標籤的影像。
- **使用範例**:
  ```bash
  # 只保留 'dog' 標籤，並排除任何包含 'person' 的影像
  python filter_dataset.py -i ./raw_dataset -o ./filtered_dogs --label dog --exclude person
  ```

`filter_labeled_voc.py` / `filter_labeled_cvat.py`
- **功能**: 清理資料集，只保留那些**確實有標註物件**的影像及其標註檔，去除所有空標註的資料。分別支援 PASCAL VOC 和 CVAT 格式。
- **使用範例 (VOC)**:
  ```bash
  python filter_labeled_voc.py -i ./voc_dataset -o ./labeled_only
  ```
- **使用範例 (CVAT)**:
  ```bash
  python filter_labeled_cvat.py -x annotations.xml -i ./images -o ./labeled_only
  ```

### 影像擴增 (Image Augmentation)

`augment_yolo.py` (推薦)
- **功能**: 強大的影像擴增工具，可同時處理影像與標註（支援 YOLO, PASCAL VOC, CVAT 格式）。除了常見的翻轉、亮度調整外，還支援**固定角度步長旋轉**，可用於快速產生大量特定角度的擴增資料。
- **使用範例**:
  ```bash
  # 每張圖以 90 度為間隔產生 4 個擴增版本
  python augment_yolo.py --images_dir ./img --labels_dir ./lab --output_images ./aug_img --output_labels ./aug_lab --rotate_step 90
  ```

`image_augmentation_script.py`
- **功能**: 針對 PASCAL VOC 格式的簡易擴增工具，支援水平翻轉、亮度對比、色調飽和度調整等。
- **使用範例**:
  ```bash
  python image_augmentation_script.py -i ./dataset -o ./augmented_dataset -a horizontal_flip mixed
  ```

### 專用與輔助工具 (Specialized & Utility Tools)

`image_brightness_detector.py`
- **功能**: 專為電子元件等需要清晰紋理的場景設計的**影像品質分析器**。它不僅僅檢查亮度，更透過分析文字清晰度、細節豐富度、對比度等進階指標，來評估一張影像是否適合作為訓練資料，並給出「可用性分數」。支援對子資料夾進行批次分析與報告生成。
- **使用範例**:
  ```bash
  # 批次分析 'all_components' 下所有子資料夾，並自動產生報告
  python image_brightness_detector.py --batch-subfolders ./all_components --auto-output
  ```

`aoi_ocr_classifier.py`
- **功能**: 專為 AOI (自動光學檢測) 設計的分類工具。它使用 OCR 技術讀取電子元件影像上的文字，自動判斷其正確方向（0/90/180/270度），並將其分類至對應角度的資料夾或 NG 資料夾。
- **使用範例**:
  ```bash
  python aoi_ocr_classifier.py -i ./component_images -o ./classified_components -t "TARGET_TEXT"
  ```

`image_rotator.py`
- **功能**: 簡單的影像旋轉工具，可將指定資料夾內的所有影像分別旋轉 90、180、270 度，並存放到對應的子資料夾。
- **使用範例**:
  ```bash
  python image_rotator.py ./input_images ./rotated_images
  ```

---

## 安裝與需求

建議您建立一個 Python 虛擬環境。本專案所需的所有套件都已列在 `requirements.txt` 中。

請執行以下指令來安裝所有相依套件：

```bash
pip install -r requirements.txt
```

一個建議的 `requirements.txt` 檔案內容如下：

```
torch
torchvision
ultralytics
opencv-python
Pillow
albumentations
scikit-learn
tqdm
PyYAML
paddleocr
```

## 通用使用方式

本工具箱中的所有腳本都支援命令列操作。您可以透過 `-h` 或 `--help` 參數來查看每個腳本的詳細用法與所有可選參數。

```bash
python <script_name>.py --help
```

## 貢獻

歡迎您透過提交 Pull Requests 或回報 Issues 來為本專案做出貢獻。如果您有新的想法或發現了 bug，請不要猶豫，讓我們知道！

1. Fork 本專案。
2. 建立您的功能分支 (`git checkout -b feature/AmazingFeature`)。
3. 提交您的變更 (`git commit -m 'Add some AmazingFeature'`)。
4. 將您的分支推送到遠端 (`git push origin feature/AmazingFeature`)。
5. 開啟一個 Pull Request。

## 授權

本專案採用 MIT 授權。詳情請參閱 `LICENSE` 檔案。

---

*This toolkit was crafted to make AI development more efficient and enjoyable.*