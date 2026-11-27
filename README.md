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

## 統一命令入口

`python main.py` 列出所有工具群組。各工具由同一個入口啟動。

| 群組 | 用途 |
| --- | --- |
| `augment` | 水平框、OBB 和純影像擴增，另有直角旋轉和 VOC 平衡 |
| `crop` | 中心、XYXY 或 VOC 物件裁切，以及影像座標選取 |
| `dataset` | 分割、抽樣、篩選、VOC 計數與合併、依產線檔名整理 |
| `convert` | 轉換 VOC、CVAT 或 YOLO 標註，整理 OBB 或遮罩 |
| `inspect` | 影像品質、AOI OCR、FOV 報表與框選介面 |
| `model` | 以 YOLO 推論產生 VOC 標註、匯出 TFLite、建立極性模組 |
| `report` | 影像網格、元件 JSON 轉 Excel |

```bash
python main.py augment --help
python main.py augment obb --help
python main.py crop center -i ./images -o ./cropped -r 0.5
python main.py dataset filter -i ./voc_data -o ./labeled --labeled-only
python main.py convert voc-to-yolo --input ./voc_data --output ./yolo_dataset
```

## 工具詳解

### 標註格式轉換 (Annotation Conversion)

`convert voc-to-yolo` (推薦)
- **功能**: 更進階的 VOC 轉 YOLO 工具。支援**分層抽樣**來確保各類別在訓練/驗證/測試集中的分佈均衡，並會自動產生 YOLO 訓練所需的 `data.yaml` 檔案。
- **使用範例**:
  ```bash
  python main.py convert voc-to-yolo --input ./voc_data --output ./yolo_dataset --split 0.7 0.2 0.1
  ```

`convert prepare-obb`
- **功能**: 將 YOLO OBB 四角點影像/TXT 配對資料整理成 `images/{train,val,test}`、`labels/{train,val,test}`，並產生可供 OBB 訓練使用的 `data.yaml`。依影像內的類別組合近似分層抽樣，同時維持整體分割數量；少量資料優先保留非空的 train 與 val。
- **輸入**: 每行為 `class_id x1 y1 x2 y2 x3 y3 x4 y4`，四角點沿框的邊依序排列，座標以原圖寬高正規化。可傳入影像/TXT 同目錄、含 `images/` 與 `labels/` 的 CVAT 匯出根目錄，或用 `--labels` 指定分離的標註目錄。遞迴掃描並保留相對子目錄；此工具不讀取 DOTA 像素座標或中心/寬高/角度格式。
- **越界修正**: 自動依所有框的最外側角點補灰邊 `(114, 114, 114)`，同步平移所有角點，重新正規化至 `[0, 1]`，保留原 OBB 的形狀與角度；不修改來源。補邊後尺寸可能增加，訓練須搭配輸出的影像與標註。
- **類別與分割**: `--classes` 接受每行一個名稱的 TXT，或含 `names` 的 YAML；未指定時自動尋找資料集的 `data.yaml` / `classes.txt`，再以 `class_0`、`class_1`…作為備用名稱。保留原始 class_id，不重新編號或依清單過濾物件；名稱列表須涵蓋所有 ID。`--split` 接受 train/val 或 train/val/test，預設 `0.7 0.2 0.1`，比例會正規化，train/val 比例須大於 0。至少需 2 張有效影像；稀少類別無法保證出現在所有分割中，`--seed` 預設 42。
- **資料處理**: 空 TXT 視為背景；缺失/無效標註與損壞影像會警告並跳過，只有加上 `--missing-as-background` 才將缺少 TXT 的影像視為背景。輸出目錄必須尚未存在；全部寫入成功後才產生正式資料集。讀寫與補邊邏輯由套件中的 OBB 工具共用。
- **使用範例**:
  ```bash
  # 影像與 TXT 放在同一目錄
  python main.py convert prepare-obb --input ./obb_data --output ./yolo_obb --classes ./classes.txt --split 0.7 0.2 0.1

  # CVAT Ultralytics YOLO OBB 匯出，包含 images/、labels/、data.yaml
  python main.py convert prepare-obb --input ./cvat_export --output ./yolo_obb

  # 影像與標註分開，只建立 train 與 val
  python main.py convert prepare-obb -i ./images -l ./labels -o ./yolo_obb --classes ./classes.txt --split 0.8 0.2

  # 使用產生的設定檔訓練 OBB 模型
  yolo obb train model=yolo11n-obb.pt data=./yolo_obb/data.yaml imgsz=640
  ```

`model predict-voc`
- **功能**: 使用 YOLO 模型對影像或影片進行物件偵測，並將偵測結果儲存為 PASCAL VOC (.xml) 格式。
- **使用範例**:
  ```bash
  python main.py model predict-voc -i ./images_folder -o ./output -m yolov5s.pt
  ```

`convert cvat-to-yolo-seg`
- **功能**: 將 CVAT 導出的多邊形分割 (Segmentation) 標註檔 (.xml) 轉換為 YOLOv8 分割格式，並自動切分資料集與產生 `data.yaml`。
- **使用範例**:
  ```bash
  python main.py convert cvat-to-yolo-seg --xml annotations.xml --img-dir ./images --out-dir ./yolo_seg_dataset
  ```

`convert yolo-to-cvat`
- **功能**: 將 YOLO 格式的資料集（包含影像、標籤檔、類別檔）打包成一個 .zip 壓縮檔，以便直接匯入 CVAT 進行標註或審閱。
- **使用範例**:
  ```bash
  python main.py convert yolo-to-cvat --images-dir ./img --labels-dir ./lab --class-names ./classes.txt --output-zip ./cvat_upload.zip
  ```

### 資料集整備與分割 (Dataset Preparation & Splitting)

`dataset count-voc`
- **功能**: 遞迴掃描輸入資料夾中的影像與 PASCAL VOC XML，統計每個 label 的 bbox instance 數量，並顯示影像/XML 配對、缺檔、空標註與解析失敗摘要。
- **使用範例**:
  ```bash
  python main.py dataset count-voc -i ./voc_dataset

  # 只掃描資料夾第一層
  python main.py dataset count-voc -i ./voc_dataset --no-recursive
  ```

`dataset sample`
- **功能**: 統計來源資料夾中各第一層子資料夾的影像數量，依其占總資料量的比例計算加權抽樣數量，隨機抽取後複製到輸出資料夾。計算值小於 1 時至少抽 1 張，其餘採四捨五入；輸出會保留子資料夾結構。
- **使用範例**:
  ```bash
  python main.py dataset sample -i ./class_images -o ./sampled_images -n 100 --seed 42
  ```

> 各子資料夾會獨立套用「至少 1 張」與四捨五入規則，因此實際輸出總數可能與 `-n` 有些微差異，CLI 會列出完整配置與實際數量。

`dataset split` (推薦)
- **功能**: 功能強大的資料集分割工具，可將影像和對應的標註檔（支援 .xml, .txt 等）按比例分割為訓練、驗證、測試集。支援傳統模式與 COCO-like (images/labels 分離) 模式。
- **使用範例**:
  ```bash
  python main.py dataset split -s ./source_data -o ./split_dataset -a .xml --val --coco
  ```

`dataset filter`
- **功能**: 根據指定的類別名稱，遞迴地篩選 PASCAL VOC 資料集。您可以選擇只保留包含特定標籤的影像，或排除包含某些標籤的影像。
- **使用範例**:
  ```bash
  # 只保留 'dog' 標籤，並排除任何包含 'person' 的影像
  python main.py dataset filter -i ./raw_dataset -o ./filtered_dogs --label dog --exclude person
  ```

使用同一個資料集篩選入口即可保留已標註的 VOC 或 CVAT 資料。

```bash
python main.py dataset filter -i ./voc_dataset -o ./labeled_only --labeled-only
python main.py dataset filter --format cvat --cvat-xml ./annotations.xml -i ./images -o ./labeled_only --labeled-only
```

### 影像擴增 (Image Augmentation)

使用 `augment annotated` 處理 VOC、CVAT 或 YOLO 水平框。YOLO 水平框的 TXT 每行含類別 ID 與四個框參數。四角點標註使用 `augment obb`。純影像使用 `augment image-only`。

#### JSON 擴增配方

內建配方在 [configs/augmentation_profiles.json](configs/augmentation_profiles.json)，包含 `smart_camera`、`hz_pde`、`hz_pde_traditional` 和 `connector`。可用 `--config` 載入專案配方。檔案只需填寫要新增或覆寫的配方。同名配方會替換整個步驟陣列。

| 區塊 | 適用範圍 |
| --- | --- |
| `shared` | 供三種擴增模式共用的光照、色彩、模糊、雜訊或壓縮處理 |
| `annotated` | VOC、CVAT 與 YOLO 水平框。內建 `rotation` 的陣列定義旋轉後的附加處理 |
| `image_only` | 純影像使用 `--image_only_profile` 選擇配方 |
| `obb` | OBB 的拍攝效果。內建 `mixed` 會先套用保留四角點的幾何處理 |

`shared` 與 `obb` 僅接受不改變座標的步驟。旋轉角度仍由命令列指定。若同一配方寫在 `shared`，其他區塊須使用不同的名稱。每個步驟有 `transform` 名稱及對應參數，`p` 為 0 到 1 的機率，`OneOf` 可用 `transforms` 列出候選步驟。程式會在寫入輸出前檢查配方。

例如將以下內容存為 `project_profiles.json`，可建立三種模式共用的亮度配方。

```json
{
  "shared": {
    "soft_light": [
      {
        "transform": "RandomBrightnessContrast",
        "brightness_limit": 0.05,
        "contrast_limit": 0.08,
        "p": 1.0
      }
    ]
  }
}
```

```bash
python main.py augment annotated -i ./images -l ./labels -o ./aug --config ./project_profiles.json -a soft_light
python main.py augment obb -i ./images -l ./obb_labels -o ./obb_aug --config ./project_profiles.json -a soft_light
python main.py augment image-only -i ./images -o ./image_aug --config ./project_profiles.json --image_only_profile soft_light
```

`augment obb`
- **功能**: 使用 Albumentations 2.x 同步擴增影像與 **YOLO OBB 四角點標註**。每行格式為 `class_id x1 y1 x2 y2 x3 y3 x4 y4`，座標以影像寬高正規化，四個頂點須沿框的邊依序排列。支援水平/垂直翻轉、亮度對比、色調飽和度、混合擴增與旋轉；透過 keypoints 保留四角點，旋轉與等比縮放時會擴張畫布。補邊為灰色 `(114, 114, 114)`，輸出影像尺寸可能改變。
- **CVAT 越界標註修正**: 輸入允許負值或大於 1 的角點；擴增前依所有框的最外側角點向外取整補邊，同步平移所有角點，依輸出尺寸重新正規化至 `[0, 1]`，保留原 OBB 的角度與像素尺寸。`-a original` 僅做必要補邊，不做其他擴增；影像與標註會一起輸出，不修改來源。此修正不會逐點截斷座標，也不會重建畫面外的物體內容。
- **輸出與數量**: 結果分放在輸出目錄的 `images/` 與 `labels/`，包含所選模式的處理結果。`-n` 指定每張原圖、每種擴增的版本數；固定步長模式從 0° 產生至小於 360°，例如步長 45 會產生 8 組，不可與 `-n` 併用。`rotation` 只做旋轉；`mixed` 包含隨機翻轉、等比縮放、旋轉及光照畫質變化。`--rotate_limit` 控制 `mixed`/`rotation` 的隨機旋轉範圍，預設 ±30°。
- **資料處理**: `-l` 省略時在影像同目錄尋找同名 `.txt`。`--recursive` 遞迴掃描並保留相對子目錄，標註目錄須有相同結構。空標註檔或缺少 `.txt` 的影像皆視為背景，會依指定模式及數量擴增，並為每張輸出建立配對的空 `.txt`，無須額外參數。無效或無法讀取的標註、損壞影像會列警告並跳過。既有輸出不覆寫，請使用新的輸出目錄；`--seed` 預設 42，可在相同環境與輸入下重現結果。
- **使用範例**:
  ```bash
  # 僅修正 CVAT 越界 OBB，不做其他擴增
  python main.py augment obb -i ./images -l ./labels -o ./obb_fixed -a original

  # 每張原圖產生 5 組混合擴增（影像與標註分開）
  python main.py augment obb -i ./images -l ./labels -o ./obb_aug -a mixed -n 5

  # 每隔 45° 產生一組，共 8 組（影像與標註同目錄）
  python main.py augment obb -i ./dataset -o ./obb_rotated -a rotation --rotate_step 45

  # 隨機旋轉 ±40°，每張產生 3 組
  python main.py augment obb -i ./images -l ./labels -o ./obb_random -a rotation --rotate_limit 40 -n 3

  # 多種擴增；保留 train/val 等子目錄結構
  python main.py augment obb -i ./images -l ./labels -o ./obb_aug_all -a horizontal_flip vertical_flip brightness_contrast hue_saturation --recursive --seed 42
  ```

`augment annotated`
- **功能**: 支援 PASCAL VOC、YOLO 與 CVAT 標註同步擴增，也可使用保守的純影像擴增管道。`-n` 指定每張原圖的擴增版本數，`--total_count COUNT` 則直接指定整批要產生的擴增總數。`augment image-only` 預設不旋轉，需以 `--image_only_rotate [MAX_ANGLE]` 明確開啟。分類資料夾可用 `--augment_per_folder COUNT` 讓每個第一層子資料夾產生固定張數的擴增影像，輸出不保留原圖。標註資料則可用 `--augment_per_class COUNT` 做類別平衡擴增（詳見下方說明）。Connector 可用 `connector` profile，只模擬拍攝條件變化，不做旋轉、翻轉、透視或剪切，適合 pin row 位移與插錯 pin 檢測。
- **類別平衡擴增 (`--augment_per_class`)**: 依標註內的類別名稱（VOC/CVAT 的 `<name>`、YOLO 的 class id）統計 bbox instance 數，反覆挑選最能補足目前缺口的原圖進行擴增，直到每個類別在輸出中都累積到指定的 instance 數。輸出只含擴增結果、不複製原圖，並會盡量把同一類別的擴增結果分散到不同原圖上（例如某類別只有 10 張原圖、要補到 100 個 instance，則每張各擴增 10 次）。由於一張影像常同時含多個類別，補足稀有類別時常見類別必然會超額，結束時會列出每個類別的「原始 / 輸出 / 目標」對照表。此選項適用於 `augment annotated`，且不能與 `-n` 或 `--total_count` 併用。
- **使用範例**:
  ```bash
  python main.py augment annotated -i ./dataset -o ./augmented_dataset -a horizontal_flip mixed

  # 純影像輕量擴增（不旋轉）
  python main.py augment image-only -i ./images -o ./augmented -n 3

  # 純影像輕量擴增，另外啟用 ±10° 旋轉
  python main.py augment image-only -i ./images -o ./augmented --image_only_rotate 10 -n 3

  # 整批總共產生 100 張擴增影像
  python main.py augment image-only -i ./images -o ./augmented --total_count 100

  # 含標註資料整批總共產生 100 組影像/標籤
  python main.py augment annotated -i ./images -l ./labels -o ./augmented -a mixed --total_count 100

  # 類別平衡擴增：讓 XML 內每個 label 在輸出中都累積到 100 個 bbox instance（輸出不含原圖）
  python main.py augment annotated -i ./dataset -o ./balanced -a mixed --augment_per_class 100

  # 分類資料夾擴增：./class_images/A、B、C 各自產生 200 張擴增影像（輸出不含原圖）
  python main.py augment image-only -i ./class_images -o ./balanced --augment_per_folder 200

  # Connector 純影像擴增：保留 pin row 幾何，不旋轉、不翻轉、不透視
  python main.py augment image-only -i ./connector_images -o ./connector_aug --image_only_profile connector --augment_per_folder 200

  # Connector 標註同步擴增
  python main.py augment annotated -i ./images -l ./labels -o ./connector_aug -a connector -n 5
  ```

### 專用與輔助工具 (Specialized & Utility Tools)

`inspect quality`
- **功能**: 專為電子元件等需要清晰紋理的場景設計的**影像品質分析器**。它不僅僅檢查亮度，更透過分析文字清晰度、細節豐富度、對比度等進階指標，來評估一張影像是否適合作為訓練資料，並給出「可用性分數」。支援對子資料夾進行批次分析與報告生成。
- **使用範例**:
  ```bash
  # 批次分析 'all_components' 下所有子資料夾，並自動產生報告
  python main.py inspect quality --batch-subfolders ./all_components --auto-output
  ```

`inspect ocr`
- **功能**: 專為 AOI (自動光學檢測) 設計的分類工具。它使用 OCR 技術讀取電子元件影像上的文字，自動判斷其正確方向（0/90/180/270度），並將其分類至對應角度的資料夾或 NG 資料夾。
- **使用範例**:
  ```bash
  python main.py inspect ocr -i ./component_images -o ./classified_components -t "TARGET_TEXT"
  ```

`crop center`
- **功能**: 依原圖寬、高的比例擷取中心區域，輸入可為單張影像或資料夾，輸出為指定資料夾。`--ratio` / `-r` 預設 `0.5`，範圍為 `0 < ratio <= 1`；例如 1920×1080 的影像設定 `0.5`，會從中心裁出 960×540（面積約為原圖的 25%）。尺寸無條件捨去至整數，最少保留 1 像素；邊界無法對稱分配時，多出的 1 像素會從右側或下側裁掉。
- **檔案處理**: 支援 JPG、JPEG、PNG、BMP、TIF、TIFF、WebP；會先依 EXIF 校正影像方向，多幀影像僅處理第一幀。輸出保留原檔名與格式。預設僅處理資料夾第一層；加上 `--recursive` 會遞迴掃描並保留子目錄結構。輸出位於輸入資料夾內時會自動排除該輸出資料夾。既有同名輸出須使用 `--overwrite` 才會覆寫，且不允許覆蓋來源影像。損壞影像會列出錯誤並繼續處理其他影像；任一影像失敗時結束碼為 1。
- **使用範例**:
  ```bash
  # 本工具只需要 Pillow（已列於 requirements.txt）
  python -m pip install Pillow

  # 單張影像：寬、高各保留 50%
  python main.py crop center -i ./photo.jpg -o ./cropped -r 0.5

  # 資料夾：寬、高各保留 80%
  python main.py crop center -i ./images -o ./cropped --ratio 0.8

  # 包含所有子資料夾，並保留原目錄結構
  python main.py crop center -i ./images -o ./cropped -r 0.6 --recursive

  # 重新產生並覆寫既有輸出
  python main.py crop center -i ./images -o ./cropped -r 0.6 --recursive --overwrite
  ```

`augment rotate`
- **功能**: 簡單的影像旋轉工具，可將指定資料夾內的所有影像分別旋轉 90、180、270 度，並存放到對應的子資料夾。
- **使用範例**:
  ```bash
  python main.py augment rotate ./input_images ./rotated_images
  ```

---

## 設定值檢查

| 工具 | 目前的設定與建議 |
| --- | --- |
| 影像擴增 | 相機與產品專用的機率、色彩範圍及尺寸已移到 JSON。OBB 的補邊、座標轉換和畫布邊界留在程式中，以維持四角點和影像同步。水平框和純影像的 `--seed` 目前只設定 Python 和 NumPy 的亂數狀態，尚未傳給 Albumentations 的 Compose。 |
| VOC 平衡與資料集分割 | `--target`、`--keep_ratio`、`--split` 和 `--seed` 已能依任務指定。VOC 平衡的 Pillow 亮度、對比乘數仍固定為 `0.7–1.3`，可在有多套重複使用的訓練設定時獨立做成平衡工具的配方。一般分割、VOC 轉換、CVAT 分割及 OBB 整理有不同的分層策略和小樣本行為，應先統一輸出規格再共用分割演算法。 |
| FOV 報表 | 內建 Windows 產線資料路徑，可透過 `--production-path` 指定。部署到多台機器時適合改用環境變數。類別順序寫為 `FOV`、`OK`，應從模型中讀取類別名稱並驗證報表需要的兩個類別。 |
| 影像品質與 AOI OCR | 亮度、中心比例等已有 CLI 參數。品質計分的權重和門檻、OCR 的英文語系仍固定在程式內。跨產線時可將品質評分放入 JSON，OCR 語系適合做成 CLI 參數。OCR 信心度在類別建構式和 CLI 的預設值分別是 `0.6`、`0.25`，應統一來源。 |
| 遮罩工具 | 多邊形的標籤、最小面積和近似係數已有 CLI 參數。上色使用固定 RGB `[32, 175, 98]`，對應單一 CVAT 標籤，也將全部非零像素當作缺陷。多類別專案適合用 JSON 定義像素值與標籤色彩的對應。 |
| 極性檢測模組 | 既有 JSON 已保存門檻、正規化值、金樣本和模型路徑。輸入尺寸 `224×224`、backbone 與產生器預設門檻 `0.7` 應跟訓練產物一起保存。設定中的 `device` 目前未被 Sample 使用，實際裝置由 CUDA 可用性決定。 |
| TFLite 匯出 | 影像尺寸及校正目錄已有 CLI 參數。校正上限為 100 張，ONNX opset 為 11，偵測頭則依特定 NPU 工作流程改寫。校正抽樣與匯出版本適合隨部署保存，偵測頭應用獨立程式與相容性測試維護。 |

## 安裝與需求

建議使用 Python 3.12 或以上版本及虛擬環境。一般工具可依 `pyproject.toml` 或 `requirements.txt` 安裝所需套件，建議以實際要執行的工具為準。擴增配方需要 Albumentations 2.x。

- 框選介面需要 PyQt5 和桌面環境。
- AOI OCR 需要 PaddleOCR 與相容的 PaddlePaddle 執行環境。
- FOV 報表需要 pandas、Matplotlib、模型權重及產線影像。
- TFLite 匯出使用 [pt_convert_tflite/requirements.txt](pt_convert_tflite/requirements.txt) 的獨立相依套件。

## 通用使用方式

```bash
python main.py
python main.py <群組> --help
python main.py <群組> <工具> --help
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
