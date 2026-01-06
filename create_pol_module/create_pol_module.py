import logging
import json
import argparse
from pathlib import Path


def create_module_task(module_name: str, part_number: str, output_dir: str):
    """
    創建可部署的模組，基於 sample 模板

    Args:
        task_id: 任務ID
        module_name: 模組名稱
        part_number: 料號
        output_dir: 輸出目錄路徑
    """
    import shutil
    import random
    from pathlib import Path
    import re

    try:
        logger = logging.getLogger(__name__)

        output_path = Path(output_dir)
        sample_template_path = Path("modules/sample")

        if not sample_template_path.exists():
            raise FileNotFoundError(f"找不到 sample 模板: {sample_template_path}")

        # 創建模組目錄結構
        module_base_path = Path("modules") / module_name
        module_base_path.mkdir(parents=True, exist_ok=True)

        # 複製整個 sample 目錄結構
        for item in sample_template_path.iterdir():
            if item.name in ["Sample.py", "configs.json"]:
                continue  # 這些文件需要特殊處理

            dst_path = module_base_path / item.name
            if item.is_dir():
                shutil.copytree(str(item), str(dst_path), dirs_exist_ok=True)
            else:
                shutil.copy2(str(item), str(dst_path))

        # 複製並修改 Sample.py
        sample_py_src = sample_template_path / "Sample.py"
        module_py_dst = module_base_path / f"{module_name}.py"

        with open(sample_py_src, 'r', encoding='utf-8') as f:
            sample_content = f.read()

        # 修改類名和 self.name
        # 1. 將 class Sample: 改為 class {module_name}:
        sample_content = re.sub(r'class Sample:', f'class {module_name}:', sample_content)

        # 2. 將 self.name = "sample" 改為 self.name = "{module_name}"
        sample_content = re.sub(r'self\.name = "sample"', f'self.name = "{module_name}"', sample_content)

        with open(module_py_dst, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        logger.info(f"創建模組文件: {module_py_dst}")

        # 複製模型文件到正確位置
        model_src = output_path / "model" / "best_model.pt"
        model_dst = module_base_path / "models" / "polarity" / "best.pt"
        model_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(model_src), str(model_dst))
        logger.info(f"複製模型文件: {model_src} -> {model_dst}")

        # 讀取 mean_std.json
        mean_std_file = output_path / "dataset" / "mean_std.json"
        with open(mean_std_file, 'r') as f:
            mean_std_data = json.load(f)

        # 處理 golden samples
        rawdata_dir = output_path / "raw_data"
        golden_sample_folders = {}
        thresholds = {}

        # 掃描 rawdata 目錄
        for class_folder in rawdata_dir.iterdir():
            if not class_folder.is_dir() or class_folder.name == "NG":
                continue

            # 解析文件夾名稱：{product_name}_{comp_name}_{light}
            folder_parts = class_folder.name.split('_')
            if len(folder_parts) < 3:
                continue

            product_name = folder_parts[0]
            comp_name = folder_parts[1]
            light = '_'.join(folder_parts[2:])  # 處理可能包含下劃線的光源名稱

            # 隨機選擇一張圖片
            image_files = [f for f in class_folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            if not image_files:
                continue

            selected_image = random.choice(image_files)

            # 創建目標目錄結構 (使用 part_number)
            golden_sample_path = module_base_path / "data" / "golden_sample" / part_number / product_name / comp_name
            golden_sample_path.mkdir(parents=True, exist_ok=True)

            # 複製選中的圖片
            dst_image_path = golden_sample_path / selected_image.name
            shutil.copy2(str(selected_image), str(dst_image_path))

            # 構建 golden_sample_folders 結構 (使用 part_number 作為 key)
            if part_number not in golden_sample_folders:
                golden_sample_folders[part_number] = {}
            if product_name not in golden_sample_folders[part_number]:
                golden_sample_folders[part_number][product_name] = {}
            if comp_name not in golden_sample_folders[part_number][product_name]:
                golden_sample_folders[part_number][product_name][comp_name] = {}

            golden_sample_folders[part_number][product_name][comp_name][light] = selected_image.name

            # 構建 thresholds 結構 (使用 part_number 作為 key)
            if part_number not in thresholds:
                thresholds[part_number] = {}
            if product_name not in thresholds[part_number]:
                thresholds[part_number][product_name] = {}
            if comp_name not in thresholds[part_number][product_name]:
                thresholds[part_number][product_name][comp_name] = {}

            thresholds[part_number][product_name][comp_name][light] = 0.7  # 預設閾值

        # 創建 configs.json (基於 sample 的結構，但替換相關字段)
        configs = {
            "ai_defect": ["Polarity"],
            "pkg_type": {
                part_number: []  # 使用 part_number 而非 module_name
            },
            "model_path": f"modules/{module_name}/models/polarity/best.pt",
            "embedding_size": 512,
            "thresholds": thresholds,
            "mean": mean_std_data.get("mean", [0, 0, 0]),
            "std": mean_std_data.get("std", [1, 1, 1]),
            "golden_sample_base_path": f"modules/{module_name}/data/golden_sample",
            "golden_sample_folders": golden_sample_folders,
            "device": "cuda"
        }

        configs_file = module_base_path / "configs.json"
        with open(configs_file, 'w', encoding='utf-8') as f:
            json.dump(configs, f, ensure_ascii=False, indent=2)
        logger.info(f"創建配置文件: {configs_file}")

        logger.info(f"模組 {module_name} 創建完成")
        logger.info(f"模組路徑: {module_base_path}")
        logger.info(f"料號: {part_number}")
        logger.info(f"包含 {len(golden_sample_folders.get(part_number, {}))} 個產品的金樣本")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"創建模組 {module_name} 失敗: {str(e)}", exc_info=True)
        raise
    
def opt_parse():
    parser = argparse.ArgumentParser(description="創建可部署的模組，基於 sample 模板")
    parser.add_argument("-m", "--module_name", type=str, required=True, help="模組名稱")
    parser.add_argument("-p", "--part_number", type=str, required=True, help="料號")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="訓練結果的輸出目錄路徑")
    return parser.parse_args()

if __name__ == "__main__":
    opt = opt_parse()
    create_module_task(opt.module_name, opt.part_number, opt.output_dir)