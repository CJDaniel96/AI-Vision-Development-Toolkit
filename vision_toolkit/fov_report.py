#!/usr/bin/env python3
"""
Production Line FOV/OK Classification Validator
For processing production data from D:\\Project\\AMR\\sfcTemp with date selection
"""

import os
import shutil
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
from datetime import datetime
import glob
import re

def find_available_models(models_path="models"):
    """掃描 models 資料夾中的可用模型"""
    models_path = Path(models_path)
    if not models_path.exists():
        return []
    
    available_models = []
    
    # 掃描所有日期資料夾
    for model_dir in models_path.iterdir():
        if model_dir.is_dir():
            model_file = model_dir / "best.pt"
            metadata_file = model_dir / "model_info.json"
            
            if model_file.exists():
                # 讀取模型元數據
                metadata = {}
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        print(f"Warning: Could not read metadata for {model_dir.name}: {e}")
                
                available_models.append({
                    'folder_name': model_dir.name,
                    'model_path': str(model_file),
                    'metadata': metadata,
                    'model_size': model_file.stat().st_size,
                    'epochs': metadata.get('epochs', 'Unknown'),
                    'accuracy': metadata.get('accuracy', 'Unknown')
                })
    
    # 按時間戳排序（新到舊）
    available_models.sort(key=lambda x: x['folder_name'], reverse=True)
    
    return available_models

def find_latest_training_model():
    """在分類訓練輸出中尋找最新的模型。"""
    base_path = Path("runs/classify")
    if base_path.exists():
        pattern = str(base_path / "fov_ok_classification*")
        matching_dirs = glob.glob(pattern)
        
        if matching_dirs:
            latest_dir = max(matching_dirs, key=os.path.getmtime)
            best_model_path = Path(latest_dir) / "weights" / "best.pt"
            
            if best_model_path.exists():
                return str(best_model_path)
    
    return None

def display_available_models(available_models):
    """顯示可用模型供使用者選擇"""
    if not available_models:
        print("❌ No models found in models folder!")
        return None
    
    print(f"\n🤖 Available AI Models:")
    print("-" * 80)
    print(f"{'Index':<6} {'Date/Time':<17} {'Epochs':<8} {'Accuracy':<10} {'Size (MB)':<12} {'Path'}")
    print("-" * 80)
    
    for i, model_info in enumerate(available_models):
        size_mb = model_info['model_size'] / (1024 * 1024)
        accuracy_str = f"{model_info['accuracy']:.3f}" if isinstance(model_info['accuracy'], (int, float)) else str(model_info['accuracy'])
        
        print(f"{i+1:<6} {model_info['folder_name']:<17} {model_info['epochs']:<8} "
              f"{accuracy_str:<10} {size_mb:<12.2f} {model_info['model_path']}")
    
    print("-" * 80)
    return available_models

def get_user_model_selection(available_models):
    """取得使用者選擇的模型"""
    while True:
        try:
            print(f"\nSelect model:")
            print(f"  - Enter index number (1-{len(available_models)}) to select specific model")
            print(f"  - Enter 'latest' or '1' to use the latest model")
            print(f"  - Enter 'q' or 'quit' to cancel")
            
            user_input = input("\nYour selection: ").strip().lower()
            
            if user_input in ['q', 'quit']:
                return None
            
            if user_input in ['latest', '']:
                return available_models[0]
            
            # 解析索引
            try:
                index = int(user_input) - 1
                if 0 <= index < len(available_models):
                    return available_models[index]
                else:
                    raise ValueError(f"Invalid index: {index + 1}")
            except ValueError:
                raise ValueError("Please enter a valid number or 'latest'")
                
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
            print("Please try again.")

class ProductionLineValidator:
    """產線 FOV/OK 分類驗證器"""
    
    def __init__(self, model_path, production_path=r"D:\Project\AMR\sfcTemp"):
        """
        初始化產線驗證器
        Args:
            model_path: 訓練好的模型路徑
            production_path: 產線資料路徑
        """
        self.model_path = Path(model_path)
        self.production_path = Path(production_path)
        self.class_names = ['FOV', 'OK']
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        if not self.production_path.exists():
            raise FileNotFoundError(f"Production data path not found: {production_path}")
            
        from ultralytics import YOLO

        # 載入模型
        print(f"Loading model from {model_path}")
        self.model = YOLO(str(model_path))
        print(f"Model loaded successfully")
        print(f"Production data path: {production_path}")

    def scan_available_dates(self):
        """掃描可用的日期資料夾"""
        available_dates = []
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{4}\d{2}\d{2}',    # YYYYMMDD
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'\d{2}\d{2}\d{4}'     # DDMMYYYY
        ]
        
        print(f"\nScanning for date folders in {self.production_path}...")
        
        for item in self.production_path.iterdir():
            if item.is_dir():
                folder_name = item.name
                # 檢查是否符合日期格式
                for pattern in date_patterns:
                    if re.match(pattern, folder_name):
                        # 檢查資料夾內是否有圖片
                        image_count = self._count_images_in_folder(item)
                        if image_count > 0:
                            available_dates.append({
                                'folder_name': folder_name,
                                'path': item,
                                'image_count': image_count,
                                'pattern': pattern
                            })
                        break
        
        # 按資料夾名稱排序（新到舊）
        available_dates.sort(key=lambda x: x['folder_name'], reverse=True)
        
        return available_dates

    def _count_images_in_folder(self, folder_path):
        """計算資料夾中的圖片數量"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        image_count = 0
        
        for ext in image_extensions:
            image_count += len(list(folder_path.glob(ext)))
            # 也檢查子資料夾
            image_count += len(list(folder_path.rglob(ext)))
        
        return image_count

    def display_available_dates(self, available_dates):
        """顯示可用日期供使用者選擇"""
        if not available_dates:
            print("❌ No date folders with images found!")
            return None
        
        print(f"\n📅 Available production dates:")
        print("-" * 60)
        print(f"{'Index':<6} {'Date Folder':<15} {'Images':<8} {'Path'}")
        print("-" * 60)
        
        for i, date_info in enumerate(available_dates):
            print(f"{i+1:<6} {date_info['folder_name']:<15} {date_info['image_count']:<8} {date_info['path']}")
        
        print("-" * 60)
        return available_dates

    def get_user_date_selection(self, available_dates):
        """取得使用者選擇的日期"""
        while True:
            try:
                print(f"\nEnter your choice:")
                print(f"  - Single date: Enter index number (1-{len(available_dates)})")
                print(f"  - Multiple dates: Enter numbers separated by commas (e.g., 1,3,5)")
                print(f"  - All dates: Enter 'all'")
                print(f"  - Cancel: Enter 'q' or 'quit'")
                
                user_input = input("\nYour selection: ").strip().lower()
                
                if user_input in ['q', 'quit']:
                    return None
                
                if user_input == 'all':
                    return available_dates
                
                # 解析選擇的索引
                selected_indices = []
                for part in user_input.split(','):
                    index = int(part.strip()) - 1
                    if 0 <= index < len(available_dates):
                        selected_indices.append(index)
                    else:
                        raise ValueError(f"Invalid index: {index + 1}")
                
                if not selected_indices:
                    raise ValueError("No valid dates selected")
                
                selected_dates = [available_dates[i] for i in selected_indices]
                return selected_dates
                
            except (ValueError, IndexError) as e:
                print(f"❌ Invalid input: {e}")
                print("Please try again.")

    def process_selected_dates(self, selected_dates, conf_threshold=0.25):
        """處理選定的日期資料"""
        all_results = []
        
        print(f"\n🔄 Processing {len(selected_dates)} date(s)...")
        
        for date_info in selected_dates:
            print(f"\n{'='*60}")
            print(f"Processing: {date_info['folder_name']} ({date_info['image_count']} images)")
            print(f"{'='*60}")
            
            # 收集該日期的所有圖片
            images = self._collect_images_from_folder(date_info['path'])
            
            if not images:
                print(f"No images found in {date_info['folder_name']}")
                continue
            
            # 進行預測
            predictions = self._predict_batch(images, conf_threshold)
            
            # 分析結果
            date_analysis = self._analyze_date_predictions(
                date_info['folder_name'], predictions
            )
            
            # 複製圖片和生成報告
            self._process_date_results(date_info['folder_name'], predictions, date_analysis)
            
            all_results.append({
                'date': date_info['folder_name'],
                'analysis': date_analysis,
                'predictions': predictions
            })
        
        # 如果處理多個日期，生成綜合報告
        if len(selected_dates) > 1:
            self._generate_multi_date_summary(all_results)
        
        return all_results

    def _collect_images_from_folder(self, folder_path):
        """收集資料夾中的所有圖片"""
        images = []
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        
        for ext in image_extensions:
            # 直接在資料夾中的圖片
            for img_file in folder_path.glob(ext):
                images.append(img_file)
            
            # 子資料夾中的圖片
            for img_file in folder_path.rglob(ext):
                if img_file not in images:  # 避免重複
                    images.append(img_file)
        
        return images

    def _predict_batch(self, images, conf_threshold):
        """批量預測圖片"""
        predictions = []
        
        print(f"Predicting {len(images)} images...")
        
        for img_path in tqdm(images, desc="Processing"):
            try:
                # 進行預測
                results = self.model(str(img_path), conf=conf_threshold, verbose=False)
                result = results[0]
                
                # 獲取預測結果
                if hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs.data.cpu().numpy()
                    top1_idx = result.probs.top1
                    top1_conf = result.probs.top1conf.item()
                    
                    predicted_class = self.class_names[top1_idx]
                    
                    prediction_info = {
                        'image_path': str(img_path),
                        'filename': img_path.name,
                        'predicted_class': predicted_class,
                        'confidence': top1_conf,
                        'class_probabilities': {
                            self.class_names[i]: float(prob) for i, prob in enumerate(probs)
                        },
                        'prediction_time': datetime.now().isoformat()
                    }
                else:
                    prediction_info = {
                        'image_path': str(img_path),
                        'filename': img_path.name,
                        'predicted_class': 'UNKNOWN',
                        'confidence': 0.0,
                        'class_probabilities': {},
                        'prediction_time': datetime.now().isoformat()
                    }
                
                predictions.append(prediction_info)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                predictions.append({
                    'image_path': str(img_path),
                    'filename': img_path.name,
                    'predicted_class': 'ERROR',
                    'confidence': 0.0,
                    'class_probabilities': {},
                    'error': str(e),
                    'prediction_time': datetime.now().isoformat()
                })
        
        return predictions

    def _analyze_date_predictions(self, date_name, predictions):
        """分析單日預測結果"""
        total_images = len(predictions)
        
        # 統計預測結果
        prediction_counts = {}
        confidences = []
        valid_predictions = []
        
        for pred in predictions:
            pred_class = pred['predicted_class']
            if pred_class not in ['ERROR', 'UNKNOWN']:
                prediction_counts[pred_class] = prediction_counts.get(pred_class, 0) + 1
                confidences.append(pred['confidence'])
                valid_predictions.append(pred)
        
        # 計算統計數據
        fov_count = prediction_counts.get('FOV', 0)
        ok_count = prediction_counts.get('OK', 0)
        valid_count = fov_count + ok_count
        
        fov_rate = (fov_count / valid_count) * 100 if valid_count > 0 else 0
        ok_rate = (ok_count / valid_count) * 100 if valid_count > 0 else 0
        
        analysis = {
            'date': date_name,
            'total_images': total_images,
            'valid_predictions': valid_count,
            'prediction_counts': prediction_counts,
            'fov_count': fov_count,
            'ok_count': ok_count,
            'fov_rate': fov_rate,
            'ok_rate': ok_rate,
            'confidence_stats': {
                'mean': float(np.mean(confidences)) if confidences else 0,
                'std': float(np.std(confidences)) if confidences else 0,
                'min': float(np.min(confidences)) if confidences else 0,
                'max': float(np.max(confidences)) if confidences else 0
            }
        }
        
        return analysis

    def _process_date_results(self, date_name, predictions, analysis):
        """處理單日結果：複製圖片和生成報告"""
        # 創建統一的結果資料夾結構
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 創建AI模型結果根目錄
        ai_result_root = Path("AI_model_FOV_result")
        ai_result_root.mkdir(exist_ok=True)
        
        # 在根目錄下創建具體的結果資料夾
        results_dir = ai_result_root / f"production_results_{date_name}_{timestamp}"
        results_dir.mkdir(exist_ok=True)
        
        # 創建子資料夾
        classified_dir = results_dir / "classified_images"
        fov_dir = classified_dir / "FOV"
        ok_dir = classified_dir / "OK"
        error_dir = classified_dir / "ERROR"
        reports_dir = results_dir / "reports"
        
        fov_dir.mkdir(parents=True, exist_ok=True)
        ok_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Copying images to {results_dir}...")
        
        # 複製圖片到對應資料夾
        for prediction in tqdm(predictions, desc="Copying images"):
            try:
                source_path = Path(prediction['image_path'])
                pred_class = prediction['predicted_class']
                
                if pred_class == 'FOV':
                    dest_dir = fov_dir
                elif pred_class == 'OK':
                    dest_dir = ok_dir
                else:
                    dest_dir = error_dir
                
                dest_path = dest_dir / source_path.name
                shutil.copy2(source_path, dest_path)
                
            except Exception as e:
                print(f"Error copying {prediction['image_path']}: {e}")
        
        # 生成可視化報告
        print("Generating visualizations...")
        try:
            self._generate_daily_visualizations(date_name, analysis, reports_dir)
            print("✅ Visualizations generated successfully")
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
        
        # 保存詳細結果
        print("Saving detailed results...")
        try:
            self._save_daily_results(date_name, analysis, predictions, reports_dir)
            print("✅ Detailed results saved successfully")
        except Exception as e:
            print(f"❌ Error saving detailed results: {e}")
            import traceback
            traceback.print_exc()
        
        # 生成摘要說明
        print("Generating summary text...")
        try:
            self._generate_daily_summary_text(analysis, results_dir)
            print("✅ Summary text generated successfully")
        except Exception as e:
            print(f"❌ Error generating summary text: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"✅ Results for {date_name} saved to: {results_dir}")

    def _generate_daily_visualizations(self, date_name, analysis, reports_dir):
        """生成每日可視化圖表"""
        print(f"Generating visualizations for {date_name}...")
        print(f"Analysis data: valid_predictions={analysis['valid_predictions']}, confidence_mean={analysis['confidence_stats']['mean']}")
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt

            plt.style.use('default')
        except Exception as e:
            print(f"Error setting up matplotlib: {e}")
            return
            return
        
        # 1. 日產圓餅圖
        print("Creating pie chart...")
        try:
            if analysis['valid_predictions'] > 0:
                _, ax = plt.subplots(figsize=(10, 8))
                
                labels = []
                sizes = []
                colors = []
                
                if analysis['fov_count'] > 0:
                    labels.append(f"FOV ({analysis['fov_count']})")
                    sizes.append(analysis['fov_count'])
                    colors.append('#E74C3C')  # 深红色，表示问题
                
                if analysis['ok_count'] > 0:
                    labels.append(f"OK ({analysis['ok_count']})")
                    sizes.append(analysis['ok_count'])
                    colors.append('#27AE60')  # 深绿色，表示正常
                
                print(f"Pie chart data: labels={labels}, sizes={sizes}")
                
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                                 colors=colors, startangle=90, textprops={'fontsize': 12})
                
                ax.set_title(f'AI FOV classification for {date_name}\nTotal Valid: {analysis["valid_predictions"]} images', 
                            fontsize=16, fontweight='bold', pad=20)
                
                plt.tight_layout()
                pie_chart_path = reports_dir / 'daily_pie_chart.png'
                plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Pie chart saved to: {pie_chart_path}")
            else:
                print("⚠️  No valid predictions for pie chart")
        except Exception as e:
            print(f"❌ Error creating pie chart: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. 置信度統計圖
        print("Creating confidence statistics chart...")
        self._create_confidence_statistics_chart(date_name, analysis, reports_dir)
        
        # 3. 品質分布圖
        print("Creating quality distribution chart...")
        self._create_quality_distribution_chart(date_name, analysis, reports_dir)

    def _create_confidence_statistics_chart(self, date_name, analysis, reports_dir):
        """創建置信度統計圖表"""
        try:
            import matplotlib.pyplot as plt

            if analysis['confidence_stats']['mean'] > 0:
                _, ax = plt.subplots(figsize=(10, 6))
                
                # 置信度統計柱狀圖，使用專業藍色系
                stats = analysis['confidence_stats']
                colors = ['#3498DB', '#5DADE2', '#85C1E9', '#AED6F1']  # 藍色系漸變
                
                bars = ax.bar(['Mean', 'Std', 'Min', 'Max'], 
                             [stats['mean'], stats['std'], stats['min'], stats['max']], 
                             color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
                
                ax.set_title(f'AI Model Confidence Statistics - {date_name}', 
                            fontsize=14, fontweight='bold', pad=20)
                ax.set_ylabel('Confidence Score', fontsize=12)
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3, axis='y')
                
                # 在柱狀圖上顯示數值
                for bar, value in zip(bars, [stats['mean'], stats['std'], stats['min'], stats['max']]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
                
                plt.tight_layout()
                stats_path = reports_dir / 'confidence_statistics.png'
                plt.savefig(stats_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Confidence statistics chart saved to: {stats_path}")
            else:
                print("⚠️  No confidence data available for statistics chart")
        except Exception as e:
            print(f"❌ Error creating confidence statistics chart: {e}")
            import traceback
            traceback.print_exc()

    def _create_quality_distribution_chart(self, date_name, analysis, reports_dir):
        """創建品質分布圖表"""
        try:
            import matplotlib.pyplot as plt

            if analysis['valid_predictions'] > 0:
                _, ax = plt.subplots(figsize=(10, 6))
                
                # 品質分布圖，使用一致的配色方案
                quality_data = [analysis['ok_rate'], analysis['fov_rate']]
                quality_labels = ['OK Rate', 'FOV Rate']
                quality_colors = ['#27AE60', '#E74C3C']  # 與餅圖一致的配色
                
                bars = ax.bar(quality_labels, quality_data, color=quality_colors, 
                             alpha=0.8, edgecolor='white', linewidth=1.5)
                
                ax.set_title(f'AI Quality Distribution Analysis - {date_name}', 
                            fontsize=14, fontweight='bold', pad=20)
                ax.set_ylabel('Percentage (%)', fontsize=12)
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3, axis='y')
                
                # 在柱狀圖上顯示數值
                for bar, value in zip(bars, quality_data):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
                
                plt.tight_layout()
                quality_path = reports_dir / 'quality_distribution.png'
                plt.savefig(quality_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Quality distribution chart saved to: {quality_path}")
            else:
                print("⚠️  No valid predictions for quality distribution chart")
        except Exception as e:
            print(f"❌ Error creating quality distribution chart: {e}")
            import traceback
            traceback.print_exc()

    def _save_daily_results(self, date_name, analysis, predictions, reports_dir):
        """保存每日詳細結果"""
        # 保存統計摘要
        summary = {
            'validation_timestamp': datetime.now().isoformat(),
            'date': date_name,
            'model_path': str(self.model_path),
            'total_images': analysis['total_images'],
            'valid_predictions': analysis['valid_predictions'],
            'prediction_counts': analysis['prediction_counts'],
            'fov_count': analysis['fov_count'],
            'ok_count': analysis['ok_count'],
            'fov_rate': analysis['fov_rate'],
            'ok_rate': analysis['ok_rate'],
            'confidence_statistics': analysis['confidence_stats']
        }
        
        with open(reports_dir / 'production_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 保存詳細預測結果
        predictions_data = []
        for pred in predictions:
            pred_data = {
                'filename': pred['filename'],
                'image_path': pred['image_path'],
                'predicted_class': pred['predicted_class'],
                'confidence': pred['confidence'],
                'prediction_time': pred['prediction_time']
            }
            
            # 添加各類別概率
            for class_name, prob in pred.get('class_probabilities', {}).items():
                pred_data[f'{class_name}_probability'] = prob
            
            if 'error' in pred:
                pred_data['error'] = pred['error']
            
            predictions_data.append(pred_data)
        
        import pandas as pd

        df = pd.DataFrame(predictions_data)
        df.to_csv(reports_dir / 'detailed_results.csv', index=False, encoding='utf-8-sig')

    def _generate_daily_summary_text(self, analysis, results_dir):
        """生成每日摘要文字檔案"""
        summary_text = f"""
Production Line Quality Analysis Report
Date: {analysis['date']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

SUMMARY STATISTICS
{'='*60}
Total Images Processed: {analysis['total_images']}
Valid Predictions: {analysis['valid_predictions']}

QUALITY DISTRIBUTION
{'='*60}
OK (Good Quality): {analysis['ok_count']} images ({analysis['ok_rate']:.1f}%)
FOV (Field of View Issues): {analysis['fov_count']} images ({analysis['fov_rate']:.1f}%)

CONFIDENCE STATISTICS
{'='*60}
Average Confidence: {analysis['confidence_stats']['mean']:.3f}
Standard Deviation: {analysis['confidence_stats']['std']:.3f}
Confidence Range: {analysis['confidence_stats']['min']:.3f} - {analysis['confidence_stats']['max']:.3f}

FILE ORGANIZATION
{'='*60}
- classified_images/FOV/: Images predicted as FOV issues
- classified_images/OK/: Images predicted as good quality
- reports/: Analysis reports and visualizations
  - daily_pie_chart.png: Quality distribution pie chart
  - confidence_statistics.png: Confidence statistics chart
  - quality_distribution.png: Quality distribution chart
  - production_summary.json: Detailed statistics
  - detailed_results.csv: Individual prediction results

INTERPRETATION
{'='*60}
FOV Rate: {analysis['fov_rate']:.1f}% - Percentage of images with field of view issues
OK Rate: {analysis['ok_rate']:.1f}% - Percentage of images with good quality

High FOV rates may indicate:
- Camera positioning issues
- Component placement problems
- Process variations

This report was generated by the FOV/OK Classification System V2.
        """.strip()
        
        with open(results_dir / 'README.txt', 'w', encoding='utf-8') as f:
            f.write(summary_text)

    def _generate_multi_date_summary(self, all_results):
        """生成多日期綜合報告"""
        if len(all_results) <= 1:
            return

        import matplotlib.pyplot as plt

        print(f"\nGenerating multi-date summary report...")
        
        # 創建綜合報告資料夾
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 創建AI模型結果根目錄
        ai_result_root = Path("AI_model_FOV_result")
        ai_result_root.mkdir(exist_ok=True)
        
        # 在根目錄下創建綜合報告資料夾
        summary_dir = ai_result_root / f"production_summary_{timestamp}"
        summary_dir.mkdir(exist_ok=True)
        
        # 準備多日期資料
        dates = []
        fov_rates = []
        ok_rates = []
        total_images = []
        
        for result in all_results:
            analysis = result['analysis']
            dates.append(analysis['date'])
            fov_rates.append(analysis['fov_rate'])
            ok_rates.append(analysis['ok_rate'])
            total_images.append(analysis['total_images'])
        
        # 生成趨勢圖
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 品質趨勢圖
        x_pos = range(len(dates))
        width = 0.35
        
        ax1.bar([x - width/2 for x in x_pos], ok_rates, width, label='OK Rate', color='#2ecc71')
        ax1.bar([x + width/2 for x in x_pos], fov_rates, width, label='FOV Rate', color='#e74c3c')
        
        ax1.set_xlabel('Production Date')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Quality Trend Analysis')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(dates, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 產量趨勢圖
        ax2.plot(x_pos, total_images, marker='o', linewidth=2, markersize=8, color='#3498db')
        ax2.set_xlabel('Production Date')
        ax2.set_ylabel('Number of Images')
        ax2.set_title('Production Volume Trend')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(dates, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(summary_dir / 'multi_date_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存綜合統計
        multi_summary = {
            'generation_time': datetime.now().isoformat(),
            'date_range': f"{dates[0]} to {dates[-1]}",
            'total_dates': len(dates),
            'daily_results': [result['analysis'] for result in all_results],
            'overall_statistics': {
                'total_images': sum(total_images),
                'average_fov_rate': float(np.mean(fov_rates)),
                'average_ok_rate': float(np.mean(ok_rates)),
                'fov_rate_std': float(np.std(fov_rates)),
                'ok_rate_std': float(np.std(ok_rates))
            }
        }
        
        with open(summary_dir / 'multi_date_summary.json', 'w', encoding='utf-8') as f:
            json.dump(multi_summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Multi-date summary saved to: {summary_dir}")

def main(argv=None):
    parser = argparse.ArgumentParser(description='Production Line FOV/OK Classification Validator')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained YOLO11 model (default: auto-find latest)')
    parser.add_argument('--production-path', type=str, 
                       default=r'D:\Project\AMR\sfcTemp',
                       help='Path to production data folder (default: D:\\Project\\AMR\\sfcTemp)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for predictions')
    
    args = parser.parse_args(argv)
    
    print("🏭 Production Line FOV/OK Classification Validator")
    print("🤖 Enhanced with Models Folder Integration")
    print("="*60)
    
    try:
        # 如果沒有指定模型路徑，顯示模型選擇界面
        if args.model is None:
            print("Scanning models folder for available models...")
            available_models = find_available_models()
            
            if not available_models:
                print("❌ No models found in models/ folder")
                print("Falling back to search in runs/classify/...")
                args.model = find_latest_training_model()
                if args.model is None:
                    print("❌ No trained model found in runs/classify/ either")
                    print("Pass a trained classification checkpoint with --model.")
                    return
                else:
                    print(f"✅ Found model in runs/classify: {args.model}")
            else:
                # 顯示可用模型
                display_available_models(available_models)
                
                # 讓使用者選擇模型
                selected_model = get_user_model_selection(available_models)
                
                if selected_model is None:
                    print("Operation cancelled by user.")
                    return
                
                args.model = selected_model['model_path']
                print(f"\n✅ Selected model: {selected_model['folder_name']}")
                print(f"Model path: {args.model}")
                if selected_model['metadata']:
                    print(f"Training info: {selected_model['epochs']} epochs, Accuracy: {selected_model['accuracy']}")
        
        # 初始化驗證器
        validator = ProductionLineValidator(args.model, args.production_path)
        
        # 掃描可用日期
        available_dates = validator.scan_available_dates()
        
        # 顯示可用日期
        if not validator.display_available_dates(available_dates):
            return
        
        # 讓使用者選擇日期
        selected_dates = validator.get_user_date_selection(available_dates)
        
        if selected_dates is None:
            print("Operation cancelled by user.")
            return
        
        print(f"\n✅ Selected {len(selected_dates)} date(s) for processing:")
        for date_info in selected_dates:
            print(f"  - {date_info['folder_name']} ({date_info['image_count']} images)")
        
        # 處理選定的日期
        results = validator.process_selected_dates(selected_dates, args.conf)
        
        print(f"\n🎉 Processing completed successfully!")
        print(f"Total dates processed: {len(results)}")
        
        # 顯示處理結果摘要
        for result in results:
            analysis = result['analysis']
            print(f"\n📊 {analysis['date']}:")
            print(f"  Total images: {analysis['total_images']}")
            print(f"  OK: {analysis['ok_count']} ({analysis['ok_rate']:.1f}%)")
            print(f"  FOV: {analysis['fov_count']} ({analysis['fov_rate']:.1f}%)")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise
