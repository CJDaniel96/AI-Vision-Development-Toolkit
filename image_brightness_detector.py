import cv2
import numpy as np
import os
import csv
from datetime import datetime
from pathlib import Path
import argparse

class ImageBrightnessDetector:
    def __init__(self, dark_threshold=80, center_region_ratio=0.7, contrast_threshold=20, edge_threshold=0.1):
        """
        初始化亮度偵測器（適用於電子元件圖像）
        
        Args:
            dark_threshold (int): 判定過暗的閾值 (0-255)，預設為80
            center_region_ratio (float): 中心區域比例 (0.1-1.0)，預設為0.7 (70%)
            contrast_threshold (float): 對比度閾值，預設為20
            edge_threshold (float): 邊緣密度閾值，預設為0.1
        """
        self.dark_threshold = dark_threshold
        self.center_region_ratio = center_region_ratio
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        
    def extract_center_region(self, gray_image):
        """
        擷取圖像中心區域
        
        Args:
            gray_image (numpy.ndarray): 灰度圖像
            
        Returns:
            numpy.ndarray: 中心區域的灰度圖像
        """
        height, width = gray_image.shape
        
        # 計算中心區域的尺寸
        center_width = int(width * self.center_region_ratio)
        center_height = int(height * self.center_region_ratio)
        
        # 計算中心區域的起始座標
        start_x = (width - center_width) // 2
        start_y = (height - center_height) // 2
        end_x = start_x + center_width
        end_y = start_y + center_height
        
        # 擷取中心區域
        center_region = gray_image[start_y:end_y, start_x:end_x]
        
        return center_region
    
    def calculate_contrast_and_edges(self, gray_image):
        """
        計算圖像的對比度和邊緣密度
        
        Args:
            gray_image (numpy.ndarray): 灰度圖像
            
        Returns:
            dict: 包含對比度和邊緣資訊的字典
        """
        # 計算對比度（標準差）
        contrast = np.std(gray_image)
        
        # 使用Canny邊緣檢測
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 計算局部對比度（使用拉普拉斯算子）
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        local_contrast = np.var(laplacian)
        
        return {
            'contrast': contrast,
            'edge_density': edge_density,
            'local_contrast': local_contrast
        }
    
    def calculate_advanced_metrics(self, gray_image):
        """
        計算進階圖像品質指標（專為電子元件優化）
        
        Args:
            gray_image (numpy.ndarray): 灰度圖像
            
        Returns:
            dict: 包含進階指標的字典
        """
        # 1. 文字清晰度評估（基於OCR友善度）
        # 使用多尺度邊緣檢測來評估文字清晰度
        edges_fine = cv2.Canny(gray_image, 100, 200)  # 細邊緣（文字）
        edges_coarse = cv2.Canny(gray_image, 50, 100)   # 粗邊緣（結構）
        
        text_clarity = np.sum(edges_fine > 0) / edges_fine.size
        structure_clarity = np.sum(edges_coarse > 0) / edges_coarse.size
        
        # 2. 頻域分析 - 高頻成分代表細節豐富度
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # 計算高頻能量比例
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # 定義高頻區域（距離中心較遠的區域）
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        high_freq_mask = distance > min(h, w) * 0.3
        
        total_energy = np.sum(magnitude_spectrum)
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
        detail_richness = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # 3. 局部標準差分析 - 評估圖像的局部變化
        # 使用滑動窗口計算局部標準差
        kernel = np.ones((5, 5), np.float32) / 25
        mean_filtered = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        squared_diff = (gray_image.astype(np.float32) - mean_filtered) ** 2
        local_variance = cv2.filter2D(squared_diff, -1, kernel)
        local_std = np.sqrt(local_variance)
        local_variation = np.mean(local_std)
        
        # 4. 文字/標記檢測友善度
        # 計算水平和垂直投影的變化，文字區域會有規律的變化
        horizontal_proj = np.sum(gray_image, axis=1)
        vertical_proj = np.sum(gray_image, axis=0)
        
        h_variation = np.std(horizontal_proj) / np.mean(horizontal_proj) if np.mean(horizontal_proj) > 0 else 0
        v_variation = np.std(vertical_proj) / np.mean(vertical_proj) if np.mean(vertical_proj) > 0 else 0
        text_pattern_score = (h_variation + v_variation) / 2
        
        # 5. 動態範圍評估
        dynamic_range = np.max(gray_image) - np.min(gray_image)
        
        return {
            'text_clarity': text_clarity,
            'structure_clarity': structure_clarity,
            'detail_richness': detail_richness,
            'local_variation': local_variation,
            'text_pattern_score': text_pattern_score,
            'dynamic_range': dynamic_range
        }
    
    def calculate_contrast_and_edges(self, gray_image):
        """
        計算圖像的對比度和邊緣密度
        
        Args:
            gray_image (numpy.ndarray): 灰度圖像
            
        Returns:
            dict: 包含對比度和邊緣資訊的字典
        """
        # 計算對比度（標準差）
        contrast = np.std(gray_image)
        
        # 使用Canny邊緣檢測
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 計算局部對比度（使用拉普拉斯算子）
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        local_contrast = np.var(laplacian)
        
        return {
            'contrast': contrast,
            'edge_density': edge_density,
            'local_contrast': local_contrast
        }
    
    def assess_usability_advanced(self, brightness, advanced_metrics):
        """
        使用進階指標評估圖像是否適合作為訓練資料（針對電子元件優化）
        
        Args:
            brightness (float): 平均亮度
            advanced_metrics (dict): 進階圖像品質指標
            
        Returns:
            dict: 包含詳細可用性評估的字典
        """
        # 進階評分系統 (總分100分)
        
        # 1. 文字清晰度 (35分) - 最重要，電子元件的標記文字
        text_score = min(advanced_metrics['text_clarity'] / 0.15 * 35, 35)
        
        # 2. 細節豐富度 (25分) - 基於頻域分析
        detail_score = min(advanced_metrics['detail_richness'] / 0.1 * 25, 25)
        
        # 3. 結構清晰度 (20分) - 元件輪廓和結構
        structure_score = min(advanced_metrics['structure_clarity'] / 0.2 * 20, 20)
        
        # 4. 局部變化度 (10分) - 圖像的細節變化
        variation_score = min(advanced_metrics['local_variation'] / 20 * 10, 10)
        
        # 5. 動態範圍 (10分) - 亮度範圍足夠
        range_score = min(advanced_metrics['dynamic_range'] / 150 * 10, 10)
        
        total_score = text_score + detail_score + structure_score + variation_score + range_score
        
        # 智慧判定邏輯（針對電子元件）
        text_ok = advanced_metrics['text_clarity'] >= 0.08   # 文字清晰度足夠
        detail_ok = advanced_metrics['detail_richness'] >= 0.05  # 細節豐富
        structure_ok = advanced_metrics['structure_clarity'] >= 0.1  # 結構清晰
        brightness_acceptable = brightness >= 30  # 最低亮度要求（很寬鬆）
        
        # 可用性判定：
        # 主要條件：文字清晰 AND (細節豐富 OR 結構清晰) AND 最低亮度
        # 或者：總分超過65分
        is_usable = ((text_ok and (detail_ok or structure_ok) and brightness_acceptable) or 
                    total_score >= 65)
        breakpoint()
        return {
            'is_usable': is_usable,
            'total_score': total_score,
            'text_score': text_score,
            'detail_score': detail_score,
            'structure_score': structure_score,
            'variation_score': variation_score,
            'range_score': range_score,
            'text_ok': text_ok,
            'detail_ok': detail_ok,
            'structure_ok': structure_ok,
            'brightness_acceptable': brightness_acceptable
        }
    
    def calculate_brightness(self, image_path):
        """
        計算單張圖像中心區域的亮度、進階品質指標和可用性
        
        Args:
            image_path (str): 圖像檔案路徑
            
        Returns:
            dict: 包含完整分析資訊的字典
        """
        try:
            # 讀取圖像
            image = cv2.imread(image_path)
            if image is None:
                print(f"警告: 無法讀取圖像 {image_path}")
                return None
            
            # 轉換為灰度圖像
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 計算整體分析
            overall_brightness = np.mean(gray)
            overall_basic = self.calculate_contrast_and_edges(gray)
            
            # 擷取中心區域
            center_region = self.extract_center_region(gray)
            
            # 計算中心區域基本分析
            center_brightness = np.mean(center_region)
            center_basic = self.calculate_contrast_and_edges(center_region)
            
            # 計算進階指標
            center_advanced = self.calculate_advanced_metrics(center_region)
            
            # 使用進階評估系統
            usability = self.assess_usability_advanced(center_brightness, center_advanced)
            
            return {
                'center_brightness': center_brightness,
                'overall_brightness': overall_brightness,
                'center_contrast': center_basic['contrast'],
                'center_edge_density': center_basic['edge_density'],
                'center_local_contrast': center_basic['local_contrast'],
                'overall_contrast': overall_basic['contrast'],
                'overall_edge_density': overall_basic['edge_density'],
                'overall_local_contrast': overall_basic['local_contrast'],
                
                # 進階指標
                'text_clarity': center_advanced['text_clarity'],
                'structure_clarity': center_advanced['structure_clarity'],
                'detail_richness': center_advanced['detail_richness'],
                'local_variation': center_advanced['local_variation'],
                'text_pattern_score': center_advanced['text_pattern_score'],
                'dynamic_range': center_advanced['dynamic_range'],
                
                'usability': usability,
                'center_region_shape': center_region.shape,
                'original_shape': gray.shape
            }
            
        except Exception as e:
            print(f"處理圖像 {image_path} 時發生錯誤: {e}")
            return None
    
    def analyze_images(self, image_paths):
        """
        分析多張圖像的中心區域品質和訓練資料可用性（使用進階評估）
        
        Args:
            image_paths (list): 圖像檔案路徑列表
            
        Returns:
            dict: 分析結果
        """
        center_brightness_values = []
        overall_brightness_values = []
        center_contrast_values = []
        center_edge_density_values = []
        text_clarity_values = []
        detail_richness_values = []
        usability_scores = []
        usable_images = []
        unusable_images = []
        valid_images = []
        invalid_images = []
        image_info = []
        
        print(f"開始進階分析電子元件圖像可用性 (中心區域比例: {self.center_region_ratio*100:.0f}%)...")
        print("-" * 90)
        
        for image_path in image_paths:
            result = self.calculate_brightness(image_path)
            if result is not None:
                center_brightness_values.append(result['center_brightness'])
                overall_brightness_values.append(result['overall_brightness'])
                center_contrast_values.append(result['center_contrast'])
                center_edge_density_values.append(result['center_edge_density'])
                text_clarity_values.append(result['text_clarity'])
                detail_richness_values.append(result['detail_richness'])
                usability_scores.append(result['usability']['total_score'])
                valid_images.append(image_path)
                image_info.append(result)
                
                # 分類可用/不可用圖像
                if result['usability']['is_usable']:
                    usable_images.append(image_path)
                else:
                    unusable_images.append(image_path)
                
                # 顯示詳細資訊
                usability = result['usability']
                print(f"✓ {os.path.basename(image_path)}:")
                print(f"   亮度: {result['center_brightness']:.1f} | 文字清晰: {result['text_clarity']:.3f} | 細節豐富: {result['detail_richness']:.3f}")
                print(f"   評分: {usability['total_score']:.1f}/100 | 可用: {'是' if usability['is_usable'] else '否'}")
                
                # 顯示判定原因
                reasons = []
                if usability['text_ok']:
                    reasons.append("文字清晰")
                if usability['detail_ok']:
                    reasons.append("細節豐富")
                if usability['structure_ok']:
                    reasons.append("結構清晰")
                if usability['brightness_acceptable']:
                    reasons.append("亮度可接受")
                
                if reasons:
                    print(f"   優勢: {', '.join(reasons)}")
                else:
                    print(f"   問題: 文字模糊且細節不足")
                    
            else:
                invalid_images.append(image_path)
                print(f"✗ {os.path.basename(image_path)}: 讀取失敗")
        
        if not center_brightness_values:
            return {
                'status': 'error',
                'message': '沒有成功讀取任何圖像',
                'valid_images': 0,
                'invalid_images': len(invalid_images)
            }
        
        # 計算統計資料
        average_center_brightness = np.mean(center_brightness_values)
        average_overall_brightness = np.mean(overall_brightness_values)
        average_center_contrast = np.mean(center_contrast_values)
        average_edge_density = np.mean(center_edge_density_values)
        average_text_clarity = np.mean(text_clarity_values)
        average_detail_richness = np.mean(detail_richness_values)
        average_usability_score = np.mean(usability_scores)
        
        # 統計可用性
        usable_count = len(usable_images)
        usable_percentage = (usable_count / len(valid_images)) * 100
        
        # 傳統亮度判定（作為對比）
        dark_center_count = sum(1 for b in center_brightness_values if b < self.dark_threshold)
        
        return {
            'status': 'success',
            'center_region_ratio': self.center_region_ratio,
            'dark_threshold': self.dark_threshold,
            'contrast_threshold': self.contrast_threshold,
            'edge_threshold': self.edge_threshold,
            
            # 基本統計
            'total_images': len(valid_images),
            'usable_images_count': usable_count,
            'unusable_images_count': len(unusable_images),
            'usable_percentage': usable_percentage,
            
            # 亮度統計
            'average_center_brightness': average_center_brightness,
            'average_overall_brightness': average_overall_brightness,
            'min_center_brightness': min(center_brightness_values),
            'max_center_brightness': max(center_brightness_values),
            'std_center_brightness': np.std(center_brightness_values),
            
            # 進階品質指標統計
            'average_text_clarity': average_text_clarity,
            'average_detail_richness': average_detail_richness,
            'average_center_contrast': average_center_contrast,
            'average_edge_density': average_edge_density,
            'average_usability_score': average_usability_score,
            'min_contrast': min(center_contrast_values),
            'max_contrast': max(center_contrast_values),
            'min_edge_density': min(center_edge_density_values),
            'max_edge_density': max(center_edge_density_values),
            
            # 傳統判定（僅亮度）
            'dark_center_images_count': dark_center_count,
            'dark_center_images_percentage': (dark_center_count / len(valid_images)) * 100,
            'is_center_dark': average_center_brightness < self.dark_threshold,
            
            # 圖像列表和詳細資料
            'center_brightness_values': center_brightness_values,
            'overall_brightness_values': overall_brightness_values,
            'center_contrast_values': center_contrast_values,
            'center_edge_density_values': center_edge_density_values,
            'text_clarity_values': text_clarity_values,
            'detail_richness_values': detail_richness_values,
            'usability_scores': usability_scores,
            'valid_images': valid_images,
            'invalid_images': invalid_images,
            'usable_images': usable_images,
            'unusable_images': unusable_images,
            'image_info': image_info
        }
    
    def print_results(self, results):
        """
        列印分析結果（電子元件訓練資料可用性 - 進階版）
        
        Args:
            results (dict): 分析結果
        """
        print("\n" + "=" * 90)
        print("電子元件圖像訓練資料可用性分析結果（進階評估）")
        print("=" * 90)
        
        if results['status'] == 'error':
            print(f"❌ 錯誤: {results['message']}")
            return
        
        print(f"📊 總共分析圖像數: {results['total_images']}")
        print(f"🎯 中心區域範圍: {results['center_region_ratio']*100:.0f}%")
        print()
        
        # 訓練資料可用性統計
        print("🎓 訓練資料可用性評估（進階算法）:")
        print(f"   ✅ 可用圖像: {results['usable_images_count']}/{results['total_images']} ({results['usable_percentage']:.1f}%)")
        print(f"   ❌ 不可用圖像: {results['unusable_images_count']}/{results['total_images']} ({100-results['usable_percentage']:.1f}%)")
        print(f"   📈 平均可用性評分: {results['average_usability_score']:.1f}/100")
        print()
        
        # 進階品質指標
        print("🔬 進階品質指標:")
        print(f"   📝 文字清晰度: {results['average_text_clarity']:.3f} (≥0.08為佳)")
        print(f"   🔍 細節豐富度: {results['average_detail_richness']:.3f} (≥0.05為佳)")
        print(f"   📊 平均對比度: {results['average_center_contrast']:.2f}")
        print(f"   ⚡ 邊緣密度: {results['average_edge_density']:.3f}")
        print()
        
        # 亮度統計
        print("💡 亮度統計:")
        print(f"   📈 中心區域平均: {results['average_center_brightness']:.2f}")
        print(f"   📊 亮度範圍: {results['min_center_brightness']:.1f} ~ {results['max_center_brightness']:.1f}")
        print(f"   📈 整體平均: {results['average_overall_brightness']:.2f}")
        print()
        
        # 主要結論
        print("\n" + "-" * 70)
        if results['usable_percentage'] >= 85:
            print("🎉 結論: 圖像品質優秀，非常適合訓練")
        elif results['usable_percentage'] >= 70:
            print("👍 結論: 圖像品質良好，適合訓練")
        elif results['usable_percentage'] >= 50:
            print("⚠️ 結論: 圖像品質一般，建議優化拍攝條件")
        else:
            print("❌ 結論: 圖像品質較差，強烈建議改善拍攝條件")
        
        print(f"📊 進階算法可用性: {results['usable_percentage']:.1f}%")
        
        # 與傳統亮度判定的比較
        traditional_usable = 100 - results['dark_center_images_percentage']
        improvement = results['usable_percentage'] - traditional_usable
        if improvement > 0:
            print(f"🚀 vs 傳統亮度判定: +{improvement:.1f}% 改善（智慧識別更多可用圖像）")
        elif improvement < -5:
            print(f"🔍 vs 傳統亮度判定: {improvement:.1f}% （更嚴格的品質要求）")
        else:
            print(f"📊 vs 傳統亮度判定: {improvement:.1f}% 差異")
        
        # 顯示整體亮度比較
        brightness_diff = results['average_center_brightness'] - results['average_overall_brightness']
        if brightness_diff > 0:
            print(f"💡 中心區域比整體亮 {brightness_diff:.2f} 點")
        elif brightness_diff < 0:
            print(f"🔦 中心區域比整體暗 {abs(brightness_diff):.2f} 點")
        else:
            print("⚖️ 中心區域與整體亮度相近")
        
        if results['invalid_images']:
            print(f"\n⚠️ 無法讀取的圖像: {len(results['invalid_images'])} 張")
    
    def export_batch_results_to_csv(self, batch_results, output_file):
        """
        將多個子資料夾的分析結果輸出到CSV檔案（使用進階評估指標）
        
        Args:
            batch_results (list): 多個分析結果的列表
            output_file (str): 輸出CSV檔案路徑
        """
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 準備進階評估的關鍵欄位標題
            headers = [
                '子資料夾名稱',
                '分析時間',
                '總圖像數',
                '可用圖像數',
                '可用率(%)',
                '可用性評分',
                '文字清晰度',
                '細節豐富度',
                '平均亮度',
                '平均對比度',
                '結果'
            ]
            
            # 寫入CSV檔案
            with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                
                # 為每個子資料夾寫入一行資料
                for subfolder_name, results in batch_results:
                    if results['status'] == 'error':
                        # 錯誤情況的處理
                        values = [
                            subfolder_name,
                            current_time,
                            0,
                            0,
                            '0.0',
                            '0.0',
                            '0.000',
                            '0.000',
                            '0.0',
                            '0.0',
                            'ERROR'
                        ]
                    else:
                        values = [
                            subfolder_name,
                            current_time,
                            results['total_images'],
                            results['usable_images_count'],
                            f"{results['usable_percentage']:.1f}",
                            f"{results['average_usability_score']:.1f}",
                            f"{results['average_text_clarity']:.3f}",
                            f"{results['average_detail_richness']:.3f}",
                            f"{results['average_center_brightness']:.1f}",
                            f"{results['average_center_contrast']:.1f}",
                            'EXCELLENT' if results['usable_percentage'] >= 85 else 
                            'GOOD' if results['usable_percentage'] >= 70 else
                            'FAIR' if results['usable_percentage'] >= 50 else 'POOR'
                        ]
                    
                    writer.writerow(values)
                
                # 計算總體統計
                successful_results = [r[1] for r in batch_results if r[1]['status'] == 'success']
                if successful_results:
                    total_images = sum(r['total_images'] for r in successful_results)
                    total_usable = sum(r['usable_images_count'] for r in successful_results)
                    overall_usable_rate = (total_usable / total_images * 100) if total_images > 0 else 0
                    
                    # 計算加權平均指標
                    weighted_text_clarity = sum(r['average_text_clarity'] * r['total_images'] for r in successful_results) / total_images
                    weighted_detail_richness = sum(r['average_detail_richness'] * r['total_images'] for r in successful_results) / total_images
                    weighted_brightness = sum(r['average_center_brightness'] * r['total_images'] for r in successful_results) / total_images
                    weighted_contrast = sum(r['average_center_contrast'] * r['total_images'] for r in successful_results) / total_images
                    weighted_score = sum(r['average_usability_score'] * r['total_images'] for r in successful_results) / total_images
                    
                    # 添加總計行
                    writer.writerow([])  # 空行
                    writer.writerow([
                        '=== 總計 ===',
                        current_time,
                        total_images,
                        total_usable,
                        f"{overall_usable_rate:.1f}",
                        f"{weighted_score:.1f}",
                        f"{weighted_text_clarity:.3f}",
                        f"{weighted_detail_richness:.3f}",
                        f"{weighted_brightness:.1f}",
                        f"{weighted_contrast:.1f}",
                        'EXCELLENT' if overall_usable_rate >= 85 else 
                        'GOOD' if overall_usable_rate >= 70 else
                        'FAIR' if overall_usable_rate >= 50 else 'POOR'
                    ])
            
            print(f"✅ 子資料夾批量分析結果已輸出到: {output_file}")
            print(f"📊 CSV格式: {len(batch_results)} 個子資料夾 + 總計行（進階評估版）")
            return True
            
        except Exception as e:
            print(f"❌ 輸出CSV檔案時發生錯誤: {e}")
            return False

def get_image_files(directory):
    """
    獲取目錄中的所有圖像檔案
    
    Args:
        directory (str): 目錄路徑
        
    Returns:
        list: 圖像檔案路徑列表
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    directory_path = Path(directory)
    if directory_path.is_dir():
        for file_path in directory_path.rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))
    
    return sorted(image_files)

def get_subfolders_with_images(root_directory):
    """
    獲取根目錄下所有包含圖像的子資料夾
    
    Args:
        root_directory (str): 根目錄路徑
        
    Returns:
        dict: {子資料夾名稱: [圖像檔案路徑列表]}
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    subfolders_data = {}
    
    root_path = Path(root_directory)
    if not root_path.is_dir():
        return subfolders_data
    
    # 掃描所有子資料夾
    for subfolder in root_path.iterdir():
        if subfolder.is_dir():
            image_files = []
            # 在子資料夾中尋找圖像檔案
            for file_path in subfolder.rglob('*'):
                if file_path.suffix.lower() in image_extensions:
                    image_files.append(str(file_path))
            
            # 如果子資料夾包含圖像，加入字典
            if image_files:
                subfolders_data[subfolder.name] = sorted(image_files)
    
    return subfolders_data

def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='偵測電子元件圖像是否適合作為訓練資料')
    parser.add_argument('input', help='圖像檔案路徑或包含圖像的目錄路徑')
    parser.add_argument('--threshold', '-t', type=int, default=80, 
                       help='亮度閾值 (0-255)，預設為80')
    parser.add_argument('--center-ratio', '-c', type=float, default=0.7,
                       help='中心區域比例 (0.1-1.0)，預設為0.7 (70%%)')
    parser.add_argument('--contrast-threshold', type=float, default=20,
                       help='對比度閾值，預設為20')
    parser.add_argument('--edge-threshold', type=float, default=0.1,
                       help='邊緣密度閾值，預設為0.1')
    parser.add_argument('--files', '-f', nargs='+', 
                       help='指定多個圖像檔案路徑')
    parser.add_argument('--output', '-o', type=str,
                       help='輸出CSV檔案路徑，例如: results.csv')
    parser.add_argument('--auto-output', '-a', action='store_true',
                       help='自動生成CSV檔案名稱 (格式: component_analysis_YYYYMMDD_HHMMSS.csv)')
    parser.add_argument('--batch-subfolders', '-b', action='store_true',
                       help='批量分析子資料夾模式：分析指定目錄下所有子資料夾內的圖像')
    
    args = parser.parse_args()
    
    # 驗證參數
    if not 0.1 <= args.center_ratio <= 1.0:
        print("❌ 錯誤: 中心區域比例必須在 0.1 到 1.0 之間")
        return
    
    if args.contrast_threshold < 0:
        print("❌ 錯誤: 對比度閾值必須大於等於 0")
        return
        
    if not 0 <= args.edge_threshold <= 1:
        print("❌ 錯誤: 邊緣密度閾值必須在 0 到 1 之間")
        return
    
    # 初始化偵測器
    detector = ImageBrightnessDetector(
        dark_threshold=args.threshold, 
        center_region_ratio=args.center_ratio,
        contrast_threshold=args.contrast_threshold,
        edge_threshold=args.edge_threshold
    )
    
    # 檢查是否使用批量子資料夾模式
    if args.batch_subfolders:
        # 批量子資料夾分析模式
        if not os.path.isdir(args.input):
            print(f"❌ 錯誤: 批量模式需要提供目錄路徑，但 '{args.input}' 不是目錄")
            return
        
        # 獲取所有包含圖像的子資料夾
        subfolders_data = get_subfolders_with_images(args.input)
        
        if not subfolders_data:
            print(f"❌ 在目錄 '{args.input}' 中沒有找到包含圖像的子資料夾")
            return
        
        print(f"🔍 找到 {len(subfolders_data)} 個包含圖像的子資料夾")
        
        # 批量分析結果
        batch_results = []
        
        for subfolder_name, image_paths in subfolders_data.items():
            print(f"\n{'='*60}")
            print(f"📁 正在分析子資料夾: {subfolder_name} ({len(image_paths)} 張圖像)")
            print(f"{'='*60}")
            
            # 分析該子資料夾的圖像
            results = detector.analyze_images(image_paths)
            batch_results.append((subfolder_name, results))
            
            # 顯示該子資料夾的結果
            detector.print_results(results)
        
        # 顯示總體統計
        print(f"\n{'='*80}")
        print("📊 批量分析總體統計")
        print(f"{'='*80}")
        
        successful_analyses = [r for r in batch_results if r[1]['status'] == 'success']
        if successful_analyses:
            total_images = sum(r[1]['total_images'] for r in successful_analyses)
            total_usable = sum(r[1]['usable_images_count'] for r in successful_analyses)
            overall_usable_rate = (total_usable / total_images * 100) if total_images > 0 else 0
            
            print(f"📁 成功分析子資料夾數: {len(successful_analyses)}/{len(batch_results)}")
            print(f"📊 總圖像數: {total_images}")
            print(f"✅ 總可用圖像數: {total_usable}")
            print(f"📈 整體可用率: {overall_usable_rate:.1f}%")
            
            # 找出最佳和最差的子資料夾
            best_folder = max(successful_analyses, key=lambda x: x[1]['usable_percentage'])
            worst_folder = min(successful_analyses, key=lambda x: x[1]['usable_percentage'])
            
            print(f"🏆 最佳子資料夾: {best_folder[0]} ({best_folder[1]['usable_percentage']:.1f}%)")
            print(f"⚠️ 最差子資料夾: {worst_folder[0]} ({worst_folder[1]['usable_percentage']:.1f}%)")
        
        # 處理CSV輸出
        if args.output or args.auto_output:
            if args.auto_output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"batch_component_analysis_{timestamp}.csv"
            else:
                output_file = args.output
            
            if not output_file.lower().endswith('.csv'):
                output_file += '.csv'
            
            detector.export_batch_results_to_csv(batch_results, output_file)
    
    else:
        # 原有的單一分析模式
        # 獲取圖像檔案列表
        if args.files:
            # 使用指定的多個檔案
            image_paths = args.files
        elif os.path.isdir(args.input):
            # 從目錄獲取所有圖像檔案
            image_paths = get_image_files(args.input)
            if not image_paths:
                print(f"❌ 在目錄 '{args.input}' 中沒有找到圖像檔案")
                print("💡 提示: 如果要分析子資料夾，請使用 --batch-subfolders 參數")
                return
        elif os.path.isfile(args.input):
            # 單個檔案
            image_paths = [args.input]
        else:
            print(f"❌ 路徑 '{args.input}' 不存在")
            return
        
        print(f"🔍 找到 {len(image_paths)} 張圖像待分析")
        
        # 分析圖像
        results = detector.analyze_images(image_paths)
        
        # 顯示結果
        detector.print_results(results)
        
        # 處理CSV輸出（單一分析模式使用原有方法）
        if args.output or args.auto_output:
            if args.auto_output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"component_analysis_{timestamp}.csv"
            else:
                output_file = args.output
            
            if not output_file.lower().endswith('.csv'):
                output_file += '.csv'
            
            # 為單一分析創建批量格式（方便統一格式）
            batch_results = [('單一分析', results)]
            detector.export_batch_results_to_csv(batch_results, output_file)

# 使用範例
if __name__ == "__main__":
    # 如果直接執行此腳本，可以使用以下範例代碼
    
    # 方法1: 批量分析子資料夾（推薦用法）
    # 假設目錄結構：
    # ./components/
    #   ├── resistors/     (包含電阻圖像)
    #   ├── capacitors/    (包含電容圖像)
    #   ├── ics/          (包含IC圖像)
    #   └── connectors/   (包含連接器圖像)
    
    # detector = ImageBrightnessDetector(
    #     dark_threshold=70,
    #     center_region_ratio=0.7,
    #     contrast_threshold=22,
    #     edge_threshold=0.12
    # )
    # subfolders_data = get_subfolders_with_images("./components")
    # batch_results = []
    # for subfolder_name, image_paths in subfolders_data.items():
    #     print(f"分析子資料夾: {subfolder_name}")
    #     results = detector.analyze_images(image_paths)
    #     batch_results.append((subfolder_name, results))
    # detector.export_batch_results_to_csv(batch_results, "batch_component_analysis.csv")
    
    # 方法2: 單一目錄分析（原有功能）
    # detector = ImageBrightnessDetector(
    #     dark_threshold=80, 
    #     center_region_ratio=0.7,
    #     contrast_threshold=20,
    #     edge_threshold=0.1
    # )
    # image_paths = get_image_files("./single_folder")
    # results = detector.analyze_images(image_paths)
    # detector.print_results(results)
    # detector.export_batch_results_to_csv([('單一分析', results)], "single_analysis.csv")
    
    # 方法3: 針對特定黑色元件的優化參數
    # detector = ImageBrightnessDetector(
    #     dark_threshold=55,      # 更低的亮度要求
    #     center_region_ratio=0.8, # 更集中於中心
    #     contrast_threshold=25,  # 更高的對比度要求
    #     edge_threshold=0.15     # 更高的邊緣清晰度要求
    # )
    
    # 執行命令列介面
    main()