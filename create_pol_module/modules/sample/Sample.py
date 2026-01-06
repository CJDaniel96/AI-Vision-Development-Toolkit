import json
import timm
import torch
import torch.nn as nn
import logging
import cv2
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
from pathlib import Path
from PIL import Image
from torchvision import transforms


class LearnableEdgeLayer(nn.Module):
    """
    可學習的邊緣檢測層，使用深度可分離卷積
    """
    def __init__(self, channels=1280, kernel_size=3, use_laplacian_init=True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        # 深度可分離卷積
        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,  # 深度可分離
            bias=False
        )
        # 可選：使用拉普拉斯核初始化
        if use_laplacian_init:
            self._initialize_with_laplacian()
        # 批次歸一化（可選）
        self.bn = nn.BatchNorm2d(channels)

    def _initialize_with_laplacian(self):
        """使用拉普拉斯核初始化權重"""
        if self.kernel_size == 3:
            # 標準拉普拉斯核
            laplacian_kernel = torch.tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ]).float()
        else:
            # 為其他核大小創建近似的拉普拉斯核
            laplacian_kernel = torch.zeros(self.kernel_size, self.kernel_size)
            center = self.kernel_size // 2
            laplacian_kernel[center, center] = -4
            laplacian_kernel[center-1, center] = 1
            laplacian_kernel[center+1, center] = 1
            laplacian_kernel[center, center-1] = 1
            laplacian_kernel[center, center+1] = 1
        # 擴展到所有通道
        with torch.no_grad():
            self.depthwise_conv.weight.data = laplacian_kernel.unsqueeze(0).unsqueeze(0).repeat(
                self.channels, 1, 1, 1
            )

    def forward(self, x):
        edge_features = self.depthwise_conv(x)
        edge_features = self.bn(edge_features)
        return edge_features


class OrthogonalFusion(nn.Module):
    def __init__(self, input_dim_local=1280, input_dim_global=1280):
        super().__init__()
        if input_dim_global != input_dim_local:
            self.projector = nn.Linear(input_dim_global, input_dim_local)
        else:
            self.projector = nn.Identity()

    def forward(self, local_feat, global_feat):
        B, C_local, H, W = local_feat.shape
        global_feat = self.projector(global_feat)

        global_feat_norm = torch.norm(global_feat, p=2, dim=1, keepdim=True) + 1e-6
        global_unit = global_feat / global_feat_norm
        local_flat = local_feat.view(B, C_local, -1)

        projection = torch.bmm(global_unit.unsqueeze(1), local_flat)
        projection = torch.bmm(global_unit.unsqueeze(2), projection).view(B, C_local, H, W)

        orthogonal_comp = local_feat - projection
        global_map = global_feat.unsqueeze(-1).unsqueeze(-1).expand_as(orthogonal_comp)

        return torch.cat([global_map, orthogonal_comp], dim=1)

class GlobalPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        return torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)


class HOAMV2(nn.Module):
    def __init__(self, backbone_name='efficientnetv2_rw_s', pretrained=False, features_only=True, embedding_size=128) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=features_only)
        local_in_channels = self.backbone.feature_info.channels()[-2]
        global_in_channels = self.backbone.feature_info.channels()[-1]

        self.local_branch_conv = nn.Sequential(
            nn.Conv2d(local_in_channels, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280, 0.001),
            nn.SiLU()
        )
        self.local_edge_layer = LearnableEdgeLayer()
        self.local_branch_attention = nn.MultiheadAttention(embed_dim=1280, num_heads=8)

        self.global_branch = nn.Sequential(
            nn.Conv2d(global_in_channels, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280, 0.001),
            nn.SiLU()
        )
        self.global_pool = GlobalPooling()

        self.orthogonal_fusion = OrthogonalFusion(1280, 2560)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280 * 2, embedding_size),
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        features = self.backbone(x)
        local_features = self.local_branch_conv(features[-2])
        local_features = self.local_edge_layer(local_features)
        B, C, H, W = local_features.shape
        local_flat = local_features.view(B, C, -1).permute(2, 0, 1)
        local_attended, _ = self.local_branch_attention(local_flat, local_flat, local_flat)
        local_features = local_attended.permute(1, 2, 0).view(B, C, H, W)

        global_features = self.global_branch(features[-1])
        global_features = self.global_pool(global_features).view(B, -1)

        fused = self.orthogonal_fusion(local_features, global_features)
        embedding = self.head(fused)
        return embedding


class Sample:
    def __init__(self):
        self.name = "sample"
        self.configs = self._get_configs()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.thresholds = self.configs["thresholds"]
        self.golden_sample_folders = self.configs["golden_sample_folders"]
        self.golden_sample_base_path = Path(self.configs["golden_sample_base_path"])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.configs["mean"], std=self.configs["std"]),
        ])

        self.inference_model = self._create_inference_model(self.configs["model_path"], self.configs["embedding_size"])

    def _get_configs(self):
        """
        Load configuration parameters from 'configs.json'.

        Returns:
            dict: Configuration parameters.
        """
        configs_file = Path(__file__).parent / "configs.json"
        with open(configs_file, "r") as f:
            configs = json.load(f)
        return configs
    
    def _create_inference_model(self, model_path: str, embedding_size: int) -> InferenceModel:
        """
        Load and configure the retrieval model with a match finder.

        Args:
            model_path: Checkpoint filepath.
            embedding_size: Expected size of model embeddings.

        Returns:
            Configured InferenceModel for matching.
        """
        model = HOAMV2(backbone_name='efficientnetv2_rw_s', pretrained=False, features_only=True, embedding_size=embedding_size)
        checkpoint = torch.load(model_path, map_location="cpu")
        weights = checkpoint.get("weights_only", checkpoint)
        model.load_state_dict(weights)
        model.to(self.device).eval()

        matcher = MatchFinder(distance=CosineSimilarity())
        inference_model = InferenceModel(model, match_finder=matcher)
        logging.info("Retrieval model ready.")
        return inference_model

    def preprocess(self, img_path: Path) -> torch.Tensor:
        """
        Load an image from disk and apply preprocessing pipeline.

        Args:
            img_path: Path to image file.

        Returns:
            4D tensor for model inference.

        Raises:
            FileNotFoundError: If the image cannot be read.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        return self.transform(pil).unsqueeze(0).to(self.device)
    
    def _get_golden_sample_path(self, part_number: str, product_name: str, comp_name: str, light: str) -> Path:
        """
        Get the golden sample path based on product_name, comp_name, and light condition.
        
        Args:
            product_name: Product name
            comp_name: Component name  
            light: Light condition ('top' or 'side')
            
        Returns:
            Path to golden sample image
        """
        try:
            # 嘗試找到對應的golden sample
            golden_filename = self.golden_sample_folders[part_number][product_name][comp_name][light]
            golden_path = self.golden_sample_base_path / part_number / product_name / comp_name / golden_filename
            
            if golden_path.exists():
                return golden_path
            else:
                logging.warning(f"Golden sample not found at {golden_path}, using default")
                # 使用預設的golden sample
                default_filename = self.golden_sample_folders[part_number]["Default"]["Default"][light]
                return self.golden_sample_base_path / part_number / "Default" / "Default" / default_filename
                
        except KeyError:
            logging.warning(f"Configuration not found for {product_name}/{comp_name}/{light}, using default")
            return None
    
    def _get_threshold(self, part_number: str, product_name: str, comp_name: str, light: str) -> float:
        """
        Get the threshold for similarity comparison.
        
        Args:
            product_name: Product name
            comp_name: Component name
            light: Light condition ('top' or 'side')
            
        Returns:
            Threshold value
        """
        try:
            return self.thresholds[part_number][product_name][comp_name][light]
        except KeyError:
            logging.warning(f"Threshold not found for {product_name}/{comp_name}/{light}, using default")
            return self.thresholds[part_number]["Default"]["Default"][light]
        except:
            logging.warning(f"Threshold not found for Default/Default, using default 0.8")
            return 0.8
    
    def _compare_similarity(self, query_tensor: torch.Tensor, reference_tensor: torch.Tensor, threshold: float = 0.8) -> float:
        """
        Calculate similarity between query and reference images using the trained model.
        
        Args:
            query_tensor: Preprocessed query image tensor
            reference_tensor: Preprocessed reference image tensor
            
        Returns:
            Similarity score (cosine similarity)
        """
        with torch.no_grad():
            is_match = self.inference_model.is_match(query_tensor, reference_tensor, threshold=threshold)
            
            return is_match
   
    def pol_predict(self, image_path: str, part_number: str, product_name: str, comp_name: str, light: str):
        """
        Predict whether the input image matches the golden sample.
        
        Args:
            image_path: Path to input image
            product_name: Product name
            comp_name: Component name
            light: Light condition ('top' or 'side')
            
        Returns:
            dict: Prediction results including similarity score and pass/fail status
        """
        try:
            # 預處理輸入圖片
            input_path = Path(image_path)
            input_tensor = self.preprocess(input_path)
            
            # 獲取對應的golden sample路徑
            golden_path = self._get_golden_sample_path(part_number, product_name, comp_name, light)
            golden_tensor = self.preprocess(golden_path)
            
            # 獲取閾值
            threshold = self._get_threshold(part_number, product_name, comp_name, light)
            
            # 比對相似程度
            is_match = self._compare_similarity(input_tensor, golden_tensor, threshold)
            
            return is_match
            
        except Exception as e:
            logging.error(f"Error in pol_predict: {str(e)}")
            return False
   
    def predict(self, data):
        """
        Main prediction method that processes data dictionary.
        
        Args:
            data: Dictionary containing side, top, comp_name, product_name
            
        Returns:
            dict: Prediction results for both side and top images
        """
        try:
            side_path = data['side']
            top_path = data['top']
            comp_name = data["comp_name"]
            product_name = data["product_name"]
            part_number = data["part_no"]
            
            # 預測side圖片
            if side_path:
                side_result = self.pol_predict(side_path, part_number, product_name, comp_name, "side")
            
            # 預測top圖片  
            if top_path:
                top_result = self.pol_predict(top_path, part_number, product_name, comp_name, "top")
            
            # 計算整體結果
            if side_result and top_result:
                return 'OK', 'OK'
            elif side_result or top_result:
                return 'NG', 'Polarity NG'
            else:
                return 'NG', 'Polarity NG'
            
        except Exception as e:
            logging.error(f"Error in predict: {str(e)}")
            return 'NG', e