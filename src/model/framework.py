import torch
import torch.nn as nn

# In a real package, imports would be like:
# from src.model.backbone import SharedDenseNet
# from src.model.fft_branch import FFTCNN
# from src.model.heads import DiseaseClassificationHead, ForgeryDetectionHead, LocalizationHead
from .backbone import SharedDenseNet
from .fft_branch import FFTCNN
from .heads import DiseaseClassificationHead, ForgeryDetectionHead, LocalizationHead

class MedicalDeepfakeDetector(nn.Module):
    def __init__(self, config):
        """
        config is a dict/OmegaConf loaded from config.yaml
        """
        super(MedicalDeepfakeDetector, self).__init__()
        
        # Parse configs
        pretrained = config['model']['backbone']['pretrained']
        fft_dim = config['model']['fft_branch']['output_dim']
        cls_hidden_dim = config['model']['classification_head']['hidden_dim']
        num_classes = config['model']['classification_head']['num_classes']
        forgery_hidden_dim = config['model']['forgery_head']['hidden_dim']
        dropout = config['model']['classification_head']['dropout']
        
        # Backbone map
        self.backbone = SharedDenseNet(pretrained=pretrained)
        backbone_dim = self.backbone.output_dim
        
        # Branches & Heads
        self.fft_branch = FFTCNN(output_dim=fft_dim)
        
        self.disease_head = DiseaseClassificationHead(
            input_dim=backbone_dim, 
            hidden_dim=cls_hidden_dim, 
            num_classes=num_classes, 
            dropout=dropout
        )
        
        # Forgery block uses both spatial and frequency features
        forgery_input_dim = backbone_dim + fft_dim
        self.forgery_head = ForgeryDetectionHead(
            input_dim=forgery_input_dim,
            hidden_dim=forgery_hidden_dim,
            dropout=dropout
        )
        
        self.localization_head = LocalizationHead()

        # FFT should see an unnormalized signal. We de-normalize the incoming
        # tensor (which is ImageNet-normalized by the dataset pipeline) before FFT.
        norm_mean = config.get("data", {}).get("normalization", {}).get("mean", [0.485, 0.456, 0.406])
        norm_std = config.get("data", {}).get("normalization", {}).get("std", [0.229, 0.224, 0.225])
        self.fft_expects_normalized_input = True
        self.register_buffer("norm_mean", torch.tensor(norm_mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("norm_std", torch.tensor(norm_std, dtype=torch.float32).view(1, -1, 1, 1))
        
    def forward(self, x):
        """
        x: (B, 1, 224, 224)
        """
        # De-normalize for FFT path to preserve frequency characteristics.
        x_fft = x
        if self.fft_expects_normalized_input and x.size(1) == self.norm_mean.size(1):
            x_fft = (x * self.norm_std) + self.norm_mean

        # 1. Feature Extraction (Spatial)
        pooled_features, spatial_features_list = self.backbone(x)
        
        # 2. Disease Classification (using spatial features)
        disease_logits = self.disease_head(pooled_features)

        # 3. FFT Branch (Frequency)
        fft_features = self.fft_branch(x_fft)

        # Debug: Verify FFT branch is producing non-zero features
        if self.training and torch.rand(1).item() < 0.01:  # 1% of batches
            fft_mean = fft_features.mean().item()
            fft_std = fft_features.std().item()
            fft_max = fft_features.max().item()
            print(f"[FFT Debug] mean={fft_mean:.6f}, std={fft_std:.6f}, max={fft_max:.6f}")

        # 4. Forgery Detection (Fusion of spatial and frequency)
        fused_features = torch.cat([pooled_features, fft_features], dim=1)
        forgery_logits = self.forgery_head(fused_features)
        
        # 5. Localization Head (U-Net decoder)
        localization_logits = self.localization_head(spatial_features_list)
        
        # Return logits. Loss functions will apply BCEWithLogitsLoss
        return {
            "disease_logits": disease_logits,
            "forgery_logits": forgery_logits,
            "localization_logits": localization_logits
        }
