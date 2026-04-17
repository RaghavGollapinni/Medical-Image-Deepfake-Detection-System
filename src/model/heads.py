import torch
import torch.nn as nn
import torch.nn.functional as F

class DiseaseClassificationHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_classes=14, dropout=0.3):
        super(DiseaseClassificationHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # returns logits, will apply sigmoid during loss / inference
        return self.fc(x)

class ForgeryDetectionHead(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=256, dropout=0.3):
        super(ForgeryDetectionHead, self).__init__()
        # Input dim is usually 1024 (DenseNet) + 512 (FFT) = 1536
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # returns logits, will apply sigmoid during loss / inference
        return self.fc(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # Handle potential size mismatch
            if x.size() != skip.size():
                x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class LocalizationHead(nn.Module):
    def __init__(self):
        super(LocalizationHead, self).__init__()
        # Input sizes from DenseNet121 features:
        # norm5: (B, 1024, 7, 7)
        # transition3: (B, 512, 7, 7) -> actually this means no spatial resolution difference, so we skip it.
        # transition2: (B, 256, 14, 14)
        # transition1: (B, 128, 28, 28)
        # relu0: (B, 64, 112, 112)
        # Input image: (B, 1, 224, 224)
        
        # We will decode from norm5
        # Up 1: 7x7 -> 14x14
        self.up1 = UpBlock(1024, 256, 256) # concats with transition2 (256)
        # Up 2: 14x14 -> 28x28
        self.up2 = UpBlock(256, 128, 128)  # concats with transition1 (128)
        # Up 3: 28x28 -> 56x56
        self.up3 = UpBlock(128, 0, 64)     # no skip connection here
        # Up 4: 56x56 -> 112x112
        self.up4 = UpBlock(64, 64, 32)     # concats with relu0 (64)
        # Final up: 112x112 -> 224x224
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, features_list):
        # unpack features
        relu0, trans1, trans2, trans3, norm5 = features_list
        # indices from earlier feature extraction test:
        # 0: relu0
        # 1: transition1
        # 2: transition2
        # 3: transition3
        # 4: norm5
        
        x = norm5 # 1024, 7x7
        
        x = self.up1(x, trans2) # -> 14x14, 256
        x = self.up2(x, trans1) # -> 28x28, 128
        x = self.up3(x)         # -> 56x56, 64
        x = self.up4(x, relu0)  # -> 112x112, 32
        
        x = self.final_up(x)    # -> 224x224, 32
        out = self.out_conv(x)  # -> 224x224, 1
        
        # Returns logits
        return out
