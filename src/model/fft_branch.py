import torch
import torch.nn as nn
import torch.fft

class FFTCNN(nn.Module):
    def __init__(self, output_dim=512):
        super(FFTCNN, self).__init__()
        
        # 4-layer CNN to process the magnitude spectrum
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Linear(256, output_dim)
        
    def forward(self, x):
        # x is assumed to be (B, 1, H, W)
        # Compute 2D FFT
        # If input has 3 channels, convert to grayscale first for FFT
        if x.size(1) == 3:
            # simple grayscale conversion: mean over channels
            x = x.mean(dim=1, keepdim=True)
            
        fft_complex = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        
        # Extract magnitude spectrum
        magnitude = torch.abs(fft_complex)
        
        # Log scaling for better numerical stability
        magnitude = torch.log(magnitude + 1e-8)
        
        # Pass through CNN
        features = self.features(magnitude)
        features = features.view(features.size(0), -1)
        
        # Output embedding
        embedding = self.fc(features)
        
        return embedding
