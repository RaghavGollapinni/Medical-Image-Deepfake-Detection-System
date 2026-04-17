import torch
import torch.nn as nn
import torchvision.models as models

class SharedDenseNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SharedDenseNet, self).__init__()
        # Load DenseNet-121
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        densenet = models.densenet121(weights=weights)
        
        # We need the feature maps for the Localization Head (U-Net decoder)
        # DenseNet feature blocks:
        self.features = densenet.features
        
        # Output features dim from DenseNet121 is 1024
        self.output_dim = 1024

    def forward(self, x):
        # x is assumed to be (B, 1, 224, 224) if grayscale.
        # Repeat to 3 channels if necessary for the pretrained weights
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Extract features and intermediate maps for the decoder
        # We will manually extract intermediate feature maps
        features = []
        
        # densenet.features is a Sequential. We can iterate over its modules to get intermediate outputs.
        # But for U-Net decoder, we specifically want output from blocks (e.g. denseblock1, 2, 3, 4)
        for name, module in self.features.named_children():
            x = module(x)
            # Store intermediate activations for skip connections
            if name in ['relu0', 'transition1', 'transition2', 'transition3', 'norm5']:
                features.append(x)
                
        # Last element of features is the output of the whole feature extractor (after norm5)
        # Apply global average pooling for classification/forgery heads
        pooled_features = nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        
        # Return the 1D global features and the list of 2D spatial maps
        return pooled_features, features
