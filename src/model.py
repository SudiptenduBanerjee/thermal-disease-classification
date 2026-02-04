import torch
import torch.nn as nn
from torchvision import models

class CBAM(nn.Module):
    """ Convolutional Block Attention Module (Spatial + Channel) """
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.channel_attn(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        x = x * self.spatial_attn(spatial)
        return x

class SOTA_Thermal_ConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super(SOTA_Thermal_ConvNeXt, self).__init__()
        # Load ConvNeXt-Tiny (SOTA backbone)
        self.backbone = models.convnext_tiny(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity() 
        
        # Dual-Attention Block
        self.attention = CBAM(num_features)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(num_features),
            nn.Dropout(0.4),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        # Accessing the feature extractor for ConvNeXt
        x = self.backbone.features(x)
        x = self.attention(x)
        x = self.head(x)
        return x