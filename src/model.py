import torch.nn as nn
from torchvision import models

# --- NEW ATTENTION BLOCK (To replace the simple GAP) ---
class SEAttentionBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Global Average Pool equivalent
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SOTA_ThermalModel(nn.Module):
    def __init__(self, num_classes):
        super(SOTA_ThermalModel, self).__init__()
        
        # 1. New SOTA Backbone: ConvNeXt-Tiny
        self.backbone = models.convnext_tiny(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[2].in_features # Features are 768
        
        # Remove the default classifier head
        self.backbone.classifier = nn.Identity() 
        
        # 2. SOTA Head: Attention + New Classifier
        self.attention = SEAttentionBlock(num_features) # Focus on important features!
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3), # Slightly more aggressive dropout for small data
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        
        # Apply Attention BEFORE the final linear layer
        attended_features = self.attention(features)
        
        logits = self.head(attended_features)
        return logits