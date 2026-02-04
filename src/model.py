import torch
import torch.nn as nn
from torchvision import models

class SelfAttention(nn.Module):
    """
    Self-Attention Block for Convolutional Networks.
    Calculates the relationship between every pixel and every other pixel.
    """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        # Pointwise convolutions to create Query, Key, and Value matrices
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        
        # Gamma controls how much we listen to the attention map. 
        # Starts at 0, so the model learns to use it gradually.
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        
        # 1. Project Features
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        
        # 2. Calculate Energy (Query * Key)
        energy = torch.bmm(proj_query, proj_key)
        
        # 3. Generate Attention Map
        attention = self.softmax(energy)
        
        # 4. Apply to Values
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        
        # 5. Add back to original input (Residual connection)
        out = self.gamma * out + x
        return out

class MobileNetV2_Attention(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2_Attention, self).__init__()
        
        # 1. Load SOTA lightweight backbone
        backbone = models.mobilenet_v2(weights='DEFAULT')
        
        # 2. Keep the feature extractor, discard the classifier
        self.features = backbone.features 
        
        # 3. Add Self-Attention
        # MobileNetV2 output channels are 1280
        self.attention = SelfAttention(in_dim=1280)
        
        # 4. New Classifier Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.features(x)    # Extract features
        x = self.attention(x)   # Apply Attention (Focus on disease)
        x = self.avgpool(x)     # Pool
        x = torch.flatten(x, 1) # Flatten
        x = self.classifier(x)  # Classify
        return x