"""
Global path using ResNeXt50
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from .convnext import convnextv2_base

class ConvNeXtGlobal(nn.Module):
    """Global path using ConvNeXtV2-Base backbone"""
    
    def __init__(self, num_classes=2, pretrained=False):
        super(ConvNeXtGlobal, self).__init__()
        
        # Load ConvNeXtV2-Base model
        # Input: (B, 3, H, W)
        # Output: (B, 1024, H/32, W/32)
        self.backbone = convnextv2_base(in_chans=3)
        
        # Upsampling layer
        # Input: 1024 channels (Base model dim)
        self.upconv = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=num_classes,
            kernel_size=8,
            stride=4,
            padding=2
        )
    
    def forward(self, x):
        """
        Returns:
            output: (B, num_classes, H, W)
        """
        x = self.backbone(x)            # (B, 1024, H/32, W/32)
        x = self.upconv(x)              # (B, num_classes, H/8, W/8)
        x = F.interpolate(
            x, 
            scale_factor=8, 
            mode='bilinear', 
            align_corners=True
        )                         # (B, num_classes, H, W)
        
        return x
        
