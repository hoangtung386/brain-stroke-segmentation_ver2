"""
Fixed LCNN Architecture
Properly combines local path (SEAN) and global path (ResNeXt)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sean import SEAN
from .global_path import ConvNeXtGlobal
from .components import AlignmentNetwork, AdaptiveFusion


class LCNN(nn.Module):
    """
    Lightweight CNN combining local and global paths
    
    Key fixes:
    1. Alignment First approach
    2. ConvNeXtV2 as Global Path
    3. Proper input handling for SEAN
    """
    
    def __init__(self, num_channels=1, num_classes=2, 
                 global_impact=0.3, local_impact=0.7, T=1, input_size=(512, 512)):
        """
        Args:
            num_channels: Input channels per slice
            num_classes: Output classes
            global_impact: Global weight
            local_impact: Local weight
            T: Number of slices
            input_size: Image dimensions for alignment
        """
        super(LCNN, self).__init__()
        
        self.global_impact = global_impact
        self.local_impact = local_impact
        self.T = T
        self.num_classes = num_classes
        
        # Alignment Network (First)
        self.alignment_net = AlignmentNetwork(input_size=input_size)
        
        # Local path (SEAN) - accepts aligned slices
        self.local_path = SEAN(
            in_channels=num_channels, 
            num_classes=num_classes, 
            T=T,
            input_size=input_size
        )
        
        # Adapter to convert grayscale stack to RGB for global path
        # Takes center slice and replicates to 3 channels
        self.to_rgb = nn.Conv2d(num_channels, 3, kernel_size=1, bias=False)
        
        # Initialize to replicate channels
        with torch.no_grad():
            self.to_rgb.weight.fill_(1.0)
        
        # Global path (ConvNeXt)
        self.global_path = ConvNeXtGlobal(
            num_classes=num_classes,
            pretrained=False
        )
        
        # Adaptive Fusion
        self.fusion = AdaptiveFusion(num_classes)
    
    def forward(self, x, return_alignment=False):
        """
        Args:
            x: (B, 2T+1, H, W) - Stack of adjacent grayscale slices
        """
        B, num_slices, H, W = x.shape
        
        # 1. Alignment First (Vectorized)
        # Flatten: (B, num_slices, H, W) -> (B*num_slices, 1, H, W)
        x_flat = x.view(B * num_slices, 1, H, W)
        
        # Alignment (batch processing)
        params = self.alignment_net(x_flat) # (B*T, 3)
        
        # Soft clamping instead of hard clamp
        # Tanh maps to (-1, 1), * 0.2 limits range appropriately
        params = torch.tanh(params) * 0.2
        
        # Apply transform
        aligned_flat, _ = self.alignment_net.apply_transform(x_flat, params)
        aligned_flat = torch.nan_to_num(aligned_flat, nan=0.0)
        
        # Reshape back
        x_aligned = aligned_flat.view(B, num_slices, H, W)
        
        # Store for return
        aligned_slices = [x_aligned[:, i:i+1] for i in range(num_slices)]
        alignment_params = [params.view(B, num_slices, 3)[:, i] for i in range(num_slices)]
        
        # Add channel dim for stack: (B, 1, num_slices, H, W) -> used later as (B, num_slices, H, W)
        # Note: Original code stacked dim 2 for (B, 1, 2T+1, H, W).
        # x_aligned is (B, num_slices, H, W).
        # We need to maintain compatibility with downstream logic.
        # x_aligned_flat needs to be (B, num_slices, H, W) which it is.
        x_aligned = x_aligned.unsqueeze(1) # (B, 1, num_slices, H, W)
        x_aligned_flat = x_aligned.squeeze(1) # (B, 2T+1, H, W)
        
        # 2. Global Path
        center_idx = num_slices // 2
        # Use aligned center slice
        x_center_aligned = x_aligned[:, :, center_idx, :, :] # (B, 1, H, W)
        x_rgb = self.to_rgb(x_center_aligned)
        global_output = self.global_path(x_rgb) 
        
        # 3. Local Path (SEAN)
        # Pass aligned stack
        local_output = self.local_path(x_aligned_flat)
        
        # 4. Combine with Adaptive Fusion
        output = self.fusion(global_output, local_output)
            
        if return_alignment:
            return output, aligned_slices, alignment_params
        
        return output
    
    def get_global_output(self, x):
        """Get only global path output"""
        center_idx = x.size(1) // 2
        x_center = x[:, center_idx, :, :].unsqueeze(1)  # (B, 1, H, W)
        x_rgb = self.to_rgb(x_center)
        return self.global_path(x_rgb)
    
    def get_local_output(self, x):
        """Get only local path output"""
        return self.local_path(x)
