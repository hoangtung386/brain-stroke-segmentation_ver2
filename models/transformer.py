import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    """
    Standard Transformer Block suitable for processing visual tokens
    LayerNorm -> MultiHeadAttention -> LayerNorm -> MLP
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, N, C)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class BottleneckTransformer(nn.Module):
    """
    Transformer module explicitly designed for the bottleneck.
    Processes 3D feature maps (B, C, D, H, W) by flattening spatial dimensions.
    """
    def __init__(self, in_channels, num_heads=8, num_layers=4, embed_dim=1024, spatial_size=None):
        """
        Args:
            in_channels: Input channels (should match embed_dim usually)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            embed_dim: Embedding dimension (internal)
            spatial_size: tuple (D, H, W) or (H, W) or None. 
                          Used for initializing fixed positional embeddings if desired.
                          For flexibility with varying input sizes, we might use learnable encoding
                          interpolated specific to input size, or just no positional encoding if CNN context is strong enough (but PE is recommended).
        """
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Projection if channels don't match
        if in_channels != embed_dim:
            self.input_proj = nn.Conv3d(in_channels, embed_dim, kernel_size=1)
            self.output_proj = nn.Conv3d(embed_dim, in_channels, kernel_size=1)
        else:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()
            
        # Positional embedding
        # We will use a learnable positional embedding that outputs compatible shape
        # For simplicity in 3D, let's use a resizeable P.E. or just rely on relative position biases if we had them.
        # Here, let's stick to a simple learnable embedding up to a max size, and interpolate if needed.
        # Assuming typical bottleneck size 32x32.
        self.max_tokens = 2048 # Plenty for 32x32 + depth
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            out: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # Project input
        if self.in_channels != self.embed_dim:
            x = self.input_proj(x)
            
        # Flatten: (B, C, D, H, W) -> (B, EmbedDim, N) -> (B, N, EmbedDim)
        x_flat = x.view(B, self.embed_dim, -1).permute(0, 2, 1)
        N = x_flat.shape[1]
        
        # Add positional embedding
        # We interpolate pos_embed to match N if necessary not typical for 1D sequence but 
        # here N is spatial. A better way for spatial PE is to keep it spatial.
        # For this implementation, we'll naive interpolate if N != max_tokens, 
        # but realistically, exact resize is better done in 2D/3D. 
        # Given simplified requirements, we'll try to just slice max_tokens if N <= max_tokens 
        # or interpolate if N > max_tokens.
        
        if N <= self.max_tokens:
            pos_emb = self.pos_embed[:, :N, :]
        else:
            # Interpolate (B, Max, C) -> (B, C, Max) -> Resize -> (B, C, N) -> (B, N, C)
            pos_emb = self.pos_embed.permute(0, 2, 1) # (1, C, Max)
            pos_emb = F.interpolate(pos_emb, size=(N,), mode='linear', align_corners=False)
            pos_emb = pos_emb.permute(0, 2, 1)
        
        x_flat = x_flat + pos_emb
        
        # Encoder
        for block in self.blocks:
            x_flat = block(x_flat)
            
        x_flat = self.norm(x_flat)
        
        # Reshape back
        x_out = x_flat.permute(0, 2, 1).view(B, self.embed_dim, D, H, W)
        
        if self.in_channels != self.embed_dim:
            x_out = self.output_proj(x_out)
            
        # Residual connection ideally happens in SEAN
        return x_out
