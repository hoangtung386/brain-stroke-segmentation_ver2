
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    
    Optimized implementation using nn.CrossEntropyLoss for stability and memory efficiency.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=-100):
        """
        Args:
            alpha (float, list, torch.Tensor): Weighting factor. 
                                   If float, constant weight (or class 1 weight for binary).
                                   If list/tensor, weights for each class.
                                   If None, no alpha weighting.
            gamma (float): Focusing parameter (default: 2.0).
            reduction (str): 'mean', 'sum', or 'none'.
            ignore_index (int): Index to ignore in reduction.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) or (B, C) - Raw unnormalized scores
            targets: (B, H, W) or (B) - Ground truth labels (long)
        """
        # CrossEntropyLoss handles log_softmax internally + optimized
        # We use reduction='none' to get per-element loss, then apply focal term
        
        # Determine alpha
        weight = None
        if self.alpha is not None:
             if isinstance(self.alpha, (list, tuple)):
                 weight = torch.tensor(self.alpha, device=logits.device)
             elif isinstance(self.alpha, torch.Tensor):
                 weight = self.alpha.to(logits.device)
             elif isinstance(self.alpha, (float, int)):
                 # If single float, we can't easily pass it to CE loss as 'weight' 
                 # because CE expects weight to be size C.
                 # We will apply it manually afterwards.
                 pass

        # Use built-in CrossEntropyLoss for numerical stability and memory efficiency
        # reduction='none' gives us -log(pt) for each pixel
        ce_loss_fn = nn.CrossEntropyLoss(
            weight=weight if isinstance(weight, torch.Tensor) else None,
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        ce_loss = ce_loss_fn(logits, targets) # shape: same as targets (B, H, W)
        
        # pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # Focal Term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Final Focal Loss: alpha * focal_term * ce_loss
        # Note: ce_loss already includes alpha if we passed 'weight' to CrossEntropyLoss
        # If alpha was a single float, we apply it here
        loss = focal_term * ce_loss
        
        if self.alpha is not None and isinstance(self.alpha, (float, int)):
             loss = self.alpha * loss

        if self.reduction == 'mean':
            # Handle ignore_index masking implicitly done by CE (it returns 0 for ignored)
            # But we need to divide by valid elements only if we want true mean
            # PyTorch CE reduction='mean' divides by total weights (or counts).
            # Here we just take mean of non-ignored elements if possible, or just .mean() 
            # (which divides by total size including zeros from ignore_index? No, CE sets ignored to 0)
            
            # Use valid mask count for strict correctness if needed, or just .mean()
            # Standard .mean() on the output of CE(reduction='none') includes 0s from ignored indices in the denominator?
            # Actually yes. If we want to be exact matching standard reduction:
            return loss.sum() / ( (targets != self.ignore_index).sum() + 1e-8 )
            
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
