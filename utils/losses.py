
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    
    Supports both binary and multi-class classification using raw logits.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=-100):
        """
        Args:
            alpha (float or list): Weighting factor. 
                                   If float, used for class 1 in binary case (1-alpha for class 0).
                                   If list, weights for each class in multi-class case.
                                   If None, no alpha weighting is applied.
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
            
        Returns:
            Computed focal loss
        """
        if logits.ndim > 2:
            # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W) -> (B*H*W, C)
            c = logits.shape[1]
            logits = logits.view(logits.size(0), c, -1).permute(0, 2, 1).contiguous()
            logits = logits.view(-1, c)
            targets = targets.view(-1)
        
        # Filter ignore_index
        valid_mask = targets != self.ignore_index
        logits = logits[valid_mask]
        targets = targets[valid_mask]
        
        if len(targets) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute log probs
        log_pt = F.log_softmax(logits, dim=1)
        
        # Get log_pt for the true classes
        pt = torch.exp(log_pt)
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal term
        focal_term = (1 - pt).pow(self.gamma)
        
        # Calculate alpha term
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # Binary case simplified or constant weight
                alpha_t = self.alpha # robust logic handles multi-class better if passed as list
                # For proper multi-class alpha, it should be a tensor matching num_classes
                # Here we assume if it's a float it's just a scalar multiplier or we handle it simply
                # If sophisticated alpha needed, pass tensor corresponding to classes
                loss = -alpha_t * focal_term * log_pt
            elif isinstance(self.alpha, (list, tuple, torch.Tensor)):
                if isinstance(self.alpha, (list, tuple)):
                    alpha = torch.tensor(self.alpha, device=logits.device)
                else:
                    alpha = self.alpha.to(logits.device)
                
                alpha_t = alpha.gather(0, targets)
                loss = -alpha_t * focal_term * log_pt
            else:
                 loss = -1 * focal_term * log_pt
        else:
            loss = -1 * focal_term * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
