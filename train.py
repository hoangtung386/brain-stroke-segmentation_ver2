"""
Main training script - FIXED VERSION
Use FixedTrainer to resolve validation NaN issues
"""
import os
import sys
import torch
import gc
from monai.utils import set_determinism

from config import Config
from dataset import create_dataloaders
from models.lcnn import LCNN

# Import fixed trainer instead of old one
try:
    from trainer_fixed import FixedTrainer as Trainer
    print("Using FixedTrainer (validation NaN fixes applied)")
except ImportError:
    print("FixedTrainer not found, falling back to original trainer")
    from trainer import Trainer


def validate_config():
    """Validate configuration before training"""
    print("\n" + "="*60)
    print("Configuration Validation")
    print("="*60)
    
    issues = []
    warnings = []
    
    # Check critical parameters
    if Config.BATCH_SIZE > 16:
        warnings.append(f"Large batch size ({Config.BATCH_SIZE}) may cause OOM on RTX 3090")
    
    if Config.LEARNING_RATE > 1e-3:
        warnings.append(f"High learning rate ({Config.LEARNING_RATE}) may cause instability")
    
    if Config.GRAD_CLIP_NORM < 0.5:
        warnings.append(f"Very low gradient clipping ({Config.GRAD_CLIP_NORM}) may allow exploding gradients")
    
    if Config.ALIGNMENT_WEIGHT > 0.5:
        warnings.append(f"High alignment weight ({Config.ALIGNMENT_WEIGHT}) may destabilize training")
    
    # Check data paths
    if not os.path.exists(Config.IMAGE_DIR):
        issues.append(f"Image directory not found: {Config.IMAGE_DIR}")
    
    if not os.path.exists(Config.MASK_DIR):
        issues.append(f"Mask directory not found: {Config.MASK_DIR}")
    
    # Print results
    if issues:
        print("\nCRITICAL ISSUES:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease fix these issues before training.")
        return False
    
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"   - {warning}")
        print("\nYou can proceed, but consider adjusting these parameters.")
    
    print("\nConfiguration validated")
    print("="*60)
    return True


def print_system_info():
    """Print system information"""
    print("\n" + "="*60)
    print("System Information")
    print("="*60)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Memory info
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        
        print(f"GPU Memory:")
        print(f"  Total: {total_mem:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
    
    print("="*60)


def main():
    """Main training function"""
    
    # Print system info first
    print_system_info()
    
    # Validate configuration
    if not validate_config():
        print("\nConfiguration validation failed. Exiting.")
        return 1
    
    # Set seed for reproducibility
    set_determinism(seed=Config.SEED)
    
    # Debug mode
    if Config.DEBUG_MODE:
        torch.autograd.set_detect_anomaly(True)
        print("🔍 Debug mode enabled - anomaly detection ON")
    
    # Create directories
    Config.create_directories()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    
    try:
        train_loader, val_loader = create_dataloaders(Config)
        print(f"Data loaded successfully")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Create model
    print("\n" + "="*60)
    print("Creating Model")
    print("="*60)
    
    try:
        model = LCNN(
            num_channels=Config.NUM_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            global_impact=Config.GLOBAL_IMPACT,
            local_impact=Config.LOCAL_IMPACT,
            T=Config.T
        )
        
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model created")
        print(f"Parameters: {num_params:.2f}M")
        print(f"Global impact: {Config.GLOBAL_IMPACT}")
        print(f"Local impact: {Config.LOCAL_IMPACT}")
    except Exception as e:
        print(f"Error creating model: {e}")
        return 1
    
    # Clean memory before training
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create trainer
    print("\n" + "="*60)
    print("Initializing Trainer")
    print("="*60)
    
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=Config,
            device=device,
            use_wandb=Config.USE_WANDB
        )
        print(f"Trainer initialized")
        print(f"Using: {trainer.__class__.__name__}")
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        return 1
    
    # Start training
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"Total epochs: {Config.NUM_EPOCHS}")
    print(f"Starting from epoch: {trainer.start_epoch}")
    print(f"Best Dice so far: {trainer.best_dice:.4f}")
    print("="*60 + "\n")
    
    try:
        trainer.train(num_epochs=Config.NUM_EPOCHS)
        print("\nTraining completed successfully!")
        return 0
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Checkpoint saved. Resume with: python train.py")
        return 0
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
    