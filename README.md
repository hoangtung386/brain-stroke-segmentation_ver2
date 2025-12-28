# Brain Stroke Segmentation - LCNN Architecture

A deep learning project for brain stroke lesion segmentation using **LCNN (Local-Global Combined Neural Network)**. This project implements a novel **Alignment-First** architecture that combines **SEAN (Symmetry Enhanced Attention Network)** with a **Transformer Bottleneck** for local details and **ConvNeXtV2-Base** for robust global context.

![Architecture Overview](https://img.shields.io/badge/Architecture-LCNN%20%2B%20SEAN%20%2B%20Transformer-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76b900)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/hoangtung386/brain-stroke-lcnn)
[![Acknowledgments](https://img.shields.io/badge/ACKNOWLEDGMENTS-Contributors-orange?style=flat-square&logo=open-source-initiative)](./ACKNOWLEDGMENTS.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE.txt)

## Quick Links 🎯

- 🤗 **[Pre-trained Model on Hugging Face](https://huggingface.co/hoangtung386/brain-stroke-lcnn)**
- 📊 **[Dataset Information](#-data-preparation)**
- 🚀 **[Quick Start Guide](#-installation)**
- 📖 **[Documentation](#-project-structure)**
- 💬 **[Contact & Support](#-contact)**

---

## Model Architecture 🔱

![Architectural Model](./Architectural_model.png)

This project features a significantly enhanced LCNN architecture designed to capture both fine-grained local anomalies and broad semantic context through a "Divide and Conquer" approach.

### Key Innovations

1.  **Alignment-First Strategy** 🔄
    *   Unlike traditional methods that align internal features, our model aligns the **input image slices first**.
    *   An **Alignment Network** predicts affine parameters to align the input with a canonical template, simplifying the task for downstream networks.
    *   **Impact**: Gradient flow is preserved through the alignment process (end-to-end differentiable).

2.  **Dual-Path Processing** 🛤️
    *   **Local Path (SEAN + Transformer)**:
        *   Takes a stack of aligned slices (2T+1 context).
        *   Uses **Encoder-Decoder** with **Symmetry Enhanced Attention** to spot asymmetry between hemispheres.
        *   **NEW**: Integrated **Bottleneck Transformer** to capture long-range dependencies within the local context.
    *   **Global Path (ConvNeXtV2)**:
        *   **NEW**: Replaced ResNeXt50 with **ConvNeXtV2-Base** (~88M params).
        *   Processes the center aligned slice to extract high-level semantic features, reducing false positives.

3.  **Combined Loss Function** 📉
    *   **Segmentation Loss**: Dice Loss + Cross Entropy.
    *   **Alignment Loss**: Symmetry Loss + Edge Consistency + Regularization.

---

## Pre-trained Model 🤗

The trained model is available on Hugging Face Hub for easy download and inference:

**🔗 Model Repository**: [hoangtung386/brain-stroke-lcnn](https://huggingface.co/hoangtung386/brain-stroke-lcnn)

### Download and Load

```python
from huggingface_hub import hf_hub_download
import torch
from models.lcnn import LCNN

# 1. Download best model checkpoint
model_path = hf_hub_download(
    repo_id="hoangtung386/brain-stroke-lcnn",
    filename="best_model.pth"
)

# 2. Initialize Model (New Architecture)
# Note: Ensure you are using the updated code from this repo due to architectural changes
model = LCNN(
    num_channels=1, 
    num_classes=2, 
    T=1, 
    global_impact=0.3, 
    local_impact=0.7
)

# 3. Load Weights
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully!")
```

---

## Project Structure 📂

```bash
brain-stroke-segmentation/
│
├── config.py                 # Central configuration (Hyperparams, Paths)
├── dataset.py                # Dataset and DataLoader (3D slice stacking)
├── trainer.py                # Training loop with Gradient Clipping & AMP
├── train.py                  # Main training entry point
├── evaluate.py               # Evaluation script
├── verify_model.py           # Verification script for architecture
├── verify_grads.py           # Verification script for gradient flow
│
├── models/
│   ├── lcnn.py               # LCNN Orchestrator (Alignment + Split)
│   ├── sean.py               # Local Path (SEAN + Transformer)
│   ├── global_path.py        # Global Path Wrapper
│   ├── convnext.py           # ConvNeXtV2 Implementation
│   ├── transformer.py        # Transformer Modules
│   └── components.py         # Shared blocks (AlignmentNet, Encoders)
│
├── utils/
│   ├── improved_alignment_loss.py  # Symmetry & Edge Consistency Loss
│   ├── visualization.py      # Plotting tools
│   └── metrics.py            # Dice, IoU metrics
│
├── checkpoints/              # Saved models
└── outputs/                  # Training logs and artifacts
```

---

## Installation 🚀

### Option 1: Auto Setup (Recommended)

```bash
git clone https://github.com/hoangtung386/brain-stroke-segmentation.git
cd brain-stroke-segmentation
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup (Conda)

```bash
conda create -n medDiffSeg_env python=3.11 -y
conda activate medDiffSeg_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install monai pandas numpy matplotlib scikit-learn seaborn tqdm wandb huggingface_hub
```

---

## Data Preparation 📊

Organize your data as follows:

```text
data/
├── images/
│   ├── patient_001/
│   │   ├── 001.png
│   │   └── ...
└── masks/
    ├── patient_001/
        ├── 001.png
        └── ...
```

---

## Training 🤖

To start training with the new architecture:

1.  **Configure**: Edit `config.py` to set paths and hyperparameters (Batch Size, Epochs, Transformer settings).
2.  **Run**:
    ```bash
    python train.py
    ```

**Monitoring**:
The training script includes comprehensive logging:
-   **Tqdm Progress Bars**: Real-time loss and gradient norm tracking.
-   **W&B Integration**: Set `USE_WANDB = True` in config to dashboard your runs.

---

## Citation 📚

If you use this work, please cite:

```bibtex
@software{le_vu_hoang_tung_2026_brain_stroke_lcnn,
  author = {Le Vu Hoang Tung},
  title = {Brain Stroke Segmentation using LCNN with ConvNeXt and Transformer},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/hoangtung386/brain-stroke-segmentation},
  note = {Enhanced architecture with Alignment-First flow}
}
```

---

## Contact ✉️

**Author**: Le Vu Hoang Tung  
**Email**: levuhoangtung1542003@gmail.com  
**GitHub**: [@hoangtung386](https://github.com/hoangtung386)       

---

## Acknowledgments 🤝🏻

Special thanks to the open-source community for providing the tools and frameworks that made this project possible, including PyTorch, MONAI, and Hugging Face.
