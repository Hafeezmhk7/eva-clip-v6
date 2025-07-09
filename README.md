# BLIP3-o DiT: Flow Matching for CLIP Embedding Generation

A PyTorch implementation of the BLIP3-o Diffusion Transformer architecture with flow matching for generating CLIP embeddings from EVA-CLIP conditioning. This implementation follows the exact methodology described in the BLIP3-o paper.

## 🚀 Overview

This repository implements:
- **BLIP3-o DiT Model**: NextDiT-based architecture for embedding generation
- **Flow Matching Training**: Velocity prediction with optimal transport paths  
- **EVA-CLIP → CLIP**: Maps 1280-dim EVA features to 768-dim CLIP embeddings
- **64-Token Format**: Compatible with 8×8 grid embeddings
- **HuggingFace Integration**: Custom trainer with full training pipeline
- **Production Ready**: Distributed training, mixed precision, checkpointing

## 📁 Project Structure

```
blip3o-dit/
├── src/
│   ├── __init__.py
│   └── modules/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── blip3o_config.py         # Model & training configurations
│       ├── models/
│       │   ├── __init__.py
│       │   ├── blip3o_dit.py           # Main BLIP3-o DiT model
│       │   └── lumina_nextdit2d.py     # NextDiT backbone
│       ├── losses/
│       │   ├── __init__.py
│       │   └── flow_matching_loss.py   # Flow matching loss implementation
│       ├── datasets/
│       │   ├── __init__.py
│       │   └── blip3o_dataset.py       # Dataset loading utilities
│       ├── trainers/
│       │   ├── __init__.py
│       │   └── blip3o_trainer.py       # Custom HuggingFace trainer
│       └── inference/
│           ├── __init__.py
│           └── blip3o_inference.py     # Inference utilities
├── train_blip3o_dit.py                 # Main training script
├── requirements.txt                    # Project dependencies
└── README.md                          # This file
```

## 🛠️ Installation

1. **Clone and setup environment:**
```bash
git clone <your-repository>
cd eva-clip-v3
conda create -n eva_clip_env python=3.11 -y
conda activate eva_clip_env
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```


## 📊 Data Preparation

To prepare the data, follow these steps:

1. **Download the Dataset**

   Start by downloading the required `.tar` files using the `download_data.py` script located in `src/data_hand`:

   ```bash
   python src/data_hand/download_data.py
   ```

2. **Extract Grid Embeddings**

   Finally, extract the EVA-CLIP and CLIP grid embeddings using the `extract_embeddings_g.py` script from `src/module`:

   ```bash
   python src/module/extract_embeddings_g.py
   ```

   > ⚠️ **Note**: This step is GPU-intensive. It's recommended to run it via a job script on a cluster or machine with GPU support.

This will generate a file at `embeddings/blip3o_grid_embeddings.pkl` containing:

* `eva_blip3o_embeddings`: shape `[N, 64, 1280]` — EVA-CLIP conditioning embeddings
* `clip_blip3o_embeddings`: shape `[N, 64, 768]` — CLIP target embeddings

---






## 🎯 Training



```bash
python train_blip3o_dit.py \
  --embeddings_path path/to/blip3o_grid_embeddings.pkl \
  --output_dir ./checkpoints/blip3o-dit \
  --num_epochs 10 \
  --batch_size 32 \
  --learning_rate 1e-4
```




### Debug Mode

For quick testing with reduced data:
```bash
python train_blip3o_dit.py \
  --embeddings_path path/to/blip3o_grid_embeddings.pkl \
  --output_dir ./debug \
  --debug
```
### Model Architecture

Key BLIP3-o DiT parameters:
- `model_dim`: Hidden dimension (default: 1792)
- `num_layers`: Transformer layers (default: 24) 
- `num_heads`: Attention heads (default: 28)
- `eva_embedding_size`: EVA-CLIP dimension (1280, fixed)
- `in_channels`: CLIP dimension (768, fixed)

### Flow Matching

Core flow matching parameters:
- `sigma_min/max`: Noise schedule range (1e-4, 1.0)
- `prediction_type`: "v_prediction" (recommended) or "epsilon"
- `schedule_type`: "linear" or "cosine"
- `regularization_weight`: Additional regularization (0.0)

### Training

Important training settings:
- `batch_size`: Training batch size (32)
- `learning_rate`: Learning rate (1e-4)
- `gradient_checkpointing`: Memory optimization (recommended)
- `fp16`: Mixed precision training (recommended)
- `eval_split`: Evaluation data fraction (0.1)

## 📈 Monitoring

The training script integrates with Weights & Biases for comprehensive monitoring:

- **Loss Components**: Flow matching loss, regularization loss
- **Quality Metrics**: Cosine similarity, L2 distance
- **Model Statistics**: Output norms, gradient norms
- **Training Progress**: Learning rate, epoch, step



