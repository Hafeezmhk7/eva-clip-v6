# BLIP3-o DiT: Flow Matching for CLIP Embedding Generation

A PyTorch implementation of the BLIP3-o Diffusion Transformer architecture with flow matching for generating CLIP embeddings from EVA-CLIP conditioning. This implementation follows the exact methodology described in the BLIP3-o paper.

## 🚀 Overview

This repository implements:
- **BLIP3-o DiT Model**: NextDiT-based architecture for embedding generation
- **Flow Matching Training**: Velocity prediction with optimal transport paths  
- **EVA-CLIP → CLIP**: Maps 4096-dim EVA features to 1024-dim CLIP embeddings
- **64-Token Format**: Compatible with 8×8 grid embeddings
- **HuggingFace Integration**: Custom trainer with full training pipeline

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

## Embedding Extraction Flow
```mermaid
flowchart TD
    A[Input Image\n3x224x224] --> B[CLIP ViT-L/14]
    A --> C[EVA-CLIP-8B]
    
    B --> D[Patch Extraction\n16x16=256 patches]
    D --> E[Feature Grid\n16x16x1024]
    E --> F[2x2 Avg Pooling]
    F --> G[CLIP BLIP3-o Tokens\n8x8=64 tokens\n1024-dim]
    
    C --> H[Patch Extraction\n16x16=256 patches]
    H --> I[Feature Grid\n16x16x4096]
    I --> J[2x2 Avg Pooling]
    J --> K[EVA BLIP3-o Tokens\n8x8=64 tokens\n4096-dim]

## blip3o dit architecture
```mermaid
flowchart LR
    A[Noisy CLIP Tokens\n64x1024] --> B[Token Embedding\n1024→1792]
    C[Timestep] --> D[Timestep Embedding\n896→1792]
    E[EVA Tokens\n64x4096] --> F[Linear Projection\n4096→1792]
    
    B --> G[DiT Block 1]
    D --> G
    F --> G
    
    G --> H[DiT Block 2]
    H --> I[DiT Block ...]
    I --> J[DiT Block N]
    
    J --> K[LayerNorm]
    K --> L[Output Projection\n1792→1024]
    L --> M[Velocity Prediction\n64x1024]
```

## DiT Block
```mermaid
flowchart TB
    subgraph DiTBlock["DiT Block (Detailed)"]
        direction TB
        A[Input Features] --> Norm1[LayerNorm]
        Norm1 --> SA[Self-Attention\nwith 3D RoPE]
        SA --> Add1[&oplus;]
        A --> Add1
        
        Add1 --> Norm2[LayerNorm]
        Norm2 --> CA[Cross-Attention\nwith EVA]
        CA --> Add2[&oplus;]
        Add1 --> Add2
        
        Add2 --> Norm3[LayerNorm]
        Norm3 --> FFN[Feed-Forward Network]
        FFN --> Add3[&oplus;]
        Add2 --> Add3
        
        Add3 --> Output[Output]
    end
    
    Timestep[Timestep Embedding] --> TimeProj[Time Projection\n768→4608]
    TimeProj --> Chunk[Split into 6 chunks]
    
    Chunk -->|Scale MSA| Norm1
    Chunk -->|Gate MSA| Add1
    Chunk -->|Scale Cross| Norm2
    Chunk -->|Gate Cross| Add2
    Chunk -->|Scale MLP| Norm3
    Chunk -->|Gate MLP| Add3
    
    EVA[Projected EVA Tokens] --> CA
    
    style SA fill:#f0f9ff,stroke:#91d5ff
    style CA fill:#f0f9ff,stroke:#91d5ff
    style FFN fill:#f0f9ff,stroke:#91d5ff
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



