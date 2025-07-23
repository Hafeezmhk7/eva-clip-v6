# BLIP3-o Enhanced Patch-Level DiT: Image-to-Text Translation via Flow Matching
An implementation* of BLIP3-o patch-level Diffusion Transformer (DiT) for image-to-text translation using flow matching. This project implements flexible training with support for both **CLS+patch (257 tokens)** and **patch-only (256 tokens)** modes, with detailed cosine similarity evaluation and overfitting verification.
## 🏗️ Architecture Overview

mermaid
graph TD
    A[Input Images] --> B[EVA-CLIP Encoder]
    A --> C[CLIP ViT Encoder]
    
    B --> D[EVA Features<br/>[B, 257, 4096]]
    C --> E[CLIP Features<br/>[B, 257, 1024]]
    
    D --> F[Cross-Attention<br/>Conditioning]
    E --> G[Flow Matching<br/>Target]
    
    H[Noise<br/>[B, 257, 1024]] --> I[Linear Interpolation<br/>x_t = (1-α)x_0 + αx_1]
    G --> I
    
    I --> J[BLIP3-o DiT Model<br/>12 Layers, 768 Hidden]
    F --> J
    
    J --> K[Velocity Prediction<br/>[B, 257, 1024]]
    
    K --> L[Flow Matching Loss<br/>MSE(v_pred, v_target)]
    G --> L
    
    style J fill:#e1f5fe
    style L fill:#ffebee
    style F fill:#f3e5f5
```


## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd eva-clip-flow-matching/eva-clip-v3

# Create conda environment
conda create -n eva_clip_env python=3.11 -y
conda activate eva_clip_env
run the requirement.txt file

# Install dependencies
pip install torch torchvision transformers datasets
pip install accelerate wandb tqdm pillow safetensors
pip install webdataset opencv-python scikit-learn
pip install matplotlib seaborn plotly pandas scipy
```

### 2. Extract Embeddings with CLS+Patch Support

```bash
# Extract CLS+patch embeddings (257 tokens per image)
python src/modules/extract_embeddings_g.py --include_cls

# OR extract patch-only embeddings (256 tokens per image)
python src/modules/extract_embeddings_g.py

# Verify embeddings
ls -la embeddings/
cat embeddings/*/embeddings_manifest.json
```

### 3. Training Options

#### Option A: CLS+Patch Training (257 tokens)
```bash
# Train with CLS token + 16x16 patches
python train_blip3o_enhanced.py \
    --chunked_embeddings_dir "./embeddings" \
    --output_dir "./checkpoints/cls_patch_training" \
    --training_mode "cls_patch" \
    --max_training_shards 1 \
    --overfitting_test \
    --enable_same_data_eval \
    --enable_detailed_eval
```

#### Option B: Patch-Only Training (256 tokens)
```bash
# Train with 16x16 patches only
python train_blip3o_enhanced.py \
    --chunked_embeddings_dir "./embeddings" \
    --output_dir "./checkpoints/patch_only_training" \
    --training_mode "patch_only" \
    --max_training_shards 1 \
    --overfitting_test \
    --enable_same_data_eval \
    --enable_detailed_eval
```

#### Option C: Full Dataset Training
```bash
# Train on all available shards
python train_blip3o_enhanced.py \
    --chunked_embeddings_dir "./embeddings" \
    --output_dir "./checkpoints/full_training" \
    --training_mode "cls_patch" \
    --num_epochs 5 \
    --batch_size 8
```

### 4. Evaluation and Analysis

```bash
# Comprehensive cosine similarity evaluation
python eval_blip3o_patch_similarity.py \
    --model_path "./checkpoints/cls_patch_training" \
    --chunked_embeddings_dir "./embeddings" \
    --output_dir "./evaluation_results" \
    --training_mode "auto" \
    --num_samples 1000 \
    --same_data_eval \
    --save_plots \
    --save_detailed_results
```

## 📊 Training Modes Comparison

| Feature | CLS+Patch Mode | Patch-Only Mode |
|---------|----------------|------------------|
| **Token Count** | 257 | 256 |
| **Input Format** | [CLS] + 16×16 patches | 16×16 patches only |
| **Token Layout** | [0]=CLS, [1:257]=patches | [0:256]=patches |
| **Global Representation** | Explicit CLS token | Pooled from patches |
| **Spatial Encoding** | 3D RoPE with CLS handling | Standard 3D RoPE |



### Training Configuration

```python
# Enhanced training parameters
training_config = {
    "training_mode": "cls_patch",           # or "patch_only"
    "max_training_shards": 1,               # For overfitting tests
    "num_epochs": 10,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "lr_scheduler": "cosine",
    "enable_same_data_eval": True,
    "enable_detailed_eval": True,
    "prediction_type": "velocity",          # BLIP3-o paper aligned
    "normalize_targets": True
}
```

## 📁 Project Structure

```
eva-clip-flow-matching/eva-clip-v3/
├── src/modules/
│   ├── models/
│   │   ├── __init__.py
│   │   └── blip3o_patch_dit.py              # Flexible DiT model (256/257 tokens)
│   ├── losses/
│   │   ├── __init__.py
│   │   └── blip3o_flow_matching_loss.py     # Pure flow matching loss
│   ├── trainers/
│   │   ├── __init__.py
│   │   └── blip3o_flexible_trainer.py       # Enhanced flexible trainer
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── blip3o_dataset.py                # Flexible dataset (256/257 support)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── blip3o_detailed_evaluator.py     # Comprehensive evaluator
│   ├── config/
│   │   ├── __init__.py
│   │   └── blip3o_config.py                 # Model configurations
│   └── extract_embeddings_g.py             # CLS+patch embedding extraction
├── job_scripts/
│   ├── train_blip3o_enhanced.job            # Enhanced training job
│   └── eval_blip3o_similarity.job           # Evaluation job
├── train_blip3o_enhanced.py                 # Main training script
├── eval_blip3o_patch_similarity.py          # Evaluation script
└── README.md                                # This file
```

## 🧪 Overfitting Verification Workflow

### Step 1: Single Shard Training
```bash
# Train on single shard to verify pipeline
python train_blip3o_enhanced.py \
    --training_mode "cls_patch" \
    --max_training_shards 1 \
    --overfitting_test \
    --num_epochs 10
```

### Step 2: Same-Data Evaluation
```bash
# Evaluate on same training data
python eval_blip3o_patch_similarity.py \
    --model_path "./checkpoints/overfitting_test" \
    --same_data_eval \
    --max_eval_shards 1
```



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## 🔗 Related Work

- [BLIP3-o Paper](https://arxiv.org/abs/your-paper-id)
- [EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP)
- [CLIP](https://github.com/openai/CLIP)
- [Flow Matching](https://arxiv.org/abs/2210.02747)

---


