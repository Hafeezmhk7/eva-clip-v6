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
cd blip3o-dit
conda create -n blip3o python=3.11 -y
conda activate blip3o
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "from src.modules.models.blip3o_dit import BLIP3oDiTModel; print('✅ Installation successful')"
```

## 📊 Data Preparation

First, extract your EVA-CLIP and CLIP embeddings using your existing `extract_embeddings_production.py`:

```bash
python extract_embeddings_production.py
```

This should create `embeddings/blip3o_grid_embeddings.pkl` with:
- `eva_blip3o_embeddings`: [N, 64, 1280] - EVA-CLIP conditioning
- `clip_blip3o_embeddings`: [N, 64, 768] - CLIP targets

**Test your dataset:**
```bash
python -c "from src.modules.datasets.blip3o_dataset import test_blip3o_dataset; test_blip3o_dataset('path/to/blip3o_grid_embeddings.pkl')"
```

## 🎯 Training

### Quick Start

```bash
python train_blip3o_dit.py \
  --embeddings_path path/to/blip3o_grid_embeddings.pkl \
  --output_dir ./checkpoints/blip3o-dit \
  --num_epochs 10 \
  --batch_size 32 \
  --learning_rate 1e-4
```

### Full Training Configuration

```bash
python train_blip3o_dit.py \
  --embeddings_path path/to/blip3o_grid_embeddings.pkl \
  --output_dir ./checkpoints/blip3o-dit-large \
  \
  --model_dim 1792 \
  --num_layers 24 \
  --num_heads 28 \
  --gradient_checkpointing \
  \
  --num_epochs 20 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --warmup_steps 2000 \
  \
  --sigma_min 1e-4 \
  --sigma_max 1.0 \
  --prediction_type v_prediction \
  --regularization_weight 0.0 \
  \
  --eval_split 0.1 \
  --normalize_embeddings \
  --num_workers 4 \
  \
  --fp16 \
  --logging_steps 100 \
  --save_steps 1000 \
  --eval_steps 1000 \
  \
  --wandb_project blip3o-experiments \
  --wandb_run_name large-model-v1
```

### Debug Mode

For quick testing with reduced data:
```bash
python train_blip3o_dit.py \
  --embeddings_path path/to/blip3o_grid_embeddings.pkl \
  --output_dir ./debug \
  --debug
```

### Resume Training

```bash
python train_blip3o_dit.py \
  --embeddings_path path/to/blip3o_grid_embeddings.pkl \
  --output_dir ./checkpoints/blip3o-dit \
  --resume_from_checkpoint ./checkpoints/blip3o-dit/checkpoint-5000
```

## 🔮 Inference

### Python API

```python
from src.modules.inference.blip3o_inference import BLIP3oInference
import torch

# Load trained model
inference = BLIP3oInference("./checkpoints/blip3o-dit")

# Generate CLIP embeddings from EVA-CLIP conditioning
eva_embeddings = torch.randn(4, 64, 1280)  # Batch of 4 samples
generated_clip = inference.generate(
    eva_embeddings=eva_embeddings,
    num_inference_steps=50
)

print(f"Generated CLIP embeddings: {generated_clip.shape}")  # [4, 64, 768]
```

### Batch Generation from Dataset

```python
from src.modules.inference.blip3o_inference import BLIP3oInference

inference = BLIP3oInference("./checkpoints/blip3o-dit")

# Generate samples from dataset
results = inference.generate_from_dataset(
    dataset_path="path/to/blip3o_grid_embeddings.pkl",
    num_samples=100,
    batch_size=8,
    num_inference_steps=50,
    output_path="./results/generated_samples.pkl",
    compute_metrics=True
)

print("Generation metrics:", results['generation_metrics'])
```

### Model Evaluation

```python
# Evaluate model with flow matching loss
eval_metrics = inference.evaluate_model(
    dataset_path="path/to/blip3o_grid_embeddings.pkl",
    batch_size=32,
    split="eval"
)

print("Evaluation metrics:", eval_metrics)
```

## ⚙️ Configuration

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
- **Quality Metrics**: Cosine similarity, L2 distance, SNR
- **Model Statistics**: Output norms, gradient norms
- **Training Progress**: Learning rate, epoch, step

Access your runs at `wandb.ai/<your-username>/<project-name>`

## 🔧 Advanced Usage

### Custom Model Configuration

```python
from src.modules.config.blip3o_config import BLIP3oDiTConfig
from src.modules.models.blip3o_dit import BLIP3oDiTModel

# Create custom configuration
config = BLIP3oDiTConfig(
    model_dim=2048,
    num_layers=32,
    num_heads=32,
    gradient_checkpointing=True,
)

# Create model
model = BLIP3oDiTModel(config)
```

### Custom Flow Matching Loss

```python
from src.modules.losses.flow_matching_loss import BLIP3oFlowMatchingLoss

loss_fn = BLIP3oFlowMatchingLoss(
    sigma_min=1e-5,
    sigma_max=2.0,
    prediction_type="v_prediction",
    regularization_weight=0.1,
)
```

### Distributed Training

Use HuggingFace Accelerate for multi-GPU training:

```bash
# Configure distributed setup
accelerate config

# Launch distributed training
accelerate launch train_blip3o_dit.py \
  --embeddings_path path/to/blip3o_grid_embeddings.pkl \
  --output_dir ./checkpoints/blip3o-dit-distributed \
  --batch_size 16 \
  --gradient_accumulation_steps 2
```

## 🚨 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce `--batch_size`
- Increase `--gradient_accumulation_steps`
- Enable `--gradient_checkpointing`
- Use `--fp16` training

**Slow Training:**
- Increase `--num_workers` for data loading
- Use multiple GPUs with Accelerate
- Enable `--fp16` for faster computation

**Poor Generation Quality:**
- Check flow matching loss convergence
- Verify data normalization with `--normalize_embeddings`
- Increase `--num_inference_steps` during generation
- Adjust `--sigma_min/max` parameters

### Debug Mode

For troubleshooting, use debug mode:
```bash
python train_blip3o_dit.py \
  --embeddings_path path/to/blip3o_grid_embeddings.pkl \
  --output_dir ./debug \
  --debug \
  --dry_run  # Just test setup without training
```

### Memory Requirements

- **Training**: 12-16 GB VRAM for batch_size=32
- **Inference**: 4-6 GB VRAM for batch_size=8  
- **Model Size**: ~1.5-3 GB depending on configuration

## 📚 Key Features

✅ **Exact BLIP3-o Implementation**: Follows paper methodology precisely  
✅ **Flow Matching Training**: Proper velocity field prediction  
✅ **Production Ready**: Memory efficient, distributed training support  
✅ **HuggingFace Integration**: Standard training pipeline with custom loss  
✅ **Comprehensive Monitoring**: Detailed metrics and logging  
✅ **Flexible Configuration**: Easy customization of all parameters  
✅ **Quality Inference**: Generation with quality metrics  

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Submit a pull request

## 📄 License

This implementation follows the same open-source principles as the original BLIP3-o research.

## 🙏 Acknowledgments

Based on the BLIP3-o paper and NextDiT architecture. Thanks to the Salesforce Research team for the original implementation and methodology.

---

**Ready to start training your BLIP3-o DiT model? Run the training script and watch the magic happen! ✨**