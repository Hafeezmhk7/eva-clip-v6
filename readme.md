# EVA-CLIP Reproduction with BLIP3-o DiT

A PyTorch implementation for reproducing clean EVA-CLIP embeddings from noisy inputs using a BLIP3-o inspired Diffusion Transformer (DiT) architecture with flow matching.

## Overview

This project implements and validates a BLIP3-o DiT architecture by training it to reproduce clean EVA-CLIP embeddings from noisy versions, conditioned on CLIP embeddings. This serves as an effective way to test if the DiT architecture is implemented correctly.

**Task**: `noisy_eva_embeddings + clip_conditioning → clean_eva_embeddings`

## Key Features

- **Fixed Architecture**: BLIP3-o DiT with 3D RoPE, Grouped-Query Attention, and Sandwich Normalization
- **Rectified Flow Matching**: Modern flow-based generative modeling
- **Comprehensive Evaluation**: Cosine similarity metrics and quality assessments
- **Overfitting Test**: Verify model can learn by overfitting on small dataset
- **Robust Training**: Fixed gradient flow, proper initialization, and numerical stability

## Quick Start

### 1. Installation

```bash
# Clone and setup
git clone <your-repo>
cd eva-reproduction
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Extract embeddings from your image-text dataset
python extract_embeddings.py \
    --input_dir /path/to/images \
    --output_dir ./embeddings \
    --batch_size 32
```

### 3. Train Model

```bash
# Basic training
python train_eva_reproduction.py \
    --chunked_embeddings_dir ./embeddings \
    --output_dir ./checkpoints \
    --batch_size 8 \
    --num_epochs 10

# Overfitting test (recommended first)
python train_eva_reproduction.py \
    --chunked_embeddings_dir ./embeddings \
    --output_dir ./checkpoints \
    --overfit_test_size 10 \
    --batch_size 4 \
    --num_epochs 20
```

### 4. Evaluate Results

```bash
python evaluate_model.py \
    --model_path ./checkpoints \
    --embeddings_dir ./embeddings \
    --output_dir ./evaluation \
    --num_samples 1000
```

## Architecture Details

### BLIP3-o DiT Components
- **3D Rotary Position Embedding (RoPE)**: Spatial-aware positional encoding
- **Grouped-Query Attention**: Efficient multi-head attention with key-value sharing
- **Sandwich Normalization**: RMSNorm before and after each sublayer
- **Adaptive Layer Normalization**: Timestep-conditioned normalization

### Flow Matching
- **Rectified Flow**: Linear interpolation between noise and target
- **Velocity Prediction**: Model predicts `v = target - noise`
- **L2 Normalized Embeddings**: Ensures stable training dynamics

## File Structure

```
├── fixed_model.py              # BLIP3-o DiT implementation
├── fixed_loss.py               # Flow matching loss function
├── fixed_dataset.py            # Data loading and preprocessing
├── fixed_trainer.py            # Training loop and optimization
├── train_eva_reproduction.py   # Main training script
├── evaluate_model.py           # Evaluation script
├── extract_embeddings.py       # Embedding extraction (implement as needed)
└── requirements.txt            # Dependencies
```

## Expected Results

### Success Indicators
- **Loss Decrease**: Training loss should decrease steadily
- **Velocity Similarity**: Should increase from ~0.01 to >0.1
- **EVA Similarity**: Evaluation similarity should reach >0.4 (good), >0.7 (excellent)
- **Overfitting Test**: Should achieve >0.8 similarity on small dataset

### Quality Thresholds
- **>0.7**: High quality reproduction
- **>0.8**: Very high quality reproduction  
- **>0.9**: Excellent quality reproduction

## Troubleshooting

### Common Issues

**Zero Gradients**
- Fixed with proper initialization and gradient flow
- Check `fixed_model.py` for initialization improvements

**NaN/Inf Values**
- L2 normalization with epsilon stability
- Proper timestep clamping in loss function

**Poor Convergence**
- Try overfitting test first with 10-50 samples
- Reduce learning rate or increase warmup steps
- Check data normalization

### Debugging Tips

1. **Start with Overfitting Test**: Use `--overfit_test_size 10` to verify model can learn
2. **Enable Debug Mode**: Use `--debug_mode` for detailed logging
3. **Monitor Gradients**: Check for zero or exploding gradients in logs
4. **Validate Data**: Ensure embeddings are properly normalized

## Configuration Options

### Model Sizes
- `tiny`: 384 dim, 6 layers (for testing)
- `small`: 512 dim, 8 layers
- `base`: 768 dim, 12 layers (recommended)
- `large`: 1024 dim, 16 layers

### Training Modes
- `patch_only`: 256 tokens (16x16 patches)
- `cls_patch`: 257 tokens (CLS + 256 patches)

## Key Fixes Applied

1. **Gradient Flow**: Fixed initialization and attention computation
2. **Data Pipeline**: Corrected input/output handling and normalization
3. **Loss Function**: Improved numerical stability and target computation
4. **Architecture**: Proper BLIP3-o implementation with all components
5. **Training Loop**: Robust error handling and monitoring

## Citation

If you use this code, please cite the relevant papers:
- BLIP3-o (original paper)
- Rectified Flow (flow matching method)

## License

MIT License - see LICENSE file for details.